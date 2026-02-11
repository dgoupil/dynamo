// cuda_external.go implements external CUDA checkpoint + restore orchestration.
// It replaces CRIU CUDA plugin hooks by running cuda-checkpoint from the DaemonSet.
package checkpoint

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

type cudaCheckpointAction string

const (
	cudaCheckpointBinary = "/usr/local/sbin/cuda-checkpoint"
	podResourcesSocket   = "/var/lib/kubelet/pod-resources/kubelet.sock"
	nvidiaGPUResource    = "nvidia.com/gpu"
	hostCgroupPath       = "/sys/fs/cgroup"
	cudaCheckpointTO     = 2 * time.Minute
	cudaDiscoverTO       = 30 * time.Second
	cudaDiscoverTick     = 1 * time.Second
	cudaStateCheckTO     = 2 * time.Second
	podGPUDiscoverTO     = 30 * time.Second
	podGPUDiscoverTick   = 1 * time.Second

	cudaActionLock       cudaCheckpointAction = "lock"
	cudaActionCheckpoint cudaCheckpointAction = "checkpoint"
	cudaActionRestore    cudaCheckpointAction = "restore"
	cudaActionUnlock     cudaCheckpointAction = "unlock"
)

func prepareExternalCUDA(ctx context.Context, req CheckpointRequest, sourcePID int, manifest *CheckpointManifest, log *logrus.Entry) error {
	cgroupPath, err := getContainerCgroupPath(sourcePID)
	if err != nil {
		return fmt.Errorf("failed to get source cgroup path: %w", err)
	}

	cgroupPIDs, err := getCgroupPIDs(cgroupPath)
	if err != nil {
		return fmt.Errorf("failed to list source cgroup PIDs: %w", err)
	}

	cudaPIDs := findCUDAPIDs(cgroupPIDs)
	if len(cudaPIDs) == 0 {
		log.Debug("No CUDA PIDs detected in source cgroup, skipping external CUDA preparation")
		return nil
	}

	locked := make([]int, 0, len(cudaPIDs))
	for _, pid := range cudaPIDs {
		if err := runCudaCheckpoint(ctx, pid, cudaActionLock, ""); err != nil {
			unlockCUDAProcesses(context.Background(), locked, log)
			return fmt.Errorf("cuda lock failed for PID %d: %w", pid, err)
		}
		locked = append(locked, pid)
	}

	for _, pid := range cudaPIDs {
		if err := runCudaCheckpoint(ctx, pid, cudaActionCheckpoint, ""); err != nil {
			unlockCUDAProcesses(context.Background(), locked, log)
			return fmt.Errorf("cuda checkpoint failed for PID %d: %w", pid, err)
		}
	}

	sourceGPUUUIDs, err := getPodGPUUUIDsWithRetry(ctx, req.PodName, req.PodNamespace, req.ContainerName, log)
	if err != nil {
		return fmt.Errorf("failed to discover source GPU UUIDs: %w", err)
	}
	if len(sourceGPUUUIDs) == 0 {
		unlockCUDAProcesses(context.Background(), locked, log)
		return fmt.Errorf("no source GPU UUIDs found for %s/%s container %s", req.PodNamespace, req.PodName, req.ContainerName)
	}

	manifest.ExternalRestore = &ExternalRestoreConfig{
		CUDA: &CUDARestoreData{
			PIDs:           cudaPIDs,
			SourceGPUUUIDs: sourceGPUUUIDs,
			Locked:         true,
		},
	}

	log.WithFields(logrus.Fields{
		"cuda_pids":         len(cudaPIDs),
		"source_gpu_uuids":  len(sourceGPUUUIDs),
		"container_cg_path": cgroupPath,
		"source_pod":        req.PodName,
		"source_namespace":  req.PodNamespace,
		"source_container":  req.ContainerName,
	}).Info("Prepared external CUDA checkpoint metadata")
	return nil
}

func unlockExternalCUDA(manifest *CheckpointManifest, log *logrus.Entry) {
	if manifest == nil || manifest.ExternalRestore == nil || manifest.ExternalRestore.CUDA == nil {
		return
	}
	if !manifest.ExternalRestore.CUDA.Locked {
		return
	}
	unlockCUDAProcesses(context.Background(), manifest.ExternalRestore.CUDA.PIDs, log)
}

func unlockCUDAProcesses(ctx context.Context, pids []int, log *logrus.Entry) {
	for _, pid := range pids {
		if err := runCudaCheckpoint(ctx, pid, cudaActionUnlock, ""); err != nil {
			log.WithError(err).WithField("pid", pid).Warn("Failed to unlock CUDA process")
		}
	}
}

func RestoreExternalCUDA(ctx context.Context, manifest *CheckpointManifest, podName, podNamespace, containerName string, containerPID int, restoredPID int, log *logrus.Entry) error {
	if manifest.ExternalRestore == nil || manifest.ExternalRestore.CUDA == nil {
		log.Debug("No CUDA restore data in manifest, skipping")
		return nil
	}

	cudaData := manifest.ExternalRestore.CUDA
	if len(cudaData.PIDs) == 0 {
		log.Debug("No CUDA PIDs recorded, skipping")
		return nil
	}

	targetUUIDs, err := getPodGPUUUIDsWithRetry(ctx, podName, podNamespace, containerName, log)
	if err != nil {
		return fmt.Errorf("failed to get target GPU UUIDs: %w", err)
	}
	if len(cudaData.SourceGPUUUIDs) == 0 {
		return fmt.Errorf("missing source GPU UUIDs in checkpoint manifest")
	}
	if len(targetUUIDs) == 0 {
		return fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", podNamespace, podName, containerName)
	}
	deviceMap, err := buildCUDADeviceMap(cudaData.SourceGPUUUIDs, targetUUIDs)
	if err != nil {
		return fmt.Errorf("failed to build CUDA device map: %w", err)
	}

	cgroupPath, err := getContainerCgroupPath(containerPID)
	if err != nil {
		return fmt.Errorf("failed to get container cgroup path: %w", err)
	}
	cudaPIDs, err := findRestoredCUDAPIDs(ctx, cgroupPath, restoredPID, log)
	if err != nil {
		return fmt.Errorf("failed to discover restored CUDA PIDs: %w", err)
	}
	if len(cudaPIDs) == 0 {
		return fmt.Errorf("checkpoint captured %d CUDA PIDs but none found in restored cgroup %s", len(cudaData.PIDs), cgroupPath)
	}

	for _, pid := range cudaPIDs {
		if err := runCudaCheckpoint(ctx, pid, cudaActionRestore, deviceMap); err != nil {
			return fmt.Errorf("cuda restore failed for PID %d: %w", pid, err)
		}
		if err := runCudaCheckpoint(ctx, pid, cudaActionUnlock, ""); err != nil {
			return fmt.Errorf("cuda unlock failed for PID %d: %w", pid, err)
		}
	}

	log.WithFields(logrus.Fields{
		"cuda_pids":   len(cudaPIDs),
		"device_map":  deviceMap,
		"containerID": containerPID,
		"restoredPID": restoredPID,
	}).Info("CUDA restore completed")
	return nil
}

func findRestoredCUDAPIDs(ctx context.Context, cgroupPath string, restoredPID int, log *logrus.Entry) ([]int, error) {
	deadlineCtx, cancel := context.WithTimeout(ctx, cudaDiscoverTO)
	defer cancel()

	ticker := time.NewTicker(cudaDiscoverTick)
	defer ticker.Stop()

	attempt := 0
	for {
		attempt++
		cgroupPIDs, err := getCgroupPIDs(cgroupPath)
		if err != nil {
			return nil, fmt.Errorf("failed to list restored cgroup PIDs from %s: %w", cgroupPath, err)
		}

		candidates := make([]int, 0, len(cgroupPIDs)+1)
		seen := make(map[int]struct{}, len(cgroupPIDs)+1)
		if restoredPID > 0 {
			candidates = append(candidates, restoredPID)
			seen[restoredPID] = struct{}{}
		}
		for _, pid := range cgroupPIDs {
			if _, ok := seen[pid]; ok {
				continue
			}
			candidates = append(candidates, pid)
			seen[pid] = struct{}{}
		}

		cudaPIDs := make([]int, 0, len(candidates))
		for _, pid := range candidates {
			ok, err := isCUDAProcess(deadlineCtx, pid)
			if err != nil {
				log.WithError(err).WithField("pid", pid).Debug("Failed CUDA state probe")
				continue
			}
			if ok {
				cudaPIDs = append(cudaPIDs, pid)
			}
		}

		log.WithFields(logrus.Fields{
			"attempt":      attempt,
			"cgroup_path":  cgroupPath,
			"cgroup_pids":  len(cgroupPIDs),
			"candidates":   len(candidates),
			"cuda_pids":    len(cudaPIDs),
			"sample_pids":  truncateIntSlice(cgroupPIDs, 16),
			"sample_cuda":  truncateIntSlice(cudaPIDs, 16),
			"discover_tmo": cudaDiscoverTO.String(),
		}).Info("CUDA restore PID discovery attempt")
		if len(cudaPIDs) > 0 {
			return cudaPIDs, nil
		}

		select {
		case <-deadlineCtx.Done():
			return nil, nil
		case <-ticker.C:
		}
	}
}

func isCUDAProcess(ctx context.Context, pid int) (bool, error) {
	if pid <= 0 {
		return false, nil
	}

	probeCtx, cancel := context.WithTimeout(ctx, cudaStateCheckTO)
	defer cancel()

	cmd := exec.CommandContext(probeCtx, cudaCheckpointBinary, "--get-state", "--pid", strconv.Itoa(pid))
	if err := cmd.Run(); err != nil {
		if probeCtx.Err() != nil {
			return false, probeCtx.Err()
		}
		return false, nil
	}
	return true, nil
}

func truncateIntSlice(values []int, limit int) []int {
	if len(values) <= limit {
		return append([]int(nil), values...)
	}
	return append([]int(nil), values[:limit]...)
}

func runCudaCheckpoint(ctx context.Context, pid int, action cudaCheckpointAction, deviceMap string) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, cudaCheckpointTO)
	defer cancel()

	args := []string{"--action", string(action), "--pid", strconv.Itoa(pid)}
	if action == cudaActionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}

	cmd := exec.CommandContext(timeoutCtx, cudaCheckpointBinary, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cuda-checkpoint %v failed for pid %d: %w (output: %s)", args, pid, err, strings.TrimSpace(string(output)))
	}
	return nil
}

func buildCUDADeviceMap(sourceUUIDs, targetUUIDs []string) (string, error) {
	if len(sourceUUIDs) != len(targetUUIDs) {
		return "", fmt.Errorf("GPU count mismatch: source has %d, target has %d", len(sourceUUIDs), len(targetUUIDs))
	}
	if len(sourceUUIDs) == 0 {
		return "", fmt.Errorf("GPU UUID list is empty")
	}
	pairs := make([]string, len(sourceUUIDs))
	for i := range sourceUUIDs {
		pairs[i] = sourceUUIDs[i] + "=" + targetUUIDs[i]
	}
	return strings.Join(pairs, ","), nil
}

// BuildCUDADeviceMap creates a cuda-checkpoint --device-map value from source and target GPU UUID lists.
func BuildCUDADeviceMap(sourceUUIDs, targetUUIDs []string) (string, error) {
	return buildCUDADeviceMap(sourceUUIDs, targetUUIDs)
}

func getContainerCgroupPath(pid int) (string, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if strings.HasPrefix(line, "0::") {
			return strings.TrimPrefix(line, "0::"), nil
		}
	}
	return "", fmt.Errorf("no cgroup v2 path found for pid %d", pid)
}

func getCgroupPIDs(cgroupPath string) ([]int, error) {
	procsPath := filepath.Join(hostCgroupPath, cgroupPath, "cgroup.procs")
	data, err := os.ReadFile(procsPath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	pids := make([]int, 0, len(lines))
	for _, line := range lines {
		if line == "" {
			continue
		}
		pid, err := strconv.Atoi(line)
		if err != nil {
			continue
		}
		pids = append(pids, pid)
	}
	return pids, nil
}

func findCUDAPIDs(pids []int) []int {
	cudaPIDs := make([]int, 0, len(pids))
	for _, pid := range pids {
		mapsPath := fmt.Sprintf("/proc/%d/maps", pid)
		data, err := os.ReadFile(mapsPath)
		if err != nil {
			continue
		}
		if strings.Contains(string(data), "libnvidia") {
			cudaPIDs = append(cudaPIDs, pid)
		}
	}
	return cudaPIDs
}

func getPodGPUUUIDs(ctx context.Context, podName, podNamespace, containerName string) ([]string, error) {
	if podName == "" || podNamespace == "" {
		return nil, nil
	}

	dialCtx, dialCancel := context.WithTimeout(ctx, 10*time.Second)
	defer dialCancel()
	conn, err := grpc.DialContext(
		dialCtx,
		"unix://"+podResourcesSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	client := podresourcesv1.NewPodResourcesListerClient(conn)
	listCtx, listCancel := context.WithTimeout(ctx, 10*time.Second)
	defer listCancel()
	resp, err := client.List(listCtx, &podresourcesv1.ListPodResourcesRequest{})
	if err != nil {
		return nil, err
	}

	for _, pod := range resp.GetPodResources() {
		if pod.GetName() != podName || pod.GetNamespace() != podNamespace {
			continue
		}
		for _, container := range pod.GetContainers() {
			if containerName != "" && container.GetName() != containerName {
				continue
			}
			for _, device := range container.GetDevices() {
				if device.GetResourceName() == nvidiaGPUResource {
					return device.GetDeviceIds(), nil
				}
			}
		}
	}

	return nil, nil
}

func getPodGPUUUIDsWithRetry(ctx context.Context, podName, podNamespace, containerName string, log *logrus.Entry) ([]string, error) {
	deadlineCtx, cancel := context.WithTimeout(ctx, podGPUDiscoverTO)
	defer cancel()

	ticker := time.NewTicker(podGPUDiscoverTick)
	defer ticker.Stop()

	attempt := 0
	var lastErr error
	for {
		attempt++
		uuids, err := getPodGPUUUIDs(deadlineCtx, podName, podNamespace, containerName)
		if err == nil && len(uuids) > 0 {
			return uuids, nil
		}
		if err != nil {
			lastErr = err
		}

		log.WithFields(logrus.Fields{
			"attempt":   attempt,
			"pod":       podName,
			"namespace": podNamespace,
			"container": containerName,
			"uuid_count": func() int {
				if uuids == nil {
					return 0
				}
				return len(uuids)
			}(),
		}).Debug("Waiting for pod GPU UUIDs in pod-resources")

		select {
		case <-deadlineCtx.Done():
			if lastErr != nil {
				return nil, lastErr
			}
			return nil, nil
		case <-ticker.C:
		}
	}
}

// GetPodGPUUUIDsWithRetry resolves GPU UUIDs for a pod/container from kubelet pod-resources API.
func GetPodGPUUUIDsWithRetry(ctx context.Context, podName, podNamespace, containerName string, log *logrus.Entry) ([]string, error) {
	return getPodGPUUUIDsWithRetry(ctx, podName, podNamespace, containerName, log)
}
