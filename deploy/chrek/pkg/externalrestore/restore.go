// restore.go orchestrates external restore from the DaemonSet.
// All operations happen externally: rootfs replay via /host/proc/<PID>/root,
// CRIU restore via nsenter + criu-helper, CUDA restore via cuda-checkpoint.
package externalrestore

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"

	"golang.org/x/sys/unix"
	"strconv"
	"strings"
	"time"

	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

const (
	// CRIUHelperBinary is the path to the criu-helper binary in the placeholder image.
	CRIUHelperBinary = "/usr/local/bin/criu-helper"

	// RestoreLogFilename is the CRIU restore log filename.
	RestoreLogFilename = "restore.log"
)

// RestorerConfig holds configuration for the external restore orchestrator.
type RestorerConfig struct {
	CheckpointBasePath string // Base path for checkpoint storage (PVC mount)
	CRIUHelperPath     string // Path to criu-helper binary (default: CRIUHelperBinary)
}

// Restorer orchestrates external restore operations from the DaemonSet.
type Restorer struct {
	cfg             RestorerConfig
	discoveryClient *checkpoint.DiscoveryClient
	log             *logrus.Entry
}

// NewRestorer creates a new external restore orchestrator.
func NewRestorer(cfg RestorerConfig, discoveryClient *checkpoint.DiscoveryClient) *Restorer {
	if cfg.CRIUHelperPath == "" {
		cfg.CRIUHelperPath = CRIUHelperBinary
	}
	return &Restorer{
		cfg:             cfg,
		discoveryClient: discoveryClient,
		log:             logrus.WithField("component", "restorer"),
	}
}

// Restore performs external restore for the given request.
func (r *Restorer) Restore(ctx context.Context, req RestoreAPIRequest) (*RestoreAPIResponse, error) {
	restoreStart := time.Now()
	r.log.WithFields(logrus.Fields{
		"checkpoint_id": req.CheckpointID,
		"pod":           req.PodName,
		"namespace":     req.PodNamespace,
		"container":     req.ContainerName,
	}).Info("=== Starting external restore ===")

	checkpointPath := filepath.Join(r.cfg.CheckpointBasePath, req.CheckpointID)

	// Load checkpoint manifest
	manifest, err := checkpoint.ReadCheckpointManifest(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	// Resolve the placeholder container to get its PID
	containerName := req.ContainerName
	if containerName == "" {
		containerName = "main"
	}

	placeholderPID, placeholderSpec, err := r.discoveryClient.ResolveContainerByPod(ctx, req.PodName, req.PodNamespace, containerName)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve placeholder container: %w", err)
	}
	r.log.WithField("pid", placeholderPID).Info("Resolved placeholder container")

	cudaDeviceMap := ""
	if manifest.ExternalRestore != nil && manifest.ExternalRestore.CUDA != nil && len(manifest.ExternalRestore.CUDA.PIDs) > 0 {
		if len(manifest.ExternalRestore.CUDA.SourceGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing source GPU UUIDs in checkpoint manifest")
		}
		targetGPUUUIDs, err := checkpoint.GetPodGPUUUIDsWithRetry(ctx, req.PodName, req.PodNamespace, containerName, r.log)
		if err != nil {
			return nil, fmt.Errorf("failed to get target GPU UUIDs: %w", err)
		}
		if len(targetGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", req.PodNamespace, req.PodName, containerName)
		}
		cudaDeviceMap, err = checkpoint.BuildCUDADeviceMap(manifest.ExternalRestore.CUDA.SourceGPUUUIDs, targetGPUUUIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to build CUDA device map: %w", err)
		}
	}

	var completedSteps []string

	// Step 1: Apply rootfs diff into placeholder rootfs via /host/proc/<PID>/root
	targetRoot := fmt.Sprintf("%s/%d/root", checkpoint.HostProcPath, placeholderPID)
	if err := applyRootfsDiff(checkpointPath, targetRoot, r.log); err != nil {
		return nil, fmt.Errorf("rootfs diff failed: %w", err)
	}
	if err := applyDeletedFiles(checkpointPath, targetRoot, r.log); err != nil {
		r.log.WithError(err).Warn("Failed to apply deleted files")
	}
	completedSteps = append(completedSteps, "rootfs")

	// Step 2: Restore /dev/shm into placeholder
	if err := restoreDevShm(checkpointPath, targetRoot, r.log); err != nil {
		r.log.WithError(err).Warn("Failed to restore /dev/shm")
	}

	// Step 2.5: Ensure /dev/net/tun exists in placeholder rootfs.
	// CRIU needs this character device (major 10, minor 200) to restore TUN/TAP
	// network devices. The unprivileged placeholder container doesn't have it,
	// but the DaemonSet (running as root) can create it via /host/proc/<PID>/root.
	tunPath := filepath.Join(targetRoot, "dev/net/tun")
	if _, statErr := os.Stat(tunPath); os.IsNotExist(statErr) {
		if err := os.MkdirAll(filepath.Dir(tunPath), 0755); err != nil {
			r.log.WithError(err).Warn("Failed to create /dev/net dir in placeholder")
		} else if err := syscall.Mknod(tunPath, syscall.S_IFCHR|0666, int(unix.Mkdev(10, 200))); err != nil {
			r.log.WithError(err).Warn("Failed to create /dev/net/tun in placeholder")
		} else {
			r.log.Info("Created /dev/net/tun in placeholder rootfs")
		}
	}

	// Step 3: Create link_remap stubs in placeholder rootfs
	if err := createLinkRemapStubs(checkpointPath, targetRoot, r.log); err != nil {
		r.log.WithError(err).Warn("Failed to create link_remap stubs")
	}

	// Step 3.25: Create restore marker so resumed workers detect restore mode.
	restoreMarkerPath := findEnvValue(placeholderSpec, "DYN_RESTORE_MARKER_FILE")
	if err := writeRestoreMarker(targetRoot, restoreMarkerPath); err != nil {
		r.log.WithError(err).Warn("Failed to write restore marker")
	}

	// Step 4: Execute nsenter + criu-helper
	restoredPID, err := r.executeCRIURestore(ctx, placeholderPID, checkpointPath, manifest, cudaDeviceMap)
	if err != nil {
		return nil, fmt.Errorf("CRIU restore failed: %w", err)
	}
	completedSteps = append(completedSteps, "criu")
	r.log.WithField("restored_pid", restoredPID).Info("CRIU restore completed")

	// Step 5: CUDA restore runs inside criu-helper after CRIU restore.
	if cudaDeviceMap != "" {
		completedSteps = append(completedSteps, "cuda")
	}

	// Step 6: Fail closed if the restored process exits quickly.
	// This prevents false-positive "restore succeeded" status when CRIU returns
	// but the resumed workload immediately dies.
	procRoot := filepath.Join(targetRoot, "proc")
	restoreLogPath := filepath.Join(targetRoot, "var", "criu-work", RestoreLogFilename)
	if err := waitForRestoredProcessStable(procRoot, restoredPID, 5*time.Second); err != nil {
		r.log.WithError(err).WithFields(logrus.Fields{
			"restored_pid": restoredPID,
			"proc_root":    procRoot,
		}).Error("Restored process failed stabilization check")
		logRestoredProcessDiagnostics(procRoot, restoredPID, restoreLogPath, r.log)
		return nil, fmt.Errorf("restored process did not stabilize: %w", err)
	}

	// Step 7: Start non-blocking liveness supervision for restored process.
	startRestoreSupervision(procRoot, restoredPID, r.log)

	totalDuration := time.Since(restoreStart)
	r.log.WithFields(logrus.Fields{
		"total_duration": totalDuration,
		"restored_pid":   restoredPID,
		"steps":          completedSteps,
	}).Info("=== External restore completed ===")

	return &RestoreAPIResponse{
		Success:        true,
		RestoredPID:    restoredPID,
		CompletedSteps: completedSteps,
	}, nil
}

// executeCRIURestore runs criu-helper inside the placeholder's namespaces via nsenter.
func (r *Restorer) executeCRIURestore(ctx context.Context, placeholderPID int, checkpointPath string, manifest *checkpoint.CheckpointManifest, cudaDeviceMap string) (int, error) {
	pidStr := strconv.Itoa(placeholderPID)

	// Build nsenter command for the namespaces CRIU restore needs.
	// We intentionally do not enter the cgroup namespace when cgroup management is ignored.
	args := []string{
		"-t", pidStr,
		"-m",
		"-n",
		"-p",
		"-i",
		"-u",
		"--", r.cfg.CRIUHelperPath,
		"--checkpoint-path", checkpointPath,
	}

	// Pass CRIU settings from manifest
	if manifest.CRIUDump.CRIU.WorkDir != "" {
		args = append(args, "--work-dir", manifest.CRIUDump.CRIU.WorkDir)
	}
	if cudaDeviceMap != "" {
		args = append(args, "--cuda-device-map", cudaDeviceMap)
	}

	cmd := exec.CommandContext(ctx, "nsenter", args...)
	r.log.WithField("cmd", cmd.String()).Debug("Executing nsenter + criu-helper")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("nsenter + criu-helper failed: %w\noutput: %s", err, strings.TrimSpace(string(output)))
	}

	// Parse RESTORED_PID=<N> from stdout
	outputStr := string(output)
	for _, line := range strings.Split(outputStr, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "RESTORED_PID=") {
			continue
		}
		r.log.WithField("line", line).Info("criu-helper")
	}
	for _, line := range strings.Split(outputStr, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "RESTORED_PID=") {
			pidStr := strings.TrimPrefix(line, "RESTORED_PID=")
			pid, err := strconv.Atoi(pidStr)
			if err != nil {
				return 0, fmt.Errorf("failed to parse RESTORED_PID from criu-helper output: %w", err)
			}
			return pid, nil
		}
	}

	return 0, fmt.Errorf("criu-helper did not output RESTORED_PID; output: %s", strings.TrimSpace(outputStr))
}

// applyRootfsDiff extracts rootfs-diff.tar into the target root.
func applyRootfsDiff(checkpointPath, targetRoot string, log *logrus.Entry) error {
	rootfsDiffPath := filepath.Join(checkpointPath, checkpoint.RootfsDiffFilename)
	if _, err := os.Stat(rootfsDiffPath); os.IsNotExist(err) {
		log.Debug("No rootfs-diff.tar, skipping")
		return nil
	}

	log.WithField("target", targetRoot).Info("Applying rootfs diff")
	cmd := exec.Command("tar", "--keep-old-files", "-C", targetRoot, "-xf", rootfsDiffPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		// tar exit codes 1-2 with --keep-old-files are non-fatal:
		// 1 = "file changed as we read it" or general warnings
		// 2 = "Cannot open: File exists" for read-only files
		// Both are expected when extracting over an existing rootfs.
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() <= 2 {
			log.WithField("exit_code", exitErr.ExitCode()).Debug("Rootfs diff applied (some files skipped)")
			return nil
		}
		return fmt.Errorf("tar extract failed: %w", err)
	}
	return nil
}

// applyDeletedFiles removes files marked as deleted in the checkpoint.
func applyDeletedFiles(checkpointPath, targetRoot string, log *logrus.Entry) error {
	deletedFilesPath := filepath.Join(checkpointPath, checkpoint.DeletedFilesFilename)
	data, err := os.ReadFile(deletedFilesPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read deleted files: %w", err)
	}

	var deletedFiles []string
	if err := json.Unmarshal(data, &deletedFiles); err != nil {
		return fmt.Errorf("failed to parse deleted files: %w", err)
	}

	count := 0
	for _, f := range deletedFiles {
		if f == "" {
			continue
		}
		target := filepath.Join(targetRoot, f)
		if _, err := os.Stat(target); os.IsNotExist(err) {
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.WithError(err).WithField("path", target).Debug("Could not delete file")
			continue
		}
		count++
	}
	log.WithField("count", count).Info("Deleted files applied")
	return nil
}

// restoreDevShm restores /dev/shm files into the target root's /dev/shm.
func restoreDevShm(checkpointPath, targetRoot string, log *logrus.Entry) error {
	srcDir := filepath.Join(checkpointPath, checkpoint.DevShmDirName)
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to read dev-shm dir: %w", err)
	}

	destDir := filepath.Join(targetRoot, "dev", "shm")
	if err := os.MkdirAll(destDir, 0777); err != nil {
		return fmt.Errorf("failed to create target /dev/shm: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		srcPath := filepath.Join(srcDir, entry.Name())
		destPath := filepath.Join(destDir, entry.Name())

		info, err := entry.Info()
		if err != nil {
			continue
		}
		uid, gid := -1, -1
		if stat, ok := info.Sys().(*syscall.Stat_t); ok {
			uid = int(stat.Uid)
			gid = int(stat.Gid)
		}

		src, err := os.Open(srcPath)
		if err != nil {
			continue
		}

		mode := info.Mode()
		if mode == 0 {
			mode = 0666
		}
		dst, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
		if err != nil {
			src.Close()
			continue
		}

		if _, err := io.Copy(dst, src); err != nil {
			src.Close()
			dst.Close()
			continue
		}
		if uid >= 0 && gid >= 0 {
			if err := dst.Chown(uid, gid); err != nil {
				src.Close()
				dst.Close()
				continue
			}
		}
		src.Close()
		dst.Close()
	}

	log.WithField("count", len(entries)).Debug("Restored /dev/shm files")
	return nil
}

func findEnvValue(spec *specs.Spec, key string) string {
	if spec == nil || spec.Process == nil {
		return ""
	}
	prefix := key + "="
	for _, env := range spec.Process.Env {
		if strings.HasPrefix(env, prefix) {
			return strings.TrimPrefix(env, prefix)
		}
	}
	return ""
}

func writeRestoreMarker(targetRoot, markerPath string) error {
	if markerPath == "" {
		return nil
	}
	cleanPath := filepath.Clean(markerPath)
	cleanPath = strings.TrimPrefix(cleanPath, "/")
	targetPath := filepath.Join(targetRoot, cleanPath)
	if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
		return err
	}
	return os.WriteFile(targetPath, []byte("restored"), 0644)
}

func startRestoreSupervision(procRoot string, restoredPID int, log *logrus.Entry) {
	if restoredPID <= 0 {
		return
	}
	entry := log.WithFields(logrus.Fields{
		"restored_pid": restoredPID,
		"proc_root":    procRoot,
	})
	entry.Info("Starting restored process supervision")

	go monitorRestoredProcess(procRoot, restoredPID, entry)
}

func monitorRestoredProcess(procRoot string, pid int, log *logrus.Entry) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for range ticker.C {
		state, err := readProcessState(procRoot, pid)
		if err == nil && state != "Z" {
			continue
		}
		if err == nil && state == "Z" {
			log.Warn("Restored process became zombie")
			return
		}
		if os.IsNotExist(err) {
			log.Info("Restored process exited")
			return
		}
		log.WithError(err).Debug("Failed to inspect restored process")
		return
	}
}

func waitForRestoredProcessStable(procRoot string, pid int, window time.Duration) error {
	if pid <= 0 {
		return fmt.Errorf("invalid restored PID %d", pid)
	}

	deadline := time.Now().Add(window)
	for {
		state, err := readProcessState(procRoot, pid)
		if err != nil {
			if os.IsNotExist(err) {
				return fmt.Errorf("process %d exited", pid)
			}
			return fmt.Errorf("failed to inspect process %d: %w", pid, err)
		}
		if state == "Z" {
			return fmt.Errorf("process %d became zombie", pid)
		}

		if time.Now().After(deadline) {
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func readProcessState(procRoot string, pid int) (string, error) {
	statusPath := filepath.Join(procRoot, strconv.Itoa(pid), "status")
	data, err := os.ReadFile(statusPath)
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "State:") {
			fields := strings.Fields(line)
			if len(fields) > 1 {
				return fields[1], nil
			}
			break
		}
	}
	return "", fmt.Errorf("state not found in %s", statusPath)
}

func logRestoredProcessDiagnostics(procRoot string, pid int, restoreLogPath string, log *logrus.Entry) {
	entry := log.WithFields(logrus.Fields{
		"restored_pid": pid,
		"proc_root":    procRoot,
	})

	statusPath := filepath.Join(procRoot, strconv.Itoa(pid), "status")
	if data, err := os.ReadFile(statusPath); err == nil {
		entry.WithField("path", statusPath).Errorf("Restored process status:\n%s", strings.TrimSpace(string(data)))
	} else {
		entry.WithError(err).WithField("path", statusPath).Error("Failed to read restored process status")
	}

	cmdlinePath := filepath.Join(procRoot, strconv.Itoa(pid), "cmdline")
	if data, err := os.ReadFile(cmdlinePath); err == nil {
		cmdline := strings.TrimSpace(strings.ReplaceAll(string(data), "\x00", " "))
		if cmdline == "" {
			cmdline = "<empty>"
		}
		entry.WithField("cmdline", cmdline).Error("Restored process cmdline")
	} else {
		entry.WithError(err).WithField("path", cmdlinePath).Error("Failed to read restored process cmdline")
	}

	statPath := filepath.Join(procRoot, strconv.Itoa(pid), "stat")
	if data, err := os.ReadFile(statPath); err == nil {
		raw, parseErr := parseProcExitCodeRaw(string(data))
		if parseErr != nil {
			entry.WithError(parseErr).WithField("path", statPath).Error("Failed to parse /proc stat exit code")
		} else {
			exitStatus, termSignal, coreDumped := decodeProcExitCode(raw)
			entry.WithFields(logrus.Fields{
				"exit_code_raw": raw,
				"exit_status":   exitStatus,
				"term_signal":   termSignal,
				"core_dumped":   coreDumped,
			}).Error("Decoded restored process exit code")
		}
	}

	childrenPath := filepath.Join(procRoot, "1", "task", "1", "children")
	if data, err := os.ReadFile(childrenPath); err == nil {
		entry.WithField("children", strings.TrimSpace(string(data))).Error("PID 1 children in restored namespace")
	}

	logCRIURestoreSummary(restoreLogPath, entry)
}

func parseProcExitCodeRaw(statLine string) (int, error) {
	statLine = strings.TrimSpace(statLine)
	if statLine == "" {
		return 0, fmt.Errorf("empty stat line")
	}
	paren := strings.LastIndex(statLine, ")")
	if paren < 0 || paren+2 > len(statLine) {
		return 0, fmt.Errorf("malformed stat line")
	}
	fields := strings.Fields(statLine[paren+2:])
	if len(fields) == 0 {
		return 0, fmt.Errorf("malformed stat fields")
	}
	raw, err := strconv.Atoi(fields[len(fields)-1])
	if err != nil {
		return 0, err
	}
	return raw, nil
}

func decodeProcExitCode(raw int) (exitStatus int, termSignal int, coreDumped bool) {
	exitStatus = (raw >> 8) & 0xff
	termSignal = raw & 0x7f
	coreDumped = (raw & 0x80) != 0
	return exitStatus, termSignal, coreDumped
}

func logCRIURestoreSummary(path string, log *logrus.Entry) {
	data, err := os.ReadFile(path)
	if err != nil {
		log.WithError(err).WithField("path", path).Error("Failed to read CRIU restore log")
		return
	}

	lines := strings.Split(string(data), "\n")
	keyLines := make([]string, 0, 64)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		lower := strings.ToLower(trimmed)
		if strings.Contains(lower, "error") ||
			strings.Contains(lower, "warn") ||
			strings.Contains(lower, "fail") ||
			strings.Contains(lower, "cuda") ||
			strings.Contains(lower, "iptables") ||
			strings.Contains(lower, "restore finished successfully") ||
			strings.Contains(lower, "tasks resumed") {
			keyLines = append(keyLines, trimmed)
			if len(keyLines) == 80 {
				break
			}
		}
	}
	if len(keyLines) > 0 {
		log.WithField("path", path).Errorf("CRIU restore key lines:\n%s", strings.Join(keyLines, "\n"))
	}

	tail := make([]string, 0, 40)
	for i := len(lines) - 1; i >= 0 && len(tail) < 40; i-- {
		trimmed := strings.TrimSpace(lines[i])
		if trimmed == "" {
			continue
		}
		tail = append(tail, trimmed)
	}
	for i, j := 0, len(tail)-1; i < j; i, j = i+1, j-1 {
		tail[i], tail[j] = tail[j], tail[i]
	}
	if len(tail) > 0 {
		log.WithField("path", path).Errorf("CRIU restore tail:\n%s", strings.Join(tail, "\n"))
	}
}
