// Package watcher provides Kubernetes pod watching for automatic checkpointing.
package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/externalrestore"
)

// SignalFile represents the content of a checkpoint completion signal file
type SignalFile struct {
	CheckpointID   string    `json:"checkpoint_id"`
	CheckpointPath string    `json:"checkpoint_path"`
	Timestamp      time.Time `json:"timestamp"`
	Success        bool      `json:"success"`
	Error          string    `json:"error,omitempty"`
}

// WatcherConfig holds watcher configuration.
type WatcherConfig struct {
	NodeName            string
	ListenAddr          string // HTTP server address for health checks (e.g., ":8080")
	RestrictedNamespace string // Optional: restrict watching to this namespace (empty = cluster-wide)
	AgentSocketPath     string // Pod-local UDS socket path exposed by the chrek API server.

	// Checkpoint configuration (from ConfigMap)
	CheckpointSpec *checkpoint.CheckpointSpec
}

// Watcher watches for pods with checkpoint/restore labels and triggers operations
type Watcher struct {
	config          WatcherConfig
	clientset       kubernetes.Interface
	dynamicClient   dynamic.Interface
	agentClient     *externalrestore.Client
	discoveryClient *checkpoint.DiscoveryClient
	log             *logrus.Entry

	// Track checkpoint status: "in_progress", "completed", or "" (not started/failed)
	checkpointed   map[string]string
	checkpointedMu sync.RWMutex

	// Track restore status: "in_progress", "completed", or "" (not started/failed)
	// Values:
	// - "in_progress": restore request issued and still running
	// - "completed": external restore succeeded
	// - "failed": external restore failed (no automatic retry for this pod)
	restored   map[string]string
	restoredMu sync.RWMutex

	stopCh chan struct{}
}

const (
	checkpointConditionType = "CheckpointState"
	restoreConditionType    = "CheckpointRestore"
)

var (
	dynamoCheckpointGVR = schema.GroupVersionResource{
		Group:    "nvidia.com",
		Version:  "v1alpha1",
		Resource: "dynamocheckpoints",
	}
	dynamoComponentDeploymentGVR = schema.GroupVersionResource{
		Group:    "nvidia.com",
		Version:  "v1alpha1",
		Resource: "dynamocomponentdeployments",
	}
)

// NewWatcher creates a new pod watcher.
func NewWatcher(cfg WatcherConfig, discoveryClient *checkpoint.DiscoveryClient) (*Watcher, error) {
	// Create in-cluster Kubernetes client
	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get in-cluster config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	dynamicClient, err := dynamic.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic kubernetes client: %w", err)
	}

	if cfg.AgentSocketPath == "" {
		return nil, fmt.Errorf("agent socket path is required")
	}

	return &Watcher{
		config:          cfg,
		clientset:       clientset,
		dynamicClient:   dynamicClient,
		agentClient:     externalrestore.NewClient(cfg.AgentSocketPath),
		discoveryClient: discoveryClient,
		log:             logrus.WithField("component", "watcher"),
		checkpointed:    make(map[string]string),
		restored:        make(map[string]string),
		stopCh:          make(chan struct{}),
	}, nil
}

// Start begins watching for pods and starts the health check server
func (w *Watcher) Start(ctx context.Context) error {
	if w.config.CheckpointSpec == nil {
		return fmt.Errorf("checkpoint spec is required")
	}

	w.log.WithFields(logrus.Fields{
		"node":            w.config.NodeName,
		"checkpoint":      checkpoint.KubeLabelCheckpointSource,
		"restore":         checkpoint.KubeLabelCheckpointRestore,
		"restore_enabled": w.agentClient != nil,
		"socket_path":     w.config.AgentSocketPath,
	}).Info("Starting pod watcher")

	// Start health check HTTP server if address is configured
	if w.config.ListenAddr != "" {
		httpServer := w.startHealthServer(ctx)
		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			httpServer.Shutdown(shutdownCtx)
		}()
	}

	// Namespace restriction options shared across informer factories
	var nsOptions []informers.SharedInformerOption
	if w.config.RestrictedNamespace != "" {
		w.log.WithField("namespace", w.config.RestrictedNamespace).Info("Restricting pod watching to namespace")
		nsOptions = append(nsOptions, informers.WithNamespace(w.config.RestrictedNamespace))
	} else {
		w.log.Info("Watching pods cluster-wide (all namespaces)")
	}

	var syncFuncs []cache.InformerSynced

	// --- Checkpoint informer: watches pods with checkpoint-source=true ---
	checkpointSelector := labels.SelectorFromSet(labels.Set{
		checkpoint.KubeLabelCheckpointSource: "true",
	}).String()

	ckptFactoryOpts := append([]informers.SharedInformerOption{
		informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
			opts.LabelSelector = checkpointSelector
		}),
	}, nsOptions...)

	ckptFactory := informers.NewSharedInformerFactoryWithOptions(
		w.clientset, 30*time.Second, ckptFactoryOpts...,
	)

	ckptInformer := ckptFactory.Core().V1().Pods().Informer()
	ckptInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			w.handleCheckpointPodEvent(ctx, obj.(*corev1.Pod))
		},
		UpdateFunc: func(_, newObj interface{}) {
			w.handleCheckpointPodEvent(ctx, newObj.(*corev1.Pod))
		},
	})
	go ckptFactory.Start(w.stopCh)
	syncFuncs = append(syncFuncs, ckptInformer.HasSynced)

	// --- Restore informer: watches pods with checkpoint-restore=true ---
	if w.agentClient != nil {
		restoreSelector := labels.SelectorFromSet(labels.Set{
			checkpoint.KubeLabelCheckpointRestore: "true",
		}).String()

		restoreFactoryOpts := append([]informers.SharedInformerOption{
			informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
				opts.LabelSelector = restoreSelector
			}),
		}, nsOptions...)

		restoreFactory := informers.NewSharedInformerFactoryWithOptions(
			w.clientset, 30*time.Second, restoreFactoryOpts...,
		)

		restoreInformer := restoreFactory.Core().V1().Pods().Informer()
		restoreInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				w.handleRestorePodEvent(ctx, obj.(*corev1.Pod))
			},
			UpdateFunc: func(_, newObj interface{}) {
				w.handleRestorePodEvent(ctx, newObj.(*corev1.Pod))
			},
		})
		go restoreFactory.Start(w.stopCh)
		syncFuncs = append(syncFuncs, restoreInformer.HasSynced)
	}

	// Wait for all caches to sync
	if !cache.WaitForCacheSync(w.stopCh, syncFuncs...) {
		return fmt.Errorf("failed to sync informer caches")
	}

	w.log.Info("Pod watcher started and caches synced")

	// Wait for context cancellation
	<-ctx.Done()
	close(w.stopCh)

	return nil
}

// HealthResponse is the response for health check endpoint
type HealthResponse struct {
	Status   string `json:"status"`
	NodeName string `json:"node_name"`
}

// startHealthServer starts an HTTP server for health checks
func (w *Watcher) startHealthServer(ctx context.Context) *http.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(rw http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(rw, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		json.NewEncoder(rw).Encode(HealthResponse{
			Status:   "healthy",
			NodeName: w.config.NodeName,
		})
	})

	server := &http.Server{
		Addr:         w.config.ListenAddr,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		w.log.WithField("addr", w.config.ListenAddr).Info("Starting health check server")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			w.log.WithError(err).Error("Health check server error")
		}
	}()

	return server
}

// Stop stops the watcher
func (w *Watcher) Stop() {
	close(w.stopCh)
}

// handleCheckpointPodEvent processes a checkpoint pod event
func (w *Watcher) handleCheckpointPodEvent(ctx context.Context, pod *corev1.Pod) {
	// Filter to pods on this node
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}

	// Check if pod is Ready
	if !w.isPodReady(pod) {
		return
	}

	// Check if we've already checkpointed this pod
	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	// Get checkpoint ID from label (uses the checkpoint hash)
	checkpointID, ok := pod.Labels[checkpoint.KubeLabelCheckpointHash]
	if !ok || checkpointID == "" {
		w.log.WithField("pod", podKey).Warn("Pod has checkpoint label but no checkpoint-hash label")
		return
	}

	// Check if checkpoint is already in progress or completed for this pod
	w.checkpointedMu.Lock()
	status := w.checkpointed[podKey]
	if status == "completed" || status == "in_progress" {
		w.checkpointedMu.Unlock()
		return
	}
	// Mark as in_progress to prevent concurrent attempts
	w.checkpointed[podKey] = "in_progress"
	w.checkpointedMu.Unlock()

	// Trigger checkpoint
	w.log.WithFields(logrus.Fields{
		"pod":           podKey,
		"checkpoint_id": checkpointID,
	}).Info("Pod ready, triggering checkpoint")
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "CheckpointRequested", fmt.Sprintf("Checkpoint requested: %s", checkpointID))
	w.setCheckpointStatus(ctx, pod, metav1.ConditionFalse, "CheckpointRequested", fmt.Sprintf("Checkpoint requested on node %s", w.config.NodeName), "Creating")

	go w.doCheckpoint(ctx, pod, checkpointID, podKey)
}

// isPodRunning checks if the pod phase is Running (containers started).
func (w *Watcher) isPodRunning(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodRunning
}

// isPodReady checks if all containers in the pod are ready
func (w *Watcher) isPodReady(pod *corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}

	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}

	return false
}

// handleRestorePodEvent processes a restore pod event.
// Triggers external restore when the placeholder pod becomes Running.
func (w *Watcher) handleRestorePodEvent(ctx context.Context, pod *corev1.Pod) {
	// Filter to pods on this node
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	// Snapshot current restore state for this pod.
	w.restoredMu.RLock()
	restoreState := w.restored[podKey]
	w.restoredMu.RUnlock()

	// Wait for pod to be Running.
	if !w.isPodRunning(pod) {
		return
	}

	// Restore should only run while the target pod is not ready.
	// Once restored and serving, readiness should flip to true and re-triggers are skipped.
	if w.isPodReady(pod) {
		// Keep DCD condition consistent with observed end state.
		switch restoreState {
		case "completed":
			w.setRestoreStatus(ctx, pod, metav1.ConditionTrue, "RestoreSucceeded", "Restore completed and pod is Ready")
		case "failed":
			w.setRestoreStatus(ctx, pod, metav1.ConditionFalse, "RestoreFailed", "External restore failed; pod became Ready via cold start")
		}
		return
	}

	// Get checkpoint hash from label
	checkpointID, ok := pod.Labels[checkpoint.KubeLabelCheckpointHash]
	if !ok || checkpointID == "" {
		w.log.WithField("pod", podKey).Warn("Restore pod has no checkpoint-hash label")
		return
	}

	// Verify checkpoint is ready on disk before attempting restore.
	checkpointDir := filepath.Join(w.config.CheckpointSpec.BasePath, checkpointID)
	doneMarker := filepath.Join(checkpointDir, checkpoint.CheckpointDoneFilename)
	if _, err := os.Stat(doneMarker); os.IsNotExist(err) {
		w.log.WithFields(logrus.Fields{
			"pod":           podKey,
			"checkpoint_id": checkpointID,
			"marker":        doneMarker,
		}).Debug("Checkpoint not ready on disk, skipping restore")
		return
	}

	// Check if restore is already in progress or completed
	w.restoredMu.Lock()
	status := w.restored[podKey]
	if status == "completed" || status == "in_progress" || status == "failed" {
		w.restoredMu.Unlock()
		return
	}
	w.restored[podKey] = "in_progress"
	w.restoredMu.Unlock()

	w.log.WithFields(logrus.Fields{
		"pod":           podKey,
		"checkpoint_id": checkpointID,
	}).Info("Restore pod running, triggering external restore")
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "RestoreRequested", fmt.Sprintf("Restore requested from checkpoint %s", checkpointID))
	w.setRestoreStatus(ctx, pod, metav1.ConditionFalse, "RestoreRequested", "Restore requested by watcher")

	go w.doRestore(ctx, pod, checkpointID, podKey)
}

// doRestore performs external restore by calling the local chrek UDS API.
func (w *Watcher) doRestore(ctx context.Context, pod *corev1.Pod, checkpointID, podKey string) {
	log := w.log.WithFields(logrus.Fields{
		"pod":           podKey,
		"checkpoint_id": checkpointID,
	})

	if w.agentClient == nil {
		err := fmt.Errorf("agent UDS client is not configured")
		log.WithError(err).Error("External restore failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.setRestoreStatus(ctx, pod, metav1.ConditionFalse, "RestoreFailed", err.Error())
		w.restoredMu.Lock()
		w.restored[podKey] = "failed"
		w.restoredMu.Unlock()
		return
	}

	// Determine the main container name
	containerName := "main"
	for _, c := range pod.Spec.Containers {
		if c.Name == "main" {
			break
		}
		// Fall back to first container if no "main" container
		if len(pod.Spec.Containers) == 1 {
			containerName = c.Name
		}
	}

	req := externalrestore.RestoreAPIRequest{
		CheckpointID:  checkpointID,
		PodName:       pod.Name,
		PodNamespace:  pod.Namespace,
		ContainerName: containerName,
		RequestID:     restoreRequestID(pod, checkpointID),
	}

	result, err := w.agentClient.Restore(ctx, req)
	if err != nil {
		log.WithError(err).Error("External restore failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.setRestoreStatus(ctx, pod, metav1.ConditionFalse, "RestoreFailed", err.Error())
		w.restoredMu.Lock()
		w.restored[podKey] = "failed"
		w.restoredMu.Unlock()
		return
	}

	log.WithFields(logrus.Fields{
		"restored_pid":    result.RestoredPID,
		"completed_steps": result.CompletedSteps,
	}).Info("External restore completed successfully")
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "RestoreSucceeded", fmt.Sprintf("Restore completed from checkpoint %s", checkpointID))
	w.setRestoreStatus(ctx, pod, metav1.ConditionFalse, "RestoreCompleted", fmt.Sprintf("Restore completed with PID %d; waiting for pod readiness", result.RestoredPID))

	w.restoredMu.Lock()
	w.restored[podKey] = "completed"
	w.restoredMu.Unlock()
}

// doCheckpoint performs the checkpoint and writes the signal file
func (w *Watcher) doCheckpoint(ctx context.Context, pod *corev1.Pod, checkpointID, podKey string) {
	log := w.log.WithFields(logrus.Fields{
		"pod":           podKey,
		"checkpoint_id": checkpointID,
	})

	if w.agentClient == nil {
		err := fmt.Errorf("agent UDS client is not configured")
		log.WithError(err).Error("Checkpoint failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		w.setCheckpointStatus(ctx, pod, metav1.ConditionFalse, "CheckpointFailed", err.Error(), "Failed")
		w.checkpointedMu.Lock()
		delete(w.checkpointed, podKey)
		w.checkpointedMu.Unlock()
		return
	}

	// Find the main container and get signal file path from env
	var containerID string
	var containerName string
	var signalFilePath string
	for _, container := range pod.Spec.Containers {
		if container.Name == "main" || len(pod.Spec.Containers) == 1 {
			containerName = container.Name
			// Get signal file path from environment
			for _, env := range container.Env {
				if env.Name == "DYN_CHECKPOINT_SIGNAL_FILE" {
					signalFilePath = env.Value
					break
				}
			}
			break
		}
	}

	// Get container ID from status
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == "main" || len(pod.Status.ContainerStatuses) == 1 {
			// Remove containerd:// prefix
			containerID = cs.ContainerID
			if len(containerID) > 13 && containerID[:13] == "containerd://" {
				containerID = containerID[13:]
			}
			break
		}
	}

	if containerID == "" {
		log.Error("Could not find container ID")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", "Could not resolve target container ID")
		w.setCheckpointStatus(ctx, pod, metav1.ConditionFalse, "CheckpointFailed", "Could not resolve target container ID", "Failed")
		w.checkpointedMu.Lock()
		delete(w.checkpointed, podKey)
		w.checkpointedMu.Unlock()
		return
	}

	if signalFilePath == "" {
		log.Warn("No DYN_CHECKPOINT_SIGNAL_FILE env var found, signal file will not be written")
	}

	log.WithFields(logrus.Fields{
		"container_id":     containerID,
		"signal_file_path": signalFilePath,
	}).Info("Found container, starting checkpoint")

	// Resolve container to get PID for signal file writing.
	containerPID, _, err := w.discoveryClient.ResolveContainer(ctx, containerID)
	if err != nil {
		log.WithError(err).Error("Failed to resolve container")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", fmt.Sprintf("Container resolve failed: %v", err))
		w.setCheckpointStatus(ctx, pod, metav1.ConditionFalse, "CheckpointFailed", fmt.Sprintf("Container resolve failed: %v", err), "Failed")
		w.checkpointedMu.Lock()
		delete(w.checkpointed, podKey)
		w.checkpointedMu.Unlock()
		return
	}

	// Validate CheckpointSpec is set
	if w.config.CheckpointSpec == nil {
		log.Error("CheckpointSpec is nil - cannot perform checkpoint")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", "CheckpointSpec is nil")
		w.setCheckpointStatus(ctx, pod, metav1.ConditionFalse, "CheckpointFailed", "CheckpointSpec is nil", "Failed")
		w.checkpointedMu.Lock()
		delete(w.checkpointed, podKey)
		w.checkpointedMu.Unlock()
		return
	}

	// Perform checkpoint via local chrek UDS API.
	req := externalrestore.CheckpointAPIRequest{
		ContainerID:   containerID,
		ContainerName: containerName,
		CheckpointID:  checkpointID,
		PodName:       pod.Name,
		PodNamespace:  pod.Namespace,
		RequestID:     fmt.Sprintf("%s/%s:%s", pod.Namespace, pod.Name, checkpointID),
	}

	result, err := w.agentClient.Checkpoint(ctx, req)
	if err != nil {
		log.WithError(err).Error("Checkpoint failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		w.setCheckpointStatus(ctx, pod, metav1.ConditionFalse, "CheckpointFailed", err.Error(), "Failed")
		// Write failure marker to PVC so restore pods know checkpoint failed
		checkpointDir := filepath.Join(w.config.CheckpointSpec.BasePath, checkpointID)
		w.writeCheckpointDoneMarker(checkpointDir, checkpointID, false, err.Error(), log)
		if signalFilePath != "" {
			w.writeSignalFileToPod(containerPID, signalFilePath, checkpointID, "", false, err.Error())
		}
		// Clear the in_progress status so checkpoint can be retried
		w.checkpointedMu.Lock()
		delete(w.checkpointed, podKey)
		w.checkpointedMu.Unlock()
		return
	}

	checkpointPath := filepath.Join(w.config.CheckpointSpec.BasePath, checkpointID)
	if result != nil && result.CheckpointID != "" {
		checkpointID = result.CheckpointID
		checkpointPath = filepath.Join(w.config.CheckpointSpec.BasePath, result.CheckpointID)
	}

	log.WithField("checkpoint_dir", checkpointPath).Info("Checkpoint completed successfully")
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "CheckpointSucceeded", fmt.Sprintf("Checkpoint completed: %s", checkpointID))
	w.setCheckpointStatus(ctx, pod, metav1.ConditionTrue, "CheckpointSucceeded", fmt.Sprintf("Checkpoint completed at %s", checkpointPath), "Ready")

	// Write checkpoint.done marker to PVC for cross-node restore detection
	w.writeCheckpointDoneMarker(checkpointPath, checkpointID, true, "", log)

	// Write signal file to pod's hostPath for checkpoint job pod to exit
	if signalFilePath != "" {
		w.writeSignalFileToPod(containerPID, signalFilePath, checkpointID, checkpointPath, true, "")
	}

	// Mark as completed so we don't checkpoint again
	w.checkpointedMu.Lock()
	w.checkpointed[podKey] = "completed"
	w.checkpointedMu.Unlock()
}

// writeSignalFileToPod writes a signal file to the checkpointed pod's filesystem
// via /proc/<pid>/root to indicate checkpoint completion
func (w *Watcher) writeSignalFileToPod(pid int, signalFilePath, checkpointID, checkpointPath string, success bool, errMsg string) {
	signal := SignalFile{
		CheckpointID:   checkpointID,
		CheckpointPath: checkpointPath,
		Timestamp:      time.Now().UTC(),
		Success:        success,
		Error:          errMsg,
	}

	data, err := json.MarshalIndent(signal, "", "  ")
	if err != nil {
		w.log.WithError(err).Error("Failed to marshal signal file")
		return
	}

	// Write to the pod's filesystem via /proc/<pid>/root
	hostSignalPath := fmt.Sprintf("%s/%d/root%s", checkpoint.HostProcPath, pid, signalFilePath)

	// Ensure signal directory exists in pod's filesystem
	signalDir := filepath.Dir(hostSignalPath)
	if err := os.MkdirAll(signalDir, 0755); err != nil {
		w.log.WithError(err).WithField("path", signalDir).Error("Failed to create signal directory in pod")
		return
	}

	if err := os.WriteFile(hostSignalPath, data, 0644); err != nil {
		w.log.WithError(err).WithField("path", hostSignalPath).Error("Failed to write signal file to pod")
		return
	}

	w.log.WithFields(logrus.Fields{
		"host_path": hostSignalPath,
		"pod_path":  signalFilePath,
		"pid":       pid,
		"success":   success,
	}).Info("Signal file written to pod filesystem")
}

// writeCheckpointDoneMarker writes a checkpoint.done marker file to the checkpoint directory on shared PVC.
func (w *Watcher) writeCheckpointDoneMarker(checkpointDir, checkpointID string, success bool, errMsg string, log *logrus.Entry) {
	markerPath := filepath.Join(checkpointDir, checkpoint.CheckpointDoneFilename)

	marker := SignalFile{
		CheckpointID:   checkpointID,
		CheckpointPath: checkpointDir,
		Timestamp:      time.Now().UTC(),
		Success:        success,
		Error:          errMsg,
	}

	data, err := json.MarshalIndent(marker, "", "  ")
	if err != nil {
		log.WithError(err).Error("Failed to marshal checkpoint.done marker")
		return
	}

	if err := os.WriteFile(markerPath, data, 0644); err != nil {
		log.WithError(err).WithField("path", markerPath).Error("Failed to write checkpoint.done marker")
		return
	}

	log.WithFields(logrus.Fields{
		"path":    markerPath,
		"success": success,
	}).Info("checkpoint.done marker written to PVC")
}

func (w *Watcher) emitPodEvent(ctx context.Context, pod *corev1.Pod, eventType, reason, message string) {
	event := &corev1.Event{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("%s-", pod.Name),
			Namespace:    pod.Namespace,
		},
		InvolvedObject: corev1.ObjectReference{
			Kind:       "Pod",
			Namespace:  pod.Namespace,
			Name:       pod.Name,
			UID:        pod.UID,
			APIVersion: "v1",
		},
		Type:    eventType,
		Reason:  reason,
		Message: message,
		Source: corev1.EventSource{
			Component: "chrek-watcher",
		},
		Count:          1,
		FirstTimestamp: metav1.Now(),
		LastTimestamp:  metav1.Now(),
	}

	if _, err := w.clientset.CoreV1().Events(pod.Namespace).Create(ctx, event, metav1.CreateOptions{}); err != nil {
		w.log.WithError(err).WithFields(logrus.Fields{
			"pod":     fmt.Sprintf("%s/%s", pod.Namespace, pod.Name),
			"reason":  reason,
			"message": message,
		}).Warn("Failed to create watcher event")
	}
}

func (w *Watcher) setCheckpointStatus(
	ctx context.Context,
	pod *corev1.Pod,
	conditionStatus metav1.ConditionStatus,
	reason string,
	message string,
	phase string,
) {
	checkpointName := pod.Labels[checkpoint.KubeLabelCheckpointName]
	if checkpointName == "" {
		w.log.WithField("pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)).Warn("Checkpoint pod missing checkpoint-name label")
		return
	}

	condition := map[string]interface{}{
		"type":               checkpointConditionType,
		"status":             string(conditionStatus),
		"reason":             reason,
		"message":            message,
		"lastTransitionTime": metav1.Now().Format(time.RFC3339),
	}

	err := w.updateStatusCondition(
		ctx,
		dynamoCheckpointGVR,
		pod.Namespace,
		checkpointName,
		condition,
		func(obj map[string]interface{}) error {
			if phase != "" {
				if err := unstructured.SetNestedField(obj, phase, "status", "phase"); err != nil {
					return err
				}
			}
			if message != "" {
				if err := unstructured.SetNestedField(obj, message, "status", "message"); err != nil {
					return err
				}
			}
			return nil
		},
	)
	if err != nil {
		w.log.WithError(err).WithFields(logrus.Fields{
			"checkpoint": checkpointName,
			"namespace":  pod.Namespace,
			"reason":     reason,
		}).Warn("Failed to update DynamoCheckpoint status")
	}
}

func (w *Watcher) setRestoreStatus(
	ctx context.Context,
	pod *corev1.Pod,
	conditionStatus metav1.ConditionStatus,
	reason string,
	message string,
) {
	dcdName := pod.Labels[checkpoint.KubeLabelDynamoSelector]
	if dcdName == "" {
		w.log.WithField("pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)).Warn("Restore pod missing DCD selector label")
		return
	}

	condition := map[string]interface{}{
		"type":               restoreConditionType,
		"status":             string(conditionStatus),
		"reason":             reason,
		"message":            message,
		"lastTransitionTime": metav1.Now().Format(time.RFC3339),
	}

	err := w.updateStatusCondition(
		ctx,
		dynamoComponentDeploymentGVR,
		pod.Namespace,
		dcdName,
		condition,
		nil,
	)
	if err != nil {
		w.log.WithError(err).WithFields(logrus.Fields{
			"dcd":       dcdName,
			"namespace": pod.Namespace,
			"reason":    reason,
		}).Warn("Failed to update DynamoComponentDeployment restore condition")
	}
}

func (w *Watcher) updateStatusCondition(
	ctx context.Context,
	gvr schema.GroupVersionResource,
	namespace string,
	name string,
	condition map[string]interface{},
	mutate func(obj map[string]interface{}) error,
) error {
	var lastErr error

	for attempt := 0; attempt < 3; attempt++ {
		obj, err := w.dynamicClient.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		conditions, _, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
		if err != nil {
			return fmt.Errorf("failed to read status.conditions: %w", err)
		}
		conditions = upsertCondition(conditions, condition)
		if err := unstructured.SetNestedSlice(obj.Object, conditions, "status", "conditions"); err != nil {
			return fmt.Errorf("failed to set status.conditions: %w", err)
		}

		if mutate != nil {
			if err := mutate(obj.Object); err != nil {
				return fmt.Errorf("failed to mutate status object: %w", err)
			}
		}

		if _, err := w.dynamicClient.Resource(gvr).Namespace(namespace).UpdateStatus(ctx, obj, metav1.UpdateOptions{}); err != nil {
			if apierrors.IsConflict(err) {
				lastErr = err
				continue
			}
			return err
		}

		return nil
	}

	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("status update failed after retries")
}

func restoreRequestID(pod *corev1.Pod, checkpointID string) string {
	if id := pod.Labels[checkpoint.KubeLabelRestoreRequestID]; id != "" {
		return id
	}
	return fmt.Sprintf("%s/%s:%s", pod.Namespace, pod.Name, checkpointID)
}

func upsertCondition(conditions []interface{}, condition map[string]interface{}) []interface{} {
	condType, _ := condition["type"].(string)
	for i, existing := range conditions {
		typed, ok := existing.(map[string]interface{})
		if !ok {
			continue
		}
		if existingType, _ := typed["type"].(string); existingType == condType {
			conditions[i] = condition
			return conditions
		}
	}

	return append(conditions, condition)
}
