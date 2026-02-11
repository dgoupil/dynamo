// Package externalrestore provides external restore orchestration for the DaemonSet.
// The DaemonSet performs all restore operations externally: rootfs replay, CRIU via
// nsenter + criu-helper, and CUDA restore. The placeholder pod just donates namespaces.
package externalrestore

import "time"

// CheckpointAPIRequest is the JSON body for POST /checkpoint.
type CheckpointAPIRequest struct {
	ContainerID   string `json:"container_id"`
	ContainerName string `json:"container_name,omitempty"`
	CheckpointID  string `json:"checkpoint_id,omitempty"`
	PodName       string `json:"pod_name,omitempty"`
	PodNamespace  string `json:"pod_namespace,omitempty"`
	RequestID     string `json:"request_id,omitempty"`
}

// CheckpointAPIResponse is the JSON response for POST /checkpoint.
type CheckpointAPIResponse struct {
	Success      bool   `json:"success"`
	CheckpointID string `json:"checkpoint_id,omitempty"`
	Message      string `json:"message,omitempty"`
	Error        string `json:"error,omitempty"`
}

// RestoreAPIRequest is the JSON body for POST /restore.
type RestoreAPIRequest struct {
	CheckpointID  string `json:"checkpoint_id"`
	PodName       string `json:"pod_name"`
	PodNamespace  string `json:"pod_namespace"`
	ContainerName string `json:"container_name"`
	RequestID     string `json:"request_id,omitempty"`
}

// RestoreAPIResponse is the JSON response for POST /restore.
type RestoreAPIResponse struct {
	Success        bool     `json:"success"`
	RestoredPID    int      `json:"restored_pid,omitempty"`
	CompletedSteps []string `json:"completed_steps,omitempty"`
	Error          string   `json:"error,omitempty"`
}

// HealthResponse is the JSON response for GET /health.
type HealthResponse struct {
	Status   string `json:"status"`
	NodeName string `json:"node_name"`
}

// CheckpointInfo represents information about a stored checkpoint.
type CheckpointInfo struct {
	ID           string    `json:"id"`
	CreatedAt    time.Time `json:"created_at"`
	SourceNode   string    `json:"source_node"`
	ContainerID  string    `json:"container_id"`
	PodName      string    `json:"pod_name"`
	PodNamespace string    `json:"pod_namespace"`
}

// ListCheckpointsResponse is the JSON response for GET /checkpoints.
type ListCheckpointsResponse struct {
	Checkpoints []CheckpointInfo `json:"checkpoints"`
}
