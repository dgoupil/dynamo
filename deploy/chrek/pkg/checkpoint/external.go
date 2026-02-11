// external.go defines types for external (DaemonSet-driven) restore.
// These are saved in the checkpoint manifest at dump time and consumed at restore time.
package checkpoint

// ExternalRestoreConfig holds metadata needed by the DaemonSet to perform
// an external restore (rootfs replay, CRIU via nsenter, CUDA restore).
// Saved in CheckpointManifest at checkpoint time, read at restore time.
type ExternalRestoreConfig struct {
	// CUDA holds per-process CUDA checkpoint data (nil when no CUDA PIDs detected).
	CUDA *CUDARestoreData `yaml:"cuda,omitempty"`
}

// CUDARestoreData captures CUDA state from checkpoint time for restore.
// The DaemonSet uses this to drive cuda-checkpoint restore+unlock after CRIU restore.
type CUDARestoreData struct {
	// PIDs are the cgroup PIDs that had CUDA contexts at checkpoint time.
	// At restore, the DaemonSet enumerates cgroup PIDs and maps them to these.
	PIDs []int `yaml:"pids"`

	// SourceGPUUUIDs are the GPU UUIDs visible to the source pod (from PodResources API).
	// Used with --device-map when source and target GPUs differ.
	SourceGPUUUIDs []string `yaml:"sourceGpuUuids"`

	// Locked indicates whether the CUDA PIDs were successfully locked before dump.
	// If true, restore must unlock after cuda-checkpoint restore.
	Locked bool `yaml:"locked"`
}
