// server.go provides the HTTP-over-UDS server for checkpoint and restore operations.
package externalrestore

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

const (
	// DefaultSocketPath is the default UDS socket path.
	DefaultSocketPath = "/var/run/chrek/chrek.sock"

	// MinWriteTimeout is the minimum write timeout for the HTTP server.
	MinWriteTimeout = 300 * time.Second
)

// ServerConfig holds configuration for the UDS server.
type ServerConfig struct {
	SocketPath     string
	NodeName       string
	CheckpointSpec *checkpoint.CheckpointSpec
	CRIUTimeout    uint32 // CRIU timeout in seconds (from config)
}

// Server is the HTTP-over-UDS server for checkpoint and restore operations.
type Server struct {
	cfg          ServerConfig
	httpServer   *http.Server
	listener     net.Listener
	restorer     *Restorer
	checkpointer *checkpoint.Checkpointer
	log          *logrus.Entry
}

// NewServer creates a new UDS server.
func NewServer(cfg ServerConfig, checkpointer *checkpoint.Checkpointer, restorer *Restorer) *Server {
	s := &Server{
		cfg:          cfg,
		restorer:     restorer,
		checkpointer: checkpointer,
		log:          logrus.WithField("component", "uds-server"),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/checkpoint", s.handleCheckpoint)
	mux.HandleFunc("/checkpoints", s.handleListCheckpoints)
	mux.HandleFunc("/restore", s.handleRestore)

	// WriteTimeout must exceed the CRIU timeout since checkpoint/restore
	// blocks until completion. Add 60s buffer for pre/post work.
	writeTimeout := time.Duration(cfg.CRIUTimeout)*time.Second + 60*time.Second
	if writeTimeout < MinWriteTimeout {
		writeTimeout = MinWriteTimeout
	}

	s.httpServer = &http.Server{
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: writeTimeout,
		IdleTimeout:  120 * time.Second,
	}

	return s
}

// Start begins listening on the UDS socket. Blocks until shutdown.
func (s *Server) Start() error {
	socketPath := s.cfg.SocketPath
	if socketPath == "" {
		socketPath = DefaultSocketPath
	}

	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(socketPath), 0755); err != nil {
		return fmt.Errorf("failed to create socket directory: %w", err)
	}

	// Remove stale socket file
	os.Remove(socketPath)

	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", socketPath, err)
	}
	s.listener = ln

	// Make socket world-accessible so sidecars can connect
	if err := os.Chmod(socketPath, 0666); err != nil {
		s.log.WithError(err).Warn("Failed to chmod socket")
	}

	s.log.WithField("socket", socketPath).Info("UDS server listening")
	return s.httpServer.Serve(ln)
}

// Shutdown gracefully shuts down the server.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, HealthResponse{
		Status:   "healthy",
		NodeName: s.cfg.NodeName,
	})
}

func (s *Server) handleCheckpoint(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CheckpointAPIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, CheckpointAPIResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid request body: %v", err),
		})
		return
	}

	if req.ContainerID == "" {
		writeJSON(w, http.StatusBadRequest, CheckpointAPIResponse{
			Success: false,
			Error:   "container_id is required",
		})
		return
	}

	if req.CheckpointID == "" {
		req.CheckpointID = fmt.Sprintf("ckpt-%d", time.Now().UnixNano())
	}

	params := checkpoint.CheckpointRequest{
		ContainerID:   req.ContainerID,
		ContainerName: req.ContainerName,
		CheckpointID:  req.CheckpointID,
		CheckpointDir: s.cfg.CheckpointSpec.BasePath,
		NodeName:      s.cfg.NodeName,
		PodName:       req.PodName,
		PodNamespace:  req.PodNamespace,
	}

	result, err := s.checkpointer.Checkpoint(r.Context(), params, s.cfg.CheckpointSpec)
	if err != nil {
		s.log.WithError(err).Error("Checkpoint failed")
		writeJSON(w, http.StatusInternalServerError, CheckpointAPIResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Write checkpoint.done marker
	donePath := result.CheckpointDir + "/" + checkpoint.CheckpointDoneFilename
	doneContent := fmt.Sprintf(`{"success":true,"timestamp":"%s"}`, time.Now().UTC().Format(time.RFC3339))
	if err := os.WriteFile(donePath, []byte(doneContent), 0644); err != nil {
		s.log.WithError(err).Error("Failed to write checkpoint.done marker")
		writeJSON(w, http.StatusInternalServerError, CheckpointAPIResponse{
			Success: false,
			Error:   fmt.Sprintf("Checkpoint succeeded but failed to write done marker: %v", err),
		})
		return
	}

	s.log.WithField("checkpoint_id", result.CheckpointID).Info("Checkpoint completed")
	writeJSON(w, http.StatusOK, CheckpointAPIResponse{
		Success:      true,
		CheckpointID: result.CheckpointID,
		Message:      fmt.Sprintf("Checkpoint created at %s", result.CheckpointDir),
	})
}

func (s *Server) handleListCheckpoints(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ids, err := checkpoint.ListCheckpoints(s.cfg.CheckpointSpec.BasePath)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	var checkpoints []CheckpointInfo
	for _, id := range ids {
		meta, err := checkpoint.ReadCheckpointManifest(filepath.Join(s.cfg.CheckpointSpec.BasePath, id))
		if err != nil {
			continue
		}
		checkpoints = append(checkpoints, CheckpointInfo{
			ID:           meta.CheckpointID,
			CreatedAt:    meta.CreatedAt,
			SourceNode:   meta.K8s.SourceNode,
			ContainerID:  meta.K8s.ContainerID,
			PodName:      meta.K8s.PodName,
			PodNamespace: meta.K8s.PodNamespace,
		})
	}

	writeJSON(w, http.StatusOK, ListCheckpointsResponse{Checkpoints: checkpoints})
}

func (s *Server) handleRestore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RestoreAPIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, RestoreAPIResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid request body: %v", err),
		})
		return
	}

	if req.CheckpointID == "" || req.PodName == "" || req.PodNamespace == "" {
		writeJSON(w, http.StatusBadRequest, RestoreAPIResponse{
			Success: false,
			Error:   "checkpoint_id, pod_name, and pod_namespace are required",
		})
		return
	}

	result, err := s.restorer.Restore(r.Context(), req)
	if err != nil {
		s.log.WithError(err).Error("Restore failed")
		writeJSON(w, http.StatusInternalServerError, RestoreAPIResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, result)
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
