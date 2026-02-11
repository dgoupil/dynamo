// Package main provides the chrek DaemonSet agent.
// The agent runs a UDS HTTP server for checkpoint/restore operations and
// optionally watches pods for automatic checkpointing.
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/externalrestore"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/watcher"
)

func main() {
	cfg, err := LoadConfigOrDefault(ConfigMapPath)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		log.Fatalf("Invalid configuration: %v", err)
	}

	discoveryClient, err := checkpoint.NewDiscoveryClient()
	if err != nil {
		log.Fatalf("Failed to create discovery client: %v", err)
	}
	defer discoveryClient.Close()

	checkpointer := checkpoint.NewCheckpointer(discoveryClient)

	// Create the external restorer
	restorer := externalrestore.NewRestorer(
		externalrestore.RestorerConfig{
			CheckpointBasePath: cfg.Checkpoint.BasePath,
		},
		discoveryClient,
	)

	// Create UDS server
	serverCfg := externalrestore.ServerConfig{
		SocketPath:     cfg.Agent.SocketPath,
		NodeName:       cfg.Agent.NodeName,
		CheckpointSpec: &cfg.Checkpoint,
		CRIUTimeout:    cfg.Checkpoint.CRIU.Timeout,
	}
	srv := externalrestore.NewServer(serverCfg, checkpointer, restorer)

	// Context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	log.Printf("Chrek agent starting (node: %s)", cfg.Agent.NodeName)
	log.Printf("Checkpoint directory: %s", cfg.Checkpoint.BasePath)
	log.Printf("UDS socket: %s", cfg.Agent.SocketPath)

	// Start optional watcher alongside UDS server
	if cfg.Agent.EnableWatcher {
		watcherCfg := watcher.WatcherConfig{
			NodeName:            cfg.Agent.NodeName,
			ListenAddr:          cfg.Agent.ListenAddr,
			RestrictedNamespace: cfg.Agent.RestrictedNamespace,
			AgentSocketPath:     cfg.Agent.SocketPath,
			CheckpointSpec:      &cfg.Checkpoint,
		}
		podWatcher, err := watcher.NewWatcher(watcherCfg, discoveryClient)
		if err != nil {
			log.Fatalf("Failed to create pod watcher: %v", err)
		}
		go func() {
			log.Printf("Pod watcher started (watching for label: %s=true)", checkpoint.KubeLabelCheckpointSource)
			if err := podWatcher.Start(ctx); err != nil {
				log.Printf("Pod watcher error: %v", err)
			}
		}()
	}

	// Handle graceful shutdown
	go func() {
		<-sigChan
		log.Println("Shutting down...")
		cancel()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutdownCancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
	}()

	// Start UDS server (blocks until shutdown)
	if err := srv.Start(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}

	log.Println("Agent stopped")
}
