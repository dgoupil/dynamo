// config.go provides configuration loading for the checkpoint agent.
package main

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/externalrestore"
)

// ConfigMapPath is the default path where the ConfigMap is mounted.
const ConfigMapPath = "/etc/chrek/config.yaml"

// FullConfig is the root configuration structure loaded from the ConfigMap.
type FullConfig struct {
	Agent      AgentConfig              `yaml:"agent"`
	Checkpoint checkpoint.CheckpointSpec `yaml:"checkpoint"`
}

// AgentConfig holds the runtime configuration for the checkpoint agent daemon.
type AgentConfig struct {
	// SocketPath is the UDS socket path for the API server.
	// Default: /var/run/chrek/chrek.sock
	SocketPath string `yaml:"socketPath"`

	// ListenAddr is the TCP address for health checks (optional, for K8s probes).
	ListenAddr string `yaml:"listenAddr"`

	// EnableWatcher enables automatic pod watching alongside the UDS server.
	EnableWatcher bool `yaml:"enableWatcher"`

	// NodeName is the Kubernetes node name (from NODE_NAME env, downward API)
	NodeName string `yaml:"-"`

	// RestrictedNamespace restricts pod watching to this namespace (optional)
	RestrictedNamespace string `yaml:"-"`
}

// LoadConfig loads the full configuration from a YAML file.
func LoadConfig(path string) (*FullConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	cfg := &FullConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file %s: %w", path, err)
	}

	cfg.Agent.loadEnvOverrides()
	cfg.applyDefaults()

	return cfg, nil
}

// LoadConfigOrDefault loads configuration from a file, falling back to defaults if the file doesn't exist.
func LoadConfigOrDefault(path string) (*FullConfig, error) {
	cfg, err := LoadConfig(path)
	if err != nil {
		if os.IsNotExist(err) {
			cfg = &FullConfig{}
			cfg.Agent.loadEnvOverrides()
			cfg.applyDefaults()
			return cfg, nil
		}
		return nil, err
	}
	return cfg, nil
}

// loadEnvOverrides applies environment variable overrides to the AgentConfig.
func (c *AgentConfig) loadEnvOverrides() {
	if v := os.Getenv("NODE_NAME"); v != "" {
		c.NodeName = v
	}
	if v := os.Getenv("RESTRICTED_NAMESPACE"); v != "" {
		c.RestrictedNamespace = v
	}
}

func (c *FullConfig) applyDefaults() {
	if c.Agent.SocketPath == "" {
		c.Agent.SocketPath = externalrestore.DefaultSocketPath
	}
}

// Validate checks that the configuration has valid values.
func (c *FullConfig) Validate() error {
	if c.Agent.SocketPath == "" {
		return fmt.Errorf("agent.socketPath cannot be empty")
	}
	return c.Checkpoint.Validate()
}
