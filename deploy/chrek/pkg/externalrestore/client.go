// client.go provides a UDS HTTP client used by the watcher to broker
// checkpoint/restore requests to the local chrek agent API.
package externalrestore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

const unixBaseURL = "http://unix"

// Client is an HTTP-over-UDS client for local agent APIs.
type Client struct {
	socketPath string
	httpClient *http.Client
}

// NewClient creates a client that connects to the local agent UDS socket.
func NewClient(socketPath string) *Client {
	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", socketPath)
		},
		ForceAttemptHTTP2: false,
	}

	return &Client{
		socketPath: socketPath,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   0, // Caller context controls request timeout.
		},
	}
}

// SocketPath returns the configured UDS path.
func (c *Client) SocketPath() string {
	return c.socketPath
}

// Checkpoint calls POST /checkpoint and validates the response.
func (c *Client) Checkpoint(ctx context.Context, req CheckpointAPIRequest) (*CheckpointAPIResponse, error) {
	var resp CheckpointAPIResponse
	if err := c.doJSON(ctx, http.MethodPost, "/checkpoint", req, &resp); err != nil {
		return nil, err
	}
	if !resp.Success {
		return nil, fmt.Errorf("checkpoint request failed: %s", resp.Error)
	}
	return &resp, nil
}

// Restore calls POST /restore and validates the response.
func (c *Client) Restore(ctx context.Context, req RestoreAPIRequest) (*RestoreAPIResponse, error) {
	var resp RestoreAPIResponse
	if err := c.doJSON(ctx, http.MethodPost, "/restore", req, &resp); err != nil {
		return nil, err
	}
	if !resp.Success {
		return nil, fmt.Errorf("restore request failed: %s", resp.Error)
	}
	return &resp, nil
}

func (c *Client) doJSON(ctx context.Context, method, path string, reqBody, respBody interface{}) error {
	var body io.Reader
	if reqBody != nil {
		payload, err := json.Marshal(reqBody)
		if err != nil {
			return fmt.Errorf("failed to marshal request: %w", err)
		}
		body = bytes.NewReader(payload)
	}

	req, err := http.NewRequestWithContext(ctx, method, unixBaseURL+path, body)
	if err != nil {
		return fmt.Errorf("failed to build request: %w", err)
	}
	if reqBody != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	start := time.Now()
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("uds request failed (%s %s): %w", method, path, err)
	}
	defer resp.Body.Close()

	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed reading response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyMsg := strings.TrimSpace(string(payload))
		if bodyMsg == "" {
			bodyMsg = "<empty>"
		}
		return fmt.Errorf("uds request failed (%s %s): status=%d body=%s elapsed=%s", method, path, resp.StatusCode, bodyMsg, time.Since(start))
	}

	if respBody != nil && len(payload) > 0 {
		if err := json.Unmarshal(payload, respBody); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}
