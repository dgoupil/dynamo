#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GTC Benchmark: 8 Aggregated EPD instances on 8 GPUs (Baseline)
#
# Architecture: No disaggregation. Uses vLLM's native data-parallel mode
# to run 8 replicas (one per GPU) behind a single API endpoint.
# Each replica handles the full Encode + Prefill + Decode pipeline.
#
# This serves as the baseline for comparison against disaggregated configs.
#
# Usage:
#   bash 8_agg_epd.sh
#   MODEL="my/model" NUM_PROMPTS=200 bash 8_agg_epd.sh

set -euo pipefail

declare -a PIDS=()

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
NUM_GPUS="${NUM_GPUS:-8}"
LOG_PATH="${LOG_PATH:-./logs/8_agg_epd}"
mkdir -p "$LOG_PATH"

PORT="${PORT:-8000}"                    # matches dynamo frontend default
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

START_TIME=$(date +"%Y%m%d_%H%M%S")

###############################################################################
# Helpers
###############################################################################
wait_for_server() {
    local port=$1
    echo "Waiting for server on port $port ..."
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/health > /dev/null 2>&1; do
            sleep 2
        done" && echo "  -> port $port ready" || { echo "TIMEOUT waiting for port $port"; return 1; }
}

cleanup() {
    echo "Stopping everything..."
    trap - INT TERM USR1

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    kill -- -$$ 2>/dev/null
    echo "All processes stopped."
    exit 0
}

trap cleanup INT TERM USR1

###############################################################################
# Launch vLLM with data-parallel across 8 GPUs (single frontend, 8 replicas)
###############################################################################
echo "Starting vLLM data-parallel server (${NUM_GPUS} replicas) on port $PORT ..."

vllm serve "$MODEL" \
    --port "$PORT" \
    --data-parallel-size "$NUM_GPUS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    >"${LOG_PATH}/server_${START_TIME}.log" 2>&1 &

PIDS+=($!)

wait_for_server "$PORT"

echo "============================================="
echo " 8 Aggregated EPD - Server is up!"
echo " Endpoint: http://localhost:$PORT"
echo " Replicas: $NUM_GPUS (one per GPU)"
echo "============================================="

###############################################################################
# Ready - wait for user to run aiperf or Ctrl+C to stop
###############################################################################
echo "Ready for benchmarking on http://localhost:$PORT"
echo "Press Ctrl+C to stop all services."
wait
