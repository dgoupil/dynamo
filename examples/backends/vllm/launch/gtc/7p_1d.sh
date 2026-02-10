#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GTC Benchmark: 7 Prefill + 1 Decode on 8 GPUs
#
# Architecture: PD disaggregation (no separate encoder)
# - GPU 0-6: 7 prefill workers (handle encoding + prefill, KV producer via NIXL)
# - GPU 7:   1 decode worker (KV consumer via NIXL, generates tokens)
# - PD proxy on port 8000 orchestrates the P -> D pipeline
#
# Each prefill instance handles the full vision encoding + prefill in one shot,
# then pushes KV cache to the decode instance via NixlConnector.
#
# Usage:
#   bash 7p_1d.sh
#   MODEL="my/model" bash 7p_1d.sh

set -euo pipefail

declare -a PIDS=()

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
LOG_PATH="${LOG_PATH:-./logs/7p_1d}"
mkdir -p "$LOG_PATH"

PREFILL_BASE_PORT="${PREFILL_BASE_PORT:-19530}"  # prefill: 19530..19536
DECODE_PORT="${DECODE_PORT:-19537}"              # decode on GPU 7
PROXY_PORT="${PROXY_PORT:-8000}"                 # matches dynamo frontend default

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_TIME=$(date +"%Y%m%d_%H%M%S")

export UCX_TLS=all
export UCX_NET_DEVICES=all

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
# 7 Prefill workers (GPU 0-6) - encoding + prefill, KV producer
###############################################################################
PREFILL_URLS=""
NIXL_PREFILL_BASE_PORT=5559

for i in $(seq 0 6); do
    PORT=$((PREFILL_BASE_PORT + i))
    NIXL_PORT=$((NIXL_PREFILL_BASE_PORT + i))
    LOG_FILE="${LOG_PATH}/prefill${i}_gpu${i}_${START_TIME}.log"

    echo "Starting prefill worker $i on GPU $i, port $PORT (NIXL: $NIXL_PORT) ..."

    CUDA_VISIBLE_DEVICES="$i" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT" \
    vllm serve "$MODEL" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer"
        }' \
        >"$LOG_FILE" 2>&1 &

    PIDS+=($!)

    if [ -z "$PREFILL_URLS" ]; then
        PREFILL_URLS="http://localhost:$PORT"
    else
        PREFILL_URLS="$PREFILL_URLS,http://localhost:$PORT"
    fi
done

###############################################################################
# Decode worker (GPU 7) - KV consumer via NIXL
###############################################################################
echo "Starting decode worker on GPU 7, port $DECODE_PORT ..."

CUDA_VISIBLE_DEVICES=7 \
VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
vllm serve "$MODEL" \
    --port "$DECODE_PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }' \
    >"${LOG_PATH}/decode_gpu7_${START_TIME}.log" 2>&1 &

PIDS+=($!)

# Wait for all instances
for i in $(seq 0 6); do
    wait_for_server $((PREFILL_BASE_PORT + i))
done
wait_for_server "$DECODE_PORT"

###############################################################################
# PD proxy on port 8000
###############################################################################
echo "Starting PD proxy on port $PROXY_PORT ..."

python "$SCRIPT_DIR/disagg_pd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --prefill-servers-urls "$PREFILL_URLS" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    >"${LOG_PATH}/proxy_${START_TIME}.log" 2>&1 &

PIDS+=($!)
sleep 3

echo "============================================="
echo " 7P + 1D - All services are up!"
echo " Proxy:   http://localhost:$PROXY_PORT"
echo " Prefill: $PREFILL_URLS (GPUs 0-6)"
echo " Decode:  http://localhost:$DECODE_PORT (GPU 7)"
echo "============================================="

###############################################################################
# Ready - wait for user to run aiperf or Ctrl+C to stop
###############################################################################
echo "Ready for benchmarking on http://localhost:$PROXY_PORT"
echo "Press Ctrl+C to stop all services."
wait
