#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# E/PD (Encode + aggregated Prefill-Decode) multimodal deployment
#
# Architecture: 2-component disaggregation (no separate processor process)
# - Encode Worker: Dedicated vision encoder that extracts image embeddings
# - PD Worker: Aggregated prefill/decode worker that receives requests from
#              the frontend, routes images to encode workers, and runs full
#              inference. (Uses --dyn-route-to-encoder for encode routing)
#
# Benefits: Decouples vision encoding from LLM inference, enables independent
#           scaling of encode vs inference, one fewer process than the old
#           Processor+Encode+PD topology.
# For full P/D disaggregation, see disagg_multimodal_epd.sh
# For standard single-worker deployment, see agg_multimodal.sh

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "E/PD Multimodal Serving"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "=================================================="

# Start frontend (HTTP endpoint)
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# Set max model length (respect env var override)
DYN_MAX_MODEL_LEN=${DYN_MAX_MODEL_LEN:-16384}

# Disable prefix caching if requested (enabled by default in vLLM)
DYN_DISABLE_PREFIX_CACHE=${DYN_DISABLE_PREFIX_CACHE:-false}
PREFIX_CACHE_ARGS=""
if [[ "$DYN_DISABLE_PREFIX_CACHE" == "true" ]]; then
    PREFIX_CACHE_ARGS="--no-enable-prefix-caching"
fi

EXTRA_ARGS="--gpu-memory-utilization 0.85 --max-model-len $DYN_MAX_MODEL_LEN $PREFIX_CACHE_ARGS"

# Multimodal embedding cache for PD worker (0 = disabled)
DYN_EMBED_CACHE_GB=${DYN_EMBED_CACHE_GB:-0}
PD_EXTRA_ARGS=""
if [[ "$DYN_EMBED_CACHE_GB" != "0" ]]; then
    PD_EXTRA_ARGS="--dyn-multimodal-embedding-cache-capacity-gb $DYN_EMBED_CACHE_GB"
fi

# Launch Encode Worker + PD Worker
DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-1}
DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.75}
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (mem: $DYN_ENCODE_GPU_MEM)..."
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME --gpu-memory-utilization $DYN_ENCODE_GPU_MEM --max-model-len $DYN_MAX_MODEL_LEN &

DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-2}
echo "Starting PD worker on GPU $DYN_PD_WORKER_GPU ..."
CUDA_VISIBLE_DEVICES=$DYN_PD_WORKER_GPU python -m dynamo.vllm --multimodal-worker --dyn-route-to-encoder --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $EXTRA_ARGS $PD_EXTRA_ARGS &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Wait for all background processes to complete
wait
