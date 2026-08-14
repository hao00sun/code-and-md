#!/usr/bin/env bash
set -euo pipefail

# Download a LeRobot MultiTaskDiT pretrained checkpoint through a local proxy.
#
# Usage:
#   ./DIT/download_dit_pretrained_with_proxy.sh
#
# Optional overrides:
#   PROXY_URL=http://127.0.0.1:7890
#   HF_MODEL_ID=NONHUMAN-RESEARCH/multi-task-dit-training-fruits
#   PRETRAINED_PATH=/media/wu/data/SUN_ht/dit/pretrained_weights/multi_task_dit_flow_matching_14d_base
#   HF_CACHE_BASE=/media/wu/data/SUN_ht/dit/cache/huggingface

HF_MODEL_ID="${HF_MODEL_ID:-NONHUMAN-RESEARCH/multi-task-dit-training-fruits}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7890}"
DIT_ROOT="${DIT_ROOT:-/media/wu/data/SUN_ht/dit}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${DIT_ROOT}/pretrained_weights/multi_task_dit_flow_matching_14d_base}"
HF_CACHE_BASE="${HF_CACHE_BASE:-${DIT_ROOT}/cache/huggingface}"

export HF_HOME="${HF_CACHE_BASE}"
export HF_HUB_CACHE="${HF_CACHE_BASE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_BASE}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"

export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"

mkdir -p "${PRETRAINED_PATH}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

echo "[INFO] HF_MODEL_ID     = ${HF_MODEL_ID}"
echo "[INFO] PRETRAINED_PATH = ${PRETRAINED_PATH}"
echo "[INFO] HF_HOME         = ${HF_HOME}"
echo "[INFO] PROXY_URL       = ${PROXY_URL}"

if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD_CMD=(hf download)
else
    HF_DOWNLOAD_CMD=(huggingface-cli download)
fi

"${HF_DOWNLOAD_CMD[@]}" "${HF_MODEL_ID}" \
    --local-dir "${PRETRAINED_PATH}"

# MultiTaskDiT uses CLIP by default. Keep a local copy so training can run offline.
"${HF_DOWNLOAD_CMD[@]}" openai/clip-vit-base-patch16

echo "[INFO] Download complete."
echo "[INFO] You can now train with:"
echo "  ./DIT/run_dit_flow_matching_train_with_logs.sh"
