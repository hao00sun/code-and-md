#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wu/miniconda3/envs/lerobot/bin/python}"
ROOT="/media/wu/data/SUN_ht/innov"
CONFIG_PATH="${1:-${SCRIPT_DIR}/flow_matching_left_arm.yaml}"
PRETRAINED_ROOT="${ROOT}/pretrained_weights"
LOCAL_CLIP_ROOT="${PRETRAINED_ROOT}/clip-vit-base-patch16"
TIME_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="flow_matching_left_arm_${TIME_TAG}"
OUTPUT_DIR="${ROOT}/runs/${RUN_NAME}"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log"

export HF_HOME="${ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${ROOT}/cache/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python not found: ${PYTHON_BIN}"
    exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG_PATH}"
    exit 1
fi
if [[ ! -f "${LOCAL_CLIP_ROOT}/config.json" || ! -f "${LOCAL_CLIP_ROOT}/pytorch_model.bin" ]]; then
    echo "[ERROR] Local CLIP weights are incomplete: ${LOCAL_CLIP_ROOT}"
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")" "${LOG_DIR}" "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}"
cp "${CONFIG_PATH}" "${LOG_DIR}/train_config_input.yaml"

echo "[INFO] config     = ${CONFIG_PATH}"
echo "[INFO] dataset    = ${ROOT}/datasets/innov_0617_1554_left_arm"
echo "[INFO] pretrained = ${LOCAL_CLIP_ROOT}"
echo "[INFO] output_dir = ${OUTPUT_DIR}"
echo "[INFO] log_file   = ${LOG_FILE}"

"${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
    --config_path "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --job_name "${RUN_NAME}" \
    2>&1 | tee "${LOG_FILE}"

LATEST_MODEL="${OUTPUT_DIR}/checkpoints/030000/pretrained_model"
if [[ -d "${LATEST_MODEL}" ]]; then
    mkdir -p "${ROOT}/inference_models"
    ln -sfn "${LATEST_MODEL}" "${ROOT}/inference_models/flow_matching_left_arm_latest"
    echo "[DONE] inference model link = ${ROOT}/inference_models/flow_matching_left_arm_latest"
fi
