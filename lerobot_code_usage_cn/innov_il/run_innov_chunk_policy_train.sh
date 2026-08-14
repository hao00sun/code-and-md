#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_KIND="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/wu/miniconda3/envs/lerobot/bin/python}"
BIG_DISK="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f"
INNOV_ROOT="${BIG_DISK}/innov"
ARTIFACT_ROOT="${INNOV_ROOT}/innov_il"

case "${POLICY_KIND}" in
    diffusion)
        CONFIG_PATH="${2:-${SCRIPT_DIR}/diffusion_innov_0617_1554.yaml}"
        RUN_PREFIX="diffusion_innov"
        POLICY_ROOT="${ARTIFACT_ROOT}/deffusion"
        ;;
    flow|flow_matching)
        CONFIG_PATH="${2:-${SCRIPT_DIR}/flow_matching_innov_0617_1554.yaml}"
        RUN_PREFIX="flow_matching_innov"
        POLICY_ROOT="${ARTIFACT_ROOT}/follow"
        ;;
    *)
        echo "Usage: $0 {diffusion|flow_matching} [config.yaml]"
        exit 2
        ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python not found: ${PYTHON_BIN}"
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG_PATH}"
    exit 1
fi

PRETRAINED_ROOT="${POLICY_ROOT}/pretrained_weights"
HF_CACHE_BASE="${PRETRAINED_ROOT}/huggingface"

export HF_HOME="${HF_CACHE_BASE}"
export HF_HUB_CACHE="${HF_CACHE_BASE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_BASE}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export TORCH_HOME="${PRETRAINED_ROOT}/torch"

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

if [[ -n "${PROXY_URL:-}" ]]; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export HTTP_PROXY="${PROXY_URL}"
    export HTTPS_PROXY="${PROXY_URL}"
    echo "[INFO] proxy      = ${PROXY_URL}"
fi

TIME_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_PREFIX}_${TIME_TAG}"
OUTPUT_DIR="${POLICY_ROOT}/runs/${RUN_NAME}"
LOG_DIR="${POLICY_ROOT}/logs/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log"

mkdir -p "$(dirname "${OUTPUT_DIR}")" "${LOG_DIR}" \
    "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" \
    "${TORCH_HOME}/hub/checkpoints"

cp "${CONFIG_PATH}" "${LOG_DIR}/train_config_input.yaml"

if [[ "${POLICY_KIND}" == "diffusion" ]]; then
    echo "[INFO] Downloading/checking ImageNet ResNet34 pretrained weights..."
    "${PYTHON_BIN}" -c \
        "from torchvision.models import ResNet34_Weights, resnet34; resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)"
else
    LOCAL_CLIP_ROOT="${PRETRAINED_ROOT}/clip-vit-base-patch16"
    if [[ ! -f "${LOCAL_CLIP_ROOT}/config.json" || ! -f "${LOCAL_CLIP_ROOT}/pytorch_model.bin" ]]; then
        echo "[ERROR] Local CLIP weights are incomplete: ${LOCAL_CLIP_ROOT}"
        exit 1
    fi
    echo "[INFO] Checking local CLIP pretrained weights..."
    "${PYTHON_BIN}" -c \
        "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('${LOCAL_CLIP_ROOT}', local_files_only=True); CLIPProcessor.from_pretrained('${LOCAL_CLIP_ROOT}', local_files_only=True)"
fi

echo "[INFO] policy     = ${POLICY_KIND}"
echo "[INFO] config     = ${CONFIG_PATH}"
echo "[INFO] pretrained = ${PRETRAINED_ROOT}"
echo "[INFO] output_dir = ${OUTPUT_DIR}"
echo "[INFO] log_file   = ${LOG_FILE}"
echo "[INFO] python     = ${PYTHON_BIN}"

"${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
    --config_path "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --job_name "${RUN_NAME}" \
    2>&1 | tee "${LOG_FILE}"





