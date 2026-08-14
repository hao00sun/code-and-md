#!/usr/bin/env bash
set -euo pipefail

# MultiTaskDiT flow-matching fine-tuning launcher.
# Usage:
#   ./DIT/run_dit_flow_matching_train_with_logs.sh DIT/multi_task_dit_flow_matching_train.yaml

CONFIG_PATH="${1:-DIT/multi_task_dit_flow_matching_train.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/wu/miniconda3/envs/lerobot/bin/python}"

DIT_ROOT="${DIT_ROOT:-/media/wu/data/SUN_ht/dit}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${DIT_ROOT}/pretrained_weights/multi_task_dit_flow_matching_14d_base}"
LOG_BASE="${LOG_BASE:-${DIT_ROOT}/logs}"
MODEL_BASE="${MODEL_BASE:-${DIT_ROOT}/runs}"
HF_CACHE_BASE="${HF_CACHE_BASE:-${DIT_ROOT}/cache/huggingface}"

export HF_HOME="${HF_CACHE_BASE}"
export HF_HUB_CACHE="${HF_CACHE_BASE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_BASE}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[ERROR] config file not found: ${CONFIG_PATH}"
    exit 1
fi

missing=()
for required in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [ ! -f "${PRETRAINED_PATH}/${required}" ]; then
        missing+=("${required}")
    fi
done

if [ "${#missing[@]}" -gt 0 ]; then
    echo "[ERROR] MultiTaskDiT flow-matching checkpoint is incomplete: ${PRETRAINED_PATH}"
    echo "[ERROR] Missing: ${missing[*]}"
    echo
    echo "Download it first:"
    echo "  HF_MODEL_ID=NONHUMAN-RESEARCH/multi-task-dit-training-fruits ./DIT/download_dit_pretrained_with_proxy.sh"
    exit 1
fi

mkdir -p "${LOG_BASE}" "${MODEL_BASE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="multi_task_dit_flow_matching_arx_${TIME_TAG}"
LOG_ROOT="${LOG_BASE}/${RUN_NAME}"
OUTPUT_DIR="${MODEL_BASE}/${RUN_NAME}"

mkdir -p "${LOG_ROOT}"

echo "[INFO] CONFIG_PATH = ${CONFIG_PATH}"
echo "[INFO] PRETRAINED  = ${PRETRAINED_PATH}"
echo "[INFO] OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "[INFO] LOG_ROOT    = ${LOG_ROOT}"
echo "[INFO] HF_HOME     = ${HF_HOME}"

cp "${CONFIG_PATH}" "${LOG_ROOT}/train_config_input.yaml"

{
    echo "========== date =========="
    date
    echo
    echo "========== hostname =========="
    hostname
    echo
    echo "========== pwd =========="
    pwd
    echo
    echo "========== config path =========="
    echo "${CONFIG_PATH}"
    echo
    echo "========== git rev =========="
    git rev-parse HEAD || true
    echo
    echo "========== git branch =========="
    git branch --show-current || true
    echo
    echo "========== git status =========="
    git status || true
    echo
    echo "========== python =========="
    echo "PYTHON_BIN=${PYTHON_BIN}"
    "${PYTHON_BIN}" --version || true
    echo
    echo "========== torch / cuda =========="
    "${PYTHON_BIN}" - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu count:", torch.cuda.device_count())
        print("gpu 0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", repr(exc))
PY
    echo
    echo "========== nvidia-smi =========="
    nvidia-smi || true
    echo
    echo "========== important env =========="
    env | grep -E "HF_|TRANSFORMERS|CUDA|PYTORCH|WANDB|NO_PROXY|no_proxy" || true
} > "${LOG_ROOT}/env_snapshot.txt" 2>&1

(
    echo "timestamp,gpu_name,gpu_util_percent,memory_used_mib,memory_total_mib,temp_c,power_w"
    while true; do
        nvidia-smi \
          --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
          --format=csv,noheader,nounits
        sleep 10
    done
) > "${LOG_ROOT}/gpu_usage.csv" 2>&1 &

GPU_LOG_PID=$!

cleanup() {
    kill "${GPU_LOG_PID}" 2>/dev/null || true
}
trap cleanup EXIT

TRAIN_CMD=(
    "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train
    "--config_path=${CONFIG_PATH}"
    "--policy.path=${PRETRAINED_PATH}"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=${RUN_NAME}"
)

printf '%q ' "${TRAIN_CMD[@]}" > "${LOG_ROOT}/train_command.txt"
printf '\n' >> "${LOG_ROOT}/train_command.txt"

printf '[INFO] TRAIN_CMD = ' | tee "${LOG_ROOT}/train_stdout.log"
printf '%q ' "${TRAIN_CMD[@]}" | tee -a "${LOG_ROOT}/train_stdout.log"
printf '\n' | tee -a "${LOG_ROOT}/train_stdout.log"
echo "[INFO] Start training..." | tee -a "${LOG_ROOT}/train_stdout.log"

set +e
"${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_ROOT}/train_stdout.log"
RET=${PIPESTATUS[0]}
set -e

echo "[INFO] train return code = ${RET}" | tee -a "${LOG_ROOT}/train_stdout.log"

{
    echo "========== output tree =========="
    find "${OUTPUT_DIR}" -maxdepth 5 -type f | sort || true
    echo
    echo "========== checkpoints =========="
    find "${OUTPUT_DIR}" \( -path "*checkpoint*" -o -path "*pretrained_model*" \) | sort || true
} > "${LOG_ROOT}/after_train_files.txt" 2>&1

exit "${RET}"
