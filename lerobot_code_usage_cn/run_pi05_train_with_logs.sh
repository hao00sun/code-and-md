#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PI0.5 / PI05 training launcher with full logs
# Usage:
#   ./run_pi05_train_with_logs.sh configs_local/arx_pi05_train.yaml
# ============================================================

CONFIG_PATH="${1:-pi05_semantic.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/wu/miniconda3/envs/lerobot/bin/python}"

PI_ROOT="/data/SUN_ht/pi"
PRETRAINED_PATH="${PI_ROOT}/pretrained_weights/pi05_base"
LOG_BASE="${PI_ROOT}/logs"
MODEL_BASE="/mnt/bigdata/SUN_ht/runs"
HF_CACHE_BASE="${PI_ROOT}/cache/huggingface"
PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-${HF_CACHE_BASE}/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}"

export HF_HOME="${HF_CACHE_BASE}"
export HF_HUB_CACHE="${HF_CACHE_BASE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_BASE}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Avoid httpx failing on proxy values like socks://127.0.0.1:7890 when
# all model/tokenizer files are already available in the local HF cache.
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY

if [ ! -f "${PRETRAINED_PATH}/config.json" ] || [ ! -f "${PRETRAINED_PATH}/model.safetensors" ]; then
    echo "[ERROR] PI0.5 pretrained weights are incomplete: ${PRETRAINED_PATH}"
    exit 1
fi

if [ ! -f "${PALIGEMMA_TOKENIZER_PATH}/config.json" ] || [ ! -f "${PALIGEMMA_TOKENIZER_PATH}/tokenizer.json" ]; then
    echo "[ERROR] Local PaliGemma tokenizer is incomplete: ${PALIGEMMA_TOKENIZER_PATH}"
    exit 1
fi

mkdir -p "${LOG_BASE}" "${MODEL_BASE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# PI0.5's processor loads this tokenizer separately from the model checkpoint.
# Validate it before spending time and GPU memory loading the 14 GB model.
POLICY_PREPROCESSOR="${PRETRAINED_PATH}/policy_preprocessor.json"
POLICY_PREPROCESSOR_BAK="${POLICY_PREPROCESSOR}.bak_tokenizer_repo_id"
"${PYTHON_BIN}" - "${POLICY_PREPROCESSOR}" "${POLICY_PREPROCESSOR_BAK}" "${PALIGEMMA_TOKENIZER_PATH}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

preprocessor = Path(sys.argv[1])
backup = Path(sys.argv[2])
tokenizer_path = sys.argv[3]

data = json.loads(preprocessor.read_text())
changed = False
for step in data.get("steps", []):
    if step.get("registry_name") == "tokenizer_processor":
        config = step.setdefault("config", {})
        if config.get("tokenizer_name") != tokenizer_path:
            if not backup.exists():
                shutil.copy2(preprocessor, backup)
            config["tokenizer_name"] = tokenizer_path
            changed = True

if changed:
    preprocessor.write_text(json.dumps(data, indent=2) + "\n")
PY

if ! "${PYTHON_BIN}" - "${PALIGEMMA_TOKENIZER_PATH}" <<'PY'
import sys
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
PY
then
    echo "[ERROR] PaliGemma tokenizer cannot be loaded locally: ${PALIGEMMA_TOKENIZER_PATH}"
    exit 1
fi

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_PREFIX="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(cfg.get("job_name") or "pi05_innov_train")
PY
)"
RUN_NAME="${RUN_PREFIX}_${TIME_TAG}"
LOG_ROOT="${LOG_BASE}/${RUN_NAME}"

mkdir -p "${LOG_ROOT}"

echo "[INFO] CONFIG_PATH = ${CONFIG_PATH}"
echo "[INFO] PRETRAINED  = ${PRETRAINED_PATH}"
echo "[INFO] TOKENIZER   = ${PALIGEMMA_TOKENIZER_PATH}"
echo "[INFO] MODEL_ROOT = ${MODEL_BASE}/${RUN_NAME}"
echo "[INFO] LOG_ROOT    = ${LOG_ROOT}"

# 保存原始训练配置
if [ -f "${CONFIG_PATH}" ]; then
    cp "${CONFIG_PATH}" "${LOG_ROOT}/train_config_input.yaml"
else
    echo "[ERROR] config file not found: ${CONFIG_PATH}"
    exit 1
fi

# 保存环境和代码快照
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
    echo "========== git diff =========="
    git diff || true

    echo
    echo "========== python =========="
    echo "PYTHON_BIN=${PYTHON_BIN}"
    "${PYTHON_BIN}" --version || true

    echo
    echo "========== conda env =========="
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"
    conda info --envs || true

    echo
    echo "========== conda list =========="
    conda list || true

    echo
    echo "========== pip freeze =========="
    "${PYTHON_BIN}" -m pip freeze || true

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
except Exception as e:
    print("torch check failed:", repr(e))
PY

    echo
    echo "========== nvidia-smi =========="
    nvidia-smi || true

    echo
    echo "========== important env =========="
    env | grep -E "HF_|TRANSFORMERS|CUDA|PYTORCH|WANDB|http_proxy|https_proxy|all_proxy|NO_PROXY|no_proxy" || true

} > "${LOG_ROOT}/env_snapshot.txt" 2>&1

# 后台记录 GPU 使用情况
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
    echo "[INFO] cleaning up..."
    kill "${GPU_LOG_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[INFO] GPU logger PID = ${GPU_LOG_PID}"

# 训练命令：权重、模型输出和日志统一存放在 PI_ROOT 下。
TRAIN_CMD=(
    "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train
    "--config_path=${CONFIG_PATH}"
    "--output_dir=${MODEL_BASE}/${RUN_NAME}"
    "--log_dir=${LOG_ROOT}"
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

# 训练结束后保存输出目录结构
{
    echo "========== output tree =========="
    find "${MODEL_BASE}/${RUN_NAME}" -maxdepth 5 -type f | sort || true

    echo
    echo "========== checkpoints =========="
    find "${MODEL_BASE}/${RUN_NAME}" \( -path "*checkpoint*" -o -path "*pretrained_model*" \) | sort || true
} > "${LOG_ROOT}/after_train_files.txt" 2>&1

exit "${RET}"
