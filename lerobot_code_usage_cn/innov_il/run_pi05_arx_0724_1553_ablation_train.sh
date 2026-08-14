#!/usr/bin/env bash
set -euo pipefail

# Sequential PI05 training jobs for the converted Innov 0730 four-camera datasets.
#
# Runs trainings one by one:
#   1) /data/SUN_ht/datasets/innov_0730_4cam_ep0_92_clean
#      v3.0 four-camera ARX bimanual data, full 14-dim action loss
#   2) /data/SUN_ht/datasets/innov_0730_4cam_ep0_92_front
#      v3.0 four-camera ARX bimanual data, full 14-dim action loss
#
# These two roots correspond to the --sources-style inputs used by
# merge_lerobot_v21_arx_bimanual.py, but are already converted in-place to
# LeRobot v3.0 for training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

PYTHON_BIN="${PYTHON_BIN:-/home/wu/miniconda3/envs/lerobot/bin/python}"
CLEAN_DATASET_REPO_ID="${CLEAN_DATASET_REPO_ID:-innov_0730_4cam_ep0_92_clean}"
CLEAN_DATASET_ROOT="${CLEAN_DATASET_ROOT:-/data/SUN_ht/datasets/innov_0730_4cam_ep0_92_clean}"
FRONT_DATASET_REPO_ID="${FRONT_DATASET_REPO_ID:-innov_0730_4cam_ep0_92_front}"
FRONT_DATASET_ROOT="${FRONT_DATASET_ROOT:-/data/SUN_ht/datasets/innov_0730_4cam_ep0_92_front}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/data/SUN_ht/pi/pretrained_weights/pi05_base}"

BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-16000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/bigdata/SUN_ht/runs}"
LOG_BASE="${LOG_BASE:-/data/SUN_ht/pi/logs}"
CONFIG_DIR="${CONFIG_DIR:-${REPO_ROOT}/innov_il/generated_train_configs}"
DRY_RUN="${DRY_RUN:-0}"
RUN_FRONT="${RUN_FRONT:-1}"

mkdir -p "${CONFIG_DIR}" "${LOG_BASE}"

export HF_HOME="${HF_HOME:-/data/SUN_ht/pi/cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CLEAN_DATASET_REPO_ID
export CLEAN_DATASET_ROOT
export FRONT_DATASET_REPO_ID
export FRONT_DATASET_ROOT
export PRETRAINED_PATH
export BATCH_SIZE
export STEPS
export SAVE_FREQ
export LOG_FREQ
export NUM_WORKERS
export OUTPUT_BASE
export LOG_BASE
export RUN_FRONT

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python not found: ${PYTHON_BIN}"
    exit 1
fi

if [[ ! -f "${PRETRAINED_PATH}/model.safetensors" ]]; then
    echo "[ERROR] PI05 pretrained weights not found: ${PRETRAINED_PATH}"
    exit 1
fi

validate_dataset() {
    local dataset_root="$1"

    if [[ ! -d "${dataset_root}" ]]; then
        echo "[ERROR] Dataset not found: ${dataset_root}"
        exit 1
    fi

    "${PYTHON_BIN}" - "${dataset_root}" <<'PY'
import json
from pathlib import Path

root = Path(__import__("sys").argv[1])
info = json.loads((root / "meta/info.json").read_text())
if info.get("codebase_version") != "v3.0":
    raise SystemExit(f"[ERROR] {root} is {info.get('codebase_version')}, expected v3.0")
video_keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
if len(video_keys) != 4:
    raise SystemExit(f"[ERROR] {root} has {len(video_keys)} video keys, expected 4: {video_keys}")
for key in ("action", "observation.state"):
    shape = info["features"][key]["shape"]
    if shape != [14]:
        raise SystemExit(f"[ERROR] {root} feature {key} shape is {shape}, expected [14]")
PY
}

validate_dataset "${CLEAN_DATASET_ROOT}"
if [[ "${RUN_FRONT}" == "1" || "${RUN_FRONT}" == "true" ]]; then
    validate_dataset "${FRONT_DATASET_ROOT}"
fi

write_config() {
    local config_path="$1"
    local job_name="$2"
    local grad_accum="$3"
    local dataset_repo_id="$4"
    local dataset_root="$5"

    "${PYTHON_BIN}" - "$config_path" "$job_name" "$grad_accum" "$dataset_repo_id" "$dataset_root" <<'PY'
from pathlib import Path
import os
import sys

config_path = Path(sys.argv[1])
job_name = sys.argv[2]
grad_accum = int(sys.argv[3])
dataset_repo_id = sys.argv[4]
dataset_root = sys.argv[5]

pretrained_path = os.environ["PRETRAINED_PATH"]
batch_size = os.environ["BATCH_SIZE"]
steps = os.environ["STEPS"]
save_freq = os.environ["SAVE_FREQ"]
log_freq = os.environ["LOG_FREQ"]
num_workers = os.environ["NUM_WORKERS"]
output_base = os.environ["OUTPUT_BASE"]
log_base = os.environ["LOG_BASE"]

text = f"""dataset:
  repo_id: {dataset_repo_id}
  root: {dataset_root}
  use_imagenet_stats: false
policy:
  type: pi05
  repo_id: wu/{job_name}
  pretrained_path: {pretrained_path}
  compile_model: false
  gradient_checkpointing: true
  dtype: bfloat16
  freeze_vision_encoder: false
  train_expert_only: false
  device: cuda
  normalization_mapping:
    ACTION: MEAN_STD
    STATE: MEAN_STD
    VISUAL: IDENTITY
  push_to_hub: false

output_dir: {output_base}/{job_name}_{{timestamp}}
log_dir: {log_base}/{job_name}_{{timestamp}}
job_name: {job_name}
batch_size: {batch_size}
gradient_accumulation_steps: {grad_accum}
steps: {steps}
save_freq: {save_freq}
log_freq: {log_freq}
num_workers: {num_workers}

wandb:
  enable: false
"""
config_path.write_text(text, encoding="utf-8")
print(config_path)
PY
}

run_one() {
    local job_name="$1"
    local grad_accum="$2"
    local dataset_repo_id="$3"
    local dataset_root="$4"
    local config_path="${CONFIG_DIR}/${job_name}.yaml"
    local run_ts
    local log_file

    if [[ ! -d "${dataset_root}" ]]; then
        echo "[ERROR] Dataset not found for ${job_name}: ${dataset_root}"
        exit 1
    fi

    write_config "${config_path}" "${job_name}" "${grad_accum}" "${dataset_repo_id}" "${dataset_root}"
    run_ts="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_BASE}/${job_name}_train_${run_ts}.log"

    echo "[INFO] =================================================="
    echo "[INFO] job_name     = ${job_name}"
    echo "[INFO] config       = ${config_path}"
    echo "[INFO] grad_accum   = ${grad_accum}"
    echo "[INFO] dataset_repo = ${dataset_repo_id}"
    echo "[INFO] dataset_root = ${dataset_root}"
    echo "[INFO] log_file     = ${log_file}"
    echo "[INFO] =================================================="

    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
        echo "[INFO] DRY_RUN enabled; generated config only, training skipped."
        return
    fi

    env -u ALL_PROXY -u all_proxy \
        HF_HOME="${HF_HOME}" \
        TRANSFORMERS_OFFLINE=1 \
        HF_HUB_OFFLINE=1 \
        "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
        --config_path="${config_path}" \
        2>&1 | tee "${log_file}"
}

cd "${REPO_ROOT}"

run_one "pi05_innov_0730_4cam_ep0_92_clean_ga1" "1" "${CLEAN_DATASET_REPO_ID}" "${CLEAN_DATASET_ROOT}"
if [[ "${RUN_FRONT}" == "1" || "${RUN_FRONT}" == "true" ]]; then
    run_one "pi05_innov_0730_4cam_ep0_92_front_ga1" "1" "${FRONT_DATASET_REPO_ID}" "${FRONT_DATASET_ROOT}"
fi

echo "[INFO] All trainings finished."
