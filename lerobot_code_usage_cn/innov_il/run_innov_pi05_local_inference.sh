#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

PYTHON_BIN="${PYTHON_BIN:-/home/wu/miniconda3/envs/lerobot/bin/python}"
ROBODEPLOY_ROOT="${ROBODEPLOY_ROOT:-/data/SUN_ht/roboploy/robodeploy}"

POLICY_PATH="${POLICY_PATH:-/mnt/bigdata/SUN_ht/runs/pi05_innov_0722_15382026-07-22_17-35-27/checkpoints/002000/pretrained_model}"

LEFT_PORT="${LEFT_PORT:-/dev/ttyACM1}"
RIGHT_PORT="${RIGHT_PORT:-/dev/ttyACM0}"
TASK="${TASK:-Put the water flosser into the box and close the lid.}"
FPS="${FPS:-30}"
DURATION_S="${DURATION_S:-120}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-3}"
ACTION_SMOOTH_MAX_STEP="${ACTION_SMOOTH_MAX_STEP:-0.05}"
ACTION_FILTER_ALPHA="${ACTION_FILTER_ALPHA:-0.5}" # 1 disables EMA filtering; lower is smoother
ACTION_MAX_DELTA="${ACTION_MAX_DELTA:-0.05}"      # max per-control-step action change; <=0 disables
ACTION_CLIP_MIN="${ACTION_CLIP_MIN:-}"            # empty disables; 1 value or 14 comma-separated values
ACTION_CLIP_MAX="${ACTION_CLIP_MAX:-}"            # empty disables; 1 value or 14 comma-separated values
POLICY_ACTION_ARMS="${POLICY_ACTION_ARMS:-auto}"  # auto | left | right | both; maps 7-dim policy actions to one arm
CAMERA_TIMEOUT_MS="${CAMERA_TIMEOUT_MS:-1000}"
CAMERA_RETRIES="${CAMERA_RETRIES:-5}"
SHOW_CAMERAS="${SHOW_CAMERAS:-1}"
CAMERA_PREVIEW_WIDTH="${CAMERA_PREVIEW_WIDTH:-420}"
PRINT_ACTION="${PRINT_ACTION:-1}"
PRINT_ACTION_EVERY="${PRINT_ACTION_EVERY:-30}"
ZERO_ARMS="${ZERO_ARMS:-both}"          # none | left | right | both; arms moved to zero/home when pressing z
ZERO_ONLY="${ZERO_ONLY:-0}"             # 1: move to zero/home pose immediately and exit before policy inference
ZERO_NO_CONFIRM="${ZERO_NO_CONFIRM:-0}" # 1: skip typing ZERO confirmation for ZERO_ONLY only
HOME_JOINTS="${HOME_JOINTS:-0,0,0,0,0,0}"
HOME_GRIPPER="${HOME_GRIPPER:-1.0}"     # gripper target for zero/home; use keep to keep current position
HOME_DURATION_S="${HOME_DURATION_S:-5}"
HOME_FPS="${HOME_FPS:-30}"
HOME_JOINT_VELOCITY="${HOME_JOINT_VELOCITY:-1.0}"
HOME_GRIPPER_VELOCITY="${HOME_GRIPPER_VELOCITY:-1.0}"
ACTION_LOG_BASE="${ACTION_LOG_BASE:-/data/SUN_ht/pi/logs}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
ACTION_LOG_DIR="${ACTION_LOG_DIR:-${ACTION_LOG_BASE}/innov_pi05_local_inference_${RUN_TS}}"
ACTION_LOG_PATH="${ACTION_LOG_PATH:-${ACTION_LOG_DIR}/actions.jsonl}"

DEFAULT_CAMERA_CONFIG="{\"front\":{\"type\":\"intelrealsense\",\"serial_number_or_name\":\"935422072733\",\"width\":848,\"height\":480,\"fps\":30},\"front_1\":{\"type\":\"intelrealsense\",\"serial_number_or_name\":\"938422076287\",\"width\":848,\"height\":480,\"fps\":30},\"left_wrist\":{\"type\":\"intelrealsense\",\"serial_number_or_name\":\"409122273564\",\"width\":640,\"height\":480,\"fps\":30},\"right_wrist\":{\"type\":\"intelrealsense\",\"serial_number_or_name\":\"409122273228\",\"width\":640,\"height\":480,\"fps\":30}}"
CAMERA_CONFIG="${INNOV_CAMERA_CONFIG:-${DEFAULT_CAMERA_CONFIG}}"

export PYTHONPATH="${ROBODEPLOY_ROOT}/src:${REPO_ROOT}/src:${PYTHONPATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_HOME="/data/SUN_ht/pi/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python not found: ${PYTHON_BIN}"
    exit 1
fi

if [[ ! -f "${POLICY_PATH}/model.safetensors" ]]; then
    echo "[ERROR] PI05 checkpoint not found: ${POLICY_PATH}"
    echo "        Set POLICY_PATH=/path/to/checkpoints/xxxxxx/pretrained_model"
    exit 1
fi

mkdir -p "${ACTION_LOG_DIR}"

echo "[INFO] Python      = ${PYTHON_BIN}"
echo "[INFO] Policy      = ${POLICY_PATH}"
echo "[INFO] Left port   = ${LEFT_PORT}"
echo "[INFO] Right port  = ${RIGHT_PORT}"
echo "[INFO] Task        = ${TASK}"
echo "[INFO] Duration    = ${DURATION_S}s"
echo "[INFO] Cameras     = ${CAMERA_CONFIG}"
echo "[INFO] Preview     = ${SHOW_CAMERAS} width=${CAMERA_PREVIEW_WIDTH}"
echo "[INFO] Action log  = ${ACTION_LOG_PATH}"
echo "[INFO] Zero arms   = ${ZERO_ARMS} (hotkey z move-to-zero arms)"
echo "[INFO] Home joints = ${HOME_JOINTS}"
echo "[INFO] Home gripper= ${HOME_GRIPPER}"
echo "[INFO] Act filter  = alpha=${ACTION_FILTER_ALPHA} max_delta=${ACTION_MAX_DELTA} smooth=${ACTION_SMOOTH_MAX_STEP}"
echo "[INFO] Act clip    = min=${ACTION_CLIP_MIN:-off} max=${ACTION_CLIP_MAX:-off}"
echo "[INFO] Policy arms = ${POLICY_ACTION_ARMS}"

EXTRA_ARGS=()
if [[ "${PRINT_ACTION}" == "1" || "${PRINT_ACTION}" == "true" ]]; then
    EXTRA_ARGS+=(--print_action)
fi
if [[ "${ZERO_ONLY}" == "1" || "${ZERO_ONLY}" == "true" ]]; then
    EXTRA_ARGS+=(--zero_only)
fi
if [[ "${ZERO_NO_CONFIRM}" == "1" || "${ZERO_NO_CONFIRM}" == "true" ]]; then
    EXTRA_ARGS+=(--zero_no_confirm)
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/innov_pi05_local_inference.py" \
    --policy_path "${POLICY_PATH}" \
    --left_port "${LEFT_PORT}" \
    --right_port "${RIGHT_PORT}" \
    --cameras "${CAMERA_CONFIG}" \
    --task "${TASK}" \
    --fps "${FPS}" \
    --duration_s "${DURATION_S}" \
    --warmup_rounds "${WARMUP_ROUNDS}" \
    --action_smooth_max_step "${ACTION_SMOOTH_MAX_STEP}" \
    --action_filter_alpha "${ACTION_FILTER_ALPHA}" \
    --action_max_delta "${ACTION_MAX_DELTA}" \
    --action_clip_min "${ACTION_CLIP_MIN}" \
    --action_clip_max "${ACTION_CLIP_MAX}" \
    --policy_action_arms "${POLICY_ACTION_ARMS}" \
    --camera_timeout_ms "${CAMERA_TIMEOUT_MS}" \
    --camera_retries "${CAMERA_RETRIES}" \
    --show_cameras "${SHOW_CAMERAS}" \
    --camera_preview_width "${CAMERA_PREVIEW_WIDTH}" \
    --action_log_path "${ACTION_LOG_PATH}" \
    --print_action_every "${PRINT_ACTION_EVERY}" \
    --zero_arms "${ZERO_ARMS}" \
    --home_joints "${HOME_JOINTS}" \
    --home_gripper "${HOME_GRIPPER}" \
    --home_duration_s "${HOME_DURATION_S}" \
    --home_fps "${HOME_FPS}" \
    --home_joint_velocity "${HOME_JOINT_VELOCITY}" \
    --home_gripper_velocity "${HOME_GRIPPER_VELOCITY}" \
    "${EXTRA_ARGS[@]}"
