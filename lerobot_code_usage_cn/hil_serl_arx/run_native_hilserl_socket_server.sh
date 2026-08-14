#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

DATASET_ROOT="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30"
REWARD_MODEL_PATH="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl/reward_classifier_ep_split_20260612_091649/checkpoints/last/pretrained_model"

HIL_SERL_OUT="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl"
LOG_DIR="$HIL_SERL_OUT/logs"

TIME_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="native_hilserl_socket_${TIME_TAG}"
OUTPUT_DIR="$HIL_SERL_OUT/$RUN_NAME"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "DATASET_ROOT=$DATASET_ROOT"
echo "REWARD_MODEL_PATH=$REWARD_MODEL_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "SERVER_PORT=5006"
echo "ALLOWED_CLIENT_HOST=192.168.0.84"

if [ ! -f "$REWARD_MODEL_PATH/config.json" ] || [ ! -f "$REWARD_MODEL_PATH/model.safetensors" ]; then
  echo "ERROR: REWARD_MODEL_PATH must point to an already-trained reward classifier pretrained_model directory."
  echo "Missing config.json or model.safetensors under: $REWARD_MODEL_PATH"
  exit 1
fi

python hil_serl_arx/native_hilserl_socket_server.py \
  --dataset_root "$DATASET_ROOT" \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --host 0.0.0.0 \
  --port 5006 \
  --allowed_client_host 192.168.0.84 \
  --device cuda \
  --storage_device cpu \
  --batch_size 256 \
  --online_ratio 0.5 \
  --updates_per_transition 1 \
  --learning_starts 32 \
  --reward_success_threshold 0.2 \
  --online_action_mode actor \
  --max_action_delta 0.03 \
  2>&1 | tee "$LOG_FILE"
