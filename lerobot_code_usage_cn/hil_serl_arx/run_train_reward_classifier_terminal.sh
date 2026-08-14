#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

SOURCE_DATASET="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30"
TERMINAL_DATASET="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_reward_classifier_terminal_v30"
BASE_CONFIG="$REPO_ROOT/hil_serl_arx/reward_classifier_train_config_arx_terminal.json"

HIL_SERL_OUT="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl"
LOG_DIR="$HIL_SERL_OUT/logs"
CONFIG_DIR="$HIL_SERL_OUT/configs"

mkdir -p "$HIL_SERL_OUT"
mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"

echo "Generating terminal reward-classifier dataset..."
python hil_serl_arx/make_reward_classifier_terminal_dataset_v30.py \
  --src "$SOURCE_DATASET" \
  --dst "$TERMINAL_DATASET" \
  --success-tail-frames 30 \
  --failure-tail-frames 30 \
  --failure-random-frames-per-episode 30 \
  --seed 2 \
  --overwrite

TIME_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="reward_classifier_terminal_${TIME_TAG}"

OUTPUT_DIR="$HIL_SERL_OUT/$RUN_NAME"
RUN_CONFIG="$CONFIG_DIR/${RUN_NAME}.json"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

python - <<PY
from pathlib import Path
import json

base_config = Path("$BASE_CONFIG")
run_config = Path("$RUN_CONFIG")

cfg = json.loads(base_config.read_text())
cfg["dataset"]["root"] = "$TERMINAL_DATASET"
cfg["reward_model"]["label_lookup_root"] = "$TERMINAL_DATASET"
cfg["output_dir"] = "$OUTPUT_DIR"
cfg["job_name"] = "$RUN_NAME"
cfg["resume"] = False

run_config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

summary_path = Path("$TERMINAL_DATASET") / "meta" / "reward_classifier_terminal_subset.json"
summary = json.loads(summary_path.read_text())

print("Generated run config:")
print(run_config)
print("Output dir:")
print(cfg["output_dir"])
print("Terminal dataset:")
print(cfg["dataset"]["root"])
print("Label lookup column:")
print(cfg["reward_model"]["label_lookup_column"])
print("Label counts:")
print(summary["label_counts"])
print("Sample type counts:")
print(summary["sample_type_counts"])
PY

echo "CONFIG_PATH=$RUN_CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_FILE=$LOG_FILE"

if command -v lerobot-train >/dev/null 2>&1; then
  lerobot-train --config_path "$RUN_CONFIG" 2>&1 | tee "$LOG_FILE"
else
  python -m lerobot.scripts.lerobot_train --config_path "$RUN_CONFIG" 2>&1 | tee "$LOG_FILE"
fi
