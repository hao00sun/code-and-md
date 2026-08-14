# 最简配置环境与启动流程

本文只保留最短流程：新机器拿到代码后，配置环境、准备权重和数据集，然后启动 PI05 训练或推理。

默认仓库路径：

```bash
/home/wu/lerobot_space/lerobot
```

## 1. 安装基础工具

```bash
sudo apt update
sudo apt install -y git git-lfs curl wget unzip zip ffmpeg build-essential
git lfs install
```

确认 GPU：

```bash
nvidia-smi
```

如果 `nvidia-smi` 不可用，先处理 NVIDIA 驱动。

## 2. 创建 conda 环境

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
pip install uv
```

确认：

```bash
python --version
uv --version
```

## 3. 拉取或解压代码

从 GitHub 拉：

```bash
mkdir -p /home/wu/lerobot_space
cd /home/wu/lerobot_space
git clone <你的代码仓库地址> lerobot
cd lerobot
```

如果是 zip：

```bash
mkdir -p /home/wu/lerobot_space/lerobot
cd /home/wu/lerobot_space/lerobot
unzip /path/to/lerobot_code_usage_cn_*.zip
```

## 4. 安装 Python 依赖

推荐：

```bash
cd /home/wu/lerobot_space/lerobot
conda activate lerobot
uv sync --locked
```

如果不想用 `.venv`，直接装到当前 conda 环境：

```bash
pip install -e .
```

本项目本地脚本默认 Python：

```bash
/home/wu/miniconda3/envs/lerobot/bin/python
```

如果你的 conda 路径不同，运行时覆盖：

```bash
PYTHON_BIN=$(which python) ./run_pi05_train_with_logs.sh
```

## 5. 准备 PI05 权重和 tokenizer

PI05 base 权重放这里：

```bash
/data/SUN_ht/pi/pretrained_weights/pi05_base
```

必须包含：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

PaliGemma tokenizer 缓存放这里：

```bash
/data/SUN_ht/pi/cache/huggingface
```

最稳妥方式是从旧机器复制：

```bash
mkdir -p /data/SUN_ht/pi/pretrained_weights
mkdir -p /data/SUN_ht/pi/cache

rsync -avP 旧机器:/data/SUN_ht/pi/pretrained_weights/pi05_base/ \
  /data/SUN_ht/pi/pretrained_weights/pi05_base/

rsync -avP 旧机器:/data/SUN_ht/pi/cache/huggingface/ \
  /data/SUN_ht/pi/cache/huggingface/
```

检查：

```bash
ls /data/SUN_ht/pi/pretrained_weights/pi05_base
du -sh /data/SUN_ht/pi/pretrained_weights/pi05_base
```

## 6. 准备数据集

本代码库当前最小训练流程不包含数据采集。训练 PI05 时，建议使用 LeRobot v3.0 格式数据。

当前 PI05 主配置默认数据集：

```bash
/data/SUN_ht/datasets/innov_0730_merged_v30
```

从旧机器复制：

```bash
mkdir -p /data/SUN_ht/datasets
rsync -avP 旧机器:/data/SUN_ht/datasets/innov_0730_merged_v30/ \
  /data/SUN_ht/datasets/innov_0730_merged_v30/
```

检查：

```bash
ls /data/SUN_ht/datasets/innov_0730_merged_v30/meta
cat /data/SUN_ht/datasets/innov_0730_merged_v30/meta/info.json | head
```

## 7. 数据转换最简方法

如果你已经有可直接训练的 v3.0 数据集，可以跳过本节。

如果你手里是旧格式数据、多个数据集、视频有问题，按下面顺序处理。

### 7.1 判断数据集版本

查看：

```bash
cat /path/to/dataset/meta/info.json | grep codebase_version
```

预期训练用：

```text
"codebase_version": "v3.0"
```

如果是 `v2.1`，需要先转换成 v3.0。

### 7.2 合并两个 v2.1 双臂数据集

本仓库提供了 v2.1 合并脚本：

```bash
merge_lerobot_v21_arx_bimanual.py
```

默认输入：

```text
/media/wu/data/SUN_ht/datasets/arx_bimanual_0624_1640_old
/media/wu/data/SUN_ht/datasets/arx_bimanual_0625_1524
```

默认输出：

```text
/media/wu/data/SUN_ht/datasets/arx_bimanual_0624_0625_v21_merged
```

运行默认合并：

```bash
cd /home/wu/lerobot_space/lerobot
python merge_lerobot_v21_arx_bimanual.py --overwrite
```

自定义输入输出：

```bash
python merge_lerobot_v21_arx_bimanual.py \
  --sources /path/to/v21_dataset_a /path/to/v21_dataset_b \
  --output /path/to/v21_merged \
  --overwrite
```

作用：

```text
重新编号 episode
重写全局 index
合并 tasks
复制 parquet 和 mp4
保持 v2.1 目录结构
```

### 7.3 修复视频时间戳

如果 v2.1 转 v3.0 时遇到：

```text
non monotonically increasing dts
```

先修视频：

```bash
python fix_augmented_videos.py --root /path/to/v21_dataset
```

如果还失败，强制固定 fps：

```bash
python fix_augmented_videos.py --root /path/to/v21_dataset --fps 30
```

这个脚本会用 ffmpeg 重新编码 mp4，修复时间戳问题。

### 7.4 v2.1 转 v3.0

优先使用 LeRobot 自带数据编辑/转换工具。先查看帮助：

```bash
lerobot-edit-dataset --help
```

如果命令不可用，用模块方式：

```bash
python -m lerobot.scripts.lerobot_edit_dataset --help
```

不同 LeRobot 版本的转换命令可能略有差异，因此以 `--help` 显示为准。转换目标是得到这样的 v3.0 结构：

```text
meta/info.json
meta/stats.json
data/chunk-xxx/episode_xxxxxx.parquet
videos/...
```

转换后检查：

```bash
cat /path/to/v30_dataset/meta/info.json | grep codebase_version
ls /path/to/v30_dataset/meta
ls /path/to/v30_dataset/data
ls /path/to/v30_dataset/videos
```

### 7.5 裁剪/统一视频尺寸

如果多路相机尺寸不一致，或策略要求固定图像尺寸，可以先备份再裁剪：

```bash
python innov_il/backup_and_crop_dataset.py \
  --dataset-root /path/to/v30_dataset \
  --width 640 \
  --height 480
```

先 dry-run：

```bash
python innov_il/backup_and_crop_dataset.py \
  --dataset-root /path/to/v30_dataset \
  --width 640 \
  --height 480 \
  --dry-run
```

### 7.6 生成左臂 7 维数据集

如果要从 14 维双臂数据生成左臂 7 维数据：

```bash
python innov_il/prepare_left_arm_dataset.py \
  --source /path/to/v30_bimanual_dataset \
  --destination /path/to/v30_left_arm_dataset \
  --overwrite
```

它会：

```text
保留 observation.state 前 7 维
保留 action 前 7 维
保留 front 和 left_wrist
移除 right_wrist
更新 meta/stats
```

### 7.7 转换后最小检查

训练 PI05 前，至少检查：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/path/to/v30_dataset")
info = json.loads((root / "meta/info.json").read_text())
print("version:", info.get("codebase_version"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("features:")
for k, v in info["features"].items():
    print(" ", k, v.get("dtype"), v.get("shape"))
PY
```

PI05 双臂训练常见要求：

```text
observation.state shape: [14]
action shape: [14]
至少有 1 路或多路 observation.images.*
codebase_version: v3.0
```

最后把训练配置中的数据集路径改成转换后的路径：

```yaml
dataset:
  repo_id: your_dataset_name
  root: /path/to/v30_dataset
```

## 8. 设置离线环境变量

训练/推理前执行：

```bash
export HF_HOME=/data/SUN_ht/pi/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

## 9. 最小验证

检查 torch/cuda：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

检查 PI05 权重：

```bash
for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  test -f "/data/SUN_ht/pi/pretrained_weights/pi05_base/$f" && echo "OK $f" || echo "MISSING $f"
done
```

## 10. 启动 PI05 训练

当前主配置：

```bash
pi05_semantic.yaml
```

启动：

```bash
cd /home/wu/lerobot_space/lerobot
conda activate lerobot
./run_pi05_train_with_logs.sh pi05_semantic.yaml
```

输出目录：

```bash
/mnt/bigdata/SUN_ht/runs
```

日志目录：

```bash
/data/SUN_ht/pi/logs
```

## 11. 启动 PI05 server 推理

把 `--policy_path` 换成你的 checkpoint：

```bash
cd /home/wu/lerobot_space/lerobot
conda activate lerobot

env -u ALL_PROXY -u all_proxy \
  HF_HOME=/data/SUN_ht/pi/cache/huggingface \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
python pi05_server.py \
  --policy_path /mnt/bigdata/SUN_ht/runs/你的run/checkpoints/014000/pretrained_model \
  --host 0.0.0.0 \
  --port 5005 \
  --device cuda \
  --transport tcp_pickle \
  --action_mode chunk \
  --client_timeout_s 0 \
  --print_action
```

client 连接：

```text
tcp://服务器IP:5005
```

## 12. 启动 Innov 本地机械臂推理

把 `POLICY_PATH` 换成你的 checkpoint：

```bash
cd /home/wu/lerobot_space/lerobot
conda activate lerobot

POLICY_PATH=/mnt/bigdata/SUN_ht/runs/你的run/checkpoints/014000/pretrained_model \
TASK="Put the water flosser into the box and close the lid." \
./innov_il/run_innov_pi05_local_inference.sh
```

如果串口反了：

```bash
LEFT_PORT=/dev/ttyACM0 RIGHT_PORT=/dev/ttyACM1 \
POLICY_PATH=/mnt/bigdata/SUN_ht/runs/你的run/checkpoints/014000/pretrained_model \
./innov_il/run_innov_pi05_local_inference.sh
```

串口权限不足时：

```bash
sudo chmod a+rw /dev/ttyACM0 /dev/ttyACM1
```

## 13. 最常见问题

权重缺失：

```text
检查 /data/SUN_ht/pi/pretrained_weights/pi05_base
```

tokenizer 缺失：

```text
检查 /data/SUN_ht/pi/cache/huggingface
```

数据集缺失：

```text
检查 /data/SUN_ht/datasets/innov_0730_merged_v30
```

CUDA 不可用：

```text
先修 NVIDIA 驱动，确认 nvidia-smi 和 torch.cuda.is_available()
```

脚本找不到 Python：

```bash
PYTHON_BIN=$(which python) ./run_pi05_train_with_logs.sh pi05_semantic.yaml
```
