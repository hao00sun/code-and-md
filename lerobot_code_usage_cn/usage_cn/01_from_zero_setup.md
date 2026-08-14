# 从零开始搭建 LeRobot 本地环境

本文面向第一次接手的新手，目标是在一台新机器上完成：

- 安装基础工具。
- 拉取代码。
- 创建 Python/conda 环境。
- 安装 LeRobot 依赖。
- 配置 Hugging Face 缓存和 token。
- 拉取或迁移 PI05 / DiT / CLIP 预训练权重。
- 做最小运行验证。

下面命令默认使用 Linux/Ubuntu。

## 1. 准备系统依赖

先安装常用工具：

```bash
sudo apt update
sudo apt install -y git git-lfs curl wget unzip zip ffmpeg build-essential
```

初始化 Git LFS：

```bash
git lfs install
```

如果要用 RealSense 相机、机械臂串口、CUDA、NVIDIA 驱动，需要另外安装对应驱动。至少先确认 GPU 是否可见：

```bash
nvidia-smi
```

如果这个命令不存在或报错，说明 NVIDIA 驱动/CUDA 环境还没准备好。

## 2. 安装 Miniconda

如果机器上还没有 conda，可以安装 Miniconda：

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

安装完成后重新打开终端，或执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

确认 conda 可用：

```bash
conda --version
```

## 3. 创建 Python 环境

当前项目要求 Python 3.12+。推荐创建名为 `lerobot` 的环境：

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

确认 Python：

```bash
which python
python --version
```

预期类似：

```text
/home/wu/miniconda3/envs/lerobot/bin/python
Python 3.12.x
```

## 4. 安装 uv

LeRobot 项目推荐用 uv 管理依赖：

```bash
pip install uv
```

确认 uv：

```bash
uv --version
```

## 5. 拉取代码

如果从 GitHub 拉取官方 LeRobot：

```bash
mkdir -p /home/wu/lerobot_space
cd /home/wu/lerobot_space
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

如果是从你自己的远程仓库拉取本地改造版，把地址换成你的仓库地址：

```bash
mkdir -p /home/wu/lerobot_space
cd /home/wu/lerobot_space
git clone <你的仓库地址> lerobot
cd lerobot
```

如果是别人发来的 zip，先解压：

```bash
mkdir -p /home/wu/lerobot_space
cd /home/wu/lerobot_space
unzip lerobot_code_docs_*.zip -d lerobot_unpacked
cd lerobot_unpacked
```

## 6. 安装项目依赖

推荐先同步基础依赖：

```bash
uv sync --locked
```

如果要跑测试和开发工具：

```bash
uv sync --locked --extra test --extra dev
```

如果要尽量安装完整功能：

```bash
uv sync --locked --extra all
```

注意：`uv sync` 通常会在项目目录创建 `.venv`。当前机器上的 `.venv` 约 `6.9G`，主要是 PyTorch、CUDA、nvidia、triton 等依赖。

如果你想直接使用 conda 环境，而不是项目 `.venv`，可以用 pip editable 安装：

```bash
conda activate lerobot
pip install -e .
```

本地脚本多数默认使用：

```bash
/home/wu/miniconda3/envs/lerobot/bin/python
```

如果你的 conda 路径不同，可以运行脚本时覆盖：

```bash
PYTHON_BIN=$(which python) ./run_pi05_train_with_logs.sh
```

## 7. 配置 Hugging Face token

如果要从 Hugging Face 拉模型或数据，需要登录：

```bash
huggingface-cli login
```

或者：

```bash
hf auth login
```

登录后 token 通常保存在 Hugging Face 自己的缓存目录。不要把 token 写进公开文档或提交到 git。

当前仓库根目录里检测到 `token_hf.json`，这是敏感文件，只能自己本机使用，不建议打包或分享。

## 8. 配置缓存目录

为了避免模型和数据散落到默认 home 目录，建议固定缓存路径。

PI05 推荐缓存：

```bash
export HF_HOME=/data/SUN_ht/pi/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
```

如果机器不能直接访问 Hugging Face，可以设置代理：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

如果权重已经完整下载到本地，训练/推理时建议离线：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

## 9. 拉取 PI05 / PI0.5 预训练权重

当前本地脚本期望 PI05 base 权重在：

```bash
/data/SUN_ht/pi/pretrained_weights/pi05_base
```

至少需要这些文件：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

### 方法 A：从已有机器迁移

如果另一台机器已经有完整权重，推荐直接复制：

```bash
mkdir -p /data/SUN_ht/pi/pretrained_weights
rsync -avP 旧机器:/data/SUN_ht/pi/pretrained_weights/pi05_base/ \
  /data/SUN_ht/pi/pretrained_weights/pi05_base/
```

同时迁移 PaliGemma tokenizer 缓存：

```bash
mkdir -p /data/SUN_ht/pi/cache/huggingface
rsync -avP 旧机器:/data/SUN_ht/pi/cache/huggingface/ \
  /data/SUN_ht/pi/cache/huggingface/
```

迁移后检查：

```bash
ls -lh /data/SUN_ht/pi/pretrained_weights/pi05_base
du -sh /data/SUN_ht/pi/pretrained_weights/pi05_base
```

当前机器上这份 PI05 base 大约 `14G`。

### 方法 B：从 Hugging Face 下载

如果你知道 PI05 base 的 Hugging Face repo id，可以用：

```bash
mkdir -p /data/SUN_ht/pi/pretrained_weights/pi05_base
hf download <PI05_BASE_REPO_ID> \
  --local-dir /data/SUN_ht/pi/pretrained_weights/pi05_base
```

如果没有 `hf` 命令：

```bash
huggingface-cli download <PI05_BASE_REPO_ID> \
  --local-dir /data/SUN_ht/pi/pretrained_weights/pi05_base
```

注意：当前仓库脚本没有写明 PI05 base 的远程 repo id，因此最稳的方法是从已有机器迁移完整目录。

### 拉取 PaliGemma tokenizer

PI05 的 processor 会单独加载 PaliGemma tokenizer。当前脚本期望缓存快照路径：

```bash
/data/SUN_ht/pi/cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c
```

可以先下载：

```bash
export HF_HOME=/data/SUN_ht/pi/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
hf download google/paligemma-3b-pt-224
```

下载后验证：

```bash
find /data/SUN_ht/pi/cache/huggingface/hub -maxdepth 2 -type d -name 'models--google--paligemma-3b-pt-224'
```

## 10. 拉取 DiT / MultiTaskDiT 权重

当前仓库已经有下载脚本：

```bash
DIT/download_dit_pretrained_with_proxy.sh
```

默认会下载：

```text
NONHUMAN-RESEARCH/multi-task-dit-training-fruits
openai/clip-vit-base-patch16
```

默认保存：

```bash
/media/wu/data/SUN_ht/dit/pretrained_weights/multi_task_dit_flow_matching_14d_base
/media/wu/data/SUN_ht/dit/cache/huggingface
```

使用代理下载：

```bash
PROXY_URL=http://127.0.0.1:7890 \
./DIT/download_dit_pretrained_with_proxy.sh
```

不用代理：

```bash
PROXY_URL= \
./DIT/download_dit_pretrained_with_proxy.sh
```

如果你的大盘路径不是 `/media/wu/data/SUN_ht/dit`，可以覆盖：

```bash
DIT_ROOT=/data/SUN_ht/dit \
./DIT/download_dit_pretrained_with_proxy.sh
```

下载完成后检查：

```bash
ls -lh /media/wu/data/SUN_ht/dit/pretrained_weights/multi_task_dit_flow_matching_14d_base
```

## 11. 拉取 CLIP 权重

多个 DiT/flow matching 配置依赖：

```text
openai/clip-vit-base-patch16
```

下载到 Hugging Face 缓存：

```bash
export HF_HOME=/data/SUN_ht/pi/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
hf download openai/clip-vit-base-patch16
```

如果某些旧脚本要求本地目录：

```bash
/media/wu/data/SUN_ht/innov/pretrained_weights/clip-vit-base-patch16
```

可以创建目录后用 `hf download --local-dir`：

```bash
mkdir -p /media/wu/data/SUN_ht/innov/pretrained_weights/clip-vit-base-patch16
hf download openai/clip-vit-base-patch16 \
  --local-dir /media/wu/data/SUN_ht/innov/pretrained_weights/clip-vit-base-patch16
```

## 12. 准备数据集目录

当前 PI05 主配置使用：

```bash
/data/SUN_ht/datasets/innov_0730_merged_v30
```

如果新机器没有数据集，需要从旧机器迁移：

```bash
mkdir -p /data/SUN_ht/datasets
rsync -avP 旧机器:/data/SUN_ht/datasets/innov_0730_merged_v30/ \
  /data/SUN_ht/datasets/innov_0730_merged_v30/
```

检查 LeRobot 数据集元信息：

```bash
ls /data/SUN_ht/datasets/innov_0730_merged_v30/meta
cat /data/SUN_ht/datasets/innov_0730_merged_v30/meta/info.json | head
```

## 13. 最小验证

进入仓库：

```bash
cd /home/wu/lerobot_space/lerobot
conda activate lerobot
```

检查 Python 和 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

检查 PI05 权重文件：

```bash
for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  test -f "/data/SUN_ht/pi/pretrained_weights/pi05_base/$f" && echo "OK $f" || echo "MISSING $f"
done
```

检查 tokenizer：

```bash
python - <<'PY'
from transformers import AutoTokenizer
path = "/data/SUN_ht/pi/cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
AutoTokenizer.from_pretrained(path, local_files_only=True)
print("tokenizer ok")
PY
```

试运行训练脚本的前置检查：

```bash
./run_pi05_train_with_logs.sh pi05_semantic.yaml
```

注意：这会真的开始训练。如果只想检查配置和路径，可以先阅读脚本或在测试机器上中断。正式训练前确认 GPU、数据集和输出目录都正确。

## 14. 常见问题

### 14.1 `nvidia-smi` 不可用

先安装/修复 NVIDIA 驱动。没有 GPU 时 PI05 训练基本不可行。

### 14.2 Hugging Face 下载失败

检查网络或代理：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

如果在内网/离线机器，建议从已有机器迁移完整权重和缓存。

### 14.3 离线模式还在联网

确认：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

同时检查权重、tokenizer、CLIP 是否真的在本地。

### 14.4 串口权限不足

临时处理：

```bash
sudo chmod a+rw /dev/ttyACM0 /dev/ttyACM1
```

长期处理：

```bash
sudo usermod -aG dialout $USER
```

然后重新登录系统。

