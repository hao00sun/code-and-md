# LeRobot 本地使用与实验整理

整理日期：2026-08-14

本文根据当前仓库文件、本地脚本、配置文件、训练日志目录和已检测到的权重/缓存路径整理。它不是 LeRobot 官方说明，而是当前机器上已经使用过、改造过、或为实验准备过的工作记录。

注意：仓库根目录存在 `token_hf.json`，本文只记录其存在，不展开任何 token 内容。

## 1. 当前项目位置

仓库目录：

```bash
/home/wu/lerobot_space/lerobot
```

当前仓库是基于 Hugging Face LeRobot 的本地改造版，主要围绕真实机器人数据、PI05/PI0.5 训练与推理、MultiTaskDiT、Innov 双臂/单臂数据、HIL-SERL 奖励分类器等任务展开。

当前一层目录中和个人实验最相关的是：

```text
pi05_semantic.yaml
run_pi05_train_with_logs.sh
PI05_启动说明.md
pi05_server.py
innov_il/
DIT/
hil_serl_arx/
merge_lerobot_v21_arx_bimanual.py
fix_augmented_videos.py
```

## 2. Python 与环境配置

当前项目目录下有 `.venv`，大小约 `6.9G`。它是 uv 创建的 Python 虚拟环境，Python 基底来自 conda 环境：

```text
home = /home/wu/miniconda3/envs/lerobot/bin
uv = 0.11.29
Python = 3.12.13
```

`.venv` 中主要空间占用：

```text
4.2G  nvidia
1.6G  torch
639M  triton
84M   cmake
81M   opencv_python_headless.libs
77M   cv2
```

也就是说，空间主要被 GPU 版 PyTorch、CUDA 运行库和视觉依赖占用。

项目脚本里更常用的是 conda 环境里的 Python：

```bash
/home/wu/miniconda3/envs/lerobot/bin/python
```

多数自定义训练脚本都设置了：

```bash
PYTHON_BIN="${PYTHON_BIN:-/home/wu/miniconda3/envs/lerobot/bin/python}"
```

因此：

- 运行 `uv run ...` 时通常会使用项目 `.venv`。
- 运行 `/home/wu/miniconda3/envs/lerobot/bin/python ...` 时使用 conda 环境。
- 当前个人脚本主要按 conda Python 路径写死，和 `.venv` 不是同一个环境入口。

常用离线环境变量：

```bash
export HF_HOME=/data/SUN_ht/pi/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

## 3. 已做的核心代码改造

当前 `git diff` 显示主要改过这些文件：

```text
src/lerobot/configs/train.py
src/lerobot/scripts/lerobot_train.py
src/lerobot/policies/pi05/configuration_pi05.py
src/lerobot/policies/pi05/modeling_pi05.py
src/lerobot/rewards/classifier/configuration_classifier.py
src/lerobot/rewards/classifier/modeling_classifier.py
AGENTS.md
```

### 3.1 训练配置增强

`src/lerobot/configs/train.py` 增加：

- `log_dir`：训练日志输出目录，支持 `{timestamp}` 和 `{job_name}` 占位符。
- `gradient_accumulation_steps`：梯度累积步数。
- `output_dir` 和 `log_dir` 的时间戳格式化。

`src/lerobot/scripts/lerobot_train.py` 增加：

- 训练日志文件写入。
- 梯度累积逻辑。
- 梯度裁剪与 optimizer step 只在累积结束后执行。
- effective batch size 日志按 `batch_size * gradient_accumulation_steps * num_processes` 计算。
- `rename_map` 后的 dataset stats 可用于预处理器 normalizer/unnormalizer。

### 3.2 PI05 action loss mask

`src/lerobot/policies/pi05/configuration_pi05.py` 增加：

```yaml
action_loss_active_dims: null
action_loss_active_arms: both
```

`src/lerobot/policies/pi05/modeling_pi05.py` 增加训练 loss mask：

- PI05 仍然预测完整 action。
- 训练 loss 只在指定动作维度上平均。
- 适合双臂数据里一只手臂基本静止、推理时要冻结该手臂的场景。

14 维双臂动作约定：

```text
0  left_joint1.pos
1  left_joint2.pos
2  left_joint3.pos
3  left_joint4.pos
4  left_joint5.pos
5  left_joint6.pos
6  left_gripper.pos
7  right_joint1.pos
8  right_joint2.pos
9  right_joint3.pos
10 right_joint4.pos
11 right_joint5.pos
12 right_joint6.pos
13 right_gripper.pos
```

只训练右臂 loss 的配置示例：

```yaml
policy:
  action_loss_active_arms: right
  action_loss_active_dims: [7, 8, 9, 10, 11, 12, 13]
```

### 3.3 Reward classifier 标签回查

`src/lerobot/rewards/classifier/configuration_classifier.py` 和 `modeling_classifier.py` 做了 ARX/HIL-SERL 适配：

- 支持 `label_key = "index"`。
- 通过 `label_lookup_root` 读取 `data/**/*.parquet`。
- 建立 `global frame index -> label` 查表。
- 支持 `label_lookup_column`，例如 `is_failure_data` 或 `reward_classifier_label`。
- 支持 `label_invert`，用于把 failure/success 编码转成 reward classifier 需要的标签。
- CNN spatial embedding 支持不同特征图尺寸，通过 adaptive pooling 对齐。

## 4. PI05 / PI0.5 训练

### 4.1 当前主配置

当前根目录配置文件：

```bash
pi05_semantic.yaml
```

当前内容指向：

```yaml
dataset:
  repo_id: innov_0730_merged_v30
  root: /data/SUN_ht/datasets/innov_0730_merged_v30
policy:
  type: pi05
  repo_id: wu/pi05-innov-0720-merged-v30
  pretrained_path: /data/SUN_ht/pi/pretrained_weights/pi05_base
  gradient_checkpointing: true
  dtype: bfloat16
  device: cuda
output_dir: /mnt/bigdata/SUN_ht/runs/pi05_innov_0730_merged_4cam_v30_{timestamp}
log_dir: /data/SUN_ht/pi/logs/pi05_innov_0730_merged_4cam_v30_{timestamp}
job_name: pi05_innov_0730_merged_4cam_v30
batch_size: 16
steps: 16000
save_freq: 2000
wandb:
  enable: false
```

启动脚本：

```bash
./run_pi05_train_with_logs.sh
```

默认等价于：

```bash
./run_pi05_train_with_logs.sh pi05_semantic.yaml
```

脚本会做这些事：

- 检查 PI05 base 权重是否完整。
- 检查本地 PaliGemma tokenizer 是否完整。
- 把 `policy_preprocessor.json` 里的 tokenizer 路径改成本地 tokenizer 快照。
- 开启 Hugging Face/Transformers 离线模式。
- 保存训练输入 yaml 到日志目录。
- 保存 `env_snapshot.txt`，包括 git 状态、conda 环境、pip freeze、torch/cuda、nvidia-smi、重要环境变量。
- 每 10 秒记录 GPU 使用到 `gpu_usage.csv`。
- 保存训练命令到 `train_command.txt`。
- 保存 stdout 到 `train_stdout.log`。
- 训练结束后保存输出目录结构到 `after_train_files.txt`。

### 4.2 PI05 base 权重

当前已检测到 PI05 base 权重：

```bash
/data/SUN_ht/pi/pretrained_weights/pi05_base
```

大小约：

```text
14G
```

必要文件存在：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
policy_preprocessor.json.bak_tokenizer_repo_id
```

其中 `policy_preprocessor.json` 已把 tokenizer 指向本地 PaliGemma：

```bash
/data/SUN_ht/pi/cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c
```

PI05 base 原始配置是 32 维 state/action，训练时本地配置会根据数据集和 processor 适配实际任务。

### 4.3 Hugging Face 缓存

当前 PI05 缓存目录：

```bash
/data/SUN_ht/pi/cache/huggingface
```

大小约：

```text
1.6G
```

检测到的 hub 模型缓存包括：

```text
models--google--paligemma-3b-pt-224
models--openai--clip-vit-base-patch16
```

### 4.4 已有 PI05 run 和日志

模型输出根目录：

```bash
/mnt/bigdata/SUN_ht/runs
```

大小约：

```text
2.9T
```

近期 PI05 run：

```text
2026-08-08 /mnt/bigdata/SUN_ht/runs/pi05_innov_0730_4cam_ep0_92_front_ga1_2026-08-08_17-47-59
2026-08-07 /mnt/bigdata/SUN_ht/runs/pi05_innov_0730_4cam_ep0_92_clean_ga1_2026-08-07_18-11-09
2026-08-03 /mnt/bigdata/SUN_ht/runs/pi05_innov_0730_merged_4cam_v30_20260803_180234
2026-07-31 /mnt/bigdata/SUN_ht/runs/pi05_innov_0730_0731_merged_stacked_fixed_20260731_191530
2026-07-31 /mnt/bigdata/SUN_ht/runs/pi05_innov_0730_merged_stacked_fixed_20260731_102123
2026-07-30 /mnt/bigdata/SUN_ht/runs/pi05_innov_0730_merged_stacked_fixed_20260730_183148
```

日志根目录：

```bash
/data/SUN_ht/pi/logs
```

大小约：

```text
632M
```

近期日志包括 PI05 训练和本地推理：

```text
pi05_innov_0730_4cam_ep0_92_front_ga1_2026-08-08_17-47-59
pi05_innov_0730_4cam_ep0_92_clean_ga1_2026-08-07_18-11-09
pi05_innov_0730_merged_4cam_v30_20260803_180234
innov_pi05_local_inference_20260810_093254
innov_pi05_local_inference_20260810_092858
```

## 5. PI05 推理与 server

核心脚本：

```bash
pi05_server.py
```

它支持：

- 加载 `checkpoints/xxxxxx/pretrained_model`。
- TCP pickle 协议：`--transport tcp_pickle`。
- OpenPI 风格 WebSocket：`--transport openpi_ws`。
- 单步动作或动作块：`--action_mode step|chunk`。
- 只启用左臂/右臂/指定动作维度。
- 冻结动作维度时使用当前 state、数据集均值/中位数/第一帧，或手动 values。

典型 TCP server：

```bash
env -u ALL_PROXY -u all_proxy \
  HF_HOME=/data/SUN_ht/pi/cache/huggingface \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
python pi05_server.py \
  --policy_path /mnt/bigdata/SUN_ht/runs/pi05_innov_0730_merged_4cam_v30_20260803_180234/checkpoints/014000/pretrained_model \
  --host 0.0.0.0 \
  --port 5005 \
  --device cuda \
  --transport tcp_pickle \
  --action_mode chunk \
  --client_timeout_s 0 \
  --print_action
```

只让右臂使用模型输出，左臂使用数据集参考位姿：

```bash
python pi05_server.py \
  --policy_path /path/to/checkpoints/014000/pretrained_model \
  --host 0.0.0.0 \
  --port 5005 \
  --device cuda \
  --transport tcp_pickle \
  --action_mode chunk \
  --active_arms right \
  --frozen_action_source dataset_mean \
  --frozen_action_dataset /data/SUN_ht/datasets/arx_0723_1401 \
  --client_timeout_s 0 \
  --print_action
```

更完整的旧启动说明在：

```bash
PI05_启动说明.md
```

注意：`PI05_启动说明.md` 里有部分旧路径仍指向 `arx_0723_1401`，而当前 `pi05_semantic.yaml` 已切到 `innov_0730_merged_v30`。使用时要以当前 yaml 或目标 checkpoint 为准。

## 6. Innov 本地机械臂 PI05 推理

启动脚本：

```bash
innov_il/run_innov_pi05_local_inference.sh
```

本地推理脚本会直接连接机械臂和 RealSense 摄像头，不走 server/client 分离。

默认配置：

```text
POLICY_PATH=/mnt/bigdata/SUN_ht/runs/pi05_innov_0722_15382026-07-22_17-35-27/checkpoints/002000/pretrained_model
LEFT_PORT=/dev/ttyACM1
RIGHT_PORT=/dev/ttyACM0
TASK="Put the water flosser into the box and close the lid."
FPS=30
DURATION_S=120
SHOW_CAMERAS=1
ACTION_FILTER_ALPHA=0.5
ACTION_MAX_DELTA=0.05
ACTION_SMOOTH_MAX_STEP=0.05
ZERO_ARMS=both
```

启动：

```bash
./innov_il/run_innov_pi05_local_inference.sh
```

指定 checkpoint 和任务：

```bash
POLICY_PATH=/mnt/bigdata/SUN_ht/runs/pi05_innov_0730_merged_4cam_v30_20260803_180234/checkpoints/014000/pretrained_model \
TASK="Place the camera into the box." \
./innov_il/run_innov_pi05_local_inference.sh
```

脚本中记录的 RealSense 摄像头：

```text
front       935422072733  848x480
front_1     938422076287  848x480
left_wrist  409122273564  640x480
right_wrist 409122273228  640x480
```

动作日志默认保存到：

```bash
/data/SUN_ht/pi/logs/innov_pi05_local_inference_时间戳/actions.jsonl
```

## 7. PI05 0730 四相机训练批处理

脚本：

```bash
innov_il/run_pi05_arx_0724_1553_ablation_train.sh
```

虽然文件名带 `arx_0724_1553_ablation`，当前内容实际是 0730 四相机数据的顺序训练脚本。

它会依次训练：

```text
/data/SUN_ht/datasets/innov_0730_4cam_ep0_92_clean
/data/SUN_ht/datasets/innov_0730_4cam_ep0_92_front
```

脚本会校验：

- 数据集为 LeRobot `v3.0`。
- 视频 key 数量为 4。
- `action` 和 `observation.state` shape 都是 `[14]`。

生成的配置目录：

```bash
innov_il/generated_train_configs
```

近期生成配置：

```text
pi05_innov_0730_4cam_ep0_92_clean_ga1.yaml
pi05_innov_0730_4cam_ep0_92_front_ga1.yaml
pi05_arx_0724_1553_freeze_left_ga2.yaml
pi05_arx_0724_1553_freeze_left_ga1.yaml
pi05_arx_0724_1553_no_freeze_ga1.yaml
```

## 8. 数据集处理与修复

### 8.1 v2.1 双臂数据合并

脚本：

```bash
merge_lerobot_v21_arx_bimanual.py
```

用途：

- 合并两个 LeRobot v2.1 ARX bimanual 数据集。
- 重新编号 episode。
- 重写 parquet 中的 `episode_index`、`index`、`task_index`。
- 保持 v2.1 目录结构。

默认输入：

```text
/media/wu/data/SUN_ht/datasets/arx_bimanual_0624_1640_old
/media/wu/data/SUN_ht/datasets/arx_bimanual_0625_1524
```

默认输出：

```text
/media/wu/data/SUN_ht/datasets/arx_bimanual_0624_0625_v21_merged
```

### 8.2 视频时间戳修复

脚本：

```bash
fix_augmented_videos.py
```

用途：

- 修复 LeRobot 数据集视频时间戳问题。
- 解决 v2.1 转 v3.0 时可能出现的 `non monotonically increasing dts`。
- 使用 ffmpeg 重新生成时间戳并转 H.264。

默认修复路径：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_augmented_fixed_v21
```

运行：

```bash
python fix_augmented_videos.py
```

强制固定 fps：

```bash
python fix_augmented_videos.py --fps 30
```

### 8.3 Innov 视频裁剪备份

脚本：

```bash
innov_il/backup_and_crop_dataset.py
```

用途：

- 备份 LeRobot v3.0 数据集。
- 将视频中心裁剪到指定尺寸。
- 更新 `meta/info.json` 里的视频 shape、codec、pix_fmt 等信息。

默认数据集：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/innov/innov_0617_1554
```

默认裁剪到：

```text
640x480
```

### 8.4 左臂单臂数据集

脚本：

```bash
innov_il/prepare_left_arm_dataset.py
```

用途：

- 从 14 维双臂数据中生成 7 维左臂数据集。
- 保留 `front` 和 `left_wrist`。
- 移除 `right_wrist`。
- 将 `observation.state` 和 `action` 截到前 7 维。

默认输入：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/innov/innov_0617_1554
```

默认输出：

```text
/media/wu/data/SUN_ht/innov/datasets/innov_0617_1554_left_arm
```

### 8.5 当前检测到的数据集

近期 `/data/SUN_ht/datasets` 下的数据集：

```text
innov_0730_4cam_ep0_92_front
innov_0730_4cam_ep0_92_clean
innov_0730_4cam_ep0_92_front_old
innov_0730_4cam_ep0_92_clean_old
innov_0730_merged_v30
innov_0730_0731_merged_stacked_fixed
innov_0731
innov_0730_1358_stacked_no_rotate
innov_0730_merged_stacked_fixed
innov_0730_merged
```

## 9. MultiTaskDiT / DiT

目录：

```bash
DIT/
```

### 9.1 预训练权重拉取

脚本：

```bash
DIT/download_dit_pretrained_with_proxy.sh
```

默认配置：

```bash
HF_MODEL_ID=NONHUMAN-RESEARCH/multi-task-dit-training-fruits
PROXY_URL=http://127.0.0.1:7890
DIT_ROOT=/media/wu/data/SUN_ht/dit
PRETRAINED_PATH=/media/wu/data/SUN_ht/dit/pretrained_weights/multi_task_dit_flow_matching_14d_base
HF_CACHE_BASE=/media/wu/data/SUN_ht/dit/cache/huggingface
```

它会下载：

```text
NONHUMAN-RESEARCH/multi-task-dit-training-fruits
openai/clip-vit-base-patch16
```

当前检测结果：

```text
未检测到 /media/wu/data/SUN_ht/dit/pretrained_weights/multi_task_dit_flow_matching_14d_base
未检测到 /media/wu/data/SUN_ht/dit/cache/huggingface
```

说明脚本已准备好，但当前机器/挂载路径下没有检测到这份 DiT 权重。

### 9.2 Flow matching 微调

配置：

```bash
DIT/multi_task_dit_flow_matching_train.yaml
```

主要设置：

```yaml
dataset:
  root: /media/wu/data/SUN_ht/datasets/arx_bimanual_0624_0625merged
policy:
  path: /media/wu/data/SUN_ht/dit/pretrained_weights/multi_task_dit_flow_matching_14d_base
  objective: flow_matching
  n_obs_steps: 2
  horizon: 32
  n_action_steps: 24
  vision_encoder_name: openai/clip-vit-base-patch16
  text_encoder_name: openai/clip-vit-base-patch16
batch_size: 32
steps: 50000
save_freq: 1000
```

训练脚本：

```bash
./DIT/run_dit_flow_matching_train_with_logs.sh
```

脚本会检查预训练 checkpoint 必要文件：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

### 9.3 Diffusion 从头训练

配置：

```bash
DIT/multi_task_dit_diffusion_train.yaml
```

主要设置：

```yaml
dataset:
  root: /media/wu/data/SUN_ht/datasets/arx_v30_297
policy:
  type: multi_task_dit
  objective: diffusion
  n_obs_steps: 2
  horizon: 32
  n_action_steps: 24
batch_size: 32
steps: 60000
save_freq: 2000
```

训练脚本：

```bash
./DIT/run_dit_diffusion_train_with_logs.sh
```

该脚本标注为从头训练 diffusion，不要求 MultiTaskDiT 预训练 checkpoint。

## 10. 早期 Innov diffusion / flow matching

目录：

```bash
innov_il/
```

### 10.1 三相机 14 维 diffusion

配置：

```bash
innov_il/diffusion_innov_0617_1554.yaml
```

数据集：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/innov/innov_0617_1554
```

策略：

```yaml
type: diffusion
vision_backbone: resnet34
n_obs_steps: 10
horizon: 64
n_action_steps: 50
batch_size: 16
steps: 30000
```

### 10.2 三相机 14 维 flow matching

配置：

```bash
innov_il/flow_matching_innov_0617_1554.yaml
```

策略：

```yaml
type: multi_task_dit
objective: flow_matching
n_obs_steps: 3
horizon: 64
n_action_steps: 50
vision_encoder_name: /media/wu/.../clip-vit-base-patch16
text_encoder_name: /media/wu/.../clip-vit-base-patch16
batch_size: 16
steps: 30000
```

启动脚本：

```bash
innov_il/run_innov_chunk_policy_train.sh diffusion
innov_il/run_innov_chunk_policy_train.sh flow_matching
```

### 10.3 左臂 7 维 flow matching

配置：

```bash
innov_il/flow_matching_left_arm.yaml
```

数据集：

```text
/media/wu/data/SUN_ht/innov/datasets/innov_0617_1554_left_arm
```

权重路径：

```text
/media/wu/data/SUN_ht/innov/pretrained_weights/clip-vit-base-patch16
```

当前检测结果：

```text
未检测到 /media/wu/data/SUN_ht/innov/pretrained_weights/clip-vit-base-patch16
```

训练脚本：

```bash
innov_il/run_left_arm_flow_matching_train.sh
```

训练完成后如果存在：

```text
checkpoints/030000/pretrained_model
```

脚本会创建软链接：

```text
/media/wu/data/SUN_ht/innov/inference_models/flow_matching_left_arm_latest
```

## 11. HIL-SERL 与 reward classifier

目录：

```bash
hil_serl_arx/
```

### 11.1 终端状态 reward classifier 数据集

脚本：

```bash
hil_serl_arx/make_reward_classifier_terminal_dataset_v30.py
```

用途：

- 从 ARX LeRobot v3 数据构造 reward classifier 训练子集。
- 成功 episode 取最后 N 帧作为正样本。
- 失败 episode 取最后 N 帧和随机帧作为负样本。
- 保留原始 `index`。
- 添加 `reward_classifier_label`。

标签含义：

```text
reward_classifier_label = 1  终端成功
reward_classifier_label = 0  失败或非成功
```

训练脚本：

```bash
hil_serl_arx/run_train_reward_classifier_terminal.sh
```

默认源数据：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30
```

默认生成数据：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_reward_classifier_terminal_v30
```

默认输出：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl
```

训练配置：

```bash
hil_serl_arx/reward_classifier_train_config_arx_terminal.json
```

关键设置：

```json
{
  "reward_model": {
    "type": "reward_classifier",
    "model_name": "/media/wu/.../hil-serl/pretrained_models/lerobot_resnet10",
    "num_cameras": 3,
    "num_classes": 2,
    "label_key": "index",
    "label_lookup_column": "reward_classifier_label",
    "device": "cuda"
  },
  "batch_size": 32,
  "steps": 3000,
  "save_freq": 300,
  "wandb": {
    "enable": false
  }
}
```

当前检测结果：

```text
未检测到 /media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl/pretrained_models/lerobot_resnet10
```

### 11.2 Native HIL-SERL socket server

启动脚本：

```bash
hil_serl_arx/run_native_hilserl_socket_server.sh
```

默认数据集：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30
```

默认 reward model：

```text
/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl/reward_classifier_ep_split_20260612_091649/checkpoints/last/pretrained_model
```

默认 server：

```text
host = 0.0.0.0
port = 5006
allowed_client_host = 192.168.0.84
device = cuda
storage_device = cpu
```

## 12. 预训练权重拉取方式汇总

### 12.1 PI05 base

当前仓库没有看到单独的 PI05 下载脚本，但训练脚本期望权重已经在：

```bash
/data/SUN_ht/pi/pretrained_weights/pi05_base
```

拉取或迁移后至少需要这些文件：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

同时需要 PaliGemma tokenizer 本地缓存：

```bash
/data/SUN_ht/pi/cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c
```

`run_pi05_train_with_logs.sh` 会检查 tokenizer，并把 PI05 的 `policy_preprocessor.json` 指到该本地路径。

### 12.2 DiT / MultiTaskDiT

使用：

```bash
PROXY_URL=http://127.0.0.1:7890 \
HF_MODEL_ID=NONHUMAN-RESEARCH/multi-task-dit-training-fruits \
./DIT/download_dit_pretrained_with_proxy.sh
```

默认下载到：

```bash
/media/wu/data/SUN_ht/dit/pretrained_weights/multi_task_dit_flow_matching_14d_base
```

并使用：

```bash
/media/wu/data/SUN_ht/dit/cache/huggingface
```

作为 Hugging Face 缓存。

### 12.3 CLIP

多个脚本依赖：

```text
openai/clip-vit-base-patch16
```

PI05 的 Hugging Face 缓存中已经检测到 CLIP 缓存：

```bash
/data/SUN_ht/pi/cache/huggingface/hub/models--openai--clip-vit-base-patch16
```

旧 Innov flow matching 脚本期望的本地 CLIP 路径目前未检测到：

```bash
/media/wu/data/SUN_ht/innov/pretrained_weights/clip-vit-base-patch16
```

## 13. 推荐使用顺序

如果要继续当前 PI05 训练：

```bash
cd /home/wu/lerobot_space/lerobot
./run_pi05_train_with_logs.sh pi05_semantic.yaml
```

如果要做 0730 四相机 clean/front 顺序训练：

```bash
cd /home/wu/lerobot_space/lerobot
./innov_il/run_pi05_arx_0724_1553_ablation_train.sh
```

如果要本地机械臂 PI05 推理：

```bash
cd /home/wu/lerobot_space/lerobot
POLICY_PATH=/path/to/checkpoints/xxxxxx/pretrained_model \
./innov_il/run_innov_pi05_local_inference.sh
```

如果要 server/client 分离推理：

```bash
cd /home/wu/lerobot_space/lerobot
python pi05_server.py \
  --policy_path /path/to/checkpoints/xxxxxx/pretrained_model \
  --host 0.0.0.0 \
  --port 5005 \
  --device cuda \
  --transport tcp_pickle \
  --action_mode chunk
```

如果要继续 DiT flow matching，先确认或重新下载权重：

```bash
./DIT/download_dit_pretrained_with_proxy.sh
./DIT/run_dit_flow_matching_train_with_logs.sh
```

## 14. 需要注意的坑

- `.venv` 和 conda `lerobot` 环境不是同一个入口；当前个人脚本主要使用 conda Python。
- 训练脚本普遍开启离线模式，如果权重/tokenizer/cache 不完整，会直接报错。
- `PI05_启动说明.md` 中有旧 ARX 路径，当前训练 yaml 已切到 Innov 0730 merged v30。
- `innov_il/run_pi05_arx_0724_1553_ablation_train.sh` 文件名和当前脚本内容不匹配，实际内容是 0730 四相机训练。
- PI05 action loss mask 只影响训练 loss，不会自动冻结推理端动作；推理冻结要在 `pi05_server.py` 或本地推理参数里单独设置。
- HIL-SERL reward classifier 的标签依赖 `index` 回查，换数据集时必须确认 `label_lookup_root` 和 `label_lookup_column`。
- 视频转换、裁剪和 v2.1/v3.0 合并脚本会改动大数据目录，运行前建议先 dry-run 或备份。

## 15. 重要文件索引

```text
PI05 训练主配置:
  pi05_semantic.yaml

PI05 带日志训练:
  run_pi05_train_with_logs.sh

PI05 server:
  pi05_server.py

PI05 本地机械臂推理:
  innov_il/run_innov_pi05_local_inference.sh
  innov_il/innov_pi05_local_inference.py

PI05 四相机批训练:
  innov_il/run_pi05_arx_0724_1553_ablation_train.sh
  innov_il/generated_train_configs/

数据集处理:
  merge_lerobot_v21_arx_bimanual.py
  fix_augmented_videos.py
  innov_il/backup_and_crop_dataset.py
  innov_il/prepare_left_arm_dataset.py

DiT:
  DIT/download_dit_pretrained_with_proxy.sh
  DIT/run_dit_flow_matching_train_with_logs.sh
  DIT/run_dit_diffusion_train_with_logs.sh
  DIT/multi_task_dit_flow_matching_train.yaml
  DIT/multi_task_dit_diffusion_train.yaml

HIL-SERL:
  hil_serl_arx/run_train_reward_classifier_terminal.sh
  hil_serl_arx/reward_classifier_train_config_arx_terminal.json
  hil_serl_arx/run_native_hilserl_socket_server.sh

旧/补充说明:
  PI05_启动说明.md
```
