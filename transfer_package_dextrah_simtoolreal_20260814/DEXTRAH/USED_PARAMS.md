# DEXTRAH 使用过的参数记录

本文档记录本机已经使用过或当前推荐使用的 DEXTRAH 训练、续训、可视化参数。除特别说明外，任务均为 `Dextrah-Kuka-Allegro`，运行目录均为：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games
conda activate /data/SUN_ht/Isaac_Gym/env_dextrah
export OMNI_KIT_ACCEPT_EULA=YES
```

## 当前稳定环境

```text
GPU: NVIDIA GeForce RTX 4090 48GB
推荐驱动: R580 系列
当前验证过的驱动: 580.178.04
不推荐驱动: 595.84
IsaacLab: /data/SUN_ht/Isaac_Gym/IsaacLab_v2.2.1
Conda: /data/SUN_ht/Isaac_Gym/env_dextrah
大文件目录: /mnt/bigdata/SUN_ht/runs/dextrah
```

`595.84` 曾导致 Isaac Sim 5.0 的 RTX/Vulkan 渲染链路在 `librtx.scenedb.plugin.so`、`libcarb.scenerenderer-rtx.plugin.so`、`libomni.hydra.rtx.plugin.so` 处段错误。降到 R580 后，仿真和 livestream 能继续启动。

## 低显存/功能测试

用于验证环境、任务、物体路径、最小训练链路是否正常。

```bash
python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 256 \
  agent.wandb_activate=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=False \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096
```

含义：

- `--num_envs 256`：只启动 256 个并行环境，主要用于排错。
- `env.use_cuda_graph=False`：关闭 CUDA graph，提高稳定性。
- `minibatch_size=4096`：与小并行数匹配，避免 rl_games batch 断言失败。

## 当前推荐完整物体集训练

这是目前在 4090 48GB 上更稳的 teacher 训练配置。

```bash
python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 8192 \
  agent.wandb_activate=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=False \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  +agent.params.config.full_experiment_name=env8192_noCudaGraph_pose45_mb32768
```

含义：

- `--num_envs 8192`：并行环境数，兼顾速度和稳定性。
- `env.objects_dir=visdex_objects`：使用 VisDex 物体集。
- `env.max_pose_angle=45.0`：目标姿态采样最大角度为 45 度。
- `env.use_cuda_graph=False`：关闭 CUDA graph，避免 `apply_object_wrench()` 附近的 CUDA/PhysX/Fabric 崩溃。
- `agent.params.config.minibatch_size=32768`：策略网络 PPO minibatch。
- `agent.params.config.central_value_config.minibatch_size=32768`：central value 网络 minibatch。
- `full_experiment_name`：固定实验目录名，方便复现和对比。

## 大并行压力测试

曾用于测试更大并行数。

```bash
python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 16384 \
  agent.wandb_activate=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=True \
  agent.params.config.minibatch_size=65536 \
  agent.params.config.central_value_config.minibatch_size=65536 \
  +agent.params.config.full_experiment_name=env16384_cudaGraph_pose45_mb65536
```

实际观察：

- 加载时间明显变长，因为每个环境会加载一个物体实例。
- `env.use_cuda_graph=True` 在本机曾触发 CUDA/PhysX/Fabric 相关崩溃。
- 不建议作为日常调参默认配置。

## 8192 无 CUDA Graph 训练

该配置曾作为表现较好的训练版本，并产生后续 resume 目录。

```bash
python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 8192 \
  agent.wandb_activate=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=False \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  +agent.params.config.full_experiment_name=env8192_noCudaGraph_pose45_mb32768
```

相关 checkpoint 目录：

```text
logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768/nn
```

## 续训版本

从已有 checkpoint 续训到 `env8192_noCudaGraph_pose45_mb32768_resume`。

```bash
RUN_DIR=logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768
CKPT=$(find -L "$RUN_DIR/nn" \
  -maxdepth 1 \
  -type f \
  -name "*.pth" \
  -printf '%T@ %p\n' \
  | sort -n \
  | tail -1 \
  | cut -d' ' -f2-)

python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 8192 \
  --checkpoint "$CKPT" \
  agent.params.load_checkpoint=True \
  agent.params.load_path="$CKPT" \
  agent.wandb_activate=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=False \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  +agent.params.config.full_experiment_name=env8192_noCudaGraph_pose45_mb32768_resume
```

最终常用 checkpoint：

```text
logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768_resume/nn/last_dextrah_lstm_ep_20000_rew__1209.8596_.pth
```

## 关闭噪声训练 5000 epoch

用于无 ADR/无训练随机化扩张的对照实验。按上次稳定训练参数运行，训练 5000 个 rl_games epoch，每 500 epoch 保存一次。

```bash
python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 8192 \
  --max_iterations 5000 \
  agent.wandb_activate=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=False \
  env.enable_adr=False \
  env.starting_adr_increments=0 \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  agent.params.config.save_frequency=500 \
  +agent.params.config.full_experiment_name=env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500
```

含义：

- `--max_iterations 5000`：映射到 rl_games 的 `max_epochs=5000`。
- `agent.params.config.save_frequency=500`：每 500 epoch 保存一次。
- `env.enable_adr=False`：关闭 ADR，不再根据成功率逐步扩大随机化范围。
- `env.starting_adr_increments=0`：从 0 随机化增量开始，配合关闭 ADR 使用。

## 子集调参

减少实际加载的 USD 物体数量，加快启动和排错。网络输入中的 object one-hot 仍保持完整物体集维度，便于后续接回完整集。

```bash
python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 4096 \
  agent.wandb_activate=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=False \
  env.use_object_subset=True \
  env.object_subset_size=64 \
  env.object_subset_start_index=0 \
  agent.params.config.minibatch_size=16384 \
  agent.params.config.central_value_config.minibatch_size=16384 \
  +agent.params.config.full_experiment_name=env4096_subset64_pose45_mb16384
```

注意：子集结果不能直接等价于完整物体集结果，只适合快速比较趋势。

## 可视化：直接 Isaac Sim GUI

```bash
export DISPLAY=:0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH

RUN_DIR=logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768_resume
CKPT=$(find -L "$RUN_DIR/nn" \
  -maxdepth 1 \
  -type f \
  -name "*.pth" \
  -printf '%T@ %p\n' \
  | sort -n \
  | tail -1 \
  | cut -d' ' -f2-)

python play.py \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 1 \
  --device cuda:0 \
  --checkpoint "$CKPT" \
  --objects_dir visdex_objects \
  --max_pose_angle 45.0 \
  --real-time \
  --speed_scale 1.0 \
  --print_every 120
```

## 可视化：Headless Livestream

```bash
export DISPLAY=:0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH

RUN_DIR=logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768_resume
CKPT=$(find -L "$RUN_DIR/nn" \
  -maxdepth 1 \
  -type f \
  -name "*.pth" \
  -printf '%T@ %p\n' \
  | sort -n \
  | tail -1 \
  | cut -d' ' -f2-)

python play.py \
  --headless \
  --livestream 2 \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 1 \
  --device cuda:0 \
  --checkpoint "$CKPT" \
  --objects_dir visdex_objects \
  --max_pose_angle 45.0 \
  --real-time \
  --speed_scale 1.0 \
  --print_every 120
```

可尝试访问：

```text
http://127.0.0.1:49100/
http://127.0.0.1:8211/streaming/client/
```

查看监听端口：

```bash
ss -ltnp | grep -E '8211|49100|8899|8011|47995|47996|47997|47998|47999|48000'
```

## 参数分类速查

- 环境规模：`--num_envs`
- 训练长度：`--max_iterations`
- 保存频率：`agent.params.config.save_frequency`
- 物体集：`env.objects_dir`
- 目标姿态范围：`env.max_pose_angle`
- 加速/稳定：`env.use_cuda_graph`
- ADR：`env.enable_adr`、`env.starting_adr_increments`、`env.success_for_adr`
- PPO batch：`agent.params.config.minibatch_size`
- Central value batch：`agent.params.config.central_value_config.minibatch_size`
- 实验命名：`+agent.params.config.full_experiment_name`
- 可视化速度：`--real-time`、`--speed_scale`、`--play_dt`
- 打印频率：`--print_every`
