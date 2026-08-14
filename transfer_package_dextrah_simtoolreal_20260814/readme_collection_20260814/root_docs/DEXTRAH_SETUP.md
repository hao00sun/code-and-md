# DEXTRAH 本地环境说明

DEXTRAH 与当前 IsaacLab 3.x 环境不是同一套栈。为了避免依赖冲突，本机保留了单独的 DEXTRAH 环境。

## 路径

```text
DEXTRAH: /data/SUN_ht/Isaac_Gym/DEXTRAH
FABRICS: /data/SUN_ht/Isaac_Gym/FABRICS
IsaacLab v2.2.1: /data/SUN_ht/Isaac_Gym/IsaacLab_v2.2.1
Conda 环境: /data/SUN_ht/Isaac_Gym/env_dextrah
本地 HOME 缓存: /data/SUN_ht/Isaac_Gym/.home
DEXTRAH checkpoint 大文件: /mnt/bigdata/SUN_ht/runs/dextrah
```

IsaacLab 3.x 灵巧手实验使用另一套环境：

```text
IsaacLab: /data/SUN_ht/Isaac_Gym/IsaacLab
Conda 环境: /data/SUN_ht/Isaac_Gym/env_isaaclab
```

## 进入 DEXTRAH 环境

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH
conda activate /data/SUN_ht/Isaac_Gym/env_dextrah
export HOME=/data/SUN_ht/Isaac_Gym/.home
export XDG_CACHE_HOME=/data/SUN_ht/Isaac_Gym/.home/.cache
export OMNI_KIT_ACCEPT_EULA=YES
```

`HOME` 和 `XDG_CACHE_HOME` 指向工作区内的可写目录，用于 Omniverse、Warp 等运行时缓存。

## 重要区别

```text
env_isaaclab: 用于 IsaacLab 3.x 官方 Reorient/Shadow Hand 基线
env_dextrah:  用于 DEXTRAH/FABRICS/Kuka-Allegro 训练
```

不要在 `DEXTRAH` 目录中执行：

```bash
./isaaclab.sh train ...
```

因为 `DEXTRAH` 目录下没有 `isaaclab.sh`。如果要跑 IsaacLab 3.x 官方基线，应切换到：

```bash
cd /data/SUN_ht/Isaac_Gym/IsaacLab
conda activate /data/SUN_ht/Isaac_Gym/env_isaaclab
```

## DEXTRAH 快速检查

```bash
python -c "import torch, warp; print(torch.__version__); print(warp.__version__)"
python -c "from rl_games.common import env_configurations, vecenv; from rl_games.algos_torch import model_builder; print('rl_games ok')"
python -c "import fabrics_sim; import dextrah_lab; print('dextrah packages ok')"
```

## Teacher 训练

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --seed -1 \
  --distributed \
  --num_envs 4096 \
  agent.params.config.minibatch_size=16384 \
  agent.params.config.central_value_config.minibatch_size=16384 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.multi_gpu=True \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=visdex_objects \
  env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
  env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
  env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=True
```

4090 单卡测试版：

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

4090 48GB 当前推荐大并行版本：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

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

小物体子集调参：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

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

子集模式只减少实际加载的 USD 物体，object one-hot 仍保持完整物体集维度，便于和完整物体集保持网络输入尺寸一致。子集适合快速调参和排错，最终结论仍需要回到完整物体集验证。

## 保存位置

当前 `train.py` 已调整为：本地只保存参数快照和 TensorBoard 日志，大文件 checkpoint 保存到大盘。

```text
本地日志:
/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/<实验名>

大文件:
/mnt/bigdata/SUN_ht/runs/dextrah/<实验名>/nn

本地 nn:
软链接到大盘 nn 目录
```

每次训练的实际参数会保存到：

```text
<本地日志>/<实验名>/params/agent.yaml
<本地日志>/<实验名>/params/env.yaml
```

## 加载与稳定性备注

当前代码已经把逐环境物体打印改为 5% 粒度进度输出：

```text
Loading objects: 409/8192 (5.0%) latest=<object_name>
Loading objects: 819/8192 (10.0%) latest=<object_name>
```

实际测试中，`8192/16384 + env.use_cuda_graph=True` 都曾在 `apply_object_wrench()` 阶段触发 CUDA/PhysX/Fabric 崩溃。`cuda_graph` 是加速项，不是训练效果项；当前优先使用 `env.use_cuda_graph=False` 保证稳定。`16384` 只作为压力测试，不建议日常调参默认使用。

## Teacher 可视化

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games
conda activate /data/SUN_ht/Isaac_Gym/env_dextrah

RUN_DIR=logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768
CKPT=$(find -L "$RUN_DIR/nn" -maxdepth 1 -type f -name "*.pth" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

echo "Run: $RUN_DIR"
echo "Checkpoint: $CKPT"

python play.py \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 16 \
  --checkpoint "$CKPT" \
  --objects_dir visdex_objects \
  --max_pose_angle 45.0
```

可视化不要加 `--headless`。本地 `nn` 是软链接到大盘，查找 checkpoint 时要用 `find -L`。`play.py` 使用普通 argparse 参数，不支持 `env.objects_dir=...` 这类 Hydra override。

## Student 蒸馏

相机 student 需要额外的 texture 数据。将 `textures.zip` 解压到：

```text
/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/assets
```

启动命令：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/distillation

python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  run_distillation.py \
  --headless \
  --distributed \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 256 \
  env.distillation=True \
  --enable_cameras \
  env.simulate_stereo=True \
  --teacher <teacher_checkpoint> \
  env.img_aug_type="rgb" \
  env.aux_coeff=10. \
  env.objects_dir="visdex_objects" \
  env.max_pose_angle=45.0 \
  env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
  env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
  env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
  env.use_cuda_graph=True
```

## 备注

DEXTRAH 是 Kuka + Allegro 手臂手系统，当前 Shadow Hand 立方体重定向实验不依赖它。当前如果只是做 IsaacLab 官方灵巧手 Reorient 消融，应优先使用 `env_isaaclab` 和 `/data/SUN_ht/Isaac_Gym/IsaacLab`。
