# DEXTRAH 本地 README

DEXTRAH 用于训练 Kuka + Allegro 的抓取策略，流程通常是先训练 privileged teacher，再进行基于相机输入的 student 蒸馏。

本仓库在当前工作区中作为独立环境使用，不和 IsaacLab 3.x 的 Shadow Hand Reorient 实验混用。

使用过的训练、续训、可视化参数已单独记录在：

```text
/data/SUN_ht/Isaac_Gym/DEXTRAH/USED_PARAMS.md
```

## 本地环境

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH
conda activate /data/SUN_ht/Isaac_Gym/env_dextrah
export HOME=/data/SUN_ht/Isaac_Gym/.home
export XDG_CACHE_HOME=/data/SUN_ht/Isaac_Gym/.home/.cache
export OMNI_KIT_ACCEPT_EULA=YES
```

相关路径：

```text
DEXTRAH: /data/SUN_ht/Isaac_Gym/DEXTRAH
FABRICS: /data/SUN_ht/Isaac_Gym/FABRICS
IsaacLab v2.2.1: /data/SUN_ht/Isaac_Gym/IsaacLab_v2.2.1
Conda 环境: /data/SUN_ht/Isaac_Gym/env_dextrah
DEXTRAH 大文件: /mnt/bigdata/SUN_ht/runs/dextrah
```

## 快速检查

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

低显存测试：

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

小物体子集调参：

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

默认 `env.use_object_subset=False`，表示使用完整物体集。子集只适合快速调参和排错，最终结论仍需要回到完整物体集验证。
子集模式只减少实际加载的 USD 物体，object one-hot 仍保持完整物体集维度，便于和完整集保持网络输入尺寸一致。

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

实际测试中，`8192/16384 + env.use_cuda_graph=True` 都曾在 `apply_object_wrench()` 阶段触发 CUDA/PhysX/Fabric 崩溃。`cuda_graph` 是加速项，不是训练效果项；当前优先使用 `env.use_cuda_graph=False` 保证稳定。`16384` 只作为压力测试，不建议日常调参默认使用。

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

已有有效 run `2026-08-04_13-55-52` 的 `nn/` 已迁移到：

```text
/mnt/bigdata/SUN_ht/runs/dextrah/2026-08-04_13-55-52/nn
```

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

可视化不要加 `--headless`。本地 `nn` 是软链接，查找 checkpoint 时使用 `find -L`。`play.py` 使用普通参数，例如 `--objects_dir`，不支持 `env.objects_dir=...` 这类 Hydra override。

## Student 蒸馏

蒸馏前需要准备 texture 数据，放到：

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
  --teacher teacher_lstm_base_best.pth \
  --max_iterations 200000 \
  --beta_schedule success_hold \
  --beta_success_target 0.30 \
  --beta_hold_iters 5000 \
  --beta_step 0.05 \
  env.distillation=True \
  --enable_cameras \
  env.simulate_stereo=True \
  env.img_aug_type="rgb" \
  env.aux_coeff=10. \
  env.objects_dir="visdex_objects" \
  env.use_object_subset=False \
  env.max_pose_angle=45.0 \
  env.enable_adr=False \
  env.num_adr_increments=0 \
  env.starting_adr_increments=0 \
  env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
  env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.005]" \
  env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
  env.use_cuda_graph=False
```

当前 teacher `teacher_lstm_base_best.pth` 对应 noADR 训练配置。因为环境代码在
`env.distillation=True` 时会默认把 `starting_adr_increments` 设为
`num_adr_increments`，如果不显式关闭 ADR，student 蒸馏环境会比 teacher 训练环境难很多。
使用上面的 noADR 配置可以让 teacher 在蒸馏环境中的表现更接近训练时分布。

默认 checkpoint 每 `5000` iteration 保存一次；需要改保存间隔时追加：

```bash
--save_every 2000
```

### Beta 调度

`beta` 表示每一步使用 teacher action 的概率：

```text
beta = 1.0  -> 100% teacher action
beta = 0.4  -> 40% teacher action, 60% student action
beta = 0.0  -> 100% student action
```

当前支持三种调度：

```text
time_staged   按 iteration 分阶段下降，默认前 30000 步保持 1.0，
              到 180000 步降到 0.4，避免过早下降。

success       按历史最高 in_success_region 直接计算 beta。
              这个模式反应快，但可能因为早期尖峰一次降得太多。

success_hold  仍按历史最高 in_success_region 计算目标 beta，
              但每次最多下降 beta_step，并且每次下降后至少保持
              beta_hold_iters 步。推荐用于避免过早下降和一次降到底。
```

推荐的保守配置：

```bash
--beta_schedule success_hold \
--beta_success_target 0.30 \
--beta_hold_iters 5000 \
--beta_step 0.05
```

如果历史最高成功率突然达到目标值，`success_hold` 也只会按如下节奏下降：

```text
0 step:     beta = 1.00
5000 step:  beta = 0.95
10000 step: beta = 0.90
15000 step: beta = 0.85
...
```

## Student 评估与数据记录

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/distillation

python eval.py \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 32 \
  --enable_cameras \
  --checkpoint <student_checkpoint> \
  --num_episodes 10 \
  env.distillation=True \
  env.simulate_stereo=True \
  env.img_aug_type="rgb" \
  env.objects_dir="visdex_objects" \
  env.max_pose_angle=45.0 \
  env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
  env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
  env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
  env.use_cuda_graph=True
```

需要记录数据时追加：

```bash
--record_data \
--max_records_per_file 100 \
--create_video
```

## 与当前 Shadow Hand 实验的关系

当前的官方灵巧手 Reorient/Repose 消融实验在 IsaacLab 3.x 中进行，路径是：

```text
/data/SUN_ht/Isaac_Gym/IsaacLab
```

对应环境是：

```text
/data/SUN_ht/Isaac_Gym/env_isaaclab
```

如果只是运行 `Isaac-Reorient-Cube-Shadow-Direct`，不要使用 DEXTRAH 环境。

## 本地修改记录

当前工作区对官方 DEXTRAH 做过以下本地适配：

- `dextrah_lab/rl_games/train.py`：将大文件输出目录迁移到 `/mnt/bigdata/SUN_ht/runs/dextrah`，本地 `nn/` 和 `videos/` 使用软链接，参数和 TensorBoard 日志仍保留在本地 run 目录。
- `dextrah_lab/rl_games/play.py`：增加 `--objects_dir`、`--max_pose_angle`、`--use_cuda_graph`、`--use_object_subset`、`--real-time`、`--speed_scale`、`--play_dt`、`--print_every` 等播放参数；默认使用确定性策略动作，便于复现实验效果。
- `dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env.py`：增加 IsaacLab 2.2.1/3.x 兼容处理，包括 PhysX/Warp 输出转 PyTorch、关节限位读取、关节摩擦写入、逐环境物体加载进度显示、物体子集加载、以及部分旧 Isaac Sim 工具函数替代。
- `dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env_cfg.py`：增加物体子集相关配置，并兼容不同 IsaacLab 版本的 `PhysxCfg`、`SimulationCfg` 导入。

运行可视化时，`595.84` 驱动曾导致 Isaac Sim 5.0 RTX/Vulkan 渲染链路段错误；当前建议使用 R580 系列驱动，已验证 `580.178.04` 能继续启动到场景和推理阶段。
