# DEXTRAH 中文总览：目录、启动与参数

本文档面向当前本机路径：

```text
/data/SUN_ht/Isaac_Gym/DEXTRAH
```

DEXTRAH 是 Kuka 机械臂 + Allegro 灵巧手的抓取策略项目，主要流程是：

```text
1. privileged teacher 强化学习训练
2. 基于相机输入的 student 蒸馏
3. student 评估、数据记录和部署参考
```

注意：DEXTRAH 使用独立环境，不要和 IsaacLab 3.x 的 Shadow Hand Reorient 实验混用。

## 一、目录功能

### 根目录

```text
/data/SUN_ht/Isaac_Gym/DEXTRAH
```

项目根目录，包含安装配置、README、依赖列表和 Python 包入口。

常见文件：

```text
README.md        本地中文 README
setup.py         pip editable install 入口
pyproject.toml   Poetry 项目配置
poetry.lock      Poetry 锁定文件
deps.txt         依赖列表
LICENSE          许可证
```

### dextrah_lab

```text
DEXTRAH/dextrah_lab
```

DEXTRAH 的主要 Python 包。任务、训练脚本、蒸馏脚本、资产和部署参考都在这里。

### dextrah_lab/assets

```text
DEXTRAH/dextrah_lab/assets
```

仿真资产目录。

子目录：

```text
kuka_allegro/     Kuka + Allegro 机器人资产
primitives/       基础几何体资产
scene_objects/    场景物体，例如 table.usd
visdex_objects/   训练用物体数据集
```

当前 teacher 训练需要：

```text
DEXTRAH/dextrah_lab/assets/visdex_objects/USD
```

其中每个物体一般是：

```text
visdex_objects/USD/<object_name>/<object_name>.usd
```

如果没有指定：

```bash
env.objects_dir=visdex_objects
```

会报：

```text
Need to specify valid directory of objects for training: ['visdex_objects']
```

### dextrah_lab/tasks

```text
DEXTRAH/dextrah_lab/tasks
```

任务定义目录。当前核心任务在：

```text
DEXTRAH/dextrah_lab/tasks/dextrah_kuka_allegro
```

### dextrah_lab/tasks/dextrah_kuka_allegro

```text
DEXTRAH/dextrah_lab/tasks/dextrah_kuka_allegro
```

Kuka + Allegro 抓取任务实现。

关键文件：

```text
gym_setup.py
dextrah_kuka_allegro_env.py
dextrah_kuka_allegro_env_cfg.py
agents/
```

功能说明：

```text
gym_setup.py                    注册 Gymnasium 任务 Dextrah-Kuka-Allegro
dextrah_kuka_allegro_env.py     环境主体逻辑：场景创建、观测、奖励、reset、物体加载
dextrah_kuka_allegro_env_cfg.py 环境默认参数：仿真频率、动作/观测、物体目录、奖励、ADR 等
agents/                         rl_games 网络和 PPO/LSTM 配置
```

当前任务名：

```text
Dextrah-Kuka-Allegro
```

### dextrah_lab/tasks/dextrah_kuka_allegro/agents

```text
DEXTRAH/dextrah_lab/tasks/dextrah_kuka_allegro/agents
```

训练和蒸馏用的 rl_games 配置。

teacher 默认配置：

```text
rl_games_ppo_lstm_cfg.yaml
```

常见 student 配置：

```text
rl_games_ppo_mono_transformer.yaml
rl_games_ppo_stereo_transformer.yaml
rl_games_mono_resnet.yaml
rl_games_ppo_lstm_scratch_cnn_aux.yaml
rl_games_ppo_lstm_scratch_cnn_aux_stereo.yaml
```

### dextrah_lab/rl_games

```text
DEXTRAH/dextrah_lab/rl_games
```

teacher 强化学习训练和播放入口。

关键文件：

```text
train.py          teacher 训练脚本
play.py           teacher 播放/评估脚本
rl_games_utils.py IsaacLab 环境与 rl_games 的适配工具
logs/             teacher 本地日志和参数快照
outputs/          Hydra 输出目录
```

当前有效 teacher run 示例：

```text
DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/2026-08-04_13-55-52
```

其中：

```text
params/agent.yaml   本次实际使用的 agent 参数快照
params/env.yaml     本次实际使用的环境参数快照
summaries/          TensorBoard event 文件
nn/                 checkpoint 软链接，真实文件在大盘
```

当前已经调整 `train.py` 的保存逻辑：大文件 checkpoint 默认保存到大盘，本地只保留日志、参数快照和软链接。

```text
本地日志目录:
DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/<实验名>

大文件目录:
/mnt/bigdata/SUN_ht/runs/dextrah/<实验名>/nn

本地软链接:
DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/<实验名>/nn
  -> /mnt/bigdata/SUN_ht/runs/dextrah/<实验名>/nn
```

如果临时想换大文件根目录，可以在命令前设置：

```bash
DEXTRAH_ARTIFACT_ROOT=/mnt/bigdata/SUN_ht/runs/dextrah_other python train.py ...
```

### dextrah_lab/distillation

```text
DEXTRAH/dextrah_lab/distillation
```

student 蒸馏、评估和数据记录脚本。

关键文件：

```text
run_distillation.py              student 蒸馏入口
run_distillation_transformer.py  transformer student 蒸馏入口
eval.py                          student 评估和数据记录
distillation.py                  DAgger/蒸馏核心逻辑
data_recorder.py                 数据记录工具
a2c_*.py                         不同 student 网络构建器
mono_encoder.py                  单目编码器
stereo_encoder.py                双目编码器
```

### dextrah_lab/deployment_scripts

```text
DEXTRAH/dextrah_lab/deployment_scripts
```

部署参考脚本，主要面向真实机器人或更完整的系统集成。

常见文件：

```text
kuka_allegro_fabric.py
kuka_allegro_state_machine.py
kuka_allegro_stereo_fgp.py
camera_calibration.py
camera_transform_publisher.py
policy_inference_stereo.py
policy_inference_transformer.py
```

这些脚本通常依赖相机、ROS 2、机器人驱动、PD 控制器等外部系统，不能保证开箱即跑。

## 二、环境准备

DEXTRAH 使用：

```text
Conda 环境: /data/SUN_ht/Isaac_Gym/env_dextrah
IsaacLab v2.2.1: /data/SUN_ht/Isaac_Gym/IsaacLab_v2.2.1
Isaac Sim: 5.0.0.0
```

每次运行前建议执行：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH
conda activate /data/SUN_ht/Isaac_Gym/env_dextrah
export HOME=/data/SUN_ht/Isaac_Gym/.home
export XDG_CACHE_HOME=/data/SUN_ht/Isaac_Gym/.home/.cache
export OMNI_KIT_ACCEPT_EULA=YES
```

`HOME` 和 `XDG_CACHE_HOME` 很重要，避免 Omniverse/Warp 缓存写到不可写目录。

快速检查：

```bash
python -c "import isaacsim; print('isaacsim ok')"
python -c "import torch, warp; print(torch.__version__); print(warp.__version__)"
python -c "import fabrics_sim; import dextrah_lab; print('dextrah packages ok')"
```

注意：不要用下面这个作为唯一判断：

```bash
python -c "import omni.kit.usd"
```

很多 `omni.*` 扩展需要在 `SimulationApp` 启动后才加载，直接 import 失败不一定代表训练不能跑。

## 三、Teacher 训练

Teacher 是 privileged RL 策略，使用状态/特权观测训练，默认配置是 LSTM actor-critic。

### 低显存小测试

先用 256 环境确认环境能启动、对象能加载、rl_games 配置没问题：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH
conda activate /data/SUN_ht/Isaac_Gym/env_dextrah
export HOME=/data/SUN_ht/Isaac_Gym/.home
export XDG_CACHE_HOME=/data/SUN_ht/Isaac_Gym/.home/.cache
export OMNI_KIT_ACCEPT_EULA=YES

cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 256 \
  agent.wandb_activate=False \
  env.use_cuda_graph=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096
```

为什么小测试要改 minibatch：

```text
batch_size = num_envs × horizon_length
256 × 16 = 4096
```

rl_games 要求：

```text
batch_size % minibatch_size == 0
```

所以 256 环境测试时不能继续用默认 `minibatch_size=16384`。

### 正式 teacher 训练

4090 上可以从 4096 环境开始：

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

如果 CUDA graph 占用过高或启动不稳定，可以先设：

```bash
env.use_cuda_graph=False
```

### 8192 大并行推荐版本

当前 4090 48GB 上建议优先使用 `8192 + noCudaGraph`。相比官方 4096 环境主要改动为：

```text
num_envs: 4096 -> 8192
minibatch_size: 16384 -> 32768
central_value_config.minibatch_size: 16384 -> 32768
env.use_cuda_graph: False
```

建议给 run 显式命名，实验名中包含关键变化：

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

保存位置：

```text
本地日志:
DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768

checkpoint 大文件:
/mnt/bigdata/SUN_ht/runs/dextrah/env8192_noCudaGraph_pose45_mb32768/nn
```

### 16384 压力测试版本

该版本用于压力测试，不建议作为日常调参默认值。实际测试中，`16384 + env.use_cuda_graph=True` 在 `apply_object_wrench()` 阶段触发过 CUDA/PhysX/Fabric 崩溃。

该版本相比官方 4096 环境主要改动为：

```text
num_envs: 4096 -> 16384
minibatch_size: 16384 -> 65536
central_value_config.minibatch_size: 16384 -> 65536
env.use_cuda_graph: False -> True
```

建议给 run 显式命名，实验名中包含关键变化：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

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

保存位置：

```text
本地日志:
DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env16384_cudaGraph_pose45_mb65536

checkpoint 大文件:
/mnt/bigdata/SUN_ht/runs/dextrah/env16384_cudaGraph_pose45_mb65536/nn
```

如果一定要测试 16384，建议先用：

```bash
env.use_cuda_graph=False
```

## 四、Teacher 播放

teacher 播放脚本在：

```text
DEXTRAH/dextrah_lab/rl_games/play.py
```

示例：

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

注意：

```text
可视化不要加 --headless。
必须使用 env_dextrah 环境，不要在 env_isaaclab 中运行 DEXTRAH play.py。
由于 nn 是软链接到大盘，查找 checkpoint 时用 find -L。
play.py 使用普通 argparse 参数，不支持 env.objects_dir=... 这类 Hydra override。
```

`play.py` 当前支持的任务相关参数：

```text
--objects_dir
--max_pose_angle
--use_cuda_graph
--use_object_subset
--object_subset_size
--object_subset_start_index
```

如果脚本参数和实际不一致，先执行：

```bash
python play.py --help
```

## 五、Student 蒸馏

student 蒸馏脚本在：

```text
DEXTRAH/dextrah_lab/distillation/run_distillation.py
```

蒸馏需要 teacher checkpoint。默认代码会找：

```text
DEXTRAH/pretrained_ckpts/new_teacher.pth
```

也可以通过：

```bash
--teacher <teacher_checkpoint_name>
```

传入 `pretrained_ckpts` 下的文件名。

示例：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/distillation

python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  run_distillation.py \
  --headless \
  --distributed \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 256 \
  --teacher new_teacher.pth \
  env.distillation=True \
  --enable_cameras \
  env.simulate_stereo=True \
  env.img_aug_type="rgb" \
  env.aux_coeff=10. \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
  env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
  env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
  env.use_cuda_graph=True
```

蒸馏输出默认在：

```text
DEXTRAH/dextrah_lab/distillation/runs/
```

## 六、Student 评估与数据记录

评估脚本：

```text
DEXTRAH/dextrah_lab/distillation/eval.py
```

示例：

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
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
  env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
  env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
  env.use_cuda_graph=True
```

记录数据时追加：

```bash
--record_data \
--max_records_per_file 100 \
--create_video
```

## 七、关键参数含义

### 通用 CLI 参数

```text
--headless
```

无窗口运行。训练时推荐打开，减少图形开销。

```text
--task=Dextrah-Kuka-Allegro
```

选择 DEXTRAH 注册的 Kuka + Allegro 任务。

```text
--num_envs
```

并行环境数量。越大采样吞吐可能越高，但显存和场景创建成本也更高。

```text
--seed
```

随机种子。`--seed -1` 表示使用随机种子；固定整数便于复现实验。

```text
--distributed
```

使用分布式启动。单卡也可以配合 `torch.distributed.run --nproc_per_node=1` 使用。

### 环境参数

```text
env.objects_dir=visdex_objects
```

指定训练物体目录。当前有效值是 `visdex_objects`。

```text
env.max_pose_angle=45.0
```

目标位姿角度范围。必须为正数，否则会报：

```text
Max pose angle must be positive
```

```text
env.use_cuda_graph=True/False
```

是否使用 CUDA graph 加速 FABRICS/仿真相关计算。`True` 更快但可能更占显存；小测试建议 `False`。

```text
env.success_for_adr=0.4
```

ADR 难度调整所需的成功阈值。达到一定成功表现后，域随机化/难度可能增加。

```text
env.distillation=True
```

启用 student 蒸馏模式，环境会使用 student 观测。

```text
env.simulate_stereo=True
```

启用双目相机模拟，常用于 stereo student。

```text
env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]"
```

覆盖 FABRICS 阻尼的 ADR 范围。

```text
env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]"
```

覆盖手指卷曲正则奖励权重。负值表示惩罚过度卷曲。

```text
env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]"
```

覆盖 lift reward 权重范围。

### rl_games/PPO 参数

这些来自：

```text
DEXTRAH/dextrah_lab/tasks/dextrah_kuka_allegro/agents/rl_games_ppo_lstm_cfg.yaml
```

```text
agent.params.config.horizon_length
```

每个环境每次 rollout 采样多少步。默认 `16`。

```text
agent.params.config.minibatch_size
```

PPO actor/critic 更新的小批量大小。必须整除：

```text
num_envs × horizon_length
```

```text
agent.params.config.central_value_config.minibatch_size
```

central value network 的 minibatch 大小，也需要与 batch size 匹配。

```text
agent.params.config.learning_rate
```

策略网络学习率。默认配置里是 `3e-4`，正式命令中常覆盖为 `1e-4`。

```text
agent.params.config.mini_epochs
```

每批 rollout 数据重复训练的 epoch 数。常用 `4`。

```text
agent.params.config.gamma
```

折扣因子。默认 `0.998`。

```text
agent.params.config.tau
```

GAE 参数。默认 `0.95`。

```text
agent.params.config.entropy_coef
```

熵正则系数，鼓励探索。默认 `0.002`。

```text
agent.params.config.grad_norm
```

梯度裁剪阈值。默认 `1.0`。

```text
agent.params.config.e_clip
```

PPO clip 范围。默认 `0.2`。

```text
agent.params.config.bounds_loss_coef
```

动作边界正则权重。默认 `0.005`。

```text
agent.params.config.seq_length
```

LSTM 序列长度。默认 `16`。

```text
agent.params.config.save_frequency
```

保存 checkpoint 的频率。默认 `200` epoch。

### 网络结构参数

teacher 默认使用 LSTM actor-critic。

Actor/Critic 主干：

```yaml
network:
  mlp:
    units: [512, 512]
    activation: elu
  rnn:
    name: lstm
    units: 1024
    layers: 1
```

Central value network：

```yaml
central_value_config:
  network:
    mlp:
      units: [1024, 512]
    rnn:
      units: 2048
      layers: 1
```

含义：

```text
MLP 提取当前观测特征
LSTM 记忆时间历史
central value network 使用非对称/特权状态估计 value
```

## 八、Reward 组成

DEXTRAH reward 主要在：

```text
DEXTRAH/dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env.py
```

核心 reward 分量包括：

```text
hand_to_object_reward      鼓励手靠近物体
object_to_goal_reward      鼓励物体接近目标位置
finger_curl_reg            手指卷曲正则，避免不合理卷曲
lift_reward                鼓励抬起物体并靠近目标高度
```

TensorBoard 中常见曲线：

```text
hand_to_object_reward/*
object_to_goal_reward/*
finger_curl_reg/*
lift_reward/*
in_success_region/*
rewards/*
shaped_rewards/*
episode_lengths/*
performance/step_fps
```

## 九、本机最近一次有效训练结果

有效 run：

```text
DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/2026-08-04_13-55-52
```

本地目录现在只保留参数快照、TensorBoard 日志和 `nn` 软链接：

```text
DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/2026-08-04_13-55-52/nn
  -> /mnt/bigdata/SUN_ht/runs/dextrah/2026-08-04_13-55-52/nn
```

大盘中已经保存模型：

```text
dextrah_lstm.pth
last_dextrah_lstm_ep_9800_rew_248.41273.pth
last_dextrah_lstm_ep_10200_rew_248.17651.pth
last_dextrah_lstm_ep_10800_rew_236.11205.pth
```

当前说明：

```text
训练已跑通，但还没有形成稳定成功策略。
目前按 checkpoint 文件名看，ep_9800 的 reward 最高，约 248.41273。
最终 checkpoint 附近到 ep_10800，最终策略文件为 dextrah_lstm.pth。
```

## 十、查看曲线

```bash
tensorboard \
  --logdir /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm \
  --host 127.0.0.1 \
  --port 6007
```

浏览器打开：

```text
http://127.0.0.1:6007
```

## 十一、常见错误

### 找不到 train.py

错误：

```text
can't open file '/data/SUN_ht/Isaac_Gym/DEXTRAH/train.py'
```

原因：在 DEXTRAH 根目录运行了 `train.py`。

解决：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games
```

### 缺 Isaac Sim 扩展

错误：

```text
omni.physx can't be satisfied
ModuleNotFoundError: No module named 'omni.kit.usd'
```

解决：安装 Isaac Sim 5 extscache。

```bash
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  "isaacsim[all,extscache]==5.0.0.0"
```

### 对象目录无效

错误：

```text
Need to specify valid directory of objects for training: ['visdex_objects']
```

解决：加参数。

```bash
env.objects_dir=visdex_objects
```

### max_pose_angle 非正数

错误：

```text
Max pose angle must be positive
```

解决：加参数。

```bash
env.max_pose_angle=45.0
```

### minibatch 不整除 batch

错误：

```text
assert(self.batch_size % self.minibatch_size == 0)
```

原因：

```text
batch_size = num_envs × horizon_length
```

小测试 `num_envs=256, horizon_length=16` 时：

```text
batch_size = 4096
```

所以 minibatch 应该用：

```text
4096, 2048, 1024, 512
```

不要用 `16384`。

### networkx 版本冲突

安装 extscache 后可能出现：

```text
urdfpy 0.0.22 requires networkx==2.2, but you have networkx 3.3
```

目前训练已能进入环境和 RL 初始化，先不处理。只有后续遇到 `urdfpy/networkx` 运行时报错时再单独修。

### 训练前加载很慢

DEXTRAH 启动慢主要来自 Isaac Sim、USD 物体实例化、PhysX/Fabric 场景构建和大规模并行环境创建。

当前代码已经把逐环境物体打印改为 5% 粒度进度输出：

```text
Loading 8192 object instances from ... (N unique objects)
Loading objects: 409/8192 (5.0%) latest=<object_name>
Loading objects: 819/8192 (10.0%) latest=<object_name>
...
```

这样可以避免每个环境打印多行导致终端刷屏和额外卡顿。

加载优化建议：

```text
调试阶段优先用 num_envs=256/4096。
大并行建议先用 num_envs=8192, env.use_cuda_graph=False。
16384 更像压力测试，不建议作为日常调参默认值。
如果使用小物体子集调参，只能验证趋势和连通性，不能直接等价替代完整物体集。
```

当前支持通过训练命令选择完整物体集或子集：

```text
env.use_object_subset=False       使用完整物体集，默认值
env.use_object_subset=True        使用子集
env.object_subset_size=64         子集物体数量
env.object_subset_start_index=0   从排序后的物体列表第几个开始取
```

实现细节：

```text
子集模式只减少实际实例化/加载的 USD 物体。
object one-hot 仍保持完整物体集维度，便于和完整物体集训练/评估保持网络输入尺寸一致。
```

子集调参示例：

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

完整物体集训练时不加这些参数，或显式写：

```bash
env.use_object_subset=False
```

## 十二、推荐工作流

1. 进入环境：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH
conda activate /data/SUN_ht/Isaac_Gym/env_dextrah
export HOME=/data/SUN_ht/Isaac_Gym/.home
export XDG_CACHE_HOME=/data/SUN_ht/Isaac_Gym/.home/.cache
export OMNI_KIT_ACCEPT_EULA=YES
```

2. 先跑 256 小测试。

3. 小测试能进入训练后，再跑 4096 teacher。

4. 用 TensorBoard 看：

```text
rewards/iter
in_success_region/iter
episode_lengths/iter
performance/step_fps
```

5. teacher 收敛后，再做 student 蒸馏。
