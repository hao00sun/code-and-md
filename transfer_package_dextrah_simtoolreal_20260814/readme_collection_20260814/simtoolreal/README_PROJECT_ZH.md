# SimToolReal 项目说明

## 它究竟解决什么问题

SimToolReal 解决的是灵巧手在真实世界中进行工具操作时，策略很难从仿真泛化到真实环境的问题。它关注的不是单纯抓取一个物体，而是让机器人能够操控工具，使工具上的关键点或姿态达到目标，从而完成锤子、螺丝刀、刷子、铲子、记号笔、橡皮等工具类任务。

核心问题可以概括为：

```text
如何在仿真中训练一个对象中心的灵巧工具操作策略，
并让它有机会零样本或少调试迁移到真实机器人。
```

## 代码库目标

这个代码库的目标接近完整闭环，但重点在 **仿真训练、评估和部署接口**。

它覆盖：

```text
训练:
  使用 Isaac Sim / Isaac Lab 或 legacy Isaac Gym 训练 RL policy。

推理:
  加载 pretrained_policy 或训练得到的 checkpoint 进行 rollout。

仿真:
  提供 SimToolReal 仿真环境、工具资产、目标采样、奖励、随机化和 benchmark。

数据采集:
  提供 recorded_data 和 DexToolBench 相关工具，用于记录、处理和可视化数据。

部署:
  提供 sim-to-sim 和 sim-to-real 部署节点，包括 policy node、goal pose node、
  fake robot/perception node 等接口。
```

它不是只提供一个训练脚本，也不是只提供一个部署脚本，而是围绕灵巧工具操作构建了从仿真训练、benchmark 评估到部署接口的较完整流程。真实部署仍依赖外部感知系统和机器人控制系统，例如 FoundationPose/SAM 感知节点和真实机器人节点。

## 适用任务、机器人本体和数据类型

适用任务：

```text
灵巧工具操作
工具姿态控制
多工具类别泛化
DexToolBench benchmark 评估
sim-to-sim / sim-to-real 策略验证
```

当前主要工具类别：

```text
hammer
screwdriver
marker
spatula
eraser
brush
```

机器人本体：

```text
Kuka 机械臂 + SHARPA 灵巧手
```

主要数据类型：

```text
机器人关节位置、速度
上一时刻 action target
手掌位置、姿态、速度
物体姿态、速度
工具关键点
目标关键点
物体尺度
success / reward / progress 等训练状态
真实或仿真记录数据
感知系统输出的物体位姿
```

当前 RL 训练中 actor 使用较少观测，critic 使用更完整状态，是非对称 actor-critic 设置。

## 输入和输出

训练阶段输入：

```text
任务配置 YAML
工具和机器人资产
目标采样配置
奖励函数配置
domain randomization 配置
RL/SAPG/PPO 超参数
可选 checkpoint
```

训练阶段输出：

```text
训练好的 policy checkpoint
TensorBoard 指标
Hydra 配置快照
训练日志
可选交互式 viewer 或视频
```

策略推理输入：

```text
当前机器人状态
工具/物体状态
目标姿态或目标关键点
上一时刻动作
可选感知系统输出
```

策略推理输出：

```text
机器人动作
机械臂目标增量
灵巧手关节目标
rollout 统计指标
successes / reward / done 原因
```

部署阶段输入：

```text
训练好的 policy
真实机器人状态
感知系统估计的物体位姿
目标姿态节点输出
坐标系标定信息
```

部署阶段输出：

```text
发送给机器人控制节点的动作命令
任务执行轨迹
成功/失败统计
记录数据
```

## 与同类代码库相比的核心优势

1. **面向工具操作，而不是简单抓取**

   很多灵巧操作代码库主要关注物体抓取、旋转或姿态重定向。SimToolReal 明确面向工具使用和工具关键点控制，更接近真实应用任务。

2. **对象中心表示**

   策略关注工具关键点、目标关键点、物体尺度和相对几何关系，这使它更适合跨不同工具实例泛化。

3. **覆盖仿真训练到部署接口**

   代码库不仅有训练环境，也包含 benchmark、数据记录、可视化、sim-to-sim 和 sim-to-real 部署节点，工程链条更完整。

4. **支持多工具类别和 DexToolBench**

   DexToolBench 提供多工具、多对象、多任务组合，便于系统评估策略是否真正具备泛化能力。

5. **训练中包含 sim-to-real 相关随机化**

   当前配置包含 reset 随机化、观测/动作延迟、物体状态噪声、关节速度噪声，以及物体被抬起后的随机外力/力矩扰动。这些设计服务于真实迁移，而不是只追求干净仿真中的高分。

6. **SAPG 探索机制**

   训练时将大量并行环境划分为不同探索强度 block，共同训练同一个 policy，用于提高探索效率和训练稳定性。

## 一句话总结

SimToolReal 是一个面向真实迁移的灵巧工具操作代码库，目标是在仿真中训练对象中心 policy，并通过 benchmark、随机化和部署接口，将策略推向真实或近真实机器人任务。

## 二、项目结构：每个目录分别负责什么

当前本地训练主要使用 Isaac Sim / Isaac Lab 版本，也就是：

```text
isaacsimenvs/
```

旧版 Isaac Gym 管线、部署、DexToolBench 和 baselines 仍然重要，但第一次阅读时可以按下面顺序抓主线：

```text
训练入口: isaacsimenvs/train.py
任务环境: isaacsimenvs/tasks/simtoolreal
训练配置: isaacsimenvs/cfg
RL 算法: rl_games
工具/benchmark: dextoolbench
部署接口: deployment
```

### 顶层目录

```text
simtoolreal/
├── assets/
├── baselines/
├── deployment/
├── dextoolbench/
├── docs/
├── isaacgymenvs/
├── isaacsimenvs/
├── recorded_data/
├── rl_games/
├── README.md
├── README_PROJECT_ZH.md
├── setup.py
├── pyproject.toml
├── download_dextoolbench_data.py
└── download_pretrained_policy.py
```

- `assets/`：机器人、桌面、工具、许可证等资产资源。
- `baselines/`：基线方法和轨迹优化 / IK 示例，主要用于对照实验或演示，不是当前 SAPG 训练主线。
- `deployment/`：部署和仿真/真实节点接口，包括 policy node、goal pose node、fake robot/perception、Isaac/MuJoCo 环境封装。
- `dextoolbench/`：DexToolBench 数据集、工具对象生成、评估、可视化和任务轨迹处理工具。
- `docs/`：安装、数据采集、部署、benchmark 和 baseline 文档。
- `isaacgymenvs/`：legacy Isaac Gym 训练环境。需要 Python 3.8，当前不作为推荐主线。
- `isaacsimenvs/`：当前推荐的 Isaac Sim / Isaac Lab 训练环境，属于主线代码。
- `recorded_data/`：真实或仿真记录数据的读取、切片、可视化工具。
- `rl_games/`：随仓库 vendored 的 RL 训练库，PPO/SAPG 训练实际由这里执行。
- `README.md`：官方 README。
- `README_PROJECT_ZH.md`：当前中文项目说明。
- `setup.py`、`pyproject.toml`：包安装和依赖配置。
- `download_dextoolbench_data.py`、`download_pretrained_policy.py`：下载数据和预训练策略的工具脚本。

### `isaacsimenvs/`

这是当前最重要的训练管线。

```text
isaacsimenvs/
├── train.py
├── play_video.py
├── cfg/
├── tasks/
├── tests/
└── utils/
```

- `train.py`：Isaac Sim / Isaac Lab 训练入口。负责启动 AppLauncher、注册 task、加载 Hydra/RL 配置、创建环境、调用 rl_games Runner 进行 PPO/SAPG 训练。
- `play_video.py`：视频播放或可视化相关入口。
- `cfg/`：任务和训练配置。包含 `task/SimToolReal.yaml`、`train/SimToolRealPPO.yaml`、`train/SimToolRealSAPG.yaml` 等。
- `tasks/`：Isaac Lab 任务实现。当前核心是 `tasks/simtoolreal/`。
- `tests/`：smoke test、环境加载、动作/观测规格、预训练 rollout 等测试脚本。
- `utils/`：Hydra、rl_games wrapper、W&B、视频捕获、HTML pose viewer 等工程工具。

实现方式上，`isaacsimenvs/train.py` 是入口，任务环境继承 Isaac Lab 的 `DirectRLEnv`，rl_games 负责训练算法。SAPG 不是完全独立算法，而是在 vendored rl_games PPO 上叠加探索系数条件化和 block 统计。

### `isaacsimenvs/tasks/simtoolreal/`

这是当前 SimToolReal Isaac Sim 版任务环境核心。

```text
tasks/simtoolreal/
├── simtoolreal_env.py
├── simtoolreal_env_cfg.py
├── pose_viewer.py
├── data/
└── utils/
```

- `simtoolreal_env.py`：环境主类 `SimToolRealEnv(DirectRLEnv)`。它本身很薄，主要负责接入 Isaac Lab 生命周期钩子，然后调用 `utils/` 中的具体任务数学。
- `simtoolreal_env_cfg.py`：环境配置类，定义机器人、场景、物体、观测、动作、reward、termination、随机化等。
- `pose_viewer.py`：姿态可视化辅助。
- `data/`：任务轨迹或目标数据，例如预生成 trajectory JSON。
- `utils/`：任务实现细节拆分目录。

`simtoolreal_env.py` 的实现链路很清晰：

```text
__init__
  -> 根据 obs/state list 计算 observation_space 和 state_space
  -> DirectRLEnv 初始化并创建场景
  -> 分配状态缓存

_setup_scene
  -> 调用 scene_utils.setup_scene

_reset_idx
  -> 调用 reset_utils.reset_env_state

_pre_physics_step
  -> action_utils.apply_action_pipeline
  -> action_utils.apply_wrench_dr

_apply_action
  -> 将当前目标写入机器人 joint target

_get_dones
  -> 更新 success tolerance curriculum
  -> 计算中间状态
  -> termination_utils.compute_terminations

_get_rewards
  -> reward_utils.compute_rewards
  -> logging_utils.log_step_metrics

_get_observations
  -> obs_utils.build_observations
```

这种实现方式的好处是：环境主类保持很薄，reward、reset、obs、termination、action pipeline 都分到独立文件，方便调试和替换。

### `isaacsimenvs/tasks/simtoolreal/utils/`

```text
utils/
├── action_utils.py
├── generate_objects.py
├── goal_sampling.py
├── logging_utils.py
├── object_size_distributions.py
├── obs_utils.py
├── reset_utils.py
├── reward_utils.py
├── scene_utils.py
└── termination_utils.py
```

- `action_utils.py`：把 policy action 转成机械臂/手的目标动作，同时处理 action delay、目标平滑和物体外力扰动。
- `generate_objects.py`：生成或加载训练物体。
- `goal_sampling.py`：目标姿态/关键点采样。
- `logging_utils.py`：训练指标、success、episode 统计写入 info/TensorBoard。
- `object_size_distributions.py`：物体尺寸分布和采样。
- `obs_utils.py`：actor observation、critic state、student observation 的构建。
- `reset_utils.py`：reset 逻辑，包括物体、机器人、目标、随机化状态。
- `reward_utils.py`：reward 项计算，包括 lifting、keypoint、distance-delta、reach-goal bonus、action penalty 等。
- `scene_utils.py`：Isaac Sim 场景创建、物体/机器人/材质/物理属性设置。
- `termination_utils.py`：done 条件、成功计数、失败原因和 tolerance curriculum。

### `isaacsimenvs/cfg/`

```text
cfg/
├── task/
└── train/
```

- `cfg/task/SimToolReal.yaml`：任务级参数，包括观测列表、状态列表、reward、termination、随机化、success tolerance 等。
- `cfg/train/SimToolRealPPO.yaml`：PPO 训练配置。
- `cfg/train/SimToolRealSAPG.yaml`：SAPG 训练配置。当前正式训练使用这个分支。

配置文件非常重要。很多行为不是写死在 Python 中，而是通过 Hydra override 修改，例如：

```text
env.scene.num_envs
agent.params.config.expl_coef_block_size
agent.params.config.max_epochs
agent.params.config.save_frequency
```

### `isaacsimenvs/utils/`

```text
utils/
├── hydra_utils.py
├── rlgames_utils.py
├── video_capture.py
├── wandb_utils.py
└── interactive_viewer/
```

- `hydra_utils.py`：处理 Hydra 配置加载和命令行 override。
- `rlgames_utils.py`：Isaac Lab env 与 rl_games 的桥接，包含环境包装、统计指标 observer、SAPG 指标处理、teacher/student obs 适配等。
- `video_capture.py`：录制视频的相机挂载和帧捕获。
- `wandb_utils.py`：W&B 日志工具。
- `interactive_viewer/`：姿态 HTML viewer，适合不启用 Isaac camera 的轻量可视化。

这些属于工程封装。训练能不能正常显示指标、TensorBoard 是否有 success/performance 曲线，通常和这里有关。

### `isaacgymenvs/`

这是 legacy Isaac Gym 版本。

```text
isaacgymenvs/
├── train.py
├── launch_training.py
├── cfg/
├── tasks/
├── utils/
└── pbt/
```

- `train.py`、`launch_training.py`：旧 Isaac Gym 训练入口。
- `cfg/`：旧版任务和训练 YAML。
- `tasks/`：旧版任务实现。
- `utils/`：旧版 rl_games wrapper、W&B、domain randomization 等工具。
- `pbt/`：population based training 相关逻辑。

当前推荐 Isaac Sim / Isaac Lab，所以新手可以先跳过 `isaacgymenvs/`。只有复现论文旧管线或对比 Isaac Gym 与 Isaac Sim 差异时再读。

### `rl_games/`

这是仓库内置的 RL 训练库。

```text
rl_games/
├── rl_games/
├── runner.py
├── docs/
├── tests/
├── notebooks/
├── pyproject.toml
└── README.md
```

- `rl_games/rl_games/`：PPO/A2C/SAPG 训练算法主体。
- `runner.py`：rl_games 通用启动入口。
- `docs/`：rl_games 配置说明。
- `tests/`、`notebooks/`：测试和示例。

一般不建议新手一开始改这里。只有当你要理解 SAPG 的 PPO overlay、exploration coefficient block、loss 细节、保存逻辑时再深入。

### `dextoolbench/`

```text
dextoolbench/
├── eval_isaacsim.py
├── eval_isaacgym.py
├── run_all_evals_isaacsim.py
├── generate_training_objects.py
├── generate_collision_meshes.py
├── interactive_create_task_trajectory.py
├── process_poses.py
├── visualize_*.py
├── metadata.py
└── objects.py
```

- `eval_isaacsim.py`、`run_all_evals_isaacsim.py`：Isaac Sim benchmark 评估。
- `eval_isaacgym.py`、`run_all_evals_isaacgym.py`：旧 Isaac Gym benchmark 评估。
- `generate_training_objects.py`、`generate_collision_meshes.py`：训练物体和碰撞网格生成。
- `interactive_create_task_trajectory.py`、`process_poses.py`：交互创建/处理任务轨迹。
- `visualize_*.py`：物体、任务、分解结果、demo 的可视化。
- `metadata.py`、`objects.py`：DexToolBench 元数据和对象定义。

这部分是 benchmark 和数据工具，不是训练主循环，但对理解“工具任务从哪里来”很关键。

### `deployment/`

```text
deployment/
├── rl_policy_node.py
├── rl_player.py
├── goal_pose_node.py
├── goal_pose_listener_node.py
├── sharpa_node.py
├── visualization_node.py
├── fake/
├── isaac/
└── mujoco/
```

- `rl_policy_node.py`：策略节点，把机器人/感知状态转成动作。
- `rl_player.py`、`rl_player_utils.py`：加载 policy 并执行推理的工具。
- `goal_pose_node.py`、`goal_pose_listener_node.py`：目标姿态发布/监听。
- `sharpa_node.py`：SHARPA 手相关接口。
- `visualization_node.py`：可视化节点。
- `fake/`：fake robot / fake perception，用于不接真实硬件时调通链路。
- `isaac/`：Isaac 部署环境封装。
- `mujoco/`：MuJoCo 部署/仿真环境封装。

如果当前只做训练，deployment 可以暂时跳过；如果要 sim-to-real 或真实系统闭环，它就是重点。

### `baselines/`

```text
baselines/
├── run_trajopt.py
├── test_trajopt_sharpa.py
├── visualize_demo_with_hand.py
├── visualize_demo_with_hand_trajopt.py
├── pyroki_snippets/
└── assets/
```

这里主要是轨迹优化、IK、可视化和 baseline demo。它用于对比和理解任务几何，不是当前 SAPG 训练主流程。

### `recorded_data/`

```text
recorded_data/
├── core.py
├── slice_recorded_data.py
└── visualize.py
```

- `core.py`：记录数据的数据结构和读写。
- `slice_recorded_data.py`：切片已有记录。
- `visualize.py`：可视化记录数据。

适合做真实/仿真数据回放和分析。

### `assets/`

```text
assets/
├── urdf/
└── licenses/
```

- `urdf/`：桌面、工具、环境相关 URDF。
- `licenses/`：资产许可证。

一般不需要先读。只有资产路径、碰撞、导入、许可问题才需要查。

## 哪些是核心算法、工程封装、示例或工具

核心算法 / 任务逻辑：

```text
isaacsimenvs/tasks/simtoolreal/simtoolreal_env.py
isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py
isaacsimenvs/tasks/simtoolreal/utils/action_utils.py
isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py
isaacsimenvs/tasks/simtoolreal/utils/reward_utils.py
isaacsimenvs/tasks/simtoolreal/utils/reset_utils.py
isaacsimenvs/tasks/simtoolreal/utils/termination_utils.py
isaacsimenvs/tasks/simtoolreal/utils/goal_sampling.py
isaacsimenvs/cfg/task/SimToolReal.yaml
isaacsimenvs/cfg/train/SimToolRealSAPG.yaml
rl_games/rl_games/
```

工程封装 / 框架适配：

```text
isaacsimenvs/train.py
isaacsimenvs/utils/rlgames_utils.py
isaacsimenvs/utils/hydra_utils.py
isaacsimenvs/utils/wandb_utils.py
isaacsimenvs/utils/video_capture.py
setup.py
pyproject.toml
```

Benchmark / 数据工具：

```text
dextoolbench/
recorded_data/
download_dextoolbench_data.py
download_pretrained_policy.py
```

部署接口：

```text
deployment/
```

示例、测试或可后读内容：

```text
baselines/
isaacsimenvs/tests/
rl_games/tests/
rl_games/notebooks/
docs/
```

legacy 管线：

```text
isaacgymenvs/
```

## 必读文件和可暂时跳过文件

第一次阅读建议顺序：

```text
1. README_PROJECT_ZH.md
2. README.md
3. docs/isaacsim_installation.md
4. isaacsimenvs/train.py
5. isaacsimenvs/cfg/task/SimToolReal.yaml
6. isaacsimenvs/cfg/train/SimToolRealSAPG.yaml
7. isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py
8. isaacsimenvs/tasks/simtoolreal/simtoolreal_env.py
9. isaacsimenvs/tasks/simtoolreal/utils/reward_utils.py
10. isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py
11. isaacsimenvs/tasks/simtoolreal/utils/action_utils.py
12. isaacsimenvs/utils/rlgames_utils.py
```

可以暂时跳过：

```text
.git/
__pycache__/
assets/ 下的大量 URDF/许可证文件
isaacgymenvs/，除非要复现 legacy Isaac Gym
deployment/，除非要做真实部署
baselines/，除非要看轨迹优化基线
rl_games/notebooks 和 tests
docs 中暂时无关的真实数据采集/部署章节
```

## 模块之间如何串起来

Isaac Sim 训练链路：

```text
isaacsimenvs/train.py
  -> AppLauncher 启动 Isaac Sim
  -> Hydra 读取 cfg/task 和 cfg/train
  -> 注册 Isaacsimenvs-SimToolReal-Direct-v0
  -> 创建 SimToolRealEnv
  -> RlGamesVecEnvWrapper / rlgames_utils 适配 rl_games
  -> rl_games Runner 执行 PPO/SAPG
  -> TensorBoard / W&B / Hydra 输出训练指标和配置
```

环境内部链路：

```text
simtoolreal_env_cfg.py 定义任务参数
  -> scene_utils 创建机器人、桌面、工具和物理属性
  -> reset_utils 初始化机器人、物体、目标和随机化状态
  -> obs_utils 构造 actor observation 和 critic state
  -> action_utils 将 policy action 转成机器人目标动作并施加扰动
  -> reward_utils 计算工具关键点、抬起、距离变化、成功奖励和动作惩罚
  -> termination_utils 判断 success、fall、hand_far、timeout 等 done 原因
  -> logging_utils 把 successes、rewards、episode_final 等指标写出
```

SAPG 训练链路：

```text
SimToolRealSAPG.yaml
  -> 开启 PPO + SAPG exploration overlay
  -> num_envs 按 expl_coef_block_size 分成多个 block
  -> 不同 block 使用不同探索系数
  -> 所有 block 共同更新同一个 policy
  -> TensorBoard 中记录 successes_per_block 等分组表现
```

Benchmark / 数据链路：

```text
dextoolbench
  -> 生成或加载工具对象、目标轨迹和碰撞网格
  -> eval_isaacsim.py / run_all_evals_isaacsim.py 批量评估 policy
  -> visualize_*.py 检查物体、任务和 demo 是否合理
```

部署链路：

```text
deployment/rl_policy_node.py
  -> 加载训练好的 policy
  -> 接收机器人状态和感知系统输出
  -> 根据目标姿态生成动作
  -> 发送给 fake/Isaac/MuJoCo/真实机器人接口
```

## 对新手最重要的判断

如果目标是继续当前训练或看指标，优先看：

```text
isaacsimenvs/train.py
isaacsimenvs/cfg/train/SimToolRealSAPG.yaml
isaacsimenvs/cfg/task/SimToolReal.yaml
isaacsimenvs/tasks/simtoolreal/utils/logging_utils.py
```

如果目标是改任务本身，优先看：

```text
simtoolreal_env_cfg.py
reward_utils.py
obs_utils.py
action_utils.py
termination_utils.py
reset_utils.py
scene_utils.py
```

如果目标是部署到真实系统，优先看：

```text
docs/deployment.md
deployment/rl_policy_node.py
deployment/goal_pose_node.py
deployment/fake/
deployment/isaac/
deployment/mujoco/
```
