# 项目定位与代码结构导读

本文回答两个问题：

1. 这个代码库到底解决什么问题。
2. 一个新手应该如何理解每个目录、每个核心模块和必读文件。

## 一、项目定位：它究竟解决什么问题

### 1.1 这个代码库的目标

这个代码库的主体是 LeRobot，本地版本又加入了 PI05、Innov、DiT、HIL-SERL 等实验脚本和代码改造。

整体目标不是只做一个单点功能，而是提供真实机器人模仿学习/强化学习的一套工作流：

```text
数据采集 -> 数据集整理/修复/转换 -> 策略训练 -> checkpoint 保存 -> 离线/在线推理 -> 机器人部署 -> 日志与实验复盘
```

它覆盖：

- 训练：`lerobot-train`、PI05 训练脚本、DiT 训练脚本、reward classifier 训练。
- 推理：`pi05_server.py`、Innov 本地推理脚本、diffusion/flow matching server。
- 数据采集：LeRobot 原生 `lerobot-record`、`lerobot-teleoperate`、`lerobot-replay` 等。
- 仿真评估：`envs/` 和 `lerobot-eval` 支持 ALOHA、PushT、LIBERO、MetaWorld、RoboCasa 等环境。
- 真实机器人部署：`robots/`、`motors/`、`cameras/`、`teleoperators/` 负责硬件抽象。
- 闭环学习：`rl/`、`rewards/`、`hil_serl_arx/` 为 HIL-SERL、奖励模型和在线交互做准备。

所以它更接近一个“机器人学习完整闭环代码库”，不是单纯的训练脚本，也不是单纯的部署脚本。

### 1.2 它主要解决的问题

它解决的是：如何把真实机器人采集到的视频、状态、动作数据，训练成可执行的策略模型，并在仿真或真实机器人上运行。

具体包括：

- 统一数据格式：用 LeRobotDataset 管理 episode、frame、video、state、action、task 等信息。
- 统一配置系统：用 dataclass + draccus，把 yaml/json 配置转成训练、推理、硬件控制参数。
- 统一策略接口：所有策略继承 `PreTrainedPolicy`，都能保存、加载、推理、上传/下载。
- 统一处理流水线：processor 负责图像、状态、动作、归一化、tokenizer、相对动作等转换。
- 支持多种策略：ACT、Diffusion、TDMPC、SAC、SmolVLA、PI0、PI05、MultiTaskDiT 等。
- 支持硬件接入：相机、舵机/电机、机械臂 follower/leader、键盘/手柄/手机遥操作。
- 支持本地实验工程化：日志、GPU 监控、离线缓存、权重检查、server/client 分离推理。

### 1.3 它适用于什么任务

适合的任务：

- 模仿学习：从人类遥操作数据训练策略。
- 视觉语言动作策略：例如 PI05，根据多相机图像、机器人状态、语言 task 输出动作。
- 动作块预测：一次预测未来多个动作，减少逐步推理开销。
- 双臂/单臂机械臂控制：本地脚本里大量围绕 14 维双臂和 7 维单臂 action。
- 奖励模型训练：从成功/失败片段训练 reward classifier。
- 仿真 benchmark：LIBERO、MetaWorld、RoboCasa 等。
- 真实机器人部署：使用串口、电机总线、RealSense 或 OpenCV 摄像头。

不太适合的任务：

- 纯文本模型训练。
- 纯视觉分类，不涉及机器人状态/动作。
- 完全不符合 LeRobot 数据格式的数据，除非先转换。
- 极低算力设备上直接训练 PI05 这类大模型。

### 1.4 它适用于什么机器人本体

官方/仓库内支持的机器人类型很多，主要在 `src/lerobot/robots/`：

```text
so100 / so101
koch
openarm
rebot
reachy2
lekiwi
unitree_g1
omx
hope_jr
earthrover
```

当前本地实验主要围绕：

- ARX/Innov 双臂数据。
- 14 维 action/state：左臂 7 维 + 右臂 7 维。
- RealSense 多相机输入。
- 本地串口 `/dev/ttyACM0`、`/dev/ttyACM1` 控制机械臂。

### 1.5 它适用于什么数据类型

典型输入数据是 LeRobot 格式数据集：

```text
meta/info.json
meta/stats.json
meta/tasks.*
meta/episodes.*
data/chunk-xxx/episode_xxxxxx.parquet
videos/.../episode_xxxxxx.mp4
```

常见字段：

```text
observation.state
observation.images.<camera_name>
action
task
episode_index
frame_index
timestamp
index
```

本地常见数据维度：

```text
observation.state: [14] 或 [7]
action: [14] 或 [7]
image: 多相机 RGB 视频
task: 自然语言任务描述
```

### 1.6 输入和输出分别是什么

训练阶段输入：

```text
训练配置 yaml/json
LeRobot 数据集
预训练权重，可选
Hugging Face tokenizer/cache，可选
```

训练阶段输出：

```text
output_dir/
  checkpoints/
    002000/
    004000/
    ...
      pretrained_model/
        config.json
        model.safetensors
        policy_preprocessor.json
        policy_postprocessor.json
        train_config.json
日志目录/
  train.log
  train_stdout.log
  env_snapshot.txt
  gpu_usage.csv
  train_command.txt
```

推理阶段输入：

```text
checkpoint/pretrained_model
当前机器人状态 observation.state
多相机图像 observation.images.*
任务语言 task
```

推理阶段输出：

```text
action
```

对 PI05 来说，本地常见输出是 14 维双臂动作，或者动作块：

```text
[left_joint1..left_joint6, left_gripper, right_joint1..right_joint6, right_gripper]
```

### 1.7 与同类代码库相比的核心优势

核心优势：

- 数据、模型、训练、部署接口统一：不是零散 notebook，而是完整工程。
- Hugging Face 集成好：模型、数据集、processor、配置都能序列化和复用。
- 策略种类多：既有传统 imitation learning，也有 VLA/PI0/PI05/MultiTaskDiT。
- 硬件抽象完整：相机、电机、机器人本体、遥操作设备都有统一接口。
- 配置驱动：大多数训练/推理行为都能通过 yaml/json 或 CLI 参数控制。
- Processor 设计清晰：归一化、图像处理、动作转换、tokenizer 和设备搬运拆成流水线。
- 本地版本已经补齐真实实验需要的工程细节：离线权重、日志快照、GPU 监控、PI05 action loss mask、server/client 分离推理、ARX/Innov 数据处理脚本。

## 二、项目结构：每个目录分别负责什么

### 2.1 仓库一级目录总览

```text
.
├── src/lerobot/              核心库代码
├── tests/                    自动化测试
├── examples/                 官方示例和教程脚本
├── docs/                     官方文档源码
├── docker/                   Docker 环境
├── scripts/                  CI/辅助脚本
├── media/                    README/文档媒体资源
├── DIT/                      本地 MultiTaskDiT 实验脚本
├── innov_il/                 本地 Innov 机器人训练/推理/数据处理
├── hil_serl_arx/             本地 HIL-SERL 和 reward classifier 实验
├── usage_cn/                 当前中文说明文档
├── outputs/                  默认输出目录
├── pyproject.toml            依赖、入口命令、工具配置
├── uv.lock                   uv 锁文件
├── README.md                 官方项目介绍
├── AGENT_GUIDE.md            面向用户的操作指南
├── AGENTS.md                 给 AI/工程 agent 的仓库说明
├── pi05_semantic.yaml        当前 PI05 主训练配置
├── pi05_server.py            当前 PI05 server 推理脚本
└── run_pi05_train_with_logs.sh PI05 带日志训练入口
```

### 2.2 `src/lerobot/` 二级目录

这是核心代码。新手读代码时优先看这里。

```text
src/lerobot/scripts/
```

CLI 入口实现。`pyproject.toml [project.scripts]` 把命令映射到这里。

典型命令：

```text
lerobot-train
lerobot-eval
lerobot-record
lerobot-replay
lerobot-teleoperate
lerobot-rollout
lerobot-dataset-viz
lerobot-find-cameras
lerobot-find-port
```

核心文件：

```text
src/lerobot/scripts/lerobot_train.py
src/lerobot/scripts/lerobot_eval.py
src/lerobot/scripts/lerobot_record.py
src/lerobot/scripts/lerobot_rollout.py
```

实现方式：

- 解析 dataclass 配置。
- 创建 dataset/env/policy/processor/optimizer。
- 执行训练循环、评估、保存 checkpoint。
- record/teleoperate/rollout 则负责真实机器人或策略运行。

```text
src/lerobot/configs/
```

全局配置系统。LeRobot 大量使用 dataclass + draccus，把配置类自动变成 CLI 参数。

核心文件：

```text
src/lerobot/configs/train.py
src/lerobot/configs/policies.py
src/lerobot/configs/types.py
src/lerobot/configs/parser.py
```

实现方式：

- `TrainPipelineConfig` 是训练根配置。
- 策略、环境、优化器、调度器都注册成可选择子类。
- CLI 里 `--policy.type=pi05` 会映射到 `PI05Config`。

```text
src/lerobot/policies/
```

策略模型目录，是核心算法区。

常见策略：

```text
act/
diffusion/
multi_task_dit/
pi0/
pi05/
smolvla/
tdmpc/
sac/
vqbet/
```

关键公共文件：

```text
src/lerobot/policies/pretrained.py
src/lerobot/policies/factory.py
```

每个策略通常按三个文件组织：

```text
configuration_<name>.py
modeling_<name>.py
processor_<name>.py
```

实现方式：

- `configuration_*.py` 定义策略超参数和输入输出 features。
- `modeling_*.py` 定义 PyTorch 模型和 `select_action` / forward / loss。
- `processor_*.py` 定义该策略需要的前处理和后处理。
- `factory.py` 根据 `policy.type` 或 `policy.path` 动态创建策略。

本地最需要关注：

```text
src/lerobot/policies/pi05/configuration_pi05.py
src/lerobot/policies/pi05/modeling_pi05.py
src/lerobot/policies/pi05/processor_pi05.py
src/lerobot/policies/multi_task_dit/
```

```text
src/lerobot/processor/
```

数据处理流水线。它连接 dataset batch 和 policy input/output。

核心文件：

```text
src/lerobot/processor/pipeline.py
src/lerobot/processor/normalize_processor.py
src/lerobot/processor/rename_processor.py
src/lerobot/processor/relative_action_processor.py
```

实现方式：

- `DataProcessorPipeline` / `PolicyProcessorPipeline` 串联多个 step。
- preprocessor：dataset batch -> policy input。
- postprocessor：policy output -> robot action。
- checkpoint 会保存 `policy_preprocessor.json` 和 `policy_postprocessor.json`。

```text
src/lerobot/datasets/
```

LeRobot 数据集读写和元信息管理。

核心文件：

```text
src/lerobot/datasets/lerobot_dataset.py
src/lerobot/datasets/dataset_metadata.py
src/lerobot/datasets/factory.py
src/lerobot/datasets/compute_stats.py
src/lerobot/datasets/video_utils.py
```

实现方式：

- `LeRobotDataset` 按 episode-aware 方式采样。
- 支持 parquet 数据和 mp4 视频。
- 根据 `meta/info.json`、`meta/stats.json` 解析 features 和归一化统计。
- 训练时 `make_dataset(cfg)` 创建 dataset。

```text
src/lerobot/envs/
```

仿真环境封装。

作用：

- 把 ALOHA、PushT、LIBERO、MetaWorld、RoboCasa、VLABench 等环境统一成 LeRobot 可评估接口。
- `lerobot-eval` 会用这里创建环境。

核心文件：

```text
src/lerobot/envs/configs.py
src/lerobot/envs/factory.py
```

```text
src/lerobot/robots/
```

真实机器人 follower 本体抽象。

作用：

- 连接真实机械臂。
- 读取 observation。
- 发送 action。
- 管理校准、关节限制、断开连接等。

每个机器人一般都有：

```text
config_*.py
*_follower.py
```

```text
src/lerobot/teleoperators/
```

遥操作设备抽象。

支持：

```text
keyboard
gamepad
phone
leader arm
homunculus
unitree_g1
```

作用：

- `lerobot-teleoperate` 和 `lerobot-record` 用它把人类输入转成机器人 action。

```text
src/lerobot/cameras/
```

相机抽象。

支持：

```text
opencv
realsense
zmq
reachy2_camera
```

作用：

- 统一读取 RGB 图像。
- 给机器人 observation 或数据采集提供图像。

```text
src/lerobot/motors/
```

电机总线和底层驱动。

支持：

```text
dynamixel
feetech
damiao
robstride
```

作用：

- 读写关节位置、速度、力矩。
- 处理串口、CAN、校准和 motor bus。

```text
src/lerobot/rewards/
```

奖励模型。

包含：

```text
classifier/
robometer/
sarm/
topreward/
```

本地重点：

```text
src/lerobot/rewards/classifier/
```

作用：

- 训练/加载 reward classifier。
- 给 HIL-SERL 或在线学习提供成功概率/奖励信号。

```text
src/lerobot/rl/
```

强化学习和 HIL-SERL 相关逻辑。

包含：

```text
algorithms/
data_sources/
actor.py
learner.py
```

作用：

- actor 负责和真实/仿真环境交互。
- learner 负责在线更新。
- algorithms 里包含 SAC 等算法配置和实现。

```text
src/lerobot/rollout/
```

策略部署和 rollout 框架。

作用：

- `lerobot-rollout` 的配置和执行逻辑。
- 支持同步推理、RTC 推理、不同 rollout strategy。

```text
src/lerobot/optim/
```

优化器和学习率调度器。

核心：

```text
optimizers.py
schedulers.py
factory.py
```

```text
src/lerobot/transforms/
src/lerobot/data_processing/
src/lerobot/common/
src/lerobot/model/
src/lerobot/templates/
src/lerobot/transport/
src/lerobot/utils/
```

这些是通用支撑模块：

- `transforms/`：图像或数据增强变换。
- `data_processing/`：额外数据处理工具。
- `common/`：通用组件。
- `model/`：部分共享模型组件。
- `templates/`：模型卡模板。
- `transport/`：通信/传输相关封装。
- `utils/`：日志、常量、导入、随机种子、文件等工具函数。

## 三、本地新增/重点目录

### 3.1 `usage_cn/`

中文说明文档。

建议新手先读：

```text
usage_cn/README.md
usage_cn/01_from_zero_setup.md
usage_cn/04_project_position_and_structure.md
usage_cn/02_local_usage_and_experiments.md
```

### 3.2 `DIT/`

本地 MultiTaskDiT 实验。

包含：

```text
download_dit_pretrained_with_proxy.sh
run_dit_flow_matching_train_with_logs.sh
run_dit_diffusion_train_with_logs.sh
multi_task_dit_flow_matching_train.yaml
multi_task_dit_diffusion_train.yaml
dit_flow_matching_server.py
dit_diffusion_server.py
```

作用：

- 下载 DiT 预训练权重。
- 训练 flow matching / diffusion。
- 启动 DiT 推理 server。

### 3.3 `innov_il/`

本地 Innov 机器人实验目录。

包含：

```text
run_innov_pi05_local_inference.sh
innov_pi05_local_inference.py
run_pi05_arx_0724_1553_ablation_train.sh
backup_and_crop_dataset.py
prepare_left_arm_dataset.py
diffusion_innov_0617_1554.yaml
flow_matching_innov_0617_1554.yaml
flow_matching_left_arm.yaml
```

作用：

- PI05 本地机械臂推理。
- 四相机 PI05 训练。
- 旧 diffusion / flow matching 训练和离线推理。
- 数据集裁剪、备份、左臂数据生成。

### 3.4 `hil_serl_arx/`

HIL-SERL 和 reward classifier 实验目录。

包含：

```text
run_train_reward_classifier_terminal.sh
make_reward_classifier_terminal_dataset_v30.py
reward_classifier_train_config_arx_terminal.json
run_native_hilserl_socket_server.sh
native_hilserl_socket_server.py
```

作用：

- 从成功/失败数据构造终端状态 reward classifier 数据集。
- 训练 reward classifier。
- 启动 native HIL-SERL socket server。

## 四、核心算法、工程封装、示例测试怎么区分

核心算法：

```text
src/lerobot/policies/
src/lerobot/rewards/
src/lerobot/rl/
src/lerobot/optim/
```

工程封装：

```text
src/lerobot/scripts/
src/lerobot/configs/
src/lerobot/processor/
src/lerobot/datasets/
src/lerobot/robots/
src/lerobot/cameras/
src/lerobot/motors/
src/lerobot/teleoperators/
src/lerobot/rollout/
```

本地实验工程：

```text
DIT/
innov_il/
hil_serl_arx/
pi05_server.py
run_pi05_train_with_logs.sh
pi05_semantic.yaml
merge_lerobot_v21_arx_bimanual.py
fix_augmented_videos.py
```

示例、测试、文档、环境：

```text
examples/
tests/
docs/
docker/
media/
scripts/
```

可以临时跳过：

```text
.github/
media/
docs/source/
docker/
tests/
examples/
```

但如果要提交代码或查 bug，`tests/` 不能长期跳过。

## 五、必须阅读和可以暂时跳过的文件

### 5.1 必须阅读

如果要理解训练主流程：

```text
pyproject.toml
src/lerobot/configs/train.py
src/lerobot/scripts/lerobot_train.py
src/lerobot/datasets/factory.py
src/lerobot/datasets/lerobot_dataset.py
src/lerobot/policies/factory.py
src/lerobot/policies/pretrained.py
src/lerobot/processor/pipeline.py
```

如果要理解 PI05：

```text
pi05_semantic.yaml
run_pi05_train_with_logs.sh
pi05_server.py
src/lerobot/policies/pi05/configuration_pi05.py
src/lerobot/policies/pi05/modeling_pi05.py
src/lerobot/policies/pi05/processor_pi05.py
usage_cn/02_local_usage_and_experiments.md
```

如果要理解数据集：

```text
src/lerobot/datasets/lerobot_dataset.py
src/lerobot/datasets/dataset_metadata.py
src/lerobot/datasets/video_utils.py
merge_lerobot_v21_arx_bimanual.py
fix_augmented_videos.py
innov_il/backup_and_crop_dataset.py
innov_il/prepare_left_arm_dataset.py
```

如果要理解真实机器人部署：

```text
src/lerobot/robots/
src/lerobot/cameras/
src/lerobot/motors/
src/lerobot/teleoperators/
innov_il/run_innov_pi05_local_inference.sh
innov_il/innov_pi05_local_inference.py
```

如果要理解 reward classifier / HIL-SERL：

```text
src/lerobot/rewards/classifier/
src/lerobot/rl/
hil_serl_arx/
```

### 5.2 可以暂时跳过

第一次读项目时可以先跳过：

```text
tests/
docs/source/
examples/
docker/
media/
.github/
src/lerobot/policies/里暂时不用的策略目录
src/lerobot/robots/里暂时不用的机器人目录
src/lerobot/envs/里暂时不用的仿真环境
```

## 六、每个核心模块是怎么串起来的

### 6.1 训练链路

训练从配置开始：

```text
yaml/json
  -> TrainPipelineConfig
  -> make_dataset
  -> make_policy
  -> make_pre_post_processors
  -> make_optimizer_and_scheduler
  -> training loop
  -> checkpoint/pretrained_model
```

对应代码：

```text
src/lerobot/scripts/lerobot_train.py
src/lerobot/configs/train.py
src/lerobot/datasets/factory.py
src/lerobot/policies/factory.py
src/lerobot/processor/pipeline.py
src/lerobot/optim/factory.py
```

### 6.2 数据进入模型的链路

```text
LeRobotDataset
  -> DataLoader batch
  -> preprocessor
  -> policy.forward / select_action
  -> loss 或 action
  -> postprocessor
```

preprocessor 常做：

- rename observation key。
- 图像 uint8 -> float。
- 图像 resize/crop/normalize。
- state/action 归一化。
- tokenizer 处理 task 文本。
- device 搬运。

postprocessor 常做：

- action 反归一化。
- action chunk 处理。
- 相对动作转绝对动作。
- 输出给机器人或 server client。

### 6.3 checkpoint 保存和加载

训练保存：

```text
pretrained_model/
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_postprocessor.json
  train_config.json
```

加载时：

```python
policy = PI05Policy.from_pretrained(policy_path, device="cuda")
```

processor 会根据 json 重建，保证训练和推理使用同一套预处理/后处理逻辑。

### 6.4 真实机器人推理链路

server/client 分离：

```text
client 采集图像和 qpos/state
  -> 发送到 pi05_server.py
  -> server 加载 PI05 checkpoint
  -> preprocessor
  -> policy predict action chunk
  -> postprocessor
  -> 可选冻结部分 action dims
  -> 返回 action
  -> client 控制机器人
```

本地一体化推理：

```text
innov_il/run_innov_pi05_local_inference.sh
  -> innov_pi05_local_inference.py
  -> 连接机械臂串口和 RealSense
  -> 加载 PI05 checkpoint
  -> 循环采集 observation
  -> 输出 action
  -> 平滑/限幅
  -> 下发到机械臂
  -> 记录 actions.jsonl
```

## 七、新手推荐读代码顺序

第一轮，只建立地图：

```text
usage_cn/README.md
usage_cn/01_from_zero_setup.md
usage_cn/04_project_position_and_structure.md
usage_cn/02_local_usage_and_experiments.md
```

第二轮，看训练如何跑：

```text
pi05_semantic.yaml
run_pi05_train_with_logs.sh
src/lerobot/scripts/lerobot_train.py
src/lerobot/configs/train.py
```

第三轮，看策略如何接入：

```text
src/lerobot/policies/factory.py
src/lerobot/policies/pretrained.py
src/lerobot/policies/pi05/configuration_pi05.py
src/lerobot/policies/pi05/modeling_pi05.py
src/lerobot/policies/pi05/processor_pi05.py
```

第四轮，看数据和 processor：

```text
src/lerobot/datasets/lerobot_dataset.py
src/lerobot/datasets/dataset_metadata.py
src/lerobot/processor/pipeline.py
src/lerobot/processor/normalize_processor.py
```

第五轮，看真实机器人部署：

```text
pi05_server.py
innov_il/run_innov_pi05_local_inference.sh
innov_il/innov_pi05_local_inference.py
src/lerobot/robots/
src/lerobot/cameras/
src/lerobot/motors/
```

