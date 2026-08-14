# DEXTRAH 项目说明

## 它究竟解决什么问题

DEXTRAH 解决的是机器人灵巧手在仿真中学习复杂物体操作策略，以及进一步把高维全状态 teacher policy 蒸馏成视觉 student policy 的问题。

在本地实验中，它主要用于研究：

```text
teacher policy 能否在 Isaac Sim / Isaac Lab 环境中完成 Kuka + Allegro 灵巧操作任务；
student policy 能否通过相机输入和蒸馏学习接近 teacher 的行为；
不同统计口径下 teacher 和 student 的成功率如何公平比较。
```

它关注的不只是训练一个 RL 模型，而是 teacher 训练、student 蒸馏、视觉输入、指标对齐和仿真验证之间的完整实验链路。

## 代码库目标

DEXTRAH 的目标是提供一个用于灵巧操作研究的仿真训练和蒸馏框架。

它覆盖：

```text
训练:
  使用 rl_games / PPO 训练 teacher policy。

推理:
  使用 play.py 加载 teacher checkpoint 进行 rollout 和指标验证。

蒸馏:
  使用 run_distillation.py 将 teacher policy 蒸馏到 student policy。

仿真:
  提供 Dextrah-Kuka-Allegro 任务环境、物体集、reward、reset、ADR 和 camera 路径。

指标分析:
  记录 reward、success、loss、beta、mean_success_all、mean_success_last_100 等指标。

部署:
  当前代码库更偏仿真训练和验证，不是完整真实机器人部署框架。
```

因此，DEXTRAH 不是完整 sim-to-real 闭环部署系统。它更准确的定位是：

```text
用于仿真中训练 teacher，并蒸馏视觉 student 的灵巧操作研究代码库。
```

## 适用任务、机器人本体和数据类型

适用任务：

```text
Dextrah-Kuka-Allegro
灵巧抓取与物体姿态控制
teacher RL 训练
student 视觉蒸馏
teacher/student rollout 验证
ADR / noADR 对比
不同 student 网络结构对比
```

机器人本体：

```text
Kuka 机械臂
Allegro Hand / dexterous hand 任务配置
```

物体与环境：

```text
visdex_objects
完整物体集或 object subset
max_pose_angle 可配置
ADR 可开启或关闭
camera/stereo 路径可开启
```

主要数据类型：

```text
teacher privileged observation
student camera observation
RGB / stereo image
机器人关节状态
物体状态
目标姿态
动作分布 mus / sigmas
reward
success 指标
imitation loss
auxiliary loss
TensorBoard event
checkpoint
```

## 输入和输出

Teacher 训练输入：

```text
任务配置
物体集
reward 配置
ADR 配置
PPO/rl_games 超参数
num_envs
max_iterations
```

Teacher 训练输出：

```text
teacher checkpoint
TensorBoard 指标
reward 曲线
success 指标
训练配置快照
```

Student 蒸馏输入：

```text
teacher checkpoint
student 网络配置
camera/stereo 图像输入
teacher action distribution
beta 调度
auxiliary loss 配置
distillation 环境配置
```

Student 蒸馏输出：

```text
student checkpoint
imitation_loss
aux_loss_object_pos
total_loss
beta
in_success_region
mean_success_all
mean_success_last_100
reward
```

推理/验证输入：

```text
teacher 或 student checkpoint
任务环境配置
物体集配置
headless 或可视化设置
max_steps / max_iterations
```

推理/验证输出：

```text
rollout 成功率
reward
episode length
success region 占比
CSV 指标或 TensorBoard event
可视化窗口中的动作表现
```

## 与同类代码库相比的核心优势

1. **同时包含 teacher 训练和 student 蒸馏链路**

   很多代码库只提供 RL 训练或只提供视觉 policy。DEXTRAH 的实验重点是先训练高性能 teacher，再蒸馏为 student，适合研究 privileged-to-vision policy transfer。

2. **支持视觉 student**

   student 可以使用 camera 输入，包含 mono/stereo 配置和图像增强路径，便于研究视觉策略在灵巧操作中的表现。

3. **高并行仿真训练**

   teacher 训练和 student 蒸馏都支持大量并行环境，适合在单机高性能 GPU 上快速收集 rollout 数据。

4. **可配置 ADR 与 noADR 对比**

   环境中可以开启或关闭 ADR，并调节 reward weight、fabric damping、pose angle、物体子集等参数，便于做系统消融。

5. **指标对齐能力**

   本地已经加入 `mean_success_all` 和 `mean_success_last_100` 等指标，用于拆分统计口径差异和真实策略能力差异。这个对 teacher/student 公平比较非常关键。

6. **beta 调度可控**

   蒸馏中支持 teacher action 与 student action 混合推进，通过 `beta` 控制 teacher 占比，并支持按时间或成功率逐步降低 beta。

7. **适合研究蒸馏失败原因**

   由于可以分别跑普通 `play.py` teacher、distillation beta=1 teacher、student rollout，因此能分析 student 表现低是来自网络能力、teacher 上限、环境差异，还是统计口径差异。

## 本地实验形成的关键认识

在本地实验中，直接用普通 `play.py` teacher 成功率和 student 蒸馏成功率比较是不公平的。普通 `play.py` 中 teacher 的 `mean_success_last_100` 可以很高，但在 `env.distillation=True` 且成功后按 `success_timeout=2.0` reset 的环境里，即使 `beta=1` 纯 teacher rollout，`mean_success_all` 也约为 0.55。

因此 noADR student 最终约 0.52 的 `mean_success_all` 并不代表完全失败，而是接近同一 distillation 统计口径下 teacher 的表现。

## 一句话总结

DEXTRAH 是一个面向仿真灵巧操作和 teacher-to-student 视觉蒸馏的研究代码库，核心价值在于训练 privileged teacher、蒸馏 camera student，并通过可控环境和指标对齐分析策略表现。

## 二、项目结构：每个目录分别负责什么

下面按当前仓库的一级和二级目录解释。阅读时可以先抓住三条主线：

```text
任务环境: dextrah_lab/tasks/dextrah_kuka_allegro
teacher 训练/验证: dextrah_lab/rl_games
student 蒸馏: dextrah_lab/distillation
```

### 顶层目录

```text
DEXTRAH/
├── dextrah_lab/
├── pretrained_ckpts/
├── README.md
├── README_PROJECT_ZH.md
├── USED_PARAMS.md
├── setup.py
├── pyproject.toml
├── poetry.lock
└── run_network_ablation_3x.sh
```

- `dextrah_lab/`：代码主体，包含任务环境、训练入口、蒸馏网络、资产和部署脚本。
- `pretrained_ckpts/`：预训练 checkpoint 放置目录。属于模型资源，不是算法实现。转移包里已排除 `.pth/.pt/.ckpt/.onnx` 等模型权重。
- `README.md`：原始项目 README，适合了解官方项目背景。
- `README_PROJECT_ZH.md`：当前中文项目说明，适合快速理解本地实验使用方式。
- `USED_PARAMS.md`：本地训练/蒸馏用过的参数记录。
- `setup.py`、`pyproject.toml`、`poetry.lock`：Python 包安装和依赖声明，属于工程配置。
- `run_network_ablation_3x.sh`：本地三网络对比实验脚本，属于实验工具脚本。

### `dextrah_lab/`

```text
dextrah_lab/
├── assets/
├── deployment_scripts/
├── distillation/
├── rl_games/
└── tasks/
```

- `assets/`：机器人、桌面、物体、光照、纹理等仿真资产。它不是算法核心，但环境创建必须依赖这里的 USD/URDF/纹理资源。
- `deployment_scripts/`：部署和真实系统接口相关脚本，例如相机标定、状态机、策略推理脚本。当前本地实验主要做仿真训练和验证，可以暂时跳过。
- `distillation/`：student policy 蒸馏核心模块，包含 student 网络、图像编码器、数据增强、蒸馏训练入口和评估脚本。
- `rl_games/`：teacher policy 的训练和验证入口，以及 Isaac Lab 环境和 rl_games 的适配层。
- `tasks/`：Isaac Lab 任务注册和任务环境实现，是 DEXTRAH 仿真环境的核心。

### `dextrah_lab/tasks/dextrah_kuka_allegro/`

这是 DEXTRAH 的环境核心。

```text
dextrah_kuka_allegro/
├── dextrah_kuka_allegro_env.py
├── dextrah_kuka_allegro_env_cfg.py
├── dextrah_adr.py
├── dextrah_kuka_allegro_utils.py
├── dextrah_kuka_allegro_constants.py
├── gym_setup.py
└── __init__.py
```

- `dextrah_kuka_allegro_env.py`：最核心文件。实现 `DextrahKukaAllegroEnv(DirectRLEnv)`，负责场景创建、机器人/物体状态缓存、reset、动作应用、reward、success、camera/distillation observation、ADR 调用等。
- `dextrah_kuka_allegro_env_cfg.py`：环境配置定义。包括机器人、物体、相机、reward 权重、ADR、成功判定、distillation 开关等参数。
- `dextrah_adr.py`：ADR 逻辑，负责自动域随机化的参数递增、成功阈值判断和随机化范围管理。
- `dextrah_kuka_allegro_utils.py`：动作缩放、张量转换、绝对动作计算等工具函数。
- `dextrah_kuka_allegro_constants.py`：动作维度、手部 PCA 范围、姿态范围等常量。
- `gym_setup.py`：任务注册 / 环境接入相关逻辑，属于工程胶水层。
- `__init__.py`：包导入和任务注册入口。

实现方式上，`dextrah_kuka_allegro_env.py` 继承 Isaac Lab 的 `DirectRLEnv`，然后重写环境生命周期钩子：创建场景、reset 指定环境、接收 action、推进物理、计算 observation/reward/done。teacher 训练时返回 privileged state observation；`env.distillation=True` 时会开启 camera/student observation 路径。

### `dextrah_lab/rl_games/`

这是 teacher policy 的训练和验证入口。

```text
rl_games/
├── train.py
├── play.py
├── rl_games_utils.py
└── wandb_utils.py
```

- `train.py`：teacher 训练入口。启动 Isaac Sim / Isaac Lab app，创建 Dextrah 任务环境，把环境交给 rl_games PPO 训练。
- `play.py`：teacher 验证/推理入口。加载 checkpoint 后 rollout，打印 reward、success、episode length 等指标。本地已加入一些 success 指标保存/打印逻辑。
- `rl_games_utils.py`：Isaac Lab 环境和 rl_games 之间的 wrapper / observer / env_info 适配。
- `wandb_utils.py`：W&B 日志相关辅助逻辑。若不使用 W&B，可暂时跳过。

实现方式上，`train.py` 和 `play.py` 本身不是算法主体，而是入口脚本。真正的 PPO 算法来自 rl_games；这里主要负责把 DEXTRAH 环境注册成 rl_games 可以消费的 vectorized environment。

### `dextrah_lab/distillation/`

这是 student 蒸馏核心。

```text
distillation/
├── run_distillation.py
├── distillation.py
├── distillation_transformer.py
├── a2c_with_aux_cnn.py
├── a2c_with_aux_cnn_stereo.py
├── a2c_mono_resnet.py
├── a2c_mono_transformer.py
├── a2c_stereo_transformer.py
├── mono_encoder.py
├── stereo_encoder.py
├── rgb_augs.py
├── depth_augs.py
├── data_recorder.py
└── eval.py
```

- `run_distillation.py`：student 蒸馏主入口。加载 teacher，创建 distillation 环境，运行 teacher/student action 混合，计算 imitation loss、aux loss、total loss，并保存 student checkpoint。
- `distillation.py`：蒸馏训练主逻辑。包含 beta 混合、loss 计算、指标记录、保存 checkpoint 等核心流程。
- `distillation_transformer.py`：Transformer 版本蒸馏逻辑。
- `a2c_with_aux_cnn.py`、`a2c_with_aux_cnn_stereo.py`：CNN student 网络和辅助损失版本。
- `a2c_mono_resnet.py`、`a2c_mono_transformer.py`、`a2c_stereo_transformer.py`：不同视觉 student 网络结构。
- `mono_encoder.py`、`stereo_encoder.py`：单目/双目图像编码器。
- `rgb_augs.py`、`depth_augs.py`：图像增强模块。
- `data_recorder.py`：记录蒸馏数据或 rollout 数据的工具。
- `eval.py`：评估辅助脚本。

实现方式上，蒸馏并不是重新训练一个 RL agent，而是让 teacher 给出动作分布或动作目标，student 从相机/图像观测中学习匹配 teacher。`beta` 控制环境推进时 teacher action 与 student action 的占比；`aux_coeff` 控制辅助损失在 total loss 中的权重。

### `dextrah_lab/assets/`

```text
assets/
├── kuka_allegro/
├── scene_objects/
├── primitives/
├── dome_light_textures/
├── curated_table_textures/
├── batch_convert_urdf.py
└── urdf_creator.py
```

- `kuka_allegro/`：Kuka + Allegro 机器人 USD 和描述文件。
- `scene_objects/`、`primitives/`：训练场景、桌面、基础几何体和物体资产。
- `dome_light_textures/`、`curated_table_textures/`：视觉随机化和渲染相关纹理。
- `batch_convert_urdf.py`、`urdf_creator.py`：资产转换工具脚本。

这些文件一般不需要先读。只有在物体加载、USD 路径、相机渲染或视觉随机化出问题时才需要深入。

### `dextrah_lab/deployment_scripts/`

```text
deployment_scripts/
├── camera_calibration.py
├── camera_transform_publisher.py
├── image_subscriber.py
├── kuka_allegro_fabric.py
├── kuka_allegro_random_targets.py
├── kuka_allegro_state_machine.py
├── kuka_allegro_stereo_fgp.py
├── policy_inference_stereo.py
└── policy_inference_transformer.py
```

这是部署/真实系统接口方向的脚本集合。当前如果只做仿真 teacher 训练和 student 蒸馏，可以暂时跳过。等需要连接相机、真实机器人、policy inference 节点时再读。

## 哪些是核心算法、工程封装、示例或工具

核心算法 / 任务逻辑：

```text
dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env.py
dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env_cfg.py
dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_adr.py
dextrah_lab/distillation/distillation.py
dextrah_lab/distillation/run_distillation.py
dextrah_lab/distillation/a2c_*.py
dextrah_lab/distillation/*_encoder.py
```

工程封装 / 框架适配：

```text
dextrah_lab/rl_games/train.py
dextrah_lab/rl_games/play.py
dextrah_lab/rl_games/rl_games_utils.py
dextrah_lab/tasks/dextrah_kuka_allegro/gym_setup.py
setup.py
pyproject.toml
```

资产 / 数据资源：

```text
dextrah_lab/assets/
pretrained_ckpts/
```

示例、工具或可后读内容：

```text
dextrah_lab/deployment_scripts/
dextrah_lab/assets/batch_convert_urdf.py
dextrah_lab/assets/urdf_creator.py
run_network_ablation_3x.sh
wandb_utils.py
```

## 必读文件和可暂时跳过文件

第一次阅读建议顺序：

```text
1. README_PROJECT_ZH.md
2. README.md
3. dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env_cfg.py
4. dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env.py
5. dextrah_lab/rl_games/train.py
6. dextrah_lab/rl_games/play.py
7. dextrah_lab/distillation/run_distillation.py
8. dextrah_lab/distillation/distillation.py
9. dextrah_lab/distillation/a2c_with_aux_cnn_stereo.py 或当前使用的 student 网络文件
```

可以暂时跳过：

```text
.git/
__pycache__/
pretrained_ckpts/ 中的模型文件
dextrah_lab/assets/ 下的大量 USD、纹理、物体文件
dextrah_lab/deployment_scripts/，除非要做真实部署
wandb 相关文件，除非要接入 W&B
```

## 模块之间如何串起来

Teacher 训练链路：

```text
train.py
  -> 注册 Dextrah-Kuka-Allegro 任务
  -> 创建 DextrahKukaAllegroEnv
  -> rl_games_utils 适配 observation/action/state space
  -> rl_games PPO 更新 teacher policy
  -> 保存 checkpoint 和 TensorBoard 指标
```

Teacher 验证链路：

```text
play.py
  -> 加载 teacher checkpoint
  -> 创建同一任务环境
  -> 使用 policy rollout
  -> 统计 reward、success、episode length
```

Student 蒸馏链路：

```text
run_distillation.py
  -> 创建 env.distillation=True 的环境
  -> 加载 teacher checkpoint
  -> student 从 camera/mono/stereo observation 编码
  -> teacher 给出动作监督
  -> beta 控制 teacher/student action 混合推进
  -> imitation loss + aux_coeff * aux loss
  -> 保存 student checkpoint 和蒸馏指标
```

环境内部链路：

```text
dextrah_kuka_allegro_env_cfg.py 定义参数
  -> dextrah_kuka_allegro_env.py 创建场景和缓存
  -> action 被转换成 Kuka/Allegro 控制目标
  -> FABRICS 负责部分运动生成/约束相关计算
  -> reward 和 success 根据物体、手、目标状态计算
  -> ADR 根据 success 调整随机化范围
```
