# SimToolReal 新手环境配置指南

本文面向第一次接触 SimToolReal 的用户，目标是从本地环境检查开始，跑通训练、指标查看和最小验证。

## 项目信息

```text
项目主页: https://simtoolreal.github.io/
代码库: https://github.com/tylerlum/simtoolreal.git
当前本机仓库: /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal
```

SimToolReal 主要面向工具操作任务，包含仿真训练、真实物体数据处理、基线、部署和 Sim-to-Real 相关流程。当前推荐使用 Isaac Sim / Isaac Lab 版本的 `isaacsimenvs` 管线。

## 0. 前提条件

推荐机器条件：

```text
系统: Ubuntu Linux
显卡: NVIDIA GPU，建议 RTX 4090 或同级别
显存: 官方规模建议 24GB 以上
内存: 建议 64GB 以上
磁盘: 建议预留 150GB 以上
```

当前本机路径：

```text
工作区: /data/SUN_ht/Isaac_Gym/SimToolReal_workspace
仓库: /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal
Python 环境: /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal
训练输出: /mnt/bigdata/SUN_ht/runs/simtoolreal
```

推荐版本组合：

```text
操作系统: Ubuntu Linux
Python: 3.11
Isaac Sim: 5.x
Isaac Lab: 2.3.2.post1
PyTorch: 2.7.x + cu126
CUDA: 12+
NVIDIA Driver: >= 525.60，建议使用较新的稳定驱动
包管理工具: uv
推荐训练管线: isaacsimenvs，也就是 Isaac Sim / Isaac Lab 版本
旧版训练管线: isaacgymenvs，需要 Python 3.8，不建议和 Isaac Sim 环境混装
```

注意：SimToolReal 的 Isaac Sim 环境和旧 Isaac Gym 环境必须分开。不要把 Isaac Gym、Python 3.8 依赖装进当前 Python 3.11 的 `env_simtoolreal` 环境。

## 0.1 从零拉取代码库

如果当前机器还没有 SimToolReal 仓库，可以按下面方式拉取。已有 `/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal` 时不要重复 clone 到同一路径，避免覆盖已有修改。

如果是从当前机器转移源码和文档，也可以直接使用已整理好的转移包：

```text
/data/SUN_ht/Isaac_Gym/transfer_package_dextrah_simtoolreal_20260814
```

对应 zip 压缩包：

```text
/data/SUN_ht/Isaac_Gym/transfer_package_dextrah_simtoolreal_20260814.zip
```

把 zip 复制到新机器后，执行：

```bash
cd /目标路径

unzip transfer_package_dextrah_simtoolreal_20260814.zip
cd transfer_package_dextrah_simtoolreal_20260814
```

解压后，SimToolReal 源码位于：

```text
transfer_package_dextrah_simtoolreal_20260814/simtoolreal
```

该转移包包含 DEXTRAH、SimToolReal 两个源码副本和相关 README / md 文档，但不包含 `.git` 历史、Python 环境、训练输出和模型权重。转移包说明见：

```text
/data/SUN_ht/Isaac_Gym/transfer_package_dextrah_simtoolreal_20260814/README_TRANSFER_ZH.md
```

如果希望保留完整 Git 历史，或者想从官方仓库重新开始，应使用下面的 `git clone` 方式。

```bash
mkdir -p /data/SUN_ht/Isaac_Gym/SimToolReal_workspace
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace

git clone https://github.com/tylerlum/simtoolreal.git simtoolreal
```

如果仓库已经存在，只需要更新代码：

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal

git pull
```

## 0.2 从零创建 Isaac Sim / Isaac Lab 环境

本机已经配置好：

```text
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal
```

如果要在新机器上重建，推荐使用 `uv` 创建 Python 3.11 虚拟环境：

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal

uv venv /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal --python 3.11
```

安装 PyTorch CUDA 12.6 版本：

```bash
uv pip install \
  --python /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python \
  torch \
  --index-url https://download.pytorch.org/whl/cu126
```

安装仓库内置的 `rl_games` 和训练依赖：

```bash
uv pip install \
  --python /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python \
  -e ./rl_games/

uv pip install \
  --python /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro "imageio[ffmpeg]" wandb termcolor
```

安装 Isaac Lab + Isaac Sim。该步骤会下载较多内容，首次启动还会编译 RTX shader：

```bash
uv pip install \
  --python /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python \
  "isaaclab[isaacsim,all]==2.3.2.post1" \
  --extra-index-url https://pypi.nvidia.com
```

安装离线碰撞分解和 `typing_extensions` 修正：

```bash
uv pip install \
  --python /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python \
  coacd "typing_extensions>=4.13"
```

注册 SimToolReal 本地包。这里必须使用 `--no-deps`，避免根目录依赖把 Python 3.11 / Isaac Sim 环境降级或装入冲突包：

```bash
uv pip install \
  --python /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python \
  -e . \
  --no-deps
```

检查安装是否成功：

```bash
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python -c "import torch, isaaclab, isaacsimenvs; print(torch.__version__); print(torch.cuda.is_available()); print(isaaclab.__file__)"
```

## 1. 检查显卡和驱动

```bash
nvidia-smi
```

能看到 NVIDIA GPU、显存和驱动版本即可继续。

## 2. 进入项目目录

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal
```

检查目录：

```bash
ls
```

应能看到：

```text
isaacsimenvs
isaacgymenvs
dextoolbench
deployment
rl_games
README.md
```

## 3. 检查 Python 环境

```bash
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python --version
```

检查 PyTorch CUDA：

```bash
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果最后输出 `True`，说明 CUDA 可用。

## 4. 设置环境变量

每次训练前建议设置：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
```

`WANDB_MODE=disabled` 用于关闭 W&B 交互登录。

## 5. 检查训练脚本

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal

/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python isaacsimenvs/train.py --help
```

如果帮助信息正常出现，说明入口脚本可用。

## 6. 小规模 smoke test

第一次运行建议先用较小并行环境，确认能启动：

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal

export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  env.scene.num_envs=2048 \
  agent.params.config.expl_coef_block_size=512 \
  agent.params.config.max_epochs=20 \
  hydra.run.dir=/mnt/bigdata/SUN_ht/runs/simtoolreal/smoke_env2048_block512
```

如果终端输出 fps、epoch、frames，说明训练链路跑通。

## 7. 正式训练指令

稳妥规模：

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal

export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  env.scene.num_envs=12288 \
  agent.params.config.expl_coef_block_size=2048 \
  hydra.run.dir=/mnt/bigdata/SUN_ht/runs/simtoolreal/simtoolreal_sapg_env12288_block2048_$(date +%m-%d-%H-%M)
```

官方规模：

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal

export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  env.scene.num_envs=24576 \
  agent.params.config.expl_coef_block_size=4096 \
  hydra.run.dir=/mnt/bigdata/SUN_ht/runs/simtoolreal/official_sapg_env24576_block4096_$(date +%m-%d-%H-%M)
```

注意保持：

```text
num_envs / expl_coef_block_size = 6
```

## 8. 启动 TensorBoard

查看所有 SimToolReal run：

```bash
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/tensorboard \
  --logdir /mnt/bigdata/SUN_ht/runs/simtoolreal \
  --host 0.0.0.0 \
  --port 6009 \
  --reload_interval 5
```

浏览器打开：

```text
http://localhost:6009
```

如果页面空白，查找 event 文件：

```bash
find /mnt/bigdata/SUN_ht/runs/simtoolreal -type f -name 'events.out.tfevents*' -printf '%p\n' | tail
```

然后把 TensorBoard 的 `--logdir` 指向 event 文件所在目录或上级目录。

## 9. 主要指标怎么看

优先看：

```text
successes
episode_final/successes
episode_final/all_goals_hit
rewards/step
episode_final/done_max_successes
episode_final/done_fall
episode_final/done_hand_far
performance/step_fps
```

`successes` 不是简单成功率，而是 episode 中成功命中目标的次数。

`successes_per_block/block_*` 是 SAPG 不同探索系数组的表现，不代表不同模型。

## 10. 当前训练是否加了噪声

默认训练开启了多类随机化：

```text
reset 初始状态随机化
obs delay
action delay
object state delay + noise
joint velocity observation noise
物体被抬起后的随机 force / torque
```

其中主动扰动物体的配置包括：

```text
force_scale = 20.0
force_prob_range = [0.001, 0.1]
torque_scale = 2.0
torque_prob_range = [0.001, 0.1]
```

## 11. 常见问题

### TensorBoard 空白

通常是 `--logdir` 指错。用 `find` 找 event 文件后重新指定路径。

### W&B 弹出登录

训练前加：

```bash
export WANDB_MODE=disabled
```

### 显存不足

降低：

```text
env.scene.num_envs
agent.params.config.expl_coef_block_size
```

并保持二者比例为 6。

例如：

```text
12288 / 2048 = 6
6144 / 1024 = 6
2048 / 512 = 4
```

如果严格按官方 SAPG block 数，建议优先保持 6。
