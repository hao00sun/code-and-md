# DEXTRAH 新手环境配置指南

本文面向第一次接触该项目的用户，目标是从零确认机器环境、进入本地目录、安装/检查依赖，并跑通最小训练或验证命令。

## 项目信息

```text
项目主页: https://sites.google.com/view/dextrah-g
代码库: https://github.com/NVlabs/DEXTRAH.git
当前本机仓库: /data/SUN_ht/Isaac_Gym/DEXTRAH
```

DEXTRAH 主要用于 Kuka 机械臂 + Allegro Hand 的灵巧操作策略训练、teacher policy 验证，以及基于相机观测的 student policy 蒸馏。

## 0. 前提条件

推荐机器条件：

```text
系统: Ubuntu Linux
显卡: NVIDIA GPU，建议 RTX 4090 或同级别
显存: 建议 24GB 以上
内存: 建议 64GB 以上
磁盘: 建议预留 150GB 以上
```

当前本机路径：

```text
项目根目录: /data/SUN_ht/Isaac_Gym/DEXTRAH
Python 环境: /data/SUN_ht/Isaac_Gym/env_dextrah
大文件输出: /mnt/bigdata/SUN_ht/runs/dextrah
```

推荐版本组合：

```text
操作系统: Ubuntu Linux
Python: 3.11
Isaac Sim: 5.0.0.0
Isaac Lab: v2.2.1
PyTorch: 使用当前 env_dextrah 中已验证的 CUDA 版本
NVIDIA Driver: 建议 R580 系列；本机验证过 580.178.04
CUDA Graph: 日常训练建议 env.use_cuda_graph=False
推荐并行环境数: 8192 用于正式 teacher 训练；16384 更偏压力测试
```

注意：DEXTRAH 不要和 IsaacLab 3.x 的 Shadow Hand Reorient 环境混用。本机将 DEXTRAH 放在独立环境 `/data/SUN_ht/Isaac_Gym/env_dextrah` 中。

## 0.1 从零拉取代码库

如果当前机器还没有 DEXTRAH 仓库，可以按下面方式拉取。已有 `/data/SUN_ht/Isaac_Gym/DEXTRAH` 时不要重复 clone 到同一路径，避免覆盖已有修改。

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

解压后，DEXTRAH 源码位于：

```text
transfer_package_dextrah_simtoolreal_20260814/DEXTRAH
```

该转移包包含 DEXTRAH、SimToolReal 两个源码副本和相关 README / md 文档，但不包含 `.git` 历史、Python 环境、训练输出和模型权重。转移包说明见：

```text
/data/SUN_ht/Isaac_Gym/transfer_package_dextrah_simtoolreal_20260814/README_TRANSFER_ZH.md
```

如果希望保留完整 Git 历史，或者想从官方仓库重新开始，应使用下面的 `git clone` 方式。

```bash
cd /data/SUN_ht/Isaac_Gym

git clone https://github.com/NVlabs/DEXTRAH.git DEXTRAH
```

如果还需要 FABRICS，请将其放在同一工作区下。本机当前路径是 `/data/SUN_ht/Isaac_Gym/FABRICS`。

如果仓库已经存在，只需要更新代码：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH

git pull
```

## 0.2 从零创建 Python 环境

本机已经配置好：

```text
/data/SUN_ht/Isaac_Gym/env_dextrah
```

如果要在新机器上重建，建议创建独立 Python 3.11 环境：

```bash
cd /data/SUN_ht/Isaac_Gym

python3.11 -m venv env_dextrah
source /data/SUN_ht/Isaac_Gym/env_dextrah/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

安装 Isaac Lab 2.2.1 / Isaac Sim 5.0.0.0 后，再安装 DEXTRAH 和 FABRICS 的本地包：

```bash
source /data/SUN_ht/Isaac_Gym/env_dextrah/bin/activate

cd /data/SUN_ht/Isaac_Gym/FABRICS
python -m pip install -e .

cd /data/SUN_ht/Isaac_Gym/DEXTRAH
python -m pip install -e .
```

如果遇到依赖冲突，以当前已验证环境为准；不要把 DEXTRAH 依赖安装进 IsaacLab 3.x 的环境中。

## 1. 检查显卡和驱动

```bash
nvidia-smi
```

能看到 NVIDIA GPU、显存和驱动版本即可继续。

## 2. 进入项目目录

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH
```

检查目录是否存在：

```bash
ls
```

应能看到 `dextrah_lab` 等目录。

## 3. 检查 Python 环境

```bash
/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python --version
```

当前 DEXTRAH 环境使用 Python 3.11。

检查 PyTorch：

```bash
/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果最后输出 `True`，说明 PyTorch 可以使用 CUDA。

## 4. 设置环境变量

每次训练或验证前建议执行：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1
```

如果不想使用 W&B：

```bash
export WANDB_MODE=disabled
```

## 5. 检查 Isaac Sim / Isaac Lab 是否能启动

运行一个轻量命令，例如查看训练脚本帮助：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python train.py --help
```

如果没有 Python import 错误，说明基本依赖可用。

## 6. Teacher policy 最小验证

已有 teacher checkpoint 示例：

```text
/mnt/bigdata/SUN_ht/runs/dextrah/env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_base/nn/last_dextrah_lstm_ep_5000_rew_4301.2305.pth
```

无窗口验证：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

CKPT=/mnt/bigdata/SUN_ht/runs/dextrah/env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_base/nn/last_dextrah_lstm_ep_5000_rew_4301.2305.pth

export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1

/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python play.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 64 \
  --device cuda:0 \
  --checkpoint "$CKPT" \
  --objects_dir visdex_objects \
  --max_pose_angle 45.0 \
  --disable_adr \
  --print_every 2000 \
  --max_steps 10000
```

如果终端出现 `success_rate`、`mean_success_last_100` 或 reward 信息，说明 teacher 验证跑通。

## 7. Student policy 可视化验证

已有 student checkpoint 示例：

```text
/mnt/bigdata/SUN_ht/runs/dextrah/student_stereo_full_noADR_successHold500_beta0_partial100k_env256_12-16-54-42/nn/dextrah_student_100000_iters.pth
```

带窗口可视化：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/distillation

STUDENT=/mnt/bigdata/SUN_ht/runs/dextrah/student_stereo_full_noADR_successHold500_beta0_partial100k_env256_12-16-54-42/nn/dextrah_student_100000_iters.pth

export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1

/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python run_distillation.py \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 1 \
  --student "$STUDENT" \
  --play_policy True \
  --max_iterations 10000 \
  env.distillation=True \
  --enable_cameras \
  env.simulate_stereo=True \
  env.img_aug_type="rgb" \
  env.objects_dir=visdex_objects \
  env.use_object_subset=False \
  env.max_pose_angle=45.0 \
  env.enable_adr=False \
  env.num_adr_increments=0 \
  env.starting_adr_increments=0 \
  env.use_cuda_graph=False
```

注意：这条命令没有 `--headless`，会打开 Isaac Sim 窗口。

## 8. 启动 TensorBoard

查看 DEXTRAH 训练结果：

```bash
/data/SUN_ht/Isaac_Gym/env_dextrah/bin/tensorboard \
  --logdir /mnt/bigdata/SUN_ht/runs/dextrah \
  --host 0.0.0.0 \
  --port 6007 \
  --reload_interval 5
```

浏览器打开：

```text
http://localhost:6007
```

如果在另一台电脑访问，需要使用训练机 IP：

```text
http://训练机IP:6007
```

## 9. 常见问题

### TensorBoard 空白

确认 `--logdir` 指向的是包含 `events.out.tfevents*` 的目录或其父目录。

查找 event 文件：

```bash
find /mnt/bigdata/SUN_ht/runs/dextrah -type f -name 'events.out.tfevents*' | tail
```

### W&B 要求登录

输入 `3` 跳过，或者训练前设置：

```bash
export WANDB_MODE=disabled
```

### 显存没有释放

检查残留进程：

```bash
pgrep -af "run_distillation|torch.distributed.run|play.py|train.py"
```

停止对应进程前先确认 PID，避免误杀其他训练。

### `env.distillation=True` 是什么

它会启用 student/camera observation 路径，并且成功后会按 `success_timeout` reset。该环境下的成功率不能直接与普通 `play.py` 成功率比较。
