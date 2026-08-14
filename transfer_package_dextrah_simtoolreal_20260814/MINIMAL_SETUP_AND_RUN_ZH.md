# DEXTRAH + SimToolReal 最简环境配置与启动流程

本文只保留最短可执行路径。更完整说明见：

```text
README_TRANSFER_ZH.md
readme_collection_20260814_CN/setup_guides/DEXTRAH_BEGINNER_SETUP_ZH.md
readme_collection_20260814_CN/setup_guides/SIMTOOLREAL_BEGINNER_SETUP_ZH.md
```

## 0. 包内容与注意事项

转移包包含：

```text
DEXTRAH/      DEXTRAH 源码
simtoolreal/  SimToolReal 源码
readme_collection_20260814_CN/  中文说明文档
```

转移包不包含：

```text
.git 历史
Python 虚拟环境
训练输出 logs/runs
模型权重 .pth/.pt/.ckpt
```

因此，新机器上需要重新配置 Python / Isaac 环境；已有 checkpoint 需要另外复制。

## 1. 解压并进入目录

```bash
cd /目标路径
unzip transfer_package_dextrah_simtoolreal_20260814.zip
cd transfer_package_dextrah_simtoolreal_20260814
```

建议设置一个工作区变量，后续命令更短：

```bash
export PKG_ROOT=$PWD
```

## 2. 基础检查

```bash
nvidia-smi
```

推荐环境：

```text
系统: Ubuntu Linux
GPU: NVIDIA，建议 24GB 显存以上
Python: 3.11
DEXTRAH: Isaac Sim 5.0.0.0 + Isaac Lab v2.2.1
SimToolReal: Isaac Sim 5.x + Isaac Lab 2.3.2.post1 + PyTorch cu126
```

DEXTRAH 和 SimToolReal 建议使用两个独立环境，不要混装。

通用环境变量：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
```

## 3. DEXTRAH 最简配置

创建环境：

```bash
cd "$PKG_ROOT"
python3.11 -m venv env_dextrah
source "$PKG_ROOT/env_dextrah/bin/activate"
python -m pip install --upgrade pip setuptools wheel
```

安装 Isaac Sim 5.0.0.0 / Isaac Lab v2.2.1 后，安装本地包：

```bash
cd "$PKG_ROOT/DEXTRAH"
python -m pip install -e .
```

如果同一工作区另有 `FABRICS`，也安装：

```bash
cd /path/to/FABRICS
python -m pip install -e .
```

检查入口：

```bash
cd "$PKG_ROOT/DEXTRAH/dextrah_lab/rl_games"
"$PKG_ROOT/env_dextrah/bin/python" train.py --help
```

Teacher policy 最小验证：

```bash
cd "$PKG_ROOT/DEXTRAH/dextrah_lab/rl_games"

CKPT=/path/to/teacher_checkpoint.pth

"$PKG_ROOT/env_dextrah/bin/python" play.py \
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

Student policy 最小验证：

```bash
cd "$PKG_ROOT/DEXTRAH/dextrah_lab/distillation"

STUDENT=/path/to/student_checkpoint.pth

"$PKG_ROOT/env_dextrah/bin/python" run_distillation.py \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 1 \
  --student "$STUDENT" \
  --play_policy True \
  --max_iterations 10000 \
  env.distillation=True
```

## 4. SimToolReal 最简配置

推荐使用 `uv`：

```bash
cd "$PKG_ROOT/simtoolreal"
uv venv "$PKG_ROOT/env_simtoolreal" --python 3.11
```

安装 PyTorch、rl_games、Isaac Lab / Isaac Sim 和本地包：

```bash
uv pip install \
  --python "$PKG_ROOT/env_simtoolreal/bin/python" \
  torch \
  --index-url https://download.pytorch.org/whl/cu126

uv pip install \
  --python "$PKG_ROOT/env_simtoolreal/bin/python" \
  -e ./rl_games/

uv pip install \
  --python "$PKG_ROOT/env_simtoolreal/bin/python" \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro "imageio[ffmpeg]" wandb termcolor

uv pip install \
  --python "$PKG_ROOT/env_simtoolreal/bin/python" \
  "isaaclab[isaacsim,all]==2.3.2.post1" \
  --extra-index-url https://pypi.nvidia.com

uv pip install \
  --python "$PKG_ROOT/env_simtoolreal/bin/python" \
  coacd "typing_extensions>=4.13"

uv pip install \
  --python "$PKG_ROOT/env_simtoolreal/bin/python" \
  -e . \
  --no-deps
```

检查入口：

```bash
cd "$PKG_ROOT/simtoolreal"
"$PKG_ROOT/env_simtoolreal/bin/python" isaacsimenvs/train.py --help
```

小规模启动测试：

```bash
cd "$PKG_ROOT/simtoolreal"

"$PKG_ROOT/env_simtoolreal/bin/python" isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  env.scene.num_envs=2048 \
  agent.params.config.expl_coef_block_size=512 \
  agent.params.config.max_epochs=20 \
  hydra.run.dir=/mnt/bigdata/SUN_ht/runs/simtoolreal/smoke_env2048_block512
```

正式训练可把规模调大，例如：

```bash
"$PKG_ROOT/env_simtoolreal/bin/python" isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  env.scene.num_envs=12288 \
  agent.params.config.expl_coef_block_size=2048 \
  hydra.run.dir=/mnt/bigdata/SUN_ht/runs/simtoolreal/train_env12288_block2048
```

## 5. TensorBoard

DEXTRAH：

```bash
"$PKG_ROOT/env_dextrah/bin/tensorboard" \
  --logdir /mnt/bigdata/SUN_ht/runs/dextrah \
  --host 0.0.0.0 \
  --port 6008
```

SimToolReal：

```bash
"$PKG_ROOT/env_simtoolreal/bin/tensorboard" \
  --logdir /mnt/bigdata/SUN_ht/runs/simtoolreal \
  --host 0.0.0.0 \
  --port 6009
```

浏览器打开：

```text
http://localhost:6008
http://localhost:6009
```

## 6. 常见失败点

```text
CUDA 不可用: 先检查 nvidia-smi 和 torch.cuda.is_available()
Isaac 首次启动很慢: 正常，可能在编译 shader
显存不足: 降低 num_envs
W&B 要登录: 设置 WANDB_MODE=disabled
找不到 checkpoint: 该转移包不含权重，需要单独复制 .pth/.pt/.ckpt
TensorBoard 空白: --logdir 指到含 events.out.tfevents* 的目录
```
