# SimToolReal README 中文版

SimToolReal 是一个面向零样本灵巧工具操作的对象中心策略项目，支持在仿真中训练，并部署到真实机器人或 sim-to-sim 环境。

## 项目内容

```text
assets/              机器人、桌子、工具和对象资产
baselines/           轨迹优化、固定抓取等 baseline
deployment/          真实部署和仿真部署节点
dextoolbench/        DexToolBench benchmark 数据和评估脚本
docs/                安装、数据、部署和 benchmark 文档
isaacsimenvs/        推荐使用的 Isaac Sim / Isaac Lab 环境
isaacgymenvs/        legacy Isaac Gym 环境
pretrained_policy/   预训练策略
recorded_data/       记录数据工具
rl_games/            vendored 强化学习算法
```

## 推荐训练入口

官方推荐 Isaac Sim 版本：

```bash
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  env.scene.num_envs=24576 \
  agent.params.config.expl_coef_block_size=4096
```

本地稳妥版使用：

```bash
env.scene.num_envs=12288
agent.params.config.expl_coef_block_size=2048
```

需要保持：

```text
num_envs / expl_coef_block_size = 6
```

## 评估入口

DexToolBench 交互式评估：

```bash
.venv_isaacsim/bin/python dextoolbench/eval_interactive_isaacsim.py \
  --config-path pretrained_policy/config.yaml \
  --checkpoint-path pretrained_policy/model.pth
```

## 当前本地重点

当前本地训练使用 SAPG，任务为 `Isaacsimenvs-SimToolReal-Direct-v0`，对象为 `handle_head_primitives`，工具类别包括 hammer、screwdriver、marker、spatula、eraser、brush。
