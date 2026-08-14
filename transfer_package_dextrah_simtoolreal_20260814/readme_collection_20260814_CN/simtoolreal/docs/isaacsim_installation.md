# SimToolReal Isaac Sim 安装说明中文版

本文档说明如何为 SimToolReal 配置推荐的 Isaac Sim / Isaac Lab 训练环境。

## 推荐环境

```text
Python = 3.11
Isaac Sim / Isaac Lab = pip 安装版本
训练入口 = isaacsimenvs/train.py
虚拟环境 = env_simtoolreal 或 .venv_isaacsim
```

## 基本步骤

```text
1. 创建 Python 3.11 虚拟环境
2. 安装 Isaac Sim / Isaac Lab 相关依赖
3. 安装本仓库及 vendored rl_games
4. 下载必要资产和预训练策略
5. 运行 smoke test 或小规模训练确认环境可用
```

## 推荐训练命令

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal

/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  env.scene.num_envs=12288 \
  agent.params.config.expl_coef_block_size=2048
```

## 注意

Isaac Sim 必须先由 `AppLauncher` 启动后才能导入 Isaac Lab 相关模块。训练脚本已经按这个顺序组织。
