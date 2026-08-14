# 灵巧手 Reorient/Repose 基线说明

IsaacLab 当前版本中，官方灵巧手立方体姿态重定向任务已经从旧名 `Repose-Cube` 改为 `Reorient-Cube`。当前实验主要使用 Shadow Hand 的 RSL-RL PPO 官方基线。

## 环境准备

```bash
cd /data/SUN_ht/Isaac_Gym/IsaacLab
conda activate /data/SUN_ht/Isaac_Gym/env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
```

## 当前已注册任务

```text
Isaac-Reorient-Cube-Shadow-Direct
Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct
Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct
Isaac-Reorient-Cube-Shadow-Camera-Direct
Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct
Isaac-Reorient-Cube-Allegro
Isaac-Reorient-Cube-Allegro-Direct
```

验证命令：

```bash
python -c "import gymnasium as gym; import isaaclab_tasks; print('\n'.join(sorted(k for k in gym.registry.keys() if 'Reorient-Cube' in k)))"
```

## 旧名到新名

```text
Isaac-Repose-Cube-Shadow-Direct-v0             -> Isaac-Reorient-Cube-Shadow-Direct
Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0   -> Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct
Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0 -> Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct
Isaac-Repose-Cube-Shadow-Vision-Direct-v0      -> Isaac-Reorient-Cube-Shadow-Camera-Direct
Isaac-Repose-Cube-Allegro-v0                   -> Isaac-Reorient-Cube-Allegro
Isaac-Repose-Cube-Allegro-Direct-v0            -> Isaac-Reorient-Cube-Allegro-Direct
```

## 官方 Shadow Hand PPO 基线

从零开始训练：

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --experiment_name shadow_hand_official
```

可视化训练：

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --num_envs 128 \
  --max_iterations 200 \
  --viz kit \
  --experiment_name shadow_hand_visual_train
```

推理播放：

```bash
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --num_envs 32 \
  --checkpoint /mnt/bigdata/SUN_ht/runs/rsl_rl/shadow_hand_official/2026-07-29_16-17-16/model_2250.pt \
  --viz kit
```

## 消融实验建议

优先从 `rot_reward_scale` 做单变量消融：

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --max_iterations 1000 \
  --experiment_name ablate_rot_reward_05 \
  env.rot_reward_scale=0.5
```

建议测试序列：

```text
0.25, 0.5, 0.75, 1.0, 1.25, 1.5
```

`rot_reward_scale=1.5` 可以使用，但它可能让策略更激进，增加动作标准差、接触不稳定、`nefc overflow` 或 NaN 的风险。

## 关键评价指标

```text
Metrics/success_rate
Diagnostics/episode_min_orientation_error_mean
Diagnostics/episode_min_orientation_error_median
Diagnostics/episode_min_orientation_error_p90
Mean episode consecutive_successes
Mean action std
```

判断训练效果时，优先看：

```text
success_rate 越高越好
orientation_error_mean 越低越好
orientation_error_median 越低越好
orientation_error_p90 越低越好
consecutive_successes 越高越好
action_std 不宜无限增大
```

不同 reward 配置之间不要只比较 `Mean reward`，因为总 reward 会随奖励权重变化而改变。

## 主要代码位置

任务源码：

```text
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/reorient/
```

Shadow Hand 配置：

```text
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/reorient/config/shadow_hand/
```

RSL-RL PPO 参数：

```text
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/reorient/config/shadow_hand/agents/rsl_rl_ppo_cfg.py
```

奖励函数：

```text
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/reorient/mdp/rewards.py
```
