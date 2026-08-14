# Shadow Hand Reorient 实验记录

本文档记录当前已经保留的 Shadow Hand 立方体重定向实验、关键指标和后续消融建议。

## 当前保留的 run

当前日志目录：

```text
/mnt/bigdata/SUN_ht/runs/rsl_rl
```

已保留两个有效实验：

```text
shadow_hand_official/2026-07-29_16-17-16
ablate_rot_reward_05/2026-07-30_17-51-00
```

之前两个可视化 smoke test 已删除：

```text
shadow_hand_visual_train/2026-07-29_15-40-07_viz_128env
shadow_hand_visual_train/2026-07-29_15-48-39_viz_4096env
```

## 实验结果

| 实验 | 主要设置 | 最后 step | success_rate | orientation_error_mean | orientation_error_median | orientation_error_p90 | Mean reward | 最新模型 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 官方参数 | `rot_reward_scale=1.0` | 2349 | 0.8667 | 0.2592 | 0.0451 | 0.3792 | 4146.15 | `model_2250.pt` |
| 消融实验 | `rot_reward_scale=0.5` | 2890 | 1.0000 | 0.0309 | 0.0261 | 0.0505 | 3949.32 | `model_2750.pt` |

初步结论：

```text
当前环境下，rot_reward_scale=0.5 的表现优于官方默认 1.0。
它的成功率更高，姿态误差更低，而且当前保留日志中没有观察到官方 run 后期那种 NaN 崩溃。
```

但这还不是最终结论，因为目前每个设置只有一次 run。正式消融应增加多个随机种子。

## 指标含义

`Metrics/success_rate`：成功率，越高越好。`1.0` 表示 100% 成功。

`Diagnostics/episode_min_orientation_error_mean`：每个 episode 中最小姿态误差的平均值，越低越好。

`Diagnostics/episode_min_orientation_error_median`：每个 episode 中最小姿态误差的中位数，越低越好。它比 mean 更不容易受极端失败样本影响。

`Diagnostics/episode_min_orientation_error_p90`：每个 episode 中最小姿态误差的 90 分位数，越低越好。它反映较差样本的表现。

`Mean episode consecutive_successes`：连续成功次数，越高越好。

`Mean action std`：策略动作标准差。过大时可能说明动作越来越激进，容易导致接触不稳定、`nefc overflow` 或 NaN。

## 推荐下一轮消融

先围绕 `rot_reward_scale` 做单变量实验：

```text
0.25, 0.5, 0.75, 1.0, 1.25, 1.5
```

建议每个值至少跑 2 到 3 个 seed。筛选阶段可先用：

```bash
cd /data/SUN_ht/Isaac_Gym/IsaacLab
conda activate /data/SUN_ht/Isaac_Gym/env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --max_iterations 1000 \
  --experiment_name ablate_rot_reward_075 \
  env.rot_reward_scale=0.75
```

`rot_reward_scale=1.5` 可以测试，但需要重点观察：

```text
Mean action std
Metrics/success_rate
Diagnostics/episode_min_orientation_error_p90
nefc overflow
NaN
```

## TensorBoard 对比

```bash
tensorboard \
  --logdir_spec official:/mnt/bigdata/SUN_ht/runs/rsl_rl/shadow_hand_official,rot05:/mnt/bigdata/SUN_ht/runs/rsl_rl/ablate_rot_reward_05 \
  --host 127.0.0.1 \
  --port 6006
```

浏览器打开：

```text
http://127.0.0.1:6006
```

## 推理播放

官方参数模型：

```bash
cd /data/SUN_ht/Isaac_Gym/IsaacLab
conda activate /data/SUN_ht/Isaac_Gym/env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --num_envs 32 \
  --checkpoint /mnt/bigdata/SUN_ht/runs/rsl_rl/shadow_hand_official/2026-07-29_16-17-16/model_2250.pt \
  --viz kit
```

`rot_reward_scale=0.5` 消融模型：

```bash
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --num_envs 32 \
  --checkpoint /mnt/bigdata/SUN_ht/runs/rsl_rl/ablate_rot_reward_05/2026-07-30_17-51-00/model_2750.pt \
  --viz kit
```
