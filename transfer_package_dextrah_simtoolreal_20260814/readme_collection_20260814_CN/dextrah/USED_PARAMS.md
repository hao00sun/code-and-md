# DEXTRAH 已使用参数中文整理

本文档记录本地 DEXTRAH 实验中反复使用过的 teacher、student、蒸馏和验证参数。

## Teacher policy

主要 teacher checkpoint：

```text
/mnt/bigdata/SUN_ht/runs/dextrah/env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_base/nn/last_dextrah_lstm_ep_5000_rew_4301.2305.pth
```

训练设置：

```text
num_envs = 8192
horizon_length = 16
max_iterations = 5000
save_frequency = 500
minibatch_size = 32768
use_cuda_graph = False
max_pose_angle = 45.0
objects_dir = visdex_objects
use_object_subset = False
enable_adr = False
network = LSTM base actor_critic
```

当前新 teacher ADR 训练设置：

```text
num_envs = 16384
max_iterations = 200000
save_frequency = 5000
enable_adr = True
num_adr_increments = 50
experiment = env16384_noCudaGraph_pose45_mb32768_ADR50_epoch200000_save5000_lstm_base
```

## Student distillation

主要 student run：

```text
student_stereo_full_noADR_successHold500_beta0_partial100k_env256_12-16-54-42
```

关键设置：

```text
num_envs = 256
env.distillation = True
enable_cameras = True
simulate_stereo = True
img_aug_type = rgb
aux_coeff = 10
objects_dir = visdex_objects
use_object_subset = False
max_pose_angle = 45.0
enable_adr = False
use_cuda_graph = False
beta_schedule = success_hold
beta_success_target = 0.30
beta_hold_iters = 500
beta_step = 0.05
```

## Beta 含义

`beta` 是 teacher action 被用于推进环境的概率：

```text
beta = 1.0 -> 100% teacher action
beta = 0.5 -> 约 50% env 用 teacher action，50% env 用 student action
beta = 0.0 -> 100% student action
```

在 `--play_policy True` 且不加 `--play_teacher` 时，student 播放强制 `beta = 0`。

## 统计口径结论

后来加入了：

```text
mean_success_all
mean_success_last_100
```

用于和 `play.py` 口径对齐。结论是：

```text
普通 play.py teacher: mean_success_all 约 0.84，last100 可到约 0.98
distillation beta=1 teacher: mean_success_all 约 0.55
noADR student 最终: mean_success_all 约 0.52
```

因此 student 在同一 distillation 口径下已经接近 teacher rollout 表现；低成功率很大程度来自成功后 2 秒 reset 与环境口径差异。
