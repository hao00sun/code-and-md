# 三个 LSTM 网络对比实验记录

生成时间: 2026-08-10T10:10:06+08:00

本文件夹只包含 TensorBoard 标量指标、参数 YAML、checkpoint 索引和总结，不包含 `.pth/.pt/.safetensors/.onnx/.ckpt/.bin` 模型权重。

## 实验对象

- `LSTM base`: `env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_base`
- `LSTM symmetric`: `env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_symmetric`
- `LSTM symmetric large`: `env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_symmetric_large`

## 关键结果对比

| 网络 | 最终 reward | 最高 reward | 最高 reward step | 最终成功区比例 | 最高成功区比例 | 最终 FPS | checkpoint 最高文件名 reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM base | 4256.98 | 4455.54 | 649199616 | 0.81604 | 0.832031 | 44562.5 | 4301.23 |
| LSTM symmetric | 12.5278 | 255.846 | 627965952 | 0.00012207 | 0.00158691 | 34745.7 | 236.056 |
| LSTM symmetric large | 4021.45 | 4351.49 | 507248640 | 0.812012 | 0.828735 | 40038.1 | 4150.39 |

## 主要指标明细

### LSTM base

- Event source: `/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_base/summaries/events.out.tfevents.1786000447.wu-Z790-UD`
- Exported scalar tags: 44
- Checkpoint files indexed, not uploaded: 12

| Metric | Last step | Last value | Max value | Max step | Min value | Min step |
|---|---:|---:|---:|---:|---:|---:|
| `rewards/step` | 655097856 | 4256.98 | 4455.54 | 649199616 | 9.28396 | 393216 |
| `shaped_rewards/step` | 655097856 | 42.5698 | 44.5554 | 649199616 | 0.0928396 | 393216 |
| `in_success_region/frame` | 655097856 | 0.81604 | 0.832031 | 645660672 | 0 | 0 |
| `object_to_goal_reward/frame` | 655228928 | 3.25005 | 3.27432 | 650641408 | 0.00193097 | 34078720 |
| `lift_reward/frame` | 655097856 | 3.827 | 3.8862 | 645660672 | 0.0959469 | 4587520 |
| `hand_to_object_reward/frame` | 655228928 | 0.329723 | 0.356826 | 288227328 | 0.00146525 | 9437184 |
| `finger_curl_reg/frame` | 655097856 | -0.0775832 | -0.00847257 | 9699328 | -0.0972672 | 312737792 |
| `episode_lengths/step` | 655097856 | 586.383 | 599 | 4849664 | 56.6667 | 393216 |
| `performance/step_fps` | 655228928 | 44562.5 | 76667.1 | 9568256 | 35891.5 | 49152000 |
| `performance/step_inference_fps` | 655228928 | 43561.1 | 74068.5 | 9568256 | 35246.9 | 49152000 |
| `performance/step_inference_rl_update_fps` | 655228928 | 25767.9 | 36954.2 | 9568256 | 22438.1 | 85852160 |
| `losses/a_loss` | 655228928 | -0.000862404 | 0.0564652 | 0 | -0.0226761 | 917504 |
| `losses/c_loss` | 655228928 | 0.0110105 | 6.7152 | 19529728 | 0.000641169 | 9043968 |
| `losses/entropy` | 655228928 | 20.2701 | 20.2701 | 655228928 | 13.6294 | 26214400 |
| `info/kl` | 655228928 | -0.000518371 | 0.247 | 0 | -0.000518803 | 655097856 |

### LSTM symmetric

- Event source: `/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_symmetric/summaries/events.out.tfevents.1786047169.wu-Z790-UD`
- Exported scalar tags: 44
- Checkpoint files indexed, not uploaded: 12

| Metric | Last step | Last value | Max value | Max step | Min value | Min step |
|---|---:|---:|---:|---:|---:|---:|
| `rewards/step` | 655097856 | 12.5278 | 255.846 | 627965952 | 4.70491 | 650248192 |
| `shaped_rewards/step` | 655097856 | 0.125278 | 2.55846 | 627965952 | 0.0470491 | 650248192 |
| `in_success_region/frame` | 655097856 | 0.00012207 | 0.00158691 | 599523328 | 0 | 0 |
| `object_to_goal_reward/frame` | 655097856 | 0.00613902 | 0.0746292 | 9699328 | 0.00160661 | 43646976 |
| `lift_reward/frame` | 655097856 | 0.1116 | 0.561993 | 9699328 | 0.0959787 | 1703936 |
| `hand_to_object_reward/frame` | 655228928 | 0.41681 | 0.439035 | 601227264 | 0.00614196 | 4849664 |
| `finger_curl_reg/frame` | 655097856 | -0.07837 | -0.00878507 | 9699328 | -0.133583 | 162136064 |
| `episode_lengths/step` | 655097856 | 34.9901 | 599 | 4849664 | 16.1635 | 650248192 |
| `performance/step_fps` | 655228928 | 34745.7 | 78774.3 | 3801088 | 26491.3 | 530710528 |
| `performance/step_inference_fps` | 655228928 | 34148 | 76067.1 | 3801088 | 26101.6 | 530710528 |
| `performance/step_inference_rl_update_fps` | 655228928 | 26152.4 | 48696.2 | 3801088 | 20609.7 | 530710528 |
| `losses/a_loss` | 655228928 | -0.00097771 | 0.155774 | 4849664 | -0.0214161 | 655360 |
| `losses/c_loss` | 655228928 | 0.0174619 | 9.98431 | 29360128 | 0.0018996 | 3276800 |
| `losses/entropy` | 655228928 | 11.0178 | 15.6356 | 262144 | 10.9278 | 601882624 |
| `info/kl` | 655228928 | -0.00334076 | 0.538812 | 4849664 | -0.00349203 | 654311424 |

### LSTM symmetric large

- Event source: `/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_symmetric_large/summaries/events.out.tfevents.1786070509.wu-Z790-UD`
- Exported scalar tags: 44
- Checkpoint files indexed, not uploaded: 9

| Metric | Last step | Last value | Max value | Max step | Min value | Min step |
|---|---:|---:|---:|---:|---:|---:|
| `rewards/step` | 537133056 | 4021.45 | 4351.49 | 507248640 | 9.04587 | 393216 |
| `shaped_rewards/step` | 537133056 | 40.2145 | 43.5149 | 507248640 | 0.0904587 | 393216 |
| `in_success_region/frame` | 537133056 | 0.812012 | 0.828735 | 506593280 | 0 | 0 |
| `object_to_goal_reward/frame` | 537264128 | 2.87434 | 3.01913 | 459276288 | 0.0023374 | 26476544 |
| `lift_reward/frame` | 537264128 | 3.77356 | 3.85223 | 506593280 | 0.0959678 | 6946816 |
| `hand_to_object_reward/frame` | 537264128 | 0.353803 | 0.353924 | 536608768 | 0.000542525 | 2752512 |
| `finger_curl_reg/frame` | 537264128 | -0.0912175 | -0.00977144 | 9699328 | -0.129071 | 46399488 |
| `episode_lengths/step` | 537133056 | 578.929 | 599 | 4849664 | 55.8571 | 393216 |
| `performance/step_fps` | 537264128 | 40038.1 | 77299.7 | 2359296 | 20877.4 | 535560192 |
| `performance/step_inference_fps` | 537264128 | 38317.1 | 71926.5 | 2359296 | 20138.7 | 535560192 |
| `performance/step_inference_rl_update_fps` | 537264128 | 19175.8 | 29592.9 | 2359296 | 10465.7 | 535560192 |
| `losses/a_loss` | 537264128 | -0.0141225 | 0.302014 | 0 | -0.0256739 | 290848768 |
| `losses/c_loss` | 537264128 | 0.00852156 | 2.15589 | 44040192 | 0.00120053 | 6684672 |
| `losses/entropy` | 537264128 | 21.272 | 21.272 | 537264128 | 14.9555 | 23068672 |
| `info/kl` | 537264128 | 0.0143041 | 1.23426 | 0 | 0.00641311 | 510787584 |

## 目录说明

- `metrics/<run>/*.csv`: 每个 TensorBoard scalar tag 导出的完整 CSV。
- `params/<run>/agent.yaml` 和 `env.yaml`: 训练参数快照。
- `summary.json`: 所有 scalar 的 last/max/min/点数统计。
- `checkpoint_index/checkpoints.json`: checkpoint 文件名、原始路径、大小和文件名中的 reward，模型文件本身未上传。
