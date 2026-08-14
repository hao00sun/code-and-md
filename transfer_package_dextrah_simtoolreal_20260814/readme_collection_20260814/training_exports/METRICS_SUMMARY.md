# DEXTRAH 训练效果指标摘要

本目录只包含指标和参数，不包含 `.pth` 模型大文件。

## env8192_noCudaGraph_pose45_mb32768

TensorBoard event: `/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768/summaries/events.out.tfevents.1785834637.wu-Z790-UD`

| 指标 | last step | last value | max step | max value | min step | min value |
|---|---:|---:|---:|---:|---:|---:|
| `in_success_region/frame` | 1508507648 | 0.377808 | 70909952 | 0.599487 | 0 | 0 |
| `object_to_goal_reward/frame` | 1508507648 | 1.00018 | 60555264 | 1.40294 | 23330816 | 0.00190112 |
| `lift_reward/frame` | 1508507648 | 1.44567 | 61734912 | 2.7047 | 178520064 | 0.0946875 |
| `hand_to_object_reward/frame` | 1508507648 | 0.266537 | 27131904 | 0.330168 | 4849664 | 0.00552893 |
| `finger_curl_reg/frame` | 1508507648 | -0.079587 | 9699328 | -0.00964765 | 217317376 | -0.152443 |
| `num_adr_increases/frame` | 1508507648 | 9 | 1455292416 | 9 | 0 | 0 |
| `rewards/step` | 1508507648 | 1253.07 | 73662464 | 2305.03 | 220725248 | -24.2602 |
| `shaped_rewards/step` | 1508507648 | 12.5307 | 73662464 | 23.0503 | 220725248 | -0.242602 |
| `episode_lengths/step` | 1508507648 | 515.827 | 4980736 | 599 | 393216 | 54.6667 |
| `performance/step_fps` | 1508507648 | 41380.7 | 1441792 | 76279.5 | 1263271936 | 31990.4 |
| `performance/step_inference_fps` | 1508507648 | 40490.5 | 1441792 | 73773.6 | 1263271936 | 31469.9 |
| `performance/step_inference_rl_update_fps` | 1508507648 | 23651 | 1572864 | 36938.6 | 1263271936 | 20935.5 |
| `losses/a_loss` | 1508507648 | -0.0089908 | 0 | 0.0566322 | 284426240 | -0.032569 |
| `losses/c_loss` | 1508507648 | 0.0370333 | 29360128 | 3.3196 | 280887296 | 4.66738e-05 |
| `losses/entropy` | 1508507648 | 42.8336 | 1508507648 | 42.8336 | 22675456 | 14.7159 |
| `info/kl` | 1508507648 | 0.0246594 | 0 | 0.248119 | 218628096 | 0.00580832 |

Latest checkpoint path, not uploaded:
`/mnt/bigdata/SUN_ht/runs/dextrah/env8192_noCudaGraph_pose45_mb32768/nn/last_dextrah_lstm_ep_11400_rew_1310.6005.pth`

## env8192_noCudaGraph_pose45_mb32768_resume

TensorBoard event: `/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768_resume/summaries/events.out.tfevents.1785895717.wu-Z790-UD`

| 指标 | last step | last value | max step | max value | min step | min value |
|---|---:|---:|---:|---:|---:|---:|
| `in_success_region/frame` | 2058092544 | 0.383911 | 1472724992 | 0.650513 | 1468006400 | 0 |
| `object_to_goal_reward/frame` | 2058092544 | 1.27727 | 1472593920 | 1.90988 | 1468530688 | 0.00358586 |
| `lift_reward/frame` | 2058092544 | 1.23857 | 1477574656 | 2.88808 | 1468530688 | 0.0964355 |
| `hand_to_object_reward/frame` | 2058092544 | 0.271748 | 1469186048 | 0.374817 | 1468268544 | 7.22622e-05 |
| `finger_curl_reg/frame` | 2058092544 | -0.0676347 | 1472856064 | -0.0261734 | 1472724992 | -0.108632 |
| `num_adr_increases/frame` | 2058092544 | 16 | 1890189312 | 16 | 1468006400 | 0 |
| `rewards/step` | 2058092544 | 1263.59 | 1513488384 | 2536.01 | 1468792832 | 13.8329 |
| `shaped_rewards/step` | 2057961472 | 14.2667 | 1513488384 | 25.3601 | 1468792832 | 0.138329 |
| `episode_lengths/step` | 2057961472 | 479.591 | 1546387456 | 598.168 | 1468792832 | 108.778 |
| `performance/step_fps` | 2058092544 | 40144.8 | 1468530688 | 72197.6 | 1492516864 | 36874.5 |
| `performance/step_inference_fps` | 2058092544 | 39273.6 | 1468530688 | 69904.9 | 1492516864 | 36185 |
| `performance/step_inference_rl_update_fps` | 2058092544 | 22838.3 | 1468530688 | 36009.7 | 2057043968 | 22558.2 |
| `losses/a_loss` | 2058092544 | -0.00309685 | 1468006400 | 0.171137 | 1702625280 | -0.0177263 |
| `losses/c_loss` | 2058092544 | 0.0347272 | 1477705728 | 0.250157 | 1472593920 | 0.0196621 |
| `losses/entropy` | 2058092544 | 48.1365 | 2058092544 | 48.1365 | 1468006400 | 42.2527 |
| `info/kl` | 2058092544 | 0.0332339 | 1468006400 | 0.903752 | 1789657088 | 0.00602366 |

Latest checkpoint path, not uploaded:
`/mnt/bigdata/SUN_ht/runs/dextrah/env8192_noCudaGraph_pose45_mb32768_resume/nn/last_dextrah_lstm_ep_15600_rew_1490.6122.pth`
