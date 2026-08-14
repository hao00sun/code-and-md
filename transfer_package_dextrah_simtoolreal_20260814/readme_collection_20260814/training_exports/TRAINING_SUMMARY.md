# DEXTRAH Training Results Export

Generated: 2026-08-07T18:05:18+08:00

This export contains training summaries, parameter snapshots, TensorBoard scalar CSV files, and JSON metric summaries. It intentionally excludes model artifacts such as `.pt`, `.pth`, `.safetensors`, `.onnx`, `.ckpt`, and `.bin`.

## Included Runs

### env8192_noCudaGraph_pose45_mb32768

- TensorBoard event source: `/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768/summaries/events.out.tfevents.1785834637.wu-Z790-UD`
- Exported scalar tags: 44
- Max points per exported scalar: 10000

| Metric | Last step | Last value | Best/max value | Best/max step | Min value | Min step |
|---|---:|---:|---:|---:|---:|---:|
| `in_success_region/frame` | 1508507648 | 0.377808 | 0.599487 | 70909952 | 0 | 0 |
| `object_to_goal_reward/frame` | 1508507648 | 1.00018 | 1.40294 | 60555264 | 0.00190112 | 23330816 |
| `lift_reward/frame` | 1508507648 | 1.44567 | 2.7047 | 61734912 | 0.0946875 | 178520064 |
| `hand_to_object_reward/frame` | 1508507648 | 0.266537 | 0.330168 | 27131904 | 0.00552893 | 4849664 |
| `finger_curl_reg/frame` | 1508507648 | -0.079587 | -0.00964765 | 9699328 | -0.152443 | 217317376 |
| `num_adr_increases/frame` | 1508507648 | 9 | 9 | 1455292416 | 0 | 0 |
| `rewards/step` | 1508507648 | 1253.07 | 2305.03 | 73662464 | -24.2602 | 220725248 |
| `shaped_rewards/step` | 1508507648 | 12.5307 | 23.0503 | 73662464 | -0.242602 | 220725248 |
| `episode_lengths/step` | 1508507648 | 515.827 | 599 | 4980736 | 54.6667 | 393216 |
| `performance/step_fps` | 1508507648 | 41380.7 | 76279.5 | 1441792 | 31990.4 | 1263271936 |
| `performance/step_inference_fps` | 1508507648 | 40490.5 | 73773.6 | 1441792 | 31469.9 | 1263271936 |
| `performance/step_inference_rl_update_fps` | 1508507648 | 23651 | 36938.6 | 1572864 | 20935.5 | 1263271936 |
| `losses/a_loss` | 1508507648 | -0.0089908 | 0.0566322 | 0 | -0.032569 | 284426240 |
| `losses/c_loss` | 1508507648 | 0.0370333 | 3.3196 | 29360128 | 4.66738e-05 | 280887296 |
| `losses/entropy` | 1508507648 | 42.8336 | 42.8336 | 1508507648 | 14.7159 | 22675456 |
| `info/kl` | 1508507648 | 0.0246594 | 0.248119 | 0 | 0.00580832 | 218628096 |

### env8192_noCudaGraph_pose45_mb32768_resume

- TensorBoard event source: `/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/env8192_noCudaGraph_pose45_mb32768_resume/summaries/events.out.tfevents.1785895717.wu-Z790-UD`
- Exported scalar tags: 44
- Max points per exported scalar: 4503

| Metric | Last step | Last value | Best/max value | Best/max step | Min value | Min step |
|---|---:|---:|---:|---:|---:|---:|
| `in_success_region/frame` | 2058092544 | 0.383911 | 0.650513 | 1472724992 | 0 | 1468006400 |
| `object_to_goal_reward/frame` | 2058092544 | 1.27727 | 1.90988 | 1472593920 | 0.00358586 | 1468530688 |
| `lift_reward/frame` | 2058092544 | 1.23857 | 2.88808 | 1477574656 | 0.0964355 | 1468530688 |
| `hand_to_object_reward/frame` | 2058092544 | 0.271748 | 0.374817 | 1469186048 | 7.22622e-05 | 1468268544 |
| `finger_curl_reg/frame` | 2058092544 | -0.0676347 | -0.0261734 | 1472856064 | -0.108632 | 1472724992 |
| `num_adr_increases/frame` | 2058092544 | 16 | 16 | 1890189312 | 0 | 1468006400 |
| `rewards/step` | 2058092544 | 1263.59 | 2536.01 | 1513488384 | 13.8329 | 1468792832 |
| `shaped_rewards/step` | 2057961472 | 14.2667 | 25.3601 | 1513488384 | 0.138329 | 1468792832 |
| `episode_lengths/step` | 2057961472 | 479.591 | 598.168 | 1546387456 | 108.778 | 1468792832 |
| `performance/step_fps` | 2058092544 | 40144.8 | 72197.6 | 1468530688 | 36874.5 | 1492516864 |
| `performance/step_inference_fps` | 2058092544 | 39273.6 | 69904.9 | 1468530688 | 36185 | 1492516864 |
| `performance/step_inference_rl_update_fps` | 2058092544 | 22838.3 | 36009.7 | 1468530688 | 22558.2 | 2057043968 |
| `losses/a_loss` | 2058092544 | -0.00309685 | 0.171137 | 1468006400 | -0.0177263 | 1702625280 |
| `losses/c_loss` | 2058092544 | 0.0347272 | 0.250157 | 1477705728 | 0.0196621 | 1472593920 |
| `losses/entropy` | 2058092544 | 48.1365 | 48.1365 | 2058092544 | 42.2527 | 1468006400 |
| `info/kl` | 2058092544 | 0.0332339 | 0.903752 | 1468006400 | 0.00602366 | 1789657088 |

## Parameter Snapshots

- `params/env8192_noCudaGraph_pose45_mb32768/agent.yaml`
- `params/env8192_noCudaGraph_pose45_mb32768/env.yaml`
- `params/env8192_noCudaGraph_pose45_mb32768_resume/agent.yaml`
- `params/env8192_noCudaGraph_pose45_mb32768_resume/env.yaml`

## Notes

- Main run used 8192 environments, CUDA Graph disabled, pose angle 45 degrees, minibatch size 32768.
- Resume run continued from the main training checkpoint and reached higher recorded peak success-region ratio than the main run.
- Checkpoint paths are referenced in the source summary for traceability but model files are not included in this export.
