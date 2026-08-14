# DEXTRAH 实验资料包

本资料包整理当前本机 DEXTRAH 训练相关资料，目标是便于同步到 GitHub private repo 做备份和复盘。

不包含 `.pth` 模型大文件。模型 checkpoint 只在指标清单中记录文件名、原始路径和大小。

## 内容

```text
docs/       修改后的中文 Markdown 文档
params/     每个 run 的 agent.yaml 和 env.yaml 参数快照
metrics/    TensorBoard 标量导出的 CSV、summary.json 和 SUMMARY.md
patches/    本次修改过的关键 Python 文件快照，便于追踪本地改动
manifests/  文件清单
```

## 主要 run

```text
env8192_noCudaGraph_pose45_mb32768
主训练 run，8192 并行环境，关闭 CUDA Graph。

env8192_noCudaGraph_pose45_mb32768_resume
从 checkpoint 续训的 run。
```

## 当前推荐训练参数

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games
conda activate /data/SUN_ht/Isaac_Gym/env_dextrah

python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 8192 \
  agent.wandb_activate=False \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0 \
  env.use_cuda_graph=False \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  +agent.params.config.full_experiment_name=env8192_noCudaGraph_pose45_mb32768
```

## 查看指标

优先看：

```text
metrics/SUMMARY.md
metrics/summary.json
```

完整 TensorBoard scalar 已按 tag 导出为 CSV：

```text
metrics/<run_name>/*.csv
```

## 注意

- checkpoint 大文件已迁移到 `/mnt/bigdata/SUN_ht/runs/dextrah/<run>/nn`，本包不上传。
- DEXTRAH 的 `nn/` 本地目录是软链接，查找 checkpoint 时需要 `find -L`。
- `8192/16384 + env.use_cuda_graph=True` 曾在 `apply_object_wrench()` 阶段触发 CUDA/PhysX/Fabric 崩溃，当前推荐关闭 CUDA Graph。
- 可视化使用 `play.py` 的普通 argparse 参数，不使用 `env.xxx=...` 这种 Hydra override。
