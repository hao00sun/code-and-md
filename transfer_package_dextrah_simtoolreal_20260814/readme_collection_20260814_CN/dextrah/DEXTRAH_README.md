# DEXTRAH README 中文版

DEXTRAH 是用于 Kuka + Allegro/Shadow Hand 灵巧操作的仿真与强化学习代码库。本地主要使用 Isaac Sim / Isaac Lab 环境运行 teacher 训练、student 蒸馏、teacher/student 验证和指标分析。

## 主要目录

```text
dextrah_lab/             任务、环境、训练和蒸馏代码
dextrah_lab/rl_games/    teacher policy 训练与 play 验证入口
dextrah_lab/distillation/student 蒸馏、student/teacher distillation 环境验证
pretrained_ckpts/        teacher/student checkpoint 或预训练模型
runs/                    本地早期实验输出
```

## 常用入口

Teacher 训练入口：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games
/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python train.py --task=Dextrah-Kuka-Allegro --headless
```

Teacher 验证入口：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games
/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python play.py --task=Dextrah-Kuka-Allegro --checkpoint <ckpt>
```

Student 蒸馏入口：

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/distillation
/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 run_distillation.py --task=Dextrah-Kuka-Allegro --distributed --headless
```

## 本地实验重点

本地重点围绕以下问题：

```text
1. teacher policy 在 noADR / ADR / distillation 环境中的表现
2. student policy 蒸馏是否接近 teacher 在同一统计口径下的表现
3. success 指标在 play.py 与 distillation 脚本中的统计口径差异
4. beta 调度、辅助损失、相机输入、stereo/mono student 网络的影响
```

## 注意事项

`env.distillation=True` 会改变环境返回结构并启用 student/camera 路径，同时成功后会按 `success_timeout` 触发 reset。因此 distillation 环境下的成功率不能直接和普通 `play.py` 的成功率比较。
