# DEXTRAH 使用指南中文版

本指南总结 DEXTRAH 在本机上的训练、验证、蒸馏和指标分析流程。

## 主要流程

```text
1. 训练 teacher policy
2. 在 play.py 中验证 teacher
3. 在 distillation 环境中用 beta=1 验证 teacher
4. 用 teacher 蒸馏 student
5. 对齐统计口径，比较 teacher/student
6. 导出指标和总结
```

## Teacher 训练示例

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 8192 \
  --device cuda:0
```

## Teacher 验证示例

```bash
/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python play.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --checkpoint <teacher_ckpt> \
  --num_envs 64
```

## Student 蒸馏示例

```bash
cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/distillation

/data/SUN_ht/Isaac_Gym/env_dextrah/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  run_distillation.py \
  --headless \
  --distributed \
  --task=Dextrah-Kuka-Allegro \
  --num_envs 256 \
  --teacher teacher_lstm_base_best.pth \
  env.distillation=True \
  --enable_cameras \
  env.simulate_stereo=True
```

## 关键结论

`play.py` 中 teacher 成功率很高，并不代表 distillation 环境下 teacher 也有同样成功率。对齐统计口径后，distillation beta=1 teacher 的 `mean_success_all` 约 0.55，而 noADR student 约 0.52。因此 student 表现应与同一 distillation 口径下 teacher 对比。

## 常用指标

```text
in_success_region       当前成功区域内 env 占比
mean_success_all        全程平均成功占比
mean_success_last_100   最近 100 次平均成功占比
imitation_loss          student 模仿 teacher 损失
aux_loss_object_pos     辅助 object position 损失
beta                    teacher action 推进概率
```
