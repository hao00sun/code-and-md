# 指标总结

本文件用于解释和汇总 DEXTRAH 训练指标。

## 常见指标

```text
rewards/step              平均 reward
in_success_region         当前 step 成功区域内 env 占比
mean_success_last_100     最近 100 次 success_rate 的平均值
mean_success_all          从开始到当前的 success_rate 平均值
imitation_loss            student 模仿 teacher 动作分布的损失
aux_loss_object_pos       student 辅助预测 object position 的损失
total_loss                imitation loss + aux_coeff * aux loss
beta                      teacher action 推进环境的概率
```

## 解释

`in_success_region` 是瞬时成功占比，不是 episode-level 成功率。`mean_success_last_100` 和 `mean_success_all` 是后来为了对齐 `play.py` 统计口径而加入的时间平均指标。
