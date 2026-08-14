# 训练结果总结

本文档用于总结已完成 DEXTRAH 训练的参数、指标和结论。

## 记录内容

```text
1. 训练任务名称
2. 使用的 teacher/student checkpoint
3. 训练环境参数
4. 训练指标
5. 成功率、reward、loss 的趋势
6. 后续结论和建议
```

## 主要结论

DEXTRAH 的 student 蒸馏表现需要放在同一 distillation 环境和同一统计口径下与 teacher 对比。直接用普通 `play.py` teacher 的高成功率作为 student 上限会产生误判。

在 noADR、完整物体集、success timeout/reset 的 distillation 环境里，teacher beta=1 的平均成功率约为 0.55，而 student 最终约为 0.52，说明 student 并非完全失败，而是接近该口径下 teacher rollout 的表现。
