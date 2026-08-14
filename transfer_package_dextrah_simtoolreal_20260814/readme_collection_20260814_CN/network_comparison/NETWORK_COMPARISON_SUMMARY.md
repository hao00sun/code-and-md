# 网络对比实验总结

本文档记录 DEXTRAH student 不同网络结构的训练对比。

## 对比对象

```text
mono transformer
stereo transformer
其他辅助网络或 CNN/ResNet 变体
```

## 关注指标

```text
imitation_loss
aux_loss_object_pos
total_loss
in_success_region
mean_success_last_100
mean_success_all
rewards/step
```

## 主要判断方式

网络结构不能只看 loss，需要结合 distillation 环境下的 rollout 指标。若某个网络 imitation loss 下降但 success 指标不升，说明它可能只拟合了动作分布的一部分，没有学到足够稳健的闭环控制能力。

stereo 网络理论上信息更多，但也更依赖相机配置、图像增强、输入同步和显存。mono 网络更轻，但缺少深度/双目几何信息。
