# FABRICS 中文说明

FABRICS 是当前 DEXTRAH 环境中使用的运动控制/约束相关模块，用于构造机器人控制中的 fabric terms、关节限制和避障等项。

## 用途

```text
1. 为机械臂和手部控制提供几何/动力学约束
2. 处理 joint limit repulsion 等项
3. 支持 DEXTRAH 任务中的低层控制逻辑
```

## 注意

运行中可能出现 `torch.tensor(sourceTensor)` 的 warning，这通常是张量构造方式提示，不一定导致训练失败。
