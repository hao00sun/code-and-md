# SimToolReal 部署说明中文版

部署部分用于把训练好的 policy 运行到 sim-to-sim 或真实机器人系统中。

## 部署节点

```text
RL policy node        运行策略，输出控制动作
goal pose node        提供目标姿态
perception node       使用 SAM + FoundationPose 等感知系统估计物体位姿
robot node            真实机器人控制或仿真机器人控制
```

## Sim-to-Sim

Sim-to-sim 用仿真节点替代真实机器人和真实感知，适合先验证 policy 与通信接口。

## Sim-to-Real

真实部署需要同时启动机器人控制、感知、目标姿态和 policy 节点。部署前应先在仿真中确认动作范围、安全边界和任务稳定性。

## 注意事项

```text
1. 真实机器人部署前必须小速度、小幅度测试
2. 感知延迟和位姿噪声会显著影响策略表现
3. policy 输入维度必须和训练 checkpoint 匹配
4. 目标坐标系和机器人坐标系必须严格对齐
```
