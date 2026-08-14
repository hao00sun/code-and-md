# SimToolReal Isaac Gym legacy 安装说明中文版

Isaac Gym 是 SimToolReal 论文结果中使用的旧训练管线。当前更推荐 Isaac Sim / Isaac Lab，但若需要复现旧结果，可使用 `isaacgymenvs`。

## 特点

```text
Python = 3.8
需要手动下载 Isaac Gym binary
训练入口 = isaacgymenvs/launch_training.py 或 isaacgymenvs/train.py
用途 = 复现实验、对比 legacy 结果
```

## 基本训练命令

```bash
python isaacgymenvs/launch_training.py --custom_experiment_name my_experiment
```

如需从 checkpoint 微调：

```bash
python isaacgymenvs/launch_training.py --checkpoint pretrained_policy/model.pth
```

## 注意

Isaac Gym 与 Isaac Sim 环境应放在不同虚拟环境中，避免 Python、CUDA、Isaac 依赖冲突。
