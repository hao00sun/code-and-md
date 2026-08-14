# IsaacLab 当前使用说明

本文档记录当前机器上可用的 IsaacLab 环境、常用命令和容易踩坑的地方。

## 当前路径

```text
工作区: /data/SUN_ht/Isaac_Gym
IsaacLab: /data/SUN_ht/Isaac_Gym/IsaacLab
Conda 环境: /data/SUN_ht/Isaac_Gym/env_isaaclab
训练输出: /mnt/bigdata/SUN_ht/runs/rsl_rl
```

## 进入环境

```bash
cd /data/SUN_ht/Isaac_Gym/IsaacLab
conda activate /data/SUN_ht/Isaac_Gym/env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
```

如果当前在 `DEXTRAH` 目录，不能直接执行 `./isaaclab.sh`。需要先切回 IsaacLab 目录：

```bash
cd /data/SUN_ht/Isaac_Gym/IsaacLab
```

## 基础检查

```bash
python -c "import isaacsim; print('isaacsim ok')"
python -c "import isaaclab; import isaaclab_tasks; print('isaaclab ok')"
python -c "import torch; print(torch.__version__)"
nvidia-smi
```

## 训练入口

当前版本使用统一入口：

```bash
./isaaclab.sh train --rl_library rsl_rl --task <任务名>
./isaaclab.sh play --rl_library rsl_rl --task <任务名> --checkpoint <模型路径>
```

注意：本机当前的 `train` 入口不接受旧参数 `--headless`。默认不加可视化参数就是无界面训练。

## 可视化训练

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --num_envs 128 \
  --max_iterations 200 \
  --viz kit \
  --experiment_name shadow_hand_visual_train \
  --run_name viz_128env
```

`--viz kit` 会打开 Isaac Sim 窗口，适合观察训练画面。可视化会降低训练速度，所以正式训练建议去掉 `--viz kit`。

## 无界面快速测试

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --num_envs 512 \
  --max_iterations 50 \
  --experiment_name shadow_hand_smoke
```

## 官方参数训练

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Reorient-Cube-Shadow-Direct \
  --experiment_name shadow_hand_official
```

不加 `--resume` 和 `--checkpoint` 时，就是从零开始训练。

## 查看训练曲线

```bash
tensorboard \
  --logdir /mnt/bigdata/SUN_ht/runs/rsl_rl \
  --host 127.0.0.1 \
  --port 6006
```

浏览器打开：

```text
http://127.0.0.1:6006
```

不要在浏览器里访问 `0.0.0.0:6006`。`0.0.0.0` 只适合做服务监听地址，不适合作为浏览器访问地址。

## 查看显存

```bash
watch -n 1 nvidia-smi
```

只看显存和进程：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

## 当前保存位置

RSL-RL 日志和 checkpoint 已改到：

```text
/mnt/bigdata/SUN_ht/runs/rsl_rl/<experiment_name>/<timestamp>/
```

每个 run 中常见文件：

```text
events.out.tfevents.*  TensorBoard 曲线数据
model_*.pt            可用于 play 或 resume 的模型
params/               训练配置快照
```
