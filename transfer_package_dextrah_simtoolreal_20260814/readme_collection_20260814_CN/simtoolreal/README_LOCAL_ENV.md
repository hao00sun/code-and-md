# SimToolReal 本地环境说明

本地 SimToolReal 工作区位于：

```text
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace
```

仓库路径：

```text
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal
```

Python 环境：

```text
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/python
```

## 训练输出

建议统一保存到：

```text
/mnt/bigdata/SUN_ht/runs/simtoolreal
```

示例：

```bash
hydra.run.dir=/mnt/bigdata/SUN_ht/runs/simtoolreal/simtoolreal_sapg_env12288_block2048_$(date +%m-%d-%H-%M)
```

## TensorBoard

```bash
/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal/bin/tensorboard \
  --logdir /mnt/bigdata/SUN_ht/runs/simtoolreal \
  --host 0.0.0.0 \
  --port 6009 \
  --reload_interval 5
```
