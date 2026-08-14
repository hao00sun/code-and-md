# DEXTRAH 环境搭建中文版

## 本地路径

```text
仓库: /data/SUN_ht/Isaac_Gym/DEXTRAH
环境: /data/SUN_ht/Isaac_Gym/env_dextrah
大文件输出: /mnt/bigdata/SUN_ht/runs/dextrah
```

## 常用环境变量

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1
```

## Teacher 训练目录

```text
/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games
```

## Student 蒸馏目录

```text
/data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/distillation
```

## 注意事项

```text
1. 大模型和 TensorBoard event 应保存到 /mnt/bigdata
2. student stereo checkpoint 必须配合 env.simulate_stereo=True
3. env.distillation=True 会启用 camera/student observation 路径
4. teacher play.py 和 distillation teacher beta=1 的成功率口径不同
```
