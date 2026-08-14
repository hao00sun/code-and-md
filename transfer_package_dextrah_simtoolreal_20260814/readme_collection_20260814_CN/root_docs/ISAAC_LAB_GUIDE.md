# Isaac Lab 使用指南中文版

本文件说明当前工作区中 Isaac Lab 的作用和常用操作。

## 作用

Isaac Lab 是 DEXTRAH 和 SimToolReal 当前仿真训练的主要底层框架，负责 Isaac Sim 启动、环境封装、任务注册、RL wrapper 和仿真配置。

## 常见注意事项

```text
1. 必须接受 OMNI_KIT_ACCEPT_EULA=YES
2. Isaac Sim 相关模块应在 AppLauncher 启动后导入
3. headless 训练适合大规模并行环境
4. 图形可视化会占用额外显存
5. 不同 Isaac Lab 版本之间 API 可能变化
```

## 常用环境变量

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export HYDRA_FULL_ERROR=1
```

## 本地相关路径

```text
/data/SUN_ht/Isaac_Gym/IsaacLab
/data/SUN_ht/Isaac_Gym/IsaacLab_v2.2.1
```
