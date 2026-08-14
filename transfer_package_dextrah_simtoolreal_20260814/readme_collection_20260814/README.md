# README 文档集合 2026-08-14

这个文件夹用于恢复 Isaac Gym 工作区中的原始 / 英文版 README 文档集合。大多数文件都直接从源代码仓库或之前导出的实验记录中复制而来，因此尽量保留了原始表述。

完整中文版本位于：

```text
/data/SUN_ht/Isaac_Gym/readme_collection_20260814_CN
```
网页主页位置
dextrah：https://sites.google.com/view/dextrah-g
simtoolreal：https://simtoolreal.github.io/
代码库位置
dextrah：https://github.com/NVlabs/DEXTRAH.git
simtoolreal：https://github.com/tylerlum/simtoolreal.git


## 推荐阅读顺序

```text
1. simtoolreal/docs/simtoolreal_current_training_hyperparams_zh.md
2. simtoolreal/SIMTOOLREAL_README.md
3. dextrah/USED_PARAMS.md
4. training_exports/TRAINING_SUMMARY.md
5. network_comparison/NETWORK_COMPARISON_SUMMARY.md
```

## 目录作用

```text
root_docs/             当前工作区根目录下的 DEXTRAH、Isaac Lab 和实验说明
dextrah/               DEXTRAH 仓库 README 与已使用参数
simtoolreal/           SimToolReal 项目说明、本地环境说明和训练超参数
simtoolreal/docs/      SimToolReal 安装、数据、部署、benchmark、baseline 文档
simtoolreal/rl_games/  vendored rl_games 训练框架说明
setup_guides/          DEXTRAH 和 SimToolReal 环境配置步骤
training_exports/      已导出的训练结果和指标总结
network_comparison/    三种网络结构对比实验说明
fabrics/               FABRICS 模块说明和变更记录
```
## 文件夹总览

### root_docs

工作区级别的通用说明和配置指南。

- `ISAAC_LAB_GUIDE.md`：Isaac Lab 相关说明和使用指南。
- `DEXTRAH_GUIDE.md`：DEXTRAH 使用说明。
- `DEXTRAH_SETUP.md`：DEXTRAH 环境配置说明。
- `REORIENT_REPOSE_BASELINE.md`：重定向 / repose 基线实验说明。
- `SHADOW_HAND_EXPERIMENTS.md`：Shadow Hand 实验说明。

### dextrah

与 DEXTRAH 仓库以及 DEXTRAH teacher / student policy 实验相关的文档。

- `DEXTRAH_README.md`：DEXTRAH 仓库原始 README。
- `USED_PARAMS.md`：DEXTRAH 实验中使用过的参数记录。
- `README_PROJECT_ZH.md`：后来补充的中文项目级说明。

### simtoolreal

与 SimToolReal 相关的文档。

- `SIMTOOLREAL_README.md`：SimToolReal 仓库原始 README。
- `README_LOCAL_ENV.md`：当前机器上的本地环境说明。
- `README_PROJECT_ZH.md`：后来补充的中文项目级说明。
- `docs/`：SimToolReal 原始文档页面。
- `rl_games/RL_GAMES_README.md`：从 SimToolReal 目录中复制出的 rl_games 原始 README。

### training_exports

之前 DEXTRAH 训练结果的导出总结。这里仅包含指标和说明，不包含模型权重。

- `README_SOURCE.md`：数据来源和导出说明。
- `TRAINING_SUMMARY.md`：训练过程总结。
- `METRICS_SUMMARY.md`：指标总结。

### network_comparison

三个网络结构对比实验的记录。

- `README.md`：导出文件夹说明。
- `NETWORK_COMPARISON_SUMMARY.md`：LSTM base、LSTM symmetric 和 LSTM symmetric large 三组实验的对比总结。

### fabrics

FABRICS 相关源码文档。

- `FABRICS_README.md`：FABRICS 原始 README。
- `FABRICS_CHANGELOG.md`：FABRICS 原始更新日志。

## 说明

- 这个目录作为原始 / 英文文档集合的恢复版本使用。
- 其中少数文件本身就是中文，因为它们是在本地后续补充的中文项目说明或实验记录。
- 这里不包含任何训练得到的模型 checkpoint 文件。
