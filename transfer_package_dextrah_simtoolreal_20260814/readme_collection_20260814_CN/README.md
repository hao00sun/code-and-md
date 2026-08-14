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

## 新手环境配置

如果是纯小白，优先阅读：

```text
setup_guides/DEXTRAH_BEGINNER_SETUP_ZH.md
setup_guides/SIMTOOLREAL_BEGINNER_SETUP_ZH.md
```

这两份文档包含从检查显卡、进入目录、检查 Python 环境、设置环境变量，到跑通最小验证/训练和 TensorBoard 的步骤。

## 使用说明

这些文档是为了快速回顾本机实验环境、训练指令、指标含义和结果结论。若需要查原始英文全文，可回到 `readme_collection_20260814`。
