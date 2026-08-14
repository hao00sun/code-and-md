# LeRobot 中文使用文档总览

整理日期：2026-08-14

这个目录用于保存当前 LeRobot 本地工程的中文说明。建议从本文件开始看，再按需要打开对应子文档。

## 文档列表

### 0. 最简配置环境与启动流程

文件：

```text
00_minimal_setup_and_run.md
```

作用：

- 只保留最短路径：安装依赖、创建 conda 环境、安装项目、准备 PI05 权重和数据集、启动训练/推理。
- 适合已经知道目标、只想快速跑起来的人。

### 1. 从零开始搭建环境

文件：

```text
01_from_zero_setup.md
```

作用：

- 面向小白，从一台新机器开始说明怎么安装系统依赖、conda、Python 环境。
- 说明怎么拉取 LeRobot 代码。
- 说明怎么安装 Python 依赖。
- 说明怎么配置 Hugging Face token、缓存目录、代理和离线模式。
- 说明 PI05、DiT、CLIP 等预训练权重的拉取或迁移方式。
- 说明如何做最小验证。

### 2. 当前本地实验与改造总览

文件：

```text
02_local_usage_and_experiments.md
```

作用：

- 记录当前机器上已经做过的本地改造和实验。
- 包括 `.venv`/conda 关系、PI05 训练、PI05 server、本地机械臂推理、数据集处理、DiT、HIL-SERL、reward classifier。
- 记录已检测到的权重、缓存、训练 run、日志目录。
- 适合作为项目交接、复盘、排错时的总地图。

### 3. 打包当前代码和文档

文件：

```text
03_pack_current_code.md
```

作用：

- 说明如何把当前代码和中文 md 文档压缩成 zip。
- 明确排除 `.venv`、`.git`、缓存、训练输出、大数据集、大权重等目录。
- 给出推荐命令、检查命令和解压验证方法。
- 适合把当前工程发给另一台机器或做阶段备份。

### 4. 项目定位与代码结构导读

文件：

```text
04_project_position_and_structure.md
```

作用：

- 说明这个代码库解决什么问题：训练、推理、数据采集、仿真、部署和闭环。
- 说明适用任务、机器人本体、数据类型、输入输出和核心优势。
- 按一级/二级目录解释代码结构。
- 区分核心算法、工程封装、示例、测试和工具脚本。
- 给出必须阅读和可以暂时跳过的文件。
- 解释训练链路、数据链路、checkpoint 链路和真实机器人推理链路。

## 推荐阅读顺序

如果你是第一次接手：

```text
00_minimal_setup_and_run.md
01_from_zero_setup.md
04_project_position_and_structure.md
02_local_usage_and_experiments.md
03_pack_current_code.md
```

如果你已经在当前机器上工作，只想知道做过什么：

```text
02_local_usage_and_experiments.md
```

如果你想先理解项目整体架构：

```text
04_project_position_and_structure.md
```

如果你只想打包项目：

```text
03_pack_current_code.md
```

## 当前目录结构

```text
usage_cn/
  README.md
  00_minimal_setup_and_run.md
  01_from_zero_setup.md
  02_local_usage_and_experiments.md
  03_pack_current_code.md
  04_project_position_and_structure.md
```
