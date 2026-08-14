# DEXTRAH + SimToolReal 转移包说明

本文件夹用于在不同机器之间转移 DEXTRAH、SimToolReal 两个代码库以及相关 README / 实验说明文档。

## 包含内容

```text
DEXTRAH/                         DEXTRAH 源码副本
simtoolreal/                     SimToolReal 源码副本
readme_collection_20260814/      原始 / 英文 README 文档集合
readme_collection_20260814_CN/   中文 README 文档集合和新手配置指南
MINIMAL_SETUP_AND_RUN_ZH.md      最简环境配置与启动流程
```

## 不包含内容

为了方便转移，本包默认不包含：

```text
.git/                 Git 历史
Python 虚拟环境        例如 env_dextrah、env_simtoolreal、.venv
训练输出目录           例如 logs、runs
模型权重文件           例如 .pth、.pt、.ckpt、.onnx
Python 缓存            例如 __pycache__、.pytest_cache、*.pyc
```

因此，这个文件夹适合用来转移源码和文档，但不等价于完整训练环境备份。

## 推荐转移方式

如果只想把源码和说明文档搬到另一台机器，可以直接复制整个文件夹：

```bash
cp -a /data/SUN_ht/Isaac_Gym/transfer_package_dextrah_simtoolreal_20260814 /目标路径/
```

也可以使用已经生成好的 zip 压缩包转移：

```text
/data/SUN_ht/Isaac_Gym/transfer_package_dextrah_simtoolreal_20260814.zip
```

将这个 zip 文件复制到新机器后，在目标目录解压：

```bash
cd /目标路径

unzip transfer_package_dextrah_simtoolreal_20260814.zip
```

解压后进入转移包：

```bash
cd transfer_package_dextrah_simtoolreal_20260814
```

然后优先阅读：

```text
README_TRANSFER_ZH.md
MINIMAL_SETUP_AND_RUN_ZH.md
readme_collection_20260814_CN/setup_guides/DEXTRAH_BEGINNER_SETUP_ZH.md
readme_collection_20260814_CN/setup_guides/SIMTOOLREAL_BEGINNER_SETUP_ZH.md
```

如果需要重新生成 zip，可以在原机器执行：

```bash
cd /data/SUN_ht/Isaac_Gym

zip -r transfer_package_dextrah_simtoolreal_20260814.zip transfer_package_dextrah_simtoolreal_20260814
```

如果更喜欢 tar.gz，也可以压缩为：

```bash
cd /data/SUN_ht/Isaac_Gym

tar -czf transfer_package_dextrah_simtoolreal_20260814.tar.gz transfer_package_dextrah_simtoolreal_20260814
```

## 与 git clone 的区别

转移包方式：

```text
优点: 保留当前本机已经整理过的源码和 md 文档，离线也能复制
缺点: 不包含 .git 历史，不能直接查看历史提交；需要重新配置 Python / Isaac 环境
```

重新拉取代码库方式：

```bash
git clone https://github.com/NVlabs/DEXTRAH.git DEXTRAH
git clone https://github.com/tylerlum/simtoolreal.git simtoolreal
```

```text
优点: 保留完整 Git 仓库能力，可以 pull、checkout、查看提交历史
缺点: 不包含本机整理过的中文 README、实验总结和当前配置说明；需要联网
```

## 环境配置入口

转移到新机器后，优先阅读：

```text
MINIMAL_SETUP_AND_RUN_ZH.md
readme_collection_20260814_CN/setup_guides/DEXTRAH_BEGINNER_SETUP_ZH.md
readme_collection_20260814_CN/setup_guides/SIMTOOLREAL_BEGINNER_SETUP_ZH.md
```

`MINIMAL_SETUP_AND_RUN_ZH.md` 是一页式最短流程，适合先跑通环境和启动命令。

两个新手配置指南中包含项目主页、代码库地址、推荐版本、拉取代码、创建环境、安装依赖、训练和 TensorBoard 查看指令。

## 重要提醒

DEXTRAH 和 SimToolReal 的 Python / Isaac 环境不要混用：

```text
DEXTRAH:     Python 3.11 + Isaac Sim 5.0.0.0 + Isaac Lab v2.2.1
SimToolReal: Python 3.11 + Isaac Sim 5.x + Isaac Lab 2.3.2.post1 + PyTorch cu126
```

模型 checkpoint 和训练指标大文件应单独从原机器的训练输出目录复制。
