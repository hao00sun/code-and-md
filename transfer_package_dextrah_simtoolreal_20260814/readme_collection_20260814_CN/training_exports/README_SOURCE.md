# 训练结果导出来源说明

本目录记录某次 DEXTRAH 训练结果导出的来源、内容和用途。导出内容主要包含训练总结、参数说明、指标数据和必要文档，不包含大体积模型文件。

## 用途

```text
1. 作为 GitHub data 仓库中的实验记录
2. 回溯训练参数和结果
3. 与后续 teacher/student 实验对比
4. 保留指标数据而避免上传模型 checkpoint
```

## 内容类型

```text
TRAINING_SUMMARY.md  训练总结
metrics/             指标数据或指标摘要
docs/                配套文档
README_SOURCE.md     本来源说明
```

## 注意

如果需要复现实验，应结合原始 run 目录、TensorBoard event、训练命令和 checkpoint 路径共同确认。
