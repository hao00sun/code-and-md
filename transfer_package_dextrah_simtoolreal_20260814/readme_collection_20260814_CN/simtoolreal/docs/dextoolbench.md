# DexToolBench 中文说明

DexToolBench 是 SimToolReal 用于评估灵巧工具操作策略的 benchmark。

## 任务组成

```text
6 类工具: hammer, marker, eraser, brush, spatula, screwdriver
每类 2 个对象
每个对象 2 个任务
总计 24 个组合
```

## 用途

```text
1. 数值评估 policy 在多工具、多任务上的泛化表现
2. 可视化对象和任务轨迹
3. 生成或检查 benchmark 数据
4. 辅助 sim-to-real 部署前验证
```

## 常见脚本

```bash
dextoolbench/run_all_evals_isaacsim.py
dextoolbench/eval_interactive_isaacsim.py
dextoolbench/visualize_all_objects.py
dextoolbench/visualize_all_tasks.py
```

## 数据

DexToolBench 数据需要单独下载，通常通过项目提供的下载脚本获取。
