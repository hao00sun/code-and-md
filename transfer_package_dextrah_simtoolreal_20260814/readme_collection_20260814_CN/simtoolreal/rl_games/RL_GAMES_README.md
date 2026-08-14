# rl_games 中文说明

`rl_games` 是强化学习训练框架，SimToolReal 中 vendored 了修改版本，用于运行 PPO/SAPG。

## 主要功能

```text
PPO / A2C 连续动作训练
LSTM / MLP actor-critic 网络
central value / asymmetric critic
checkpoint 保存和恢复
TensorBoard 指标记录
多环境并行采样
SAPG 探索机制扩展
```

## 常见配置字段

```text
max_epochs          最大训练 epoch
horizon_length      每个 epoch 每个 env 采样步数
minibatch_size      优化 batch 大小
mini_epochs         每批数据重复训练次数
learning_rate       学习率
gamma               折扣因子
tau                 GAE 参数
e_clip              PPO clip 范围
entropy_coef        entropy bonus 权重
save_frequency      定期保存间隔
save_best_after     从多少 epoch 后开始保存 best
```

## SimToolReal 中的特殊点

SimToolReal 使用 SAPG，因此会出现：

```text
expl_coef_block_size
expl_type
expl_reward_type
use_others_experience
off_policy_ratio
successes_per_block/block_*
```

不同 block 是训练时不同探索强度分组，最终仍训练并保存同一个 policy。
