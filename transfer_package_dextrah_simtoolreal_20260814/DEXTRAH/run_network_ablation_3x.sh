#!/usr/bin/env bash
set -euo pipefail

# 顺序启动 3 组 DEXTRAH teacher 训练：
# 1. 当前参数基线：LSTM，关闭 ADR/噪声，5000 epoch，500 epoch 保存一次
# 2. MLP：删除 actor/critic 与 central value 的 rnn 配置
# 3. LSTM 对称网络：保留 LSTM，让 actor/critic 与 central value 使用一致的 MLP/LSTM 容量
#
# 注意：rl_games 用 “rnn 键是否存在” 判断是否启用 RNN，所以 MLP 版本必须使用
# Hydra 删除键语法：'~agent.params.network.rnn'。

cd /data/SUN_ht/Isaac_Gym/DEXTRAH/dextrah_lab/rl_games

export OMNI_KIT_ACCEPT_EULA=YES
export DEXTRAH_ARTIFACT_ROOT="${DEXTRAH_ARTIFACT_ROOT:-/mnt/bigdata/SUN_ht/runs/dextrah}"

# 避免外部 CUDA 运行库路径干扰 Isaac Sim/Omniverse 组件。
unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH

COMMON_ARGS=(
  --headless
  --task=Dextrah-Kuka-Allegro
  --num_envs 8192
  --max_iterations 5000
  agent.wandb_activate=False
  env.objects_dir=visdex_objects
  env.max_pose_angle=45.0
  env.use_cuda_graph=False
  env.enable_adr=False
  env.starting_adr_increments=0
  agent.params.config.minibatch_size=32768
  agent.params.config.central_value_config.minibatch_size=32768
  agent.params.config.save_frequency=500
)

run_train() {
  local name="$1"
  shift

  echo
  echo "================================================================"
  echo "Starting experiment: ${name}"
  echo "================================================================"
  echo

  python train.py \
    "${COMMON_ARGS[@]}" \
    +agent.params.config.full_experiment_name="${name}" \
    "$@"
}

run_train \
  env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_base

run_train \
  env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_mlp \
  '~agent.params.network.rnn' \
  '~agent.params.config.central_value_config.network.rnn' \
  'agent.params.network.mlp.units=[512,512,256,128]' \
  'agent.params.config.central_value_config.network.mlp.units=[512,512,256,128]' \
  agent.params.config.seq_length=4 \
  agent.params.config.zero_rnn_on_done=False

run_train \
  env8192_noCudaGraph_pose45_mb32768_noADR_epoch5000_save500_lstm_symmetric \
  'agent.params.network.mlp.units=[512,512]' \
  'agent.params.config.central_value_config.network.mlp.units=[512,512]' \
  agent.params.network.rnn.units=1024 \
  agent.params.config.central_value_config.network.rnn.units=1024

echo
echo "All 3 network ablation experiments finished."
