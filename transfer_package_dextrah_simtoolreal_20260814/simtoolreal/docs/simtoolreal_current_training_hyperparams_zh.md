# SimToolReal 当前训练超参数与指标说明

本文整理当前 SimToolReal 训练 run 的任务、环境、随机化、算法和保存配置。

当前 run:

```text
/mnt/bigdata/SUN_ht/runs/simtoolreal/simtoolreal_sapg_env12288_block2048_08-14-11-07
```

启动时显式修改的参数:

```text
env.scene.num_envs = 12288
agent.params.config.expl_coef_block_size = 2048
```

## 一、超参数罗列

### 1. 任务与运行规模

```text
task = Isaacsimenvs-SimToolReal-Direct-v0
agent = rl_games_sapg_cfg_entry_point
algorithm = SAPG
num_envs = 12288
expl_coef_block_size = 2048
num_exploration_blocks = 6
headless = true
WANDB_MODE = disabled
```

### 2. 仿真环境参数

```text
sim.dt = 0.00833333333
decimation = 2
control_rate = 60 Hz
physics_rate = 120 Hz
episode_length_s = 10.0
termination.episode_length = 600
scene.env_spacing = 1.2
scene.replicate_physics = false
scene.clone_in_fabric = false
scene.filter_collisions = true
scene.lazy_sensor_update = true
```

### 3. 资产与任务对象

```text
robot_urdf = assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf
table_urdf = assets/urdf/table_narrow.urdf
object_name = handle_head_primitives
handle_head_types = [hammer, screwdriver, marker, spatula, eraser, brush]
num_assets_per_type = 100
shuffle_assets = true
modify_asset_frictions = true
robot_friction = 0.5
finger_tip_friction = 1.5
object_friction = 0.5
table_friction = 0.5
```

### 4. 观测空间

```text
actor obs_list =
  joint_pos
  joint_vel
  prev_action_targets
  palm_pos
  palm_rot
  object_rot
  fingertip_pos_rel_palm
  keypoints_rel_palm
  keypoints_rel_goal
  object_scales

critic state_list =
  joint_pos
  joint_vel
  prev_action_targets
  palm_pos
  palm_rot
  palm_vel
  object_rot
  object_vel
  fingertip_pos_rel_palm
  keypoints_rel_palm
  keypoints_rel_goal
  object_scales
  closest_keypoint_max_dist
  closest_fingertip_dist
  lifted_object
  progress
  successes
  reward

obs.clamp_abs_observations = 10.0
```

### 5. 动作空间与动作处理

```text
action_space = 29
action.arm_moving_average = 0.1
action.hand_moving_average = 0.1
action.dof_speed_scale = 1.5
```

### 6. 目标采样与成功条件

```text
reset.goal_sampling_type = delta
reset.delta_goal_distance = 0.1
reset.delta_rotation_degrees = 90.0
reset.target_volume_mins = [-0.35, -0.2, 0.6]
reset.target_volume_maxs = [0.35, 0.2, 0.95]
reset.target_volume_region_scale = 1.0

termination.success_tolerance = 0.075
termination.target_success_tolerance = 0.01
termination.eval_success_tolerance = null
termination.success_steps = 10
termination.max_consecutive_successes = 50
termination.force_consecutive_near_goal_steps = false
termination.tolerance_curriculum_increment = 0.9
termination.tolerance_curriculum_interval = 3000
termination.tolerance_curriculum_success_threshold = 3.0
```

### 7. 奖励参数

```text
reward.keypoint_rew_scale = 200.0
reward.keypoint_scale = 1.5
reward.object_base_size = 0.04
reward.fixed_size = [0.141, 0.03025, 0.0271]
reward.fixed_size_keypoint_reward = true
reward.lifting_rew_scale = 20.0
reward.lifting_bonus = 300.0
reward.lifting_bonus_threshold = 0.15
reward.distance_delta_rew_scale = 50.0
reward.reach_goal_bonus = 1000.0
reward.kuka_actions_penalty_scale = 0.03
reward.hand_actions_penalty_scale = 0.003
```

### 8. Reset 初始状态随机化

```text
reset.reset_position_noise_x = 0.1
reset.reset_position_noise_y = 0.1
reset.reset_position_noise_z = 0.02
reset.fixed_start_pose = null
reset.reset_dof_pos_random_interval_arm = 0.1
reset.reset_dof_pos_random_interval_fingers = 0.1
reset.reset_dof_vel_random_interval = 0.5
reset.start_arm_higher = false
reset.table_reset_z = 0.38
reset.table_reset_z_range = 0.01
reset.table_object_z_offset = 0.25
reset.table_reset_xy_range_m = [0.0, 0.0]
reset.table_reset_yaw_range_deg = 0.0
```

### 9. Domain Randomization 与主动噪声

```text
domain_randomization.use_obs_delay = true
domain_randomization.obs_delay_max = 3
domain_randomization.use_action_delay = true
domain_randomization.action_delay_max = 3

domain_randomization.use_object_state_delay_noise = true
domain_randomization.object_state_delay_max = 10
domain_randomization.object_state_xyz_noise_std = 0.01
domain_randomization.object_state_rotation_noise_degrees = 5.0
domain_randomization.object_scale_noise_multiplier_range = [1.0, 1.0]

domain_randomization.joint_velocity_obs_noise_std = 0.1

domain_randomization.force_scale = 20.0
domain_randomization.force_prob_range = [0.001, 0.1]
domain_randomization.force_decay = 0.0
domain_randomization.force_decay_interval = 0.08
domain_randomization.force_only_when_lifted = true

domain_randomization.torque_scale = 2.0
domain_randomization.torque_prob_range = [0.001, 0.1]
domain_randomization.torque_decay = 0.0
domain_randomization.torque_decay_interval = 0.08
domain_randomization.torque_only_when_lifted = true

domain_randomization.object_friction_scale_range = [1.0, 1.0]
domain_randomization.fingertip_friction_scale_range = [1.0, 1.0]
domain_randomization.friction_n_buckets = 16
```

### 10. PPO/SAPG 训练参数

```text
agent.params.config.name = 0_simtoolreal_sapg
agent.params.config.full_experiment_name = 0_simtoolreal_sapg
agent.params.config.env_name = rlgpu
agent.params.config.device = cuda:0
agent.params.config.device_name = cuda:0
agent.params.config.multi_gpu = false
agent.params.config.ppo = true
agent.params.config.mixed_precision = true
agent.params.config.normalize_input = true
agent.params.config.normalize_value = true
agent.params.config.value_bootstrap = true
agent.params.config.normalize_advantage = true
agent.params.config.reward_shaper.scale_value = 0.01
agent.params.config.num_actors = 12288
agent.params.config.gamma = 0.99
agent.params.config.tau = 0.95
agent.params.config.learning_rate = 1e-4
agent.params.config.lr_schedule = adaptive
agent.params.config.schedule_type = standard
agent.params.config.kl_threshold = 0.016
agent.params.config.max_epochs = 1000000
agent.params.config.save_best_after = 100
agent.params.config.save_frequency = 3000
agent.params.config.grad_norm = 1.0
agent.params.config.entropy_coef = 0.0
agent.params.config.truncate_grads = true
agent.params.config.e_clip = 0.1
agent.params.config.minibatch_size = 98304
agent.params.config.mini_epochs = 2
agent.params.config.critic_coef = 4.0
agent.params.config.clip_value = true
agent.params.config.horizon_length = 16
agent.params.config.seq_length = 16
agent.params.config.bounds_loss_coef = 0.0001
```

### 11. Central Value / 非对称 Critic 参数

```text
central_value_config.minibatch_size = 98304
central_value_config.mini_epochs = 2
central_value_config.learning_rate = 1e-4
central_value_config.kl_threshold = 0.016
central_value_config.clip_value = true
central_value_config.normalize_input = true
central_value_config.truncate_grads = true
central_value_config.network.name = actor_critic
central_value_config.network.central_value = true
central_value_config.network.mlp.units = [1024, 1024, 512, 512]
central_value_config.network.mlp.activation = elu
central_value_config.network.mlp.d2rl = false
```

### 12. SAPG 探索参数

```text
agent.params.config.use_others_experience = lf
agent.params.config.off_policy_ratio = 1.0
agent.params.config.expl_type = mixed_expl_learn_param
agent.params.config.expl_reward_type = entropy
agent.params.config.expl_reward_coef_embd_size = 32
agent.params.config.expl_reward_coef_scale = 0.002
agent.params.config.expl_coef_block_size = 2048
```

### 13. 保存与总训练步数

```text
max_epochs = 1000000
num_envs = 12288
horizon_length = 16
frames_per_epoch = 196608
total_frames = 196608000000
save_frequency = 3000 epochs
frames_per_regular_save = 589824000
save_best_after = 100 epochs
```

模型保存位置:

```text
/mnt/bigdata/SUN_ht/runs/simtoolreal/simtoolreal_sapg_env12288_block2048_08-14-11-07/0_simtoolreal_sapg/nn/0_simtoolreal_sapg.pth
/mnt/bigdata/SUN_ht/runs/simtoolreal/simtoolreal_sapg_env12288_block2048_08-14-11-07/0_simtoolreal_sapg/last/model.pth
```

## 二、详细说明

### 1. 当前训练任务是什么

当前训练的是 `Isaacsimenvs-SimToolReal-Direct-v0`，也就是 SimToolReal 的 Isaac Sim / Isaac Lab 版本任务。任务目标是训练 Kuka 机械臂加 SHARPA 灵巧手，对多类工具进行灵巧操作，使工具关键点逐步到达目标姿态。

当前对象集合为 `handle_head_primitives`，包含六类工具:

```text
hammer, screwdriver, marker, spatula, eraser, brush
```

每类工具有 100 个资产变体，因此训练不是单一物体，而是跨工具类别、跨几何实例的泛化训练。

### 2. `num_envs` 与 `expl_coef_block_size`

`num_envs = 12288` 表示同时并行运行 12288 个仿真环境。

`expl_coef_block_size = 2048` 是 SAPG 的探索系数组大小。当前:

```text
12288 / 2048 = 6
```

所以 TensorBoard 中会看到 `block_0` 到 `block_5`。这些 block 不是不同模型，而是同一个 policy 在训练时使用的不同探索强度分组。所有 block 的经验共同训练同一个模型，最终不会保存成 6 个推理模型。

### 3. 仿真频率与 episode 长度

`sim.dt = 1/120` 表示物理仿真频率为 120 Hz。

`decimation = 2` 表示 policy 每 2 个物理步输出一次动作，因此控制频率为:

```text
120 / 2 = 60 Hz
```

`termination.episode_length = 600`，所以每个 episode 最多:

```text
600 / 60 = 10 秒
```

### 4. 观测空间与非对称训练

Actor 使用 `obs_list`，Critic 使用 `state_list`。Critic 看到的信息更多，包括速度、最近距离、是否抬起、progress、successes 和 reward 等。

这属于非对称 actor-critic 训练:

```text
actor: 使用较少、接近策略实际可用的观测
critic: 使用更完整的状态帮助价值估计和训练稳定
```

这种设置常用于机器人强化学习，因为 critic 只在训练时使用，推理时主要依赖 actor。

### 5. 动作处理参数

`action_space = 29` 表示 policy 输出 29 维动作，其中包含机械臂和手部动作。

`arm_moving_average = 0.1` 与 `hand_moving_average = 0.1` 用于平滑动作目标，避免控制目标剧烈跳变。

`dof_speed_scale = 1.5` 控制机械臂动作增量的速度尺度。数值越大，机械臂动作变化越快，但也更容易不稳定。

### 6. 目标采样与成功条件

`goal_sampling_type = delta` 表示目标不是每次完全随机绝对姿态，而是在当前目标基础上采样一个相对变化。

`delta_goal_distance = 0.1` 表示目标位置变化尺度约 0.1 m。

`delta_rotation_degrees = 90.0` 表示目标旋转变化尺度可到 90 度。

`success_tolerance = 0.075` 是初始成功容差，训练会通过 curriculum 逐渐变严，目标为 `target_success_tolerance = 0.01`。

`success_steps = 10` 表示需要连续满足成功条件若干步才累计成功。

`max_consecutive_successes = 50` 表示一个 episode 中最多追踪 50 次连续目标成功，达到后可触发 `done_max_successes`。

### 7. 奖励项作用

`keypoint_rew_scale = 200.0` 是主要姿态匹配奖励，鼓励工具关键点接近目标关键点。

`lifting_rew_scale = 20.0` 与 `lifting_bonus = 300.0` 鼓励把物体抬离桌面。`lifting_bonus_threshold = 0.15` 表示达到一定高度后给额外奖励。

`distance_delta_rew_scale = 50.0` 奖励物体相对目标距离的改善。

`reach_goal_bonus = 1000.0` 是到达目标时的大额奖励。

`kuka_actions_penalty_scale = 0.03` 和 `hand_actions_penalty_scale = 0.003` 惩罚过大的动作，帮助减少抖动和不自然控制。

### 8. Reset 随机化

Reset 随机化用于让每个 episode 初始状态不同，提升泛化能力。

当前开启了物体初始位置噪声:

```text
x/y/z = 0.1 / 0.1 / 0.02
```

也开启了机械臂和手指关节位置、速度随机化:

```text
arm pos interval = 0.1
finger pos interval = 0.1
dof vel interval = 0.5
```

桌面 z 方向有小范围随机:

```text
table_reset_z_range = 0.01
```

但桌面 xy 和 yaw 当前没有随机:

```text
table_reset_xy_range_m = [0.0, 0.0]
table_reset_yaw_range_deg = 0.0
```

### 9. 观测、动作延迟与观测噪声

当前开启了观测延迟和动作延迟:

```text
obs_delay_max = 3
action_delay_max = 3
```

这意味着策略看到的观测、实际执行的动作可能来自短时间队列中的历史值，用于模拟真实系统中的通信和执行延迟。

物体状态观测也加入延迟和噪声:

```text
object_state_delay_max = 10
object_state_xyz_noise_std = 0.01
object_state_rotation_noise_degrees = 5.0
```

这相当于让 policy 面对带误差的物体位姿估计，贴近真实感知系统。

`joint_velocity_obs_noise_std = 0.1` 则是关节速度观测噪声。

### 10. 主动扰动物体的随机力和力矩

当前训练开启了主动物体扰动。代码会在每个 env 中按概率采样随机外力和随机力矩:

```text
force_scale = 20.0
torque_scale = 2.0
force_prob_range = [0.001, 0.1]
torque_prob_range = [0.001, 0.1]
```

并且只在物体被抬起后施加:

```text
force_only_when_lifted = true
torque_only_when_lifted = true
```

因此这是主动动力学扰动，不只是观测噪声。它会让策略学习在物体受到外部扰动时仍保持控制。

### 11. 未实际扩展的随机化项

虽然配置中存在物体尺度和摩擦随机化字段，但当前范围都是 `[1.0, 1.0]`，等价于没有随机:

```text
object_scale_noise_multiplier_range = [1.0, 1.0]
object_friction_scale_range = [1.0, 1.0]
fingertip_friction_scale_range = [1.0, 1.0]
```

所以当前主要随机化来自 reset、延迟、观测噪声和随机外力/力矩，而不是物体尺寸或摩擦变化。

### 12. PPO/SAPG 训练参数

`gamma = 0.99` 是折扣因子，决定未来奖励的重要性。

`tau = 0.95` 是 GAE 参数，用于平衡优势估计的偏差和方差。

`learning_rate = 1e-4` 是策略和价值网络学习率。

`lr_schedule = adaptive` 与 `kl_threshold = 0.016` 表示学习率会根据 KL 散度自适应调节，避免 policy 更新过猛。

`e_clip = 0.1` 是 PPO clip 范围，限制新旧策略变化。

`entropy_coef = 0.0` 表示普通 PPO entropy bonus 没有额外权重；探索主要由 SAPG 机制处理。

`minibatch_size = 98304` 与 `mini_epochs = 2` 控制每轮 rollout 后的优化 batch 大小和重复训练轮数。

`horizon_length = 16` 表示每个 epoch 每个环境采样 16 个 policy step。

### 13. SAPG 探索机制

当前使用:

```text
expl_type = mixed_expl_learn_param
expl_reward_type = entropy
expl_reward_coef_embd_size = 32
expl_reward_coef_scale = 0.002
off_policy_ratio = 1.0
use_others_experience = lf
```

SAPG 会让不同 block 使用不同探索系数，同时收集经验并共同训练一个 policy。TensorBoard 中的 `successes_per_block/block_*` 可以用来判断哪个探索强度当前更有效。

### 14. 保存频率与总训练量

当前总训练 epoch:

```text
max_epochs = 1000000
```

每个 epoch 的环境 frames:

```text
num_envs * horizon_length = 12288 * 16 = 196608
```

总环境 frames:

```text
1000000 * 196608 = 196608000000
```

定期保存间隔:

```text
save_frequency = 3000 epochs
```

对应:

```text
3000 * 196608 = 589824000 frames
```

此外，`save_best_after = 100` 表示训练超过 100 epoch 后，如果表现刷新 best，会保存或覆盖 best checkpoint。

### 15. TensorBoard 中关键指标含义

建议优先关注:

```text
successes
episode_final/successes
episode_final/all_goals_hit
rewards/step
episode_final/done_max_successes
episode_final/done_fall
episode_final/done_hand_far
performance/step_fps
```

其中 `successes` 表示最近 episode 平均成功次数，不是简单二分类成功率。

`episode_final/all_goals_hit` 表示 episode 结束时是否达到最大连续成功目标。

`successes_per_block/block_*` 表示不同 SAPG 探索系数组的表现，但所有 block 训练的是同一个 policy。

`done_max_successes` 越高通常越好，说明更多 episode 因达到最大连续成功数而结束。

`done_fall` 和 `done_hand_far` 越低越好，说明失败终止减少。

