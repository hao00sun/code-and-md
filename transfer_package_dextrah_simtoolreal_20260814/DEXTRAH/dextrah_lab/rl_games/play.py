# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--objects_dir", type=str, default="visdex_objects", help="Object asset directory to load.")
parser.add_argument("--max_pose_angle", type=float, default=45.0, help="Maximum target pose angle in degrees.")
parser.add_argument("--use_cuda_graph", action="store_true", default=False, help="Enable CUDA graph in the env.")
parser.add_argument("--disable_adr", action="store_true", default=False, help="Disable ADR during play.")
parser.add_argument("--reset_on_success_timeout", action="store_true", default=False, help="Reset envs after success_timeout seconds in success region.")
parser.add_argument("--success_timeout", type=float, default=2.0, help="Seconds to keep success before reset when --reset_on_success_timeout is set.")
parser.add_argument("--use_object_subset", action="store_true", default=False, help="Use an object subset for play.")
parser.add_argument("--object_subset_size", type=int, default=64, help="Number of unique objects in the subset.")
parser.add_argument("--object_subset_start_index", type=int, default=0, help="Start index of the object subset.")
parser.add_argument(
    "--disable_play_material_randomization",
    action="store_true",
    default=False,
    help="Disable robot/object material reset randomization during play. Useful for multi-env visualization.",
)
parser.add_argument("--stochastic_policy", action="store_true", default=False, help="Sample stochastic actions during play.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--speed_scale",
    type=float,
    default=1.0,
    help="Playback speed multiplier used with --real-time. Use 0.5 for half speed, 0.25 for quarter speed.",
)
parser.add_argument("--play_dt", type=float, default=0.0, help="Minimum wall-clock seconds for each play step.")
parser.add_argument("--print_every", type=int, default=1, help="Print play metrics every N steps.")
parser.add_argument("--max_steps", type=int, default=5000, help="Maximum number of environment steps to run.")
parser.add_argument("--save_metrics_every", type=int, default=0, help="Save play metrics every N steps. Disabled when 0.")
parser.add_argument("--metrics_path", type=str, default=None, help="CSV path for saved play metrics.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import dextrah_lab.tasks.dextrah_kuka_allegro.gym_setup

def main():
    """Play with RL-Games agent."""
    if args_cli.speed_scale <= 0.0:
        raise ValueError("--speed_scale must be positive.")
    if args_cli.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")
    if args_cli.save_metrics_every < 0:
        raise ValueError("--save_metrics_every must be non-negative.")

    def wait_with_gui_updates(deadline: float):
        """Keep Kit responsive while limiting policy playback speed."""
        while simulation_app.is_running():
            remaining_time = deadline - time.time()
            if remaining_time <= 0.0:
                break
            simulation_app.update()
            time.sleep(min(remaining_time, 0.002))

    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task,  device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.objects_dir = args_cli.objects_dir
    env_cfg.max_pose_angle = args_cli.max_pose_angle
    env_cfg.use_cuda_graph = args_cli.use_cuda_graph
    if args_cli.disable_adr:
        env_cfg.enable_adr = False
        env_cfg.num_adr_increments = 0
        env_cfg.starting_adr_increments = 0
    env_cfg.reset_on_success_timeout = args_cli.reset_on_success_timeout
    env_cfg.success_timeout = args_cli.success_timeout
    env_cfg.use_object_subset = args_cli.use_object_subset
    env_cfg.object_subset_size = args_cli.object_subset_size
    env_cfg.object_subset_start_index = args_cli.object_subset_start_index
    if args_cli.disable_play_material_randomization:
        # Keep the terms available for DextrahADR, but stop applying them at reset.
        # On IsaacLab 3 multi-env visualization, the material reset path can create
        # a one-env material buffer and then index it with all visible env ids.
        env_cfg.events.robot_physics_material.mode = "disabled_play"
        env_cfg.events.object_physics_material.mode = "disabled_play"
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    isaac_env = env.unwrapped
    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    count = 0
    num_evals = args_cli.max_steps
    sr = torch.zeros(num_evals + 1, device=args_cli.device, dtype=torch.float32)
    target_step_dt = args_cli.play_dt
    if args_cli.real_time:
        target_step_dt = max(target_step_dt, isaac_env.step_dt / args_cli.speed_scale)
    metrics_file = None
    if args_cli.metrics_path is not None:
        metrics_dir = os.path.dirname(args_cli.metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = open(args_cli.metrics_path, "w", encoding="utf-8")
        metrics_file.write("step,success_rate,mean_success_last_100,mean_success_all\n")

    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=not args_cli.stochastic_policy)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            wait_with_gui_updates(start_time + target_step_dt)
            success_rate = isaac_env.in_success_region.float().mean()
            sr[count] = success_rate
            step = count + 1
            mean_success_last_100 = sr[max(0, step - 100):step].mean()
            mean_success_all = sr[:step].mean()
            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                print("count", step, "sr: ", success_rate, "mean_last_100:", mean_success_last_100)
            if (
                metrics_file is not None
                and args_cli.save_metrics_every > 0
                and (step % args_cli.save_metrics_every == 0 or step == num_evals)
            ):
                metrics_file.write(
                    f"{step},{float(success_rate.detach().cpu())},"
                    f"{float(mean_success_last_100.detach().cpu())},"
                    f"{float(mean_success_all.detach().cpu())}\n"
                )
                metrics_file.flush()
            count += 1

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

            if count >= num_evals:
                break

    if metrics_file is not None:
        metrics_file.close()
    # close the simulator
    env.close()

    print("final sr: ", sr[max(0, num_evals - 100):num_evals].mean())


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
