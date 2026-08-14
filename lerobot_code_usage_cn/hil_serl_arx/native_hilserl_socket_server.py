#!/usr/bin/env python3
"""Native LeRobot HIL-SERL over a plain socket action server.

This script intentionally does not import hil_serl_socket_server_buffer_sac.py.
It uses LeRobot's native RL pieces:
  - GaussianActorPolicy
  - SACAlgorithm
  - ReplayBuffer
  - reward classifier

The robot/client transport is a small pickle-over-TCP protocol so the server
does not depend on LeRobot robot classes or robot environment sockets.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.rewards.classifier.modeling_classifier import Classifier
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.utils.constants import ACTION, OBS_STATE


def send_msg(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    received = 0
    while received < n:
        chunk = sock.recv(n - received)
        if not chunk:
            raise ConnectionError("socket connection closed")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> Any:
    header = recv_exact(sock, 4)
    msg_len = struct.unpack("!I", header)[0]
    return pickle.loads(recv_exact(sock, msg_len))


def stack_column(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in df[col].tolist()], axis=0)


def image_to_tensor(img: Any, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"bad image shape: {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        chw = arr
    else:
        chw = np.transpose(arr, (2, 0, 1))
    if chw.dtype == np.uint8:
        x = torch.from_numpy(np.ascontiguousarray(chw)).float() / 255.0
    else:
        x = torch.from_numpy(np.ascontiguousarray(chw)).float()
        if float(x.max()) > 2.0:
            x = x / 255.0
    return x.unsqueeze(0).to(device)


def payload_image_key(model_key: str) -> str:
    key = model_key.lower()
    if "top" in key:
        return "top"
    if "left" in key:
        return "left_hand"
    if "right" in key:
        return "right_hand"
    raise KeyError(f"Cannot map reward model image key to payload key: {model_key}")


@dataclass
class ActionScaler:
    action_min: np.ndarray
    action_max: np.ndarray

    @property
    def action_scale(self) -> np.ndarray:
        return np.maximum(self.action_max - self.action_min, 1e-6).astype(np.float32)

    def raw_to_norm(self, action_raw: np.ndarray) -> np.ndarray:
        action_raw = np.asarray(action_raw, dtype=np.float32)
        return (2.0 * (action_raw - self.action_min) / self.action_scale - 1.0).astype(np.float32)

    def norm_to_raw(self, action_norm: np.ndarray) -> np.ndarray:
        action_norm = np.asarray(action_norm, dtype=np.float32)
        return ((action_norm + 1.0) * 0.5 * self.action_scale + self.action_min).astype(np.float32)


def make_state(qpos: Any, obs_dim: int, device: torch.device) -> dict[str, torch.Tensor]:
    arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
    obs = np.zeros(obs_dim, dtype=np.float32)
    obs[: min(obs_dim, len(arr))] = arr[:obs_dim]
    return {OBS_STATE: torch.from_numpy(obs).unsqueeze(0).to(device)}


def state_to_numpy(state: dict[str, torch.Tensor]) -> np.ndarray:
    return state[OBS_STATE].detach().cpu().numpy().reshape(-1)


def build_offline_buffer(
    dataset_root: str,
    device: str,
    storage_device: str,
    capacity: int | None,
    success_only: bool,
) -> tuple[ReplayBuffer, ActionScaler, int, int]:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    parquet_files = sorted((dataset_root_path / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {dataset_root_path / 'data'}")

    frames = [pd.read_parquet(pq) for pq in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    required = ["observation.state", ACTION, "episode_index", "is_failure_data"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")

    sort_cols = ["episode_index"]
    if "frame_index" in df.columns:
        sort_cols.append("frame_index")
    elif "index" in df.columns:
        sort_cols.append("index")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    all_actions = stack_column(df, ACTION)
    scaler = ActionScaler(action_min=all_actions.min(axis=0), action_max=all_actions.max(axis=0))
    obs_dim = stack_column(df.iloc[:1], "observation.state").shape[1]
    action_dim = all_actions.shape[1]

    transition_count = 0
    for _, ep_df in df.groupby("episode_index"):
        if len(ep_df) < 2:
            continue
        if success_only and int(ep_df["is_failure_data"].iloc[0]) != 0:
            continue
        transition_count += len(ep_df) - 1

    if transition_count == 0:
        raise RuntimeError("No offline transitions available after filtering.")

    buffer = ReplayBuffer(
        capacity=max(capacity or transition_count, transition_count),
        device=device,
        state_keys=[OBS_STATE],
        use_drq=False,
        storage_device=storage_device,
    )

    added = 0
    for _, ep_df in df.groupby("episode_index"):
        ep_df = ep_df.reset_index(drop=True)
        if len(ep_df) < 2:
            continue
        if success_only and int(ep_df["is_failure_data"].iloc[0]) != 0:
            continue

        obs = stack_column(ep_df, "observation.state")
        actions_raw = stack_column(ep_df, ACTION)
        failures = ep_df["is_failure_data"].astype(int).to_numpy()

        for i in range(len(ep_df) - 1):
            state = {OBS_STATE: torch.from_numpy(obs[i]).float().unsqueeze(0)}
            next_state = {OBS_STATE: torch.from_numpy(obs[i + 1]).float().unsqueeze(0)}
            action = torch.from_numpy(scaler.raw_to_norm(actions_raw[i])).float().unsqueeze(0)
            reward = 1.0 if int(failures[i + 1]) == 0 else 0.0
            done = i == len(ep_df) - 2
            buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=done,
            )
            added += 1

    print("[OFFLINE] dataset_root:", dataset_root_path)
    print("[OFFLINE] transitions:", added)
    print("[OFFLINE] obs_dim:", obs_dim)
    print("[OFFLINE] action_dim:", action_dim)
    print("[OFFLINE] action_min:", scaler.action_min)
    print("[OFFLINE] action_max:", scaler.action_max)
    return buffer, scaler, obs_dim, action_dim


class RewardClassifierBridge:
    def __init__(self, pretrained_path: str, device: torch.device, threshold: float):
        self.device = device
        self.threshold = threshold
        model_dir = Path(pretrained_path).expanduser()
        missing = [name for name in ("config.json", "model.safetensors") if not (model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Expected an already-trained reward classifier directory at {model_dir}, "
                f"but missing: {missing}"
            )

        print("[REWARD] loading pretrained reward classifier:", model_dir)
        self.model = Classifier.from_pretrained(str(model_dir))
        self.model.to(device)
        self.model.eval()
        self.image_keys = list(getattr(self.model, "image_keys", []))
        if not self.image_keys:
            self.image_keys = list(self.model.config.input_features.keys())
        print("[REWARD] image_keys:", self.image_keys)

    @torch.inference_mode()
    def compute(self, payload: dict[str, Any]) -> dict[str, Any]:
        images_payload = payload.get("images")
        if not images_payload:
            return {"reward": 0.0, "prob_success": None, "success": False, "has_images": False}

        batch = {}
        for model_key in self.image_keys:
            key = payload_image_key(model_key)
            if key not in images_payload:
                raise KeyError(f"payload['images'] missing key: {key}")
            batch[model_key] = image_to_tensor(images_payload[key], self.device)

        out = self.model.predict([batch[k] for k in self.image_keys])
        probs = out.probabilities
        if probs.ndim == 0:
            prob_success = float(probs.item())
        elif probs.ndim == 1:
            prob_success = float(probs.reshape(-1)[0].item())
        else:
            prob_success = float(probs[:, -1].reshape(-1)[0].item())

        success = prob_success >= self.threshold
        return {
            "reward": 1.0 if success else 0.0,
            "prob_success": prob_success,
            "success": success,
            "has_images": True,
        }


def make_policy(obs_dim: int, action_dim: int, device: str, storage_device: str) -> GaussianActorPolicy:
    cfg = GaussianActorConfig(
        device=device,
        storage_device=storage_device,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(obs_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        vision_encoder_name=None,
        freeze_vision_encoder=True,
        shared_encoder=True,
    )
    return GaussianActorPolicy(cfg).to(device)


def select_action_norm(policy: GaussianActorPolicy, state: dict[str, torch.Tensor], deterministic: bool) -> np.ndarray:
    policy.eval()
    with torch.inference_mode():
        if deterministic:
            obs_features = policy.actor.encoder(state)
            action = torch.tanh(policy.actor.mean_layer(policy.actor.network(obs_features)))
        else:
            action = policy.select_action(state)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def maybe_payload_action_norm(payload: dict[str, Any], scaler: ActionScaler, fallback: np.ndarray) -> np.ndarray:
    for key in ("executed_action", "teleop_action", "previous_action"):
        if key in payload:
            arr = np.asarray(payload[key], dtype=np.float32).reshape(-1)
            if len(arr) == len(fallback):
                if np.all(arr >= -1.05) and np.all(arr <= 1.05):
                    return arr.astype(np.float32)
                return scaler.raw_to_norm(arr)
    return fallback


def train_updates(
    algorithm: SACAlgorithm,
    online_buffer: ReplayBuffer,
    offline_buffer: ReplayBuffer,
    batch_size: int,
    online_ratio: float,
    updates: int,
) -> dict[str, float]:
    if len(online_buffer) == 0 or len(offline_buffer) == 0:
        return {}

    online_bs = max(1, int(round(batch_size * online_ratio)))
    offline_bs = max(1, batch_size - online_bs)
    stats_dict: dict[str, float] = {}

    for _ in range(updates):
        online_batch = online_buffer.sample(online_bs)
        offline_batch = offline_buffer.sample(offline_bs)
        batch = concatenate_batch_transitions(online_batch, offline_batch)

        def batch_iter():
            while True:
                yield batch

        stats = algorithm.update(batch_iter())
        stats_dict = stats.to_log_dict()

    return stats_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30",
    )
    parser.add_argument(
        "--reward_model_path",
        default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl/reward_classifier_ep_split_20260612_091649/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--output_dir", default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl/native_hilserl_socket")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--allowed_client_host", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage_device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--online_ratio", type=float, default=0.5)
    parser.add_argument("--updates_per_transition", type=int, default=1)
    parser.add_argument("--online_buffer_capacity", type=int, default=100000)
    parser.add_argument("--offline_buffer_capacity", type=int, default=None)
    parser.add_argument("--offline_success_only", action="store_true")
    parser.add_argument("--learning_starts", type=int, default=32)
    parser.add_argument("--reward_success_threshold", type=float, default=0.5)
    parser.add_argument("--online_action_mode", choices=["actor", "hold"], default="hold")
    parser.add_argument("--deterministic_actor", action="store_true")
    parser.add_argument("--max_action_delta", type=float, default=0.03)
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    offline_buffer, scaler, obs_dim, action_dim = build_offline_buffer(
        dataset_root=args.dataset_root,
        device=str(device),
        storage_device=args.storage_device,
        capacity=args.offline_buffer_capacity,
        success_only=args.offline_success_only,
    )

    (output_dir / "action_scaler.json").write_text(
        json.dumps(
            {
                "action_min": scaler.action_min.tolist(),
                "action_max": scaler.action_max.tolist(),
                "action_scale": scaler.action_scale.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    policy = make_policy(obs_dim=obs_dim, action_dim=action_dim, device=str(device), storage_device=args.storage_device)
    algo_cfg = SACAlgorithmConfig.from_policy_config(policy.config)
    algorithm = SACAlgorithm(policy=policy, config=algo_cfg)
    algorithm.make_optimizers_and_scheduler()

    online_buffer = ReplayBuffer(
        capacity=args.online_buffer_capacity,
        device=str(device),
        state_keys=[OBS_STATE],
        use_drq=False,
        storage_device=args.storage_device,
    )

    reward_model = RewardClassifierBridge(
        pretrained_path=args.reward_model_path,
        device=device,
        threshold=args.reward_success_threshold,
    )

    config_snapshot = vars(args).copy()
    config_snapshot["device_resolved"] = str(device)
    config_snapshot["obs_dim"] = obs_dim
    config_snapshot["action_dim"] = action_dim
    (output_dir / "run_config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    log_path = output_dir / "native_hilserl_socket_log.csv"
    log_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "time",
            "server_step",
            "client_step",
            "mode",
            "online_buffer",
            "offline_buffer",
            "prob_success",
            "reward",
            "success",
            "has_images",
            "is_intervention",
            "latency_s",
            "loss_critic",
            "loss_actor",
            "action_norm",
        ],
    )
    if log_path.stat().st_size == 0:
        writer.writeheader()
        log_file.flush()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print("================ Native LeRobot HIL-SERL Socket Server ================")
    print("[SERVER] listening:", f"{args.host}:{args.port}")
    print("[SERVER] output_dir:", output_dir)
    print("[SERVER] online_action_mode:", args.online_action_mode)
    print("[SERVER] this server does not use LeRobot robot env/socket transport")
    print("=======================================================================")

    conn: socket.socket | None = None
    last_state: dict[str, torch.Tensor] | None = None
    last_action_norm: np.ndarray | None = None
    server_step = 0
    train_step = 0

    try:
        conn, addr = server.accept()
        if args.allowed_client_host is not None and addr[0] != args.allowed_client_host:
            print(
                f"[SERVER] rejected client {addr[0]}:{addr[1]} "
                f"(allowed_client_host={args.allowed_client_host})"
            )
            send_msg(
                conn,
                {
                    "ok": False,
                    "error": f"client host {addr[0]} is not allowed",
                    "allowed_client_host": args.allowed_client_host,
                },
            )
            conn.close()
            return
        print("[SERVER] client connected:", addr)

        while True:
            t0 = time.time()
            payload = recv_msg(conn)
            client_step = payload.get("client_step", -1)

            state = make_state(payload.get("qpos", np.zeros(obs_dim, dtype=np.float32)), obs_dim, device)
            reward_info = reward_model.compute(payload)
            done = bool(payload.get("done", False) or reward_info["success"])
            truncated = bool(payload.get("truncated", False))
            is_intervention = bool(payload.get("is_intervention", payload.get("intervention", False)))

            if last_state is not None and last_action_norm is not None:
                executed_action_norm = maybe_payload_action_norm(payload, scaler, last_action_norm)
                online_buffer.add(
                    state={k: v.detach().cpu() for k, v in last_state.items()},
                    action=torch.from_numpy(executed_action_norm).float().unsqueeze(0),
                    reward=float(reward_info["reward"]),
                    next_state={k: v.detach().cpu() for k, v in state.items()},
                    done=done,
                    truncated=truncated,
                    complementary_info={"is_intervention": float(is_intervention)},
                )

            train_logs: dict[str, float] = {}
            if len(online_buffer) >= args.learning_starts:
                train_logs = train_updates(
                    algorithm=algorithm,
                    online_buffer=online_buffer,
                    offline_buffer=offline_buffer,
                    batch_size=args.batch_size,
                    online_ratio=args.online_ratio,
                    updates=args.updates_per_transition,
                )
                train_step += args.updates_per_transition

            if args.online_action_mode == "hold":
                qpos = state_to_numpy(state)
                action_raw = qpos[:action_dim].copy()
                action_norm = scaler.raw_to_norm(action_raw)
            else:
                action_norm = select_action_norm(policy, state, deterministic=args.deterministic_actor)
                action_raw = scaler.norm_to_raw(action_norm)
                qpos = state_to_numpy(state)
                if len(qpos) >= action_dim:
                    current = qpos[:action_dim]
                    delta = np.clip(action_raw - current, -args.max_action_delta, args.max_action_delta)
                    action_raw = current + delta
                    action_norm = scaler.raw_to_norm(action_raw)

            reply = {
                "ok": True,
                "action": action_raw.astype(float).tolist(),
                "action_norm": action_norm.astype(float).tolist(),
                "server_step": server_step,
                "client_step": int(client_step) if client_step is not None else -1,
                "mode": args.online_action_mode,
                "prob_success": reward_info["prob_success"],
                "reward": reward_info["reward"],
                "success": reward_info["success"],
                "has_images": reward_info["has_images"],
                "online_buffer": len(online_buffer),
                "offline_buffer": len(offline_buffer),
                "latency_s": float(time.time() - t0),
            }
            send_msg(conn, reply)

            writer.writerow(
                {
                    "time": time.time(),
                    "server_step": server_step,
                    "client_step": client_step,
                    "mode": args.online_action_mode,
                    "online_buffer": len(online_buffer),
                    "offline_buffer": len(offline_buffer),
                    "prob_success": reward_info["prob_success"],
                    "reward": reward_info["reward"],
                    "success": reward_info["success"],
                    "has_images": reward_info["has_images"],
                    "is_intervention": is_intervention,
                    "latency_s": reply["latency_s"],
                    "loss_critic": train_logs.get("loss_critic"),
                    "loss_actor": train_logs.get("loss_actor"),
                    "action_norm": action_norm.tolist(),
                }
            )
            log_file.flush()

            if server_step % 10 == 0:
                print(
                    f"[SERVER] step={server_step} client_step={client_step} "
                    f"mode={args.online_action_mode} online={len(online_buffer)} offline={len(offline_buffer)} "
                    f"prob_success={reward_info['prob_success']} reward={reward_info['reward']} "
                    f"latency={reply['latency_s']:.3f}s"
                )

            if args.save_every > 0 and server_step > 0 and server_step % args.save_every == 0:
                save_dir = output_dir / "checkpoints" / f"step_{server_step:06d}" / "pretrained_model"
                policy.save_pretrained(save_dir)
                print("[SAVE] policy:", save_dir)

            last_state = state
            last_action_norm = action_norm
            if done:
                last_state = None
                last_action_norm = None

            server_step += 1

    except KeyboardInterrupt:
        print("[SERVER] KeyboardInterrupt")
    except ConnectionError:
        print("[SERVER] client disconnected")
    finally:
        final_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
        policy.save_pretrained(final_dir)
        print("[SAVE] final policy:", final_dir)
        if conn is not None:
            conn.close()
        server.close()
        log_file.close()


if __name__ == "__main__":
    main()
