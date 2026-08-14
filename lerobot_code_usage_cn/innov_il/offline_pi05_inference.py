"""Offline sanity-check inference for a PI05 checkpoint on a LeRobot dataset sample.

This script does not connect to a robot. It loads one sample from a local
LeRobot dataset, runs PI05 action-chunk inference, and prints the predicted
action chunk plus the dataset action at that frame.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


HF_CACHE_BASE = "/data/SUN_ht/pi/cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_BASE
os.environ["HF_HUB_CACHE"] = f"{HF_CACHE_BASE}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_BASE}/transformers"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


DEFAULT_CHECKPOINT = (
    "/mnt/bigdata/SUN_ht/runs/pi05_arx_0723_1401_2026-07-24_09-53-01/"
    "checkpoints/000020/pretrained_model"
)
DEFAULT_DATASET_ROOT = "/data/SUN_ht/datasets/arx_0723_1401"
DEFAULT_TOKENIZER = (
    "/data/SUN_ht/pi/cache/huggingface/hub/models--google--paligemma-3b-pt-224/"
    "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
)
DEFAULT_TASK = "Place the camera into the box."

ACTION_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo_id", default="arx_0723_1401")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--num_print_steps", type=int, default=5)
    return parser.parse_args()


def resolve_checkpoint(path: str) -> Path:
    raw = Path(path).expanduser()
    candidates = [raw, raw / "pretrained_model"]
    if raw.name.isdigit():
        candidates.append(raw.with_name(f"{int(raw.name):06d}") / "pretrained_model")
    for candidate in candidates:
        if (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find PI05 pretrained_model at: {path}")


def to_batched_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = value.to(device)
    if value.ndim in {1, 3}:
        return value.unsqueeze(0)
    return value


def extract_action_chunk(action_obj, action_dim: int = 14) -> np.ndarray:
    if isinstance(action_obj, dict):
        action_obj = action_obj.get("action", action_obj.get("actions", action_obj.get("ACTION")))
    if torch.is_tensor(action_obj):
        action_obj = action_obj.detach().float().cpu().numpy()
    actions = np.asarray(action_obj, dtype=np.float32)
    if actions.ndim == 3:
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[-1] < action_dim:
        raise ValueError(f"Expected action dim >= {action_dim}, got {actions.shape}")
    return actions[:, :action_dim]


def main() -> None:
    args = parse_args()
    for proxy_key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        os.environ.pop(proxy_key, None)

    checkpoint = resolve_checkpoint(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Re-run with --device cpu, or run this in a GPU-enabled shell.")
    device = torch.device(args.device)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    print(f"[OFFLINE] checkpoint = {checkpoint}")
    print(f"[OFFLINE] dataset    = {args.dataset_root}")
    print(f"[OFFLINE] sample     = {args.sample_index}")
    print(f"[OFFLINE] task       = {args.task}")
    print(f"[OFFLINE] device     = {device}")

    dataset = LeRobotDataset(repo_id=args.repo_id, root=Path(args.dataset_root))
    sample = dataset[args.sample_index]

    policy = PI05Policy.from_pretrained(str(checkpoint), device=str(device))
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        str(checkpoint),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": args.tokenizer_path},
            "device_processor": {"device": str(device)},
        },
    )

    batch = {"task": [args.task]}
    for key in policy.config.input_features:
        if key == "observation.state":
            batch[key] = to_batched_tensor(sample[key].float(), device)
        elif key.startswith("observation.images."):
            batch[key] = to_batched_tensor(sample[key].float(), device)

    with torch.inference_mode():
        processed = preprocessor(batch)
        action_obj = policy.predict_action_chunk(processed)
        action_obj = postprocessor(action_obj)

    actions = extract_action_chunk(action_obj, action_dim=14)
    gt_action = sample["action"].detach().float().cpu().numpy()[:14]

    print(f"[OFFLINE] predicted actions shape = {actions.shape}")
    print(f"[OFFLINE] gt action shape          = {gt_action.shape}")
    print("[OFFLINE] gt action:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {i:02d} {name:14s} {gt_action[i]: .6f}")

    print(f"[OFFLINE] first {min(args.num_print_steps, len(actions))} predicted actions:")
    for t, action in enumerate(actions[: args.num_print_steps]):
        print(f"  step {t:02d}: " + ", ".join(f"{x:.6f}" for x in action))

    first = actions[0]
    print("[OFFLINE] first-step pred - gt:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {i:02d} {name:14s} pred={first[i]: .6f} gt={gt_action[i]: .6f} diff={first[i] - gt_action[i]: .6f}")


if __name__ == "__main__":
    main()
