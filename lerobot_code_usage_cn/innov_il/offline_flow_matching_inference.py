"""Run a Flow Matching checkpoint on samples from its offline training dataset.

This diagnostic intentionally prints actions both before and after the saved
postprocessor. It helps distinguish unstable Flow Matching sampling from a
normalization/statistics problem.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch


DEFAULT_CHECKPOINT = (
    "/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/innov/innov_il/follow/"
    "runs/flow_matching_innov_20260618_161409/checkpoints/030000/pretrained_model"
)
DEFAULT_DATASET_ROOT = (
    "/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/innov/innov_0617_1554"
)
DEFAULT_DATASET_REPO_ID = "innov_0617_1554"


def configure_environment() -> None:
    cache_root = "/tmp/lerobot_offline_hf_cache"
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = f"{cache_root}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{cache_root}/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_root}/transformers"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for proxy_key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        os.environ.pop(proxy_key, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--index", type=int, default=100)
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat stochastic Flow Matching sampling on the same observation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-integration-steps",
        type=int,
        default=None,
        help="Optional inference-only override for the checkpoint setting.",
    )
    parser.add_argument(
        "--integration-method",
        choices=["euler", "rk4"],
        default=None,
        help="Optional inference-only override for the checkpoint setting.",
    )
    return parser.parse_args()


def tensor_summary(name: str, value: torch.Tensor) -> None:
    value = value.detach().float().cpu()
    finite = torch.isfinite(value)
    print(
        f"[OFFLINE] {name}: shape={tuple(value.shape)} "
        f"finite={int(finite.sum())}/{value.numel()} "
        f"min={value.min().item():.6f} max={value.max().item():.6f} "
        f"mean={value.mean().item():.6f} std={value.std().item():.6f}"
    )


def action_per_dimension(name: str, value: torch.Tensor) -> None:
    value = value.detach().float().cpu()
    flat = value.reshape(-1, value.shape[-1])
    mins = flat.min(dim=0).values.numpy()
    maxs = flat.max(dim=0).values.numpy()
    means = flat.mean(dim=0).numpy()
    print(f"[OFFLINE] {name} per action dimension:")
    for index, (min_val, max_val, mean_val) in enumerate(zip(mins, maxs, means, strict=True)):
        arm = "left" if index < 7 else "right"
        print(
            f"  dim={index:02d} arm={arm:<5} "
            f"min={min_val: .6f} max={max_val: .6f} mean={mean_val: .6f}"
        )


def scalar(value) -> int | float | str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return str(value.tolist())
    return value


def main() -> None:
    configure_environment()
    args = parse_args()

    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.multi_task_dit import MultiTaskDiTConfig, MultiTaskDiTPolicy

    checkpoint = str(Path(args.checkpoint).expanduser())
    dataset_root = Path(args.dataset_root).expanduser()
    device = torch.device(args.device)

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, MultiTaskDiTConfig):
        raise TypeError(f"Expected MultiTaskDiTConfig, got {type(config).__name__}")
    if config.objective != "flow_matching":
        raise ValueError(f"Expected flow_matching checkpoint, got {config.objective!r}")

    config.device = str(device)
    if args.num_integration_steps is not None:
        config.num_integration_steps = args.num_integration_steps
    if args.integration_method is not None:
        config.integration_method = args.integration_method

    print(f"[OFFLINE] checkpoint={checkpoint}")
    print(f"[OFFLINE] dataset={dataset_root}")
    print(
        f"[OFFLINE] device={device} n_obs_steps={config.n_obs_steps} "
        f"horizon={config.horizon} n_action_steps={config.n_action_steps} "
        f"integration={config.integration_method}/{config.num_integration_steps}"
    )

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index {args.index} is outside dataset length {len(dataset)}")

    sample = dataset[args.index]
    print(
        f"[OFFLINE] index={args.index}/{len(dataset) - 1} "
        f"episode={scalar(sample.get('episode_index', '?'))} "
        f"frame={scalar(sample.get('frame_index', '?'))} "
        f"task={sample.get('task', '?')!r}"
    )

    policy = MultiTaskDiTPolicy.from_pretrained(checkpoint, config=config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    model_input = {}
    for key, value in sample.items():
        if key not in config.input_features and key not in {"action", "task"}:
            continue
        # Temporal dataset samples are shaped (T, ...). The generic
        # AddBatchDimension processor only handles single-frame tensors, so
        # explicitly prepend the offline batch dimension here.
        if isinstance(value, torch.Tensor) and (key in config.input_features or key == "action"):
            value = value.unsqueeze(0)
        model_input[key] = value
    processed = preprocessor(model_input)

    tensor_summary("dataset state (physical)", sample["observation.state"])
    tensor_summary("processed state (normalized)", processed["observation.state"])
    tensor_summary("dataset action target (physical)", sample["action"])
    tensor_summary("processed action target (normalized)", processed["action"])

    # MultiTaskDiT action_delta_indices begin at 1 - n_obs_steps. The model
    # discards the first n_obs_steps - 1 predictions and returns n_action_steps.
    target_start = config.n_obs_steps - 1
    target_end = target_start + config.n_action_steps
    target_actions = sample["action"][target_start:target_end].float().cpu()
    action_per_dimension("aligned dataset target (physical)", target_actions)

    for repeat_index in range(args.repeat):
        torch.manual_seed(args.seed + repeat_index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + repeat_index)

        with torch.inference_mode():
            # The offline sample already contains the complete n_obs_steps
            # temporal window. Calling _generate_actions avoids select_action's
            # online observation queues and exercises the same Flow Matching
            # sampler used by predict_action_chunk.
            prepared = policy._prepare_batch(processed)  # noqa: SLF001
            normalized_actions = policy._generate_actions(prepared)  # noqa: SLF001
            physical_actions = postprocessor(normalized_actions)

        if isinstance(physical_actions, dict):
            physical_actions = physical_actions["action"]
        normalized_actions = normalized_actions.detach().float().cpu()
        physical_actions = physical_actions.detach().float().cpu()

        print(f"\n[OFFLINE] ===== repeat={repeat_index} seed={args.seed + repeat_index} =====")
        tensor_summary("predicted action before postprocessor (normalized)", normalized_actions)
        tensor_summary("predicted action after postprocessor (physical)", physical_actions)
        action_per_dimension("predicted action before postprocessor", normalized_actions)
        action_per_dimension("predicted action after postprocessor", physical_actions)

        prediction = physical_actions.squeeze(0)
        if prediction.shape == target_actions.shape:
            error = prediction - target_actions
            print(
                f"[OFFLINE] target comparison: MAE={error.abs().mean().item():.6f} "
                f"max_abs_error={error.abs().max().item():.6f}"
            )
        print(f"[OFFLINE] first normalized action={normalized_actions[0, 0].numpy()}")
        print(f"[OFFLINE] first physical action={physical_actions[0, 0].numpy()}")
        print(f"[OFFLINE] first target action={target_actions[0].numpy()}")


if __name__ == "__main__":
    main()
