#!/usr/bin/env python3
"""Build a terminal-state reward-classifier dataset for ARX LeRobot v3 data.

The source ARX dataset labels whole episodes with ``is_failure_data``:
  - 0 means a successful episode
  - 1 means a failed episode

For reward-classifier training, that is too broad because early frames from a
successful episode are not necessarily successful states. This script creates a
smaller dataset that keeps:
  - the last N frames from successful episodes as positive samples
  - the last N frames from failed episodes as negative samples
  - random frames from failed episodes as extra negative samples

It preserves original ``index`` values and adds ``reward_classifier_label``:
  - 1 means terminal success
  - 0 means failure / non-success
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_source_frames(src_root: Path) -> pd.DataFrame:
    parquet_files = sorted((src_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {src_root / 'data'}")

    dfs = []
    for pq in parquet_files:
        df = pd.read_parquet(pq)
        df["_source_parquet"] = str(pq)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    required = ["episode_index", "frame_index", "index", "is_failure_data"]
    missing = [col for col in required if col not in all_df.columns]
    if missing:
        raise KeyError(f"Source dataset missing required columns: {missing}")

    return all_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def select_rows(
    all_df: pd.DataFrame,
    success_tail_frames: int,
    failure_tail_frames: int,
    failure_random_frames_per_episode: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    selected_parts = []
    summary = {
        "success_tail_frames": success_tail_frames,
        "failure_tail_frames": failure_tail_frames,
        "failure_random_frames_per_episode": failure_random_frames_per_episode,
        "seed": seed,
        "episodes": [],
    }

    for ep, ep_df in all_df.groupby("episode_index", sort=True):
        ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
        label_values = sorted(set(ep_df["is_failure_data"].astype(int).tolist()))
        if len(label_values) != 1:
            raise RuntimeError(f"Episode {ep} has mixed is_failure_data labels: {label_values}")

        is_failure = int(label_values[0])
        ep_len = len(ep_df)

        if is_failure == 0:
            n_tail = min(success_tail_frames, ep_len)
            pos = ep_df.tail(n_tail).copy()
            pos["reward_classifier_label"] = 1
            pos["reward_classifier_sample_type"] = "success_tail"
            selected_parts.append(pos)
            summary["episodes"].append(
                {
                    "episode_index": int(ep),
                    "source_label": "success",
                    "selected_success_tail": int(len(pos)),
                    "selected_failure_tail": 0,
                    "selected_failure_random": 0,
                }
            )
        else:
            n_tail = min(failure_tail_frames, ep_len)
            tail = ep_df.tail(n_tail).copy()
            tail["reward_classifier_label"] = 0
            tail["reward_classifier_sample_type"] = "failure_tail"

            remaining = ep_df.drop(index=tail.index, errors="ignore")
            n_random = min(failure_random_frames_per_episode, len(remaining))
            if n_random > 0:
                random_indices = rng.choice(remaining.index.to_numpy(), size=n_random, replace=False)
                random_neg = remaining.loc[random_indices].copy()
            else:
                random_neg = remaining.iloc[:0].copy()

            random_neg["reward_classifier_label"] = 0
            random_neg["reward_classifier_sample_type"] = "failure_random"
            selected_parts.extend([tail, random_neg])
            summary["episodes"].append(
                {
                    "episode_index": int(ep),
                    "source_label": "failure",
                    "selected_success_tail": 0,
                    "selected_failure_tail": int(len(tail)),
                    "selected_failure_random": int(len(random_neg)),
                }
            )

    if not selected_parts:
        raise RuntimeError("No frames selected.")

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    selected = selected.drop(columns=["_source_parquet"], errors="ignore")

    counts = selected["reward_classifier_label"].value_counts().sort_index().to_dict()
    type_counts = selected["reward_classifier_sample_type"].value_counts().sort_index().to_dict()
    summary["total_selected_frames"] = int(len(selected))
    summary["label_counts"] = {str(int(k)): int(v) for k, v in counts.items()}
    summary["sample_type_counts"] = {str(k): int(v) for k, v in type_counts.items()}
    summary["selected_episodes"] = sorted(int(x) for x in selected["episode_index"].unique().tolist())

    selected = selected.drop(columns=["reward_classifier_sample_type"])

    return selected, summary


def copy_dataset_scaffold(src_root: Path, dst_root: Path, selected_count: int, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise RuntimeError(f"Destination exists: {dst_root}. Use --overwrite.")
        print("[WARN] removing old destination:", dst_root)
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    if (src_root / "meta").exists():
        shutil.copytree(src_root / "meta", dst_root / "meta", dirs_exist_ok=True)
    if (src_root / "videos").exists():
        shutil.copytree(src_root / "videos", dst_root / "videos", dirs_exist_ok=True)

    info_path = dst_root / "meta" / "info.json"
    if info_path.exists():
        info = read_json(info_path)
        info["codebase_version"] = "v3.0"
        info["total_frames"] = selected_count
        info["data_path"] = "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet"
        info["video_path"] = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4"
        info["data_files_size_in_mb"] = info.get("data_files_size_in_mb", 100)
        info["video_files_size_in_mb"] = info.get("video_files_size_in_mb", 500)
        info.pop("total_chunks", None)
        info.pop("total_videos", None)
        for feature in info.get("features", {}).values():
            if isinstance(feature, dict) and feature.get("dtype") != "video":
                feature.setdefault("fps", int(info.get("fps", 30)))
        info.setdefault("features", {})
        info["features"]["reward_classifier_label"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": int(info.get("fps", 30)),
        }
        info.pop("notes", None)
        write_json(info_path, info)

    write_tasks_parquet(src_root, dst_root)


def write_tasks_parquet(src_root: Path, dst_root: Path) -> None:
    src_tasks_parquet = src_root / "meta" / "tasks.parquet"
    dst_tasks_parquet = dst_root / "meta" / "tasks.parquet"
    if src_tasks_parquet.exists():
        shutil.copy2(src_tasks_parquet, dst_tasks_parquet)
        return

    src_tasks_jsonl = src_root / "meta" / "tasks.jsonl"
    if not src_tasks_jsonl.exists():
        raise FileNotFoundError(f"Missing tasks metadata: {src_tasks_jsonl}")

    rows = []
    for line in src_tasks_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"No tasks loaded from {src_tasks_jsonl}")

    tasks = pd.DataFrame(rows)
    if "task" not in tasks.columns or "task_index" not in tasks.columns:
        raise KeyError(f"Expected task/task_index columns in {src_tasks_jsonl}")

    tasks = tasks[["task", "task_index"]].drop_duplicates("task").set_index("task")
    tasks.index.name = "task"
    dst_tasks_parquet.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(dst_tasks_parquet)


def write_episodes_parquet(selected: pd.DataFrame, src_root: Path, dst_root: Path) -> None:
    episodes_jsonl = src_root / "meta" / "episodes.jsonl"
    if not episodes_jsonl.exists():
        raise FileNotFoundError(episodes_jsonl)

    legacy_rows = {}
    for line in episodes_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            legacy_rows[int(row["episode_index"])] = row

    episode_rows = []
    running_from = 0
    video_keys = [
        "observation.images.top",
        "observation.images.left_hand",
        "observation.images.right_hand",
    ]

    for ep, ep_df in selected.groupby("episode_index", sort=True):
        ep = int(ep)
        length = int(len(ep_df))
        legacy = legacy_rows.get(ep, {"tasks": ["pick and place"]})
        row = {
            "episode_index": ep,
            "tasks": legacy.get("tasks", ["pick and place"]),
            "length": length,
            "dataset_from_index": running_from,
            "dataset_to_index": running_from + length,
            "data/chunk_index": 0,
            "data/file_index": ep,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        for video_key in video_keys:
            row[f"videos/{video_key}/chunk_index"] = 0
            row[f"videos/{video_key}/file_index"] = ep
            row[f"videos/{video_key}/from_timestamp"] = 0.0
            row[f"videos/{video_key}/to_timestamp"] = float(legacy.get("length", length)) / 30.0

        episode_rows.append(row)
        running_from += length

    episodes_dir = dst_root / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episodes_dir / "file-000.parquet", index=False)


def write_selected_parquets(selected: pd.DataFrame, dst_root: Path) -> None:
    data_dir = dst_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    for ep, ep_df in selected.groupby("episode_index", sort=True):
        out = data_dir / f"episode_{int(ep):06d}.parquet"
        ep_df.to_parquet(out, index=False)


def vector_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    std = values.std(axis=0)
    std = np.maximum(std, 1e-6)
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def write_stats_json(selected: pd.DataFrame, dst_root: Path) -> None:
    stats = {}

    for key in [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "is_failure_data",
        "is_infer_data",
        "reward_classifier_label",
    ]:
        if key not in selected.columns:
            continue

        if key in {"action", "observation.state"}:
            values = np.stack(
                [np.asarray(v, dtype=np.float32).reshape(-1) for v in selected[key].tolist()],
                axis=0,
            )
        else:
            values = selected[key].to_numpy(dtype=np.float32).reshape(-1, 1)

        stats[key] = vector_stats(values)

    # The training config uses ImageNet stats for cameras, but LeRobot still
    # expects a stats entry for every camera key before it overwrites mean/std.
    image_stats = {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.485]], [[0.456]], [[0.406]]],
        "std": [[[0.229]], [[0.224]], [[0.225]]],
        "count": [int(len(selected))],
    }
    for key in [
        "observation.images.top",
        "observation.images.left_hand",
        "observation.images.right_hand",
    ]:
        stats[key] = image_stats

    write_json(dst_root / "meta" / "stats.json", stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30",
    )
    parser.add_argument(
        "--dst",
        default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_reward_classifier_terminal_v30",
    )
    parser.add_argument("--success-tail-frames", type=int, default=30)
    parser.add_argument("--failure-tail-frames", type=int, default=30)
    parser.add_argument("--failure-random-frames-per-episode", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(src_root)
    if args.success_tail_frames <= 0 or args.failure_tail_frames <= 0:
        raise ValueError("Tail frame counts must be positive.")
    if args.failure_random_frames_per_episode < 0:
        raise ValueError("--failure-random-frames-per-episode must be >= 0.")

    all_df = load_source_frames(src_root)
    selected, summary = select_rows(
        all_df=all_df,
        success_tail_frames=args.success_tail_frames,
        failure_tail_frames=args.failure_tail_frames,
        failure_random_frames_per_episode=args.failure_random_frames_per_episode,
        seed=args.seed,
    )

    copy_dataset_scaffold(src_root, dst_root, selected_count=len(selected), overwrite=args.overwrite)
    write_selected_parquets(selected, dst_root)
    write_episodes_parquet(selected, src_root, dst_root)
    write_stats_json(selected, dst_root)

    summary["src"] = str(src_root)
    summary["dst"] = str(dst_root)
    write_json(dst_root / "meta" / "reward_classifier_terminal_subset.json", summary)

    print("================ Reward Classifier Terminal Dataset ================")
    print("[SRC]", src_root)
    print("[DST]", dst_root)
    print("[SELECTED]", len(selected))
    print("[LABEL COUNTS]", summary["label_counts"], "(0=failure/non-success, 1=terminal success)")
    print("[TYPE COUNTS]", summary["sample_type_counts"])
    print("[EPISODES]", summary["selected_episodes"])
    print("[SUMMARY]", dst_root / "meta" / "reward_classifier_terminal_subset.json")
    print("====================================================================")


if __name__ == "__main__":
    main()
