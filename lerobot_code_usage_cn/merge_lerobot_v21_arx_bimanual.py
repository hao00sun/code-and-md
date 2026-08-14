#!/usr/bin/env python3
"""Merge two LeRobot v2.1 ARX bimanual datasets into one v2.1 dataset.

The merge keeps the v2.1 layout:

    data/chunk-000/episode_000000.parquet
    videos/chunk-000/{video_key}/episode_000000.mp4
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    meta/tasks.jsonl
    meta/info.json

It renumbers episodes from 0..N-1 and rewrites parquet columns:

    episode_index -> new episode index
    index         -> global contiguous frame index
    task_index    -> remapped task index

frame_index and timestamp stay episode-local.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SOURCES = [
    "/media/wu/data/SUN_ht/datasets/arx_bimanual_0624_1640_old",
    "/media/wu/data/SUN_ht/datasets/arx_bimanual_0625_1524",
]
DEFAULT_OUTPUT = "/media/wu/data/SUN_ht/datasets/arx_bimanual_0624_0625_v21_merged"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def comparable_features(info: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must match for safe v2.1 merge."""
    return info["features"]


def episode_chunk(ep_idx: int, chunks_size: int) -> int:
    return ep_idx // chunks_size


def validate_sources(sources: list[Path]) -> None:
    if len(sources) < 2:
        raise ValueError("At least two source datasets are required.")

    base_info: dict[str, Any] | None = None
    for root in sources:
        info_path = root / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        info = read_json(info_path)
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{root} is {info.get('codebase_version')}, expected v2.1")
        if base_info is None:
            base_info = info
            continue

        checks = ["fps", "robot_type", "data_path", "video_path", "chunks_size"]
        for key in checks:
            if info.get(key) != base_info.get(key):
                raise ValueError(f"{root}: info['{key}'] differs from the first dataset.")

        if comparable_features(info) != comparable_features(base_info):
            raise ValueError(f"{root}: features differ from the first dataset.")


def load_tasks(root: Path) -> tuple[list[dict[str, Any]], dict[int, int], dict[str, int]]:
    rows = read_jsonl(root / "meta/tasks.jsonl")
    task_to_new_idx: dict[str, int] = {}
    old_to_new: dict[int, int] = {}
    for row in rows:
        task = row["task"]
        if task not in task_to_new_idx:
            task_to_new_idx[task] = len(task_to_new_idx)
        old_to_new[int(row["task_index"])] = task_to_new_idx[task]
    merged_rows = [{"task_index": idx, "task": task} for task, idx in sorted(task_to_new_idx.items(), key=lambda x: x[1])]
    return merged_rows, old_to_new, task_to_new_idx


def merge_datasets(sources: list[Path], output: Path, overwrite: bool = False) -> None:
    validate_sources(sources)

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    base_info = read_json(sources[0] / "meta/info.json")
    chunks_size = int(base_info.get("chunks_size", 1000))
    video_keys = [key for key, value in base_info["features"].items() if value.get("dtype") == "video"]

    output.mkdir(parents=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    all_task_rows: list[dict[str, Any]] = []
    task_name_to_new_idx: dict[str, int] = {}
    source_task_maps: list[dict[int, int]] = []

    for src in sources:
        rows = read_jsonl(src / "meta/tasks.jsonl")
        old_to_new: dict[int, int] = {}
        for row in rows:
            task = row["task"]
            if task not in task_name_to_new_idx:
                task_name_to_new_idx[task] = len(task_name_to_new_idx)
                all_task_rows.append({"task_index": task_name_to_new_idx[task], "task": task})
            old_to_new[int(row["task_index"])] = task_name_to_new_idx[task]
        source_task_maps.append(old_to_new)

    merged_episodes: list[dict[str, Any]] = []
    merged_episode_stats: list[dict[str, Any]] = []
    total_frames = 0
    total_episodes = 0

    for src_idx, src in enumerate(sources):
        info = read_json(src / "meta/info.json")
        episodes = {int(row["episode_index"]): row for row in read_jsonl(src / "meta/episodes.jsonl")}
        episode_stats = {
            int(row["episode_index"]): row for row in read_jsonl(src / "meta/episodes_stats.jsonl")
        }
        task_map = source_task_maps[src_idx]

        for old_ep_idx in sorted(episodes):
            ep_row = dict(episodes[old_ep_idx])
            new_ep_idx = total_episodes
            new_chunk = episode_chunk(new_ep_idx, chunks_size)

            src_chunk = episode_chunk(old_ep_idx, int(info.get("chunks_size", chunks_size)))
            src_data = src / f"data/chunk-{src_chunk:03d}/episode_{old_ep_idx:06d}.parquet"
            dst_data = output / f"data/chunk-{new_chunk:03d}/episode_{new_ep_idx:06d}.parquet"
            if not src_data.is_file():
                raise FileNotFoundError(src_data)

            df = pd.read_parquet(src_data)
            length = int(len(df))
            expected_length = int(ep_row["length"])
            if length != expected_length:
                raise ValueError(f"{src_data}: parquet length {length} != meta length {expected_length}")

            df = df.copy()
            df["episode_index"] = new_ep_idx
            df["index"] = range(total_frames, total_frames + length)
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].map(lambda x: task_map[int(x)])
            dst_data.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_data, index=False)

            for video_key in video_keys:
                src_video = (
                    src
                    / f"videos/chunk-{src_chunk:03d}/{video_key}/episode_{old_ep_idx:06d}.mp4"
                )
                dst_video = (
                    output
                    / f"videos/chunk-{new_chunk:03d}/{video_key}/episode_{new_ep_idx:06d}.mp4"
                )
                if not src_video.is_file():
                    raise FileNotFoundError(src_video)
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)

            ep_row["episode_index"] = new_ep_idx
            ep_row["tasks"] = sorted(set(ep_row.get("tasks", [])))
            ep_row["length"] = length
            merged_episodes.append(ep_row)

            stats_row = dict(episode_stats[old_ep_idx])
            stats_row["episode_index"] = new_ep_idx
            merged_episode_stats.append(stats_row)

            total_frames += length
            total_episodes += 1

    merged_info = dict(base_info)
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(all_task_rows)
    merged_info["total_chunks"] = episode_chunk(total_episodes - 1, chunks_size) + 1 if total_episodes else 0
    merged_info["total_videos"] = len(video_keys)
    merged_info["splits"] = {"train": f"0:{total_episodes}"}

    write_json(output / "meta/info.json", merged_info)
    write_jsonl(output / "meta/tasks.jsonl", all_task_rows)
    write_jsonl(output / "meta/episodes.jsonl", merged_episodes)
    write_jsonl(output / "meta/episodes_stats.jsonl", merged_episode_stats)

    print(f"[OK] merged v2.1 dataset: {output}")
    print(f"[OK] episodes={total_episodes} frames={total_frames} tasks={len(all_task_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [Path(src).expanduser().resolve() for src in args.sources]
    output = Path(args.output).expanduser().resolve()
    merge_datasets(sources, output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
