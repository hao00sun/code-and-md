#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def get_dir_size(path: Path) -> str:
    try:
        out = subprocess.check_output(["du", "-sh", str(path)], text=True)
        return out.strip().split()[0]
    except Exception:
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return f"{total / (1024 ** 3):.3f}G"


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_label_column(df: pd.DataFrame, preferred: str | None):
    if preferred and preferred in df.columns:
        return preferred

    candidates = [
        "is_failure_data",
        "failure",
        "success",
        "reward",
        "label",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        name = c.lower()
        if "failure" in name or "success" in name or "reward" in name or "label" in name:
            return c

    raise RuntimeError(
        "Cannot find label column. Available columns:\n"
        + "\n".join(map(str, df.columns))
    )


def copy_videos_for_episode(src_root: Path, dst_root: Path, old_ep: int, new_ep: int):
    src_video_root = src_root / "videos"
    if not src_video_root.exists():
        return 0

    old_name = f"episode_{old_ep:06d}"
    new_name = f"episode_{new_ep:06d}"

    copied = 0

    for f in src_video_root.rglob(f"{old_name}.*"):
        rel = f.relative_to(src_video_root)
        new_rel = rel.with_name(rel.name.replace(old_name, new_name))
        dst = dst_root / "videos" / new_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        copied += 1

    return copied


def update_meta(src_root, dst_root, selected_old_eps, old_to_new, total_frames, total_videos):
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"

    if not src_meta.exists():
        print("[WARN] source meta directory not found:", src_meta)
        return

    shutil.copytree(src_meta, dst_meta, dirs_exist_ok=True)

    # info.json
    info_path = dst_meta / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))

            for key in ["total_episodes", "num_episodes"]:
                if key in info:
                    info[key] = len(selected_old_eps)

            for key in ["total_frames", "num_frames"]:
                if key in info:
                    info[key] = total_frames

            if "total_videos" in info:
                info["total_videos"] = total_videos

            if "splits" in info and isinstance(info["splits"], dict):
                info["splits"] = {"train": f"0:{len(selected_old_eps)}"}

            info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
            print("[META] updated:", info_path)
        except Exception as e:
            print("[WARN] failed to update info.json:", repr(e))

    # episodes.jsonl
    ep_path = src_meta / "episodes.jsonl"
    if ep_path.exists():
        rows = read_jsonl(ep_path)
        new_rows = []

        selected_set = set(selected_old_eps)

        for row in rows:
            ep = row.get("episode_index", row.get("episode_id", None))
            if ep is None:
                continue
            ep = int(ep)
            if ep not in selected_set:
                continue

            new_row = dict(row)
            new_ep = old_to_new[ep]

            if "episode_index" in new_row:
                new_row["episode_index"] = new_ep
            if "episode_id" in new_row:
                new_row["episode_id"] = new_ep

            new_rows.append(new_row)

        write_jsonl(dst_meta / "episodes.jsonl", new_rows)
        print("[META] updated:", dst_meta / "episodes.jsonl")

    # episodes_stats.jsonl
    stats_path = src_meta / "episodes_stats.jsonl"
    if stats_path.exists():
        rows = read_jsonl(stats_path)
        new_rows = []

        selected_set = set(selected_old_eps)

        for row in rows:
            ep = row.get("episode_index", row.get("episode_id", None))
            if ep is None:
                continue
            ep = int(ep)
            if ep not in selected_set:
                continue

            new_row = dict(row)
            new_ep = old_to_new[ep]

            if "episode_index" in new_row:
                new_row["episode_index"] = new_ep
            if "episode_id" in new_row:
                new_row["episode_id"] = new_ep

            new_rows.append(new_row)

        write_jsonl(dst_meta / "episodes_stats.jsonl", new_rows)
        print("[META] updated:", dst_meta / "episodes_stats.jsonl")

    print("[WARN] global stats may still be copied from source dataset.")
    print("[WARN] If you use this dataset for normalization-sensitive training, recompute stats later.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        type=str,
        default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_success_only_v30",
    )
    parser.add_argument("--label-col", type=str, default="is_failure_data")
    parser.add_argument("--success-value", type=int, default=0)
    parser.add_argument("--min-success-ratio", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(src_root)

    if dst_root.exists():
        if not args.overwrite:
            raise RuntimeError(f"dst exists: {dst_root}. Use --overwrite.")
        print("[WARN] removing old dst:", dst_root)
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted((src_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under: {src_root / 'data'}")

    print("[INFO] src:", src_root)
    print("[INFO] dst:", dst_root)
    print("[INFO] parquet files:", len(parquet_files))

    all_dfs = []
    for pq in parquet_files:
        df = pd.read_parquet(pq)
        df["_source_parquet"] = str(pq)
        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)

    if "episode_index" not in all_df.columns:
        raise RuntimeError("No episode_index column found. This script expects LeRobot v3.0 episode_index.")

    label_col = find_label_column(all_df, args.label_col)

    print("[INFO] label column:", label_col)
    print("[INFO] success value:", args.success_value)
    print("[INFO] min success ratio:", args.min_success_ratio)
    print()
    print("[INFO] total frames:", len(all_df))
    print("[INFO] original label counts:")
    print(all_df[label_col].value_counts().sort_index())
    print()

    selected = []
    skipped = []

    for ep, ep_df in all_df.groupby("episode_index"):
        ep = int(ep)
        labels = ep_df[label_col].astype(int)
        success_ratio = float((labels == args.success_value).mean())
        counts = labels.value_counts().sort_index().to_dict()

        item = {
            "old_ep": ep,
            "num_frames": len(ep_df),
            "success_ratio": success_ratio,
            "label_counts": counts,
        }

        if success_ratio >= args.min_success_ratio:
            selected.append(item)
        else:
            skipped.append(item)

    selected = sorted(selected, key=lambda x: x["old_ep"])
    skipped = sorted(skipped, key=lambda x: x["old_ep"])

    print("================ Selection Summary ================")
    print("selected success episodes:", [x["old_ep"] for x in selected])
    print("skipped episodes:", [x["old_ep"] for x in skipped])
    print()

    if not selected:
        raise RuntimeError(
            "No success episode selected. "
            "Try --min-success-ratio 0.9 or check label values."
        )

    dst_data_dir = dst_root / "data" / "chunk-000"
    dst_data_dir.mkdir(parents=True, exist_ok=True)

    old_to_new = {}
    total_frames = 0
    total_videos = 0

    for new_ep, item in enumerate(selected):
        old_ep = item["old_ep"]
        old_to_new[old_ep] = new_ep

        ep_df = all_df[all_df["episode_index"] == old_ep].copy()

        if "_source_parquet" in ep_df.columns:
            ep_df = ep_df.drop(columns=["_source_parquet"])

        n = len(ep_df)

        ep_df["episode_index"] = new_ep

        if "frame_index" in ep_df.columns:
            ep_df["frame_index"] = list(range(n))

        if "index" in ep_df.columns:
            ep_df["index"] = list(range(total_frames, total_frames + n))

        dst_pq = dst_data_dir / f"episode_{new_ep:06d}.parquet"
        ep_df.to_parquet(dst_pq, index=False)

        copied_videos = copy_videos_for_episode(
            src_root=src_root,
            dst_root=dst_root,
            old_ep=old_ep,
            new_ep=new_ep,
        )

        total_frames += n
        total_videos += copied_videos

        print(
            f"[COPY] old_ep={old_ep:03d} -> new_ep={new_ep:03d} "
            f"frames={n} videos={copied_videos} "
            f"success_ratio={item['success_ratio']:.3f} "
            f"label_counts={item['label_counts']}"
        )

    selected_old_eps = [x["old_ep"] for x in selected]

    update_meta(
        src_root=src_root,
        dst_root=dst_root,
        selected_old_eps=selected_old_eps,
        old_to_new=old_to_new,
        total_frames=total_frames,
        total_videos=total_videos,
    )

    mapping = {
        "src": str(src_root),
        "dst": str(dst_root),
        "label_col": label_col,
        "success_value": args.success_value,
        "min_success_ratio": args.min_success_ratio,
        "old_to_new": old_to_new,
        "selected_old_episodes": selected_old_eps,
        "skipped_old_episodes": [x["old_ep"] for x in skipped],
        "total_success_episodes": len(selected),
        "total_success_frames": total_frames,
        "total_copied_videos": total_videos,
    }

    mapping_path = dst_root / "success_episode_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("================ Output Summary ================")
    print("dst:", dst_root)
    print("success episodes:", len(selected))
    print("success frames:", total_frames)
    print("copied videos:", total_videos)
    print("folder size:", get_dir_size(dst_root))
    print("mapping file:", mapping_path)

    print()
    print("[CHECK] new label counts:")
    new_dfs = []
    for pq in sorted((dst_root / "data").glob("**/*.parquet")):
        df = pd.read_parquet(pq)
        new_dfs.append(df[[label_col, "episode_index"]])
    new_all = pd.concat(new_dfs, ignore_index=True)
    print(new_all[label_col].value_counts().sort_index())

    print()
    print("[CHECK] per-episode label counts:")
    print(pd.crosstab(new_all["episode_index"], new_all[label_col]))

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()
