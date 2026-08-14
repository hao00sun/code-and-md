#!/usr/bin/env python3
"""Back up a LeRobot v3 dataset, then center-crop its videos in place."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET_ROOT = Path(
    "/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/innov/innov_0617_1554"
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def probe_video(path: Path) -> dict[str, int | str]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,codec_name,pix_fmt,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, got {len(streams)}")

    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "codec": str(stream.get("codec_name", "")),
        "pix_fmt": str(stream.get("pix_fmt", "")),
    }


def encode_cropped_video(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    encoder: str,
    crf: int,
    preset: int,
) -> None:
    crop_filter = f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        crop_filter,
        "-an",
        "-c:v",
        encoder,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if encoder == "libsvtav1":
        command.extend(["-preset", str(preset)])
    elif encoder == "libaom-av1":
        command.extend(["-cpu-used", str(preset)])
    command.extend(["-movflags", "+faststart", str(destination)])
    run(command)


def update_info(
    info_path: Path,
    updated_video_keys: set[str],
    width: int,
    height: int,
    codec: str,
) -> None:
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected a LeRobot v3.0 dataset, got {info.get('codebase_version')!r}"
        )

    features = info.get("features", {})
    for key in updated_video_keys:
        feature = features.get(key)
        if not feature or feature.get("dtype") != "video":
            raise KeyError(f"Missing video feature metadata for {key}")
        feature["shape"] = [height, width, 3]
        video_info = feature.setdefault("info", {})
        video_info["video.height"] = height
        video_info["video.width"] = width
        video_info["video.codec"] = codec
        video_info["video.pix_fmt"] = "yuv420p"
        video_info["video.channels"] = 3

    info_path.write_text(json.dumps(info, indent=4) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Back up a LeRobot v3.0 dataset, then center-crop all video features "
            "that are larger than the requested dimensions."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>_backup_before_crop.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--encoder",
        choices=["libsvtav1", "libaom-av1"],
        default="libsvtav1",
    )
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument(
        "--preset",
        type=int,
        default=8,
        help="SVT-AV1 preset or libaom cpu-used value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect planned operations without copying or modifying files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    backup = (
        args.backup_root.expanduser().resolve()
        if args.backup_root
        else root.with_name(f"{root.name}_backup_before_crop")
    )

    if args.width <= 0 or args.height <= 0:
        raise ValueError("Target width and height must be positive")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {root}")

    info_path = root / "meta" / "info.json"
    videos_root = root / "videos"
    if not info_path.is_file() or not videos_root.is_dir():
        raise ValueError(f"Not a LeRobot dataset root: {root}")

    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"This script only accepts v3.0 datasets, got {info.get('codebase_version')!r}"
        )

    video_paths = sorted(videos_root.glob("*/chunk-*/*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No v3.0 video files found under {videos_root}")

    plans: list[tuple[Path, dict[str, int | str]]] = []
    for video_path in video_paths:
        metadata = probe_video(video_path)
        source_width = int(metadata["width"])
        source_height = int(metadata["height"])
        if source_width < args.width or source_height < args.height:
            raise ValueError(
                f"{video_path} is {source_width}x{source_height}, smaller than "
                f"target {args.width}x{args.height}; cropping cannot enlarge images"
            )
        if source_width != args.width or source_height != args.height:
            plans.append((video_path, metadata))

    print(f"[CHECK] dataset: {root}")
    print(f"[CHECK] version: {info['codebase_version']}")
    print(f"[CHECK] videos: {len(video_paths)}")
    print(f"[CHECK] target: {args.width}x{args.height}")
    print(f"[CHECK] videos requiring crop: {len(plans)}")
    for path, metadata in plans:
        print(
            f"[PLAN] {path.relative_to(root)}: "
            f"{metadata['width']}x{metadata['height']} -> {args.width}x{args.height}"
        )

    if args.dry_run:
        print("[DRY RUN] No files were copied or modified.")
        return 0

    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}. Choose another --backup-root; "
            "the script will not overwrite backups."
        )

    print(f"[BACKUP] copying dataset to: {backup}")
    shutil.copytree(root, backup)

    temporary_files: list[tuple[Path, Path, dict[str, int | str]]] = []
    try:
        for index, (source, before) in enumerate(plans, start=1):
            temporary = source.with_name(f".{source.stem}.crop_tmp.mp4")
            print(f"[CROP] {index}/{len(plans)}: {source.relative_to(root)}")
            encode_cropped_video(
                source=source,
                destination=temporary,
                width=args.width,
                height=args.height,
                encoder=args.encoder,
                crf=args.crf,
                preset=args.preset,
            )
            after = probe_video(temporary)
            if (after["width"], after["height"]) != (args.width, args.height):
                raise RuntimeError(
                    f"Unexpected output shape for {temporary}: "
                    f"{after['width']}x{after['height']}"
                )
            if after["frames"] != before["frames"]:
                raise RuntimeError(
                    f"Frame count changed for {source}: "
                    f"{before['frames']} -> {after['frames']}"
                )
            temporary_files.append((source, temporary, after))

        updated_keys: set[str] = set()
        output_codec = "av1"
        for source, temporary, after in temporary_files:
            os.replace(temporary, source)
            updated_keys.add(source.relative_to(videos_root).parts[0])
            output_codec = str(after["codec"])

        update_info(
            info_path=info_path,
            updated_video_keys=updated_keys,
            width=args.width,
            height=args.height,
            codec=output_codec,
        )
    except Exception:
        for _, temporary, _ in temporary_files:
            temporary.unlink(missing_ok=True)
        for temporary in root.glob("videos/*/chunk-*/.*.crop_tmp.mp4"):
            temporary.unlink(missing_ok=True)
        print(f"[ERROR] Processing failed. Original backup is available at: {backup}")
        raise

    print(f"[DONE] cropped dataset: {root}")
    print(f"[DONE] untouched backup: {backup}")
    print("[DONE] all processed videos preserved their original frame counts")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise
