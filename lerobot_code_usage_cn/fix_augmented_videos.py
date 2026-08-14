#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复 LeRobot 数据集视频时间戳问题。

用途：
    修复 arx_bimanual_augmented_fixed_v21/videos 下所有 mp4，
    解决 v2.1 -> v3.0 转换时出现的：
    non monotonically increasing dts

默认修复路径：
    /media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_augmented_fixed_v21

运行：
    python fix_augmented_videos.py

如果普通修复后仍然转换失败，可以强制固定 fps：
    python fix_augmented_videos.py --fps 30
"""

from pathlib import Path
import argparse
import subprocess
import sys


DEFAULT_FIX_ROOT = Path(
    "/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/"
    "datasets/arx5/arx_bimanual_augmented_fixed_v21"
)


def run_cmd(cmd: list[str]) -> None:
    """运行 shell 命令，失败时直接退出。"""
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败，返回码：{result.returncode}")


def build_ffmpeg_cmd(
    input_video: Path,
    output_video: Path,
    fps: int | None,
    crf: int,
    preset: str,
) -> list[str]:
    """
    生成 ffmpeg 命令。

    fps=None:
        普通修复，只重新生成时间戳和重新编码。

    fps=30:
        强制把视频转换为固定 fps，更适合处理严重时间戳异常。
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-an",
    ]

    if fps is not None:
        cmd += [
            "-vf",
            f"fps={fps}",
        ]

    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_video),
    ]

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="修复 LeRobot 数据集 mp4 视频时间戳问题"
    )

    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_FIX_ROOT),
        help="要修复的数据集根目录，里面应该包含 videos/ 目录",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="可选。强制固定 fps，例如 --fps 30。不填则不强制改 fps。",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="视频质量参数，越小质量越高、文件越大。默认 18。",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        help="x264 编码速度参数。默认 veryfast。",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要处理的视频，不实际修改。",
    )

    args = parser.parse_args()

    fix_root = Path(args.root).expanduser().resolve()
    video_root = fix_root / "videos"

    print("=" * 100)
    print("数据集根目录:", fix_root)
    print("视频目录:", video_root)
    print("强制 fps:", args.fps)
    print("crf:", args.crf)
    print("preset:", args.preset)
    print("dry-run:", args.dry_run)
    print("=" * 100)

    if not fix_root.exists():
        print("错误：数据集根目录不存在：", fix_root)
        sys.exit(1)

    if not video_root.exists():
        print("错误：videos 目录不存在：", video_root)
        sys.exit(1)

    videos = sorted(video_root.rglob("*.mp4"))

    # 排除上次中断可能留下来的临时文件
    videos = [
        v for v in videos
        if not v.name.endswith(".fixed.mp4")
    ]

    print(f"找到 mp4 视频数量: {len(videos)}")

    if len(videos) == 0:
        print("错误：没有找到任何 mp4 视频。")
        sys.exit(1)

    # 路径安全检查，避免出现 /wu/... 这种异常路径
    for v in videos:
        resolved = v.resolve()
        if not str(resolved).startswith(str(fix_root)):
            print("错误：发现异常视频路径：", resolved)
            sys.exit(1)

    if args.dry_run:
        print("\n以下是将要处理的视频：")
        for v in videos:
            print(v)
        print("\ndry-run 完成，没有修改任何文件。")
        return

    failed_files: list[Path] = []

    for idx, video in enumerate(videos, start=1):
        video = video.resolve()
        tmp = video.with_name(video.stem + ".fixed.mp4")

        print("\n" + "-" * 100)
        print(f"[{idx}/{len(videos)}] fixing:")
        print(video)

        if tmp.exists():
            print("发现旧的临时文件，删除：", tmp)
            tmp.unlink()

        cmd = build_ffmpeg_cmd(
            input_video=video,
            output_video=tmp,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
        )

        try:
            run_cmd(cmd)
        except Exception as e:
            print("修复失败：", video)
            print("错误信息：", e)
            failed_files.append(video)

            if tmp.exists():
                tmp.unlink()

            continue

        if not tmp.exists():
            print("修复失败：没有生成临时文件：", tmp)
            failed_files.append(video)
            continue

        # 用修复后的视频替换原视频
        tmp.replace(video)
        print("完成：", video)

    print("\n" + "=" * 100)

    if failed_files:
        print("有视频修复失败，数量：", len(failed_files))
        for f in failed_files:
            print("FAILED:", f)
        sys.exit(1)

    print("所有 mp4 修复完成。")
    print("下一步可以重新执行 v2.1 -> v3.0 转换。")


if __name__ == "__main__":
    main()