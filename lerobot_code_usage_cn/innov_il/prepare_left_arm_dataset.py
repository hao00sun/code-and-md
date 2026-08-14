"""Create a left-arm-only LeRobot dataset from the existing dual-arm dataset."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path(
    "/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/innov/innov_0617_1554"
)
DEFAULT_DESTINATION = Path("/media/wu/data/SUN_ht/innov/datasets/innov_0617_1554_left_arm")
LEFT_ARM_DIM = 7
KEPT_CAMERAS = (
    "observation.images.front",
    "observation.images.left_wrist",
)
REMOVED_CAMERA = "observation.images.right_wrist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination dataset.",
    )
    return parser.parse_args()


def copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.rglob("*"):
        relative_path = source_path.relative_to(source)
        destination_path = destination / relative_path
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def slice_vector(value):
    return list(value)[:LEFT_ARM_DIM]


def convert_data(source: Path, destination: Path) -> None:
    for parquet_path in sorted((source / "data").glob("*/*.parquet")):
        relative_path = parquet_path.relative_to(source)
        output_path = destination / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe = pd.read_parquet(parquet_path)
        for key in ("observation.state", "action"):
            dataframe[key] = dataframe[key].map(slice_vector)
        dataframe.to_parquet(output_path, index=False)
        print(f"[DATA] converted {relative_path}")


def convert_info(source: Path, destination: Path) -> None:
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    features = info["features"]

    for key in ("observation.state", "action"):
        features[key]["shape"] = [LEFT_ARM_DIM]
        if features[key].get("names") is not None:
            features[key]["names"] = features[key]["names"][:LEFT_ARM_DIM]

    features.pop(REMOVED_CAMERA, None)
    (destination / "meta").mkdir(parents=True, exist_ok=True)
    (destination / "meta/info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    )


def convert_stats(source: Path, destination: Path) -> None:
    stats = json.loads((source / "meta/stats.json").read_text())
    for key in ("observation.state", "action"):
        for stat_name, value in stats[key].items():
            if stat_name == "count":
                continue
            stats[key][stat_name] = value[:LEFT_ARM_DIM]
    stats.pop(REMOVED_CAMERA, None)
    (destination / "meta/stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    )


def convert_episode_metadata(source: Path, destination: Path) -> None:
    right_video_prefix = f"videos/{REMOVED_CAMERA}/"
    right_stats_prefix = f"stats/{REMOVED_CAMERA}/"

    for parquet_path in sorted((source / "meta/episodes").glob("*/*.parquet")):
        relative_path = parquet_path.relative_to(source)
        output_path = destination / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe = pd.read_parquet(parquet_path)
        drop_columns = [
            column
            for column in dataframe.columns
            if column.startswith(right_video_prefix) or column.startswith(right_stats_prefix)
        ]
        dataframe = dataframe.drop(columns=drop_columns)

        for feature_key in ("action", "observation.state"):
            prefix = f"stats/{feature_key}/"
            for column in dataframe.columns:
                if column.startswith(prefix) and not column.endswith("/count"):
                    dataframe[column] = dataframe[column].map(slice_vector)

        dataframe.to_parquet(output_path, index=False)
        print(f"[META] converted {relative_path}")


def validate(destination: Path) -> None:
    info = json.loads((destination / "meta/info.json").read_text())
    features = info["features"]
    assert features["observation.state"]["shape"] == [LEFT_ARM_DIM]
    assert features["action"]["shape"] == [LEFT_ARM_DIM]
    assert REMOVED_CAMERA not in features

    data_files = sorted((destination / "data").glob("*/*.parquet"))
    if not data_files:
        raise RuntimeError("No converted parquet files were created")
    sample = pd.read_parquet(data_files[0], columns=["observation.state", "action"]).iloc[0]
    assert len(sample["observation.state"]) == LEFT_ARM_DIM
    assert len(sample["action"]) == LEFT_ARM_DIM

    for camera in KEPT_CAMERAS:
        if not any((destination / "videos" / camera).glob("*/*.mp4")):
            raise RuntimeError(f"Missing video for {camera}")
    if (destination / "videos" / REMOVED_CAMERA).exists():
        raise RuntimeError("Right wrist video directory should not exist")

    print(f"[DONE] left-arm dataset ready: {destination}")


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(destination)

    destination.mkdir(parents=True)
    convert_data(source, destination)
    convert_info(source, destination)
    convert_stats(source, destination)
    convert_episode_metadata(source, destination)
    shutil.copy2(source / "meta/tasks.parquet", destination / "meta/tasks.parquet")

    for camera in KEPT_CAMERAS:
        copy_tree(source / "videos" / camera, destination / "videos" / camera)

    validate(destination)


if __name__ == "__main__":
    main()
