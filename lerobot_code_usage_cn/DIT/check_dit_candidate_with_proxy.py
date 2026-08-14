#!/usr/bin/env python
"""Check whether a Hugging Face repo looks like a LeRobot MultiTaskDiT checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError


REQUIRED_FILES = {
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
}


def check_model(model_id: str) -> bool:
    api = HfApi()
    try:
        files = set(api.list_repo_files(model_id, repo_type="model"))
    except (HfHubHTTPError, RepositoryNotFoundError) as exc:
        print(f"model_id: {model_id}")
        print(f"ERROR: cannot list repo files: {exc}")
        return False

    missing = sorted(REQUIRED_FILES - files)

    print(f"model_id: {model_id}")
    print(f"required files: {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
    if "config.json" not in files:
        print("\nVERDICT: not ideal")
        print("- config.json is missing")
        return False

    try:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = hf_hub_download(
                model_id,
                filename="config.json",
                repo_type="model",
                local_dir=tmp,
            )
            config = json.loads(Path(config_path).read_text())
    except (EntryNotFoundError, HfHubHTTPError) as exc:
        print("\nVERDICT: not ideal")
        print(f"- could not download config.json: {exc}")
        return False

    print(f"type: {config.get('type')}")
    print(f"objective: {config.get('objective')}")
    print(f"n_obs_steps: {config.get('n_obs_steps')}")
    print(f"horizon: {config.get('horizon')}")
    print(f"n_action_steps: {config.get('n_action_steps')}")

    input_features = config.get("input_features") or {}
    output_features = config.get("output_features") or {}
    action_feature = output_features.get("action") or {}
    state_feature = input_features.get("observation.state") or {}
    image_keys = [key for key in input_features if key.startswith("observation.images.")]

    print(f"state shape: {state_feature.get('shape')}")
    print(f"action shape: {action_feature.get('shape')}")
    print(f"image keys: {image_keys}")

    problems = []
    if missing:
        problems.append("missing required LeRobot checkpoint files")
    if config.get("type") != "multi_task_dit":
        problems.append("config type is not multi_task_dit")
    if config.get("objective") != "diffusion":
        problems.append("objective is not diffusion")
    if action_feature.get("shape") not in ([14], (14,)):
        problems.append("action shape is not [14]")
    if state_feature.get("shape") not in ([14], (14,)):
        problems.append("state shape is not [14]")

    if problems:
        print("\nVERDICT: not ideal")
        for problem in problems:
            print(f"- {problem}")
        return False

    print("\nVERDICT: good candidate")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", nargs="+", help="Hugging Face model id(s), e.g. user/model")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--hf-cache", default="/media/wu/data/SUN_ht/dit/cache/huggingface")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HF_HUB_CACHE"] = f"{args.hf_cache}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{args.hf_cache}/transformers"
    os.environ["http_proxy"] = args.proxy
    os.environ["https_proxy"] = args.proxy
    os.environ["HTTP_PROXY"] = args.proxy
    os.environ["HTTPS_PROXY"] = args.proxy
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    good = []
    for index, model_id in enumerate(args.model_id):
        if index:
            print("\n" + "=" * 80 + "\n")
        if check_model(model_id):
            good.append(model_id)

    if len(args.model_id) > 1:
        print("\n" + "=" * 80)
        print("GOOD CANDIDATES:")
        if good:
            for model_id in good:
                print(model_id)
        else:
            print("(none)")


if __name__ == "__main__":
    main()
