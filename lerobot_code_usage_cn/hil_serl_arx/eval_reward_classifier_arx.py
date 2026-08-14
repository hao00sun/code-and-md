#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def find_latest_run(base_dir: str) -> Path:
    base = Path(base_dir)
    runs = sorted(
        [p for p in base.glob("reward_classifier_official_mapped_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No reward_classifier_official_mapped_* run dirs found under {base}")
    return runs[0]


def find_model_dir(run_or_ckpt: str) -> Path:
    p = Path(run_or_ckpt)

    candidates = []

    if p.is_dir():
        candidates.append(p)
        candidates.append(p / "pretrained_model")
        candidates.append(p / "policy")
        candidates.append(p / "checkpoints" / "last" / "pretrained_model")
        candidates.append(p / "checkpoints" / "last")

        ckpt_root = p / "checkpoints"
        if ckpt_root.exists():
            for d in sorted(ckpt_root.glob("*"), key=lambda x: x.name):
                if d.is_dir():
                    candidates.append(d / "pretrained_model")
                    candidates.append(d)

    for c in candidates:
        if not c.exists() or not c.is_dir():
            continue

        files = {x.name for x in c.iterdir() if x.is_file()}
        if (
            "config.json" in files
            or "model.safetensors" in files
            or "pytorch_model.bin" in files
            or "model.pt" in files
            or "model.pth" in files
        ):
            return c

    # If no clean model dir found, print useful debug info.
    print("\n[ERROR] Could not automatically find a model directory.")
    print("Searched candidates:")
    for c in candidates:
        print("  ", c)

    print("\nExisting files under run dir:")
    if p.exists():
        for f in list(p.rglob("*"))[:200]:
            print("  ", f)
    raise FileNotFoundError(f"Could not find saved model under: {p}")


def find_latest_config(config_base: str) -> Path:
    base = Path(config_base)
    configs = sorted(
        base.glob("reward_classifier_official_mapped_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not configs:
        raise FileNotFoundError(f"No run configs found under {base}")
    return configs[0]


def load_model(model_dir: Path, config_json: Path | None, device: str):
    from lerobot.rewards.classifier.modeling_classifier import Classifier

    # First try official from_pretrained.
    try:
        model = Classifier.from_pretrained(str(model_dir))
        model.to(device)
        model.eval()
        print(f"[OK] Loaded model with Classifier.from_pretrained: {model_dir}")
        return model
    except Exception as e:
        print("[WARN] Classifier.from_pretrained failed, trying fallback state_dict loading.")
        print("       error:", repr(e))

    if config_json is None:
        raise RuntimeError("Fallback loading requires --config-json")

    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig

    cfg = json.loads(Path(config_json).read_text())
    rm_cfg = cfg["reward_model"]

    model = Classifier(RewardClassifierConfig(**rm_cfg))

    # Find state dict.
    state_files = []
    for name in ["model.safetensors", "pytorch_model.bin", "model.pt", "model.pth"]:
        f = model_dir / name
        if f.exists():
            state_files.append(f)

    if not state_files:
        # Search recursively.
        for pattern in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
            state_files.extend(model_dir.rglob(pattern))

    if not state_files:
        raise FileNotFoundError(f"No model weight file found under {model_dir}")

    state_path = state_files[0]
    print(f"[INFO] Loading weights from: {state_path}")

    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(state_path))
    else:
        state = torch.load(str(state_path), map_location="cpu")

    if isinstance(state, dict):
        for key in ["model", "state_dict", "policy", "reward_model"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    clean_state = {}
    for k, v in state.items():
        nk = k
        for prefix in ["module.", "policy.", "reward_model.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"[INFO] load_state_dict missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[INFO] missing examples:", missing[:10])
    if unexpected:
        print("[INFO] unexpected examples:", unexpected[:10])

    model.to(device)
    model.eval()
    return model


def load_label_table(dataset_root: str, label_col: str = "is_failure_data") -> pd.DataFrame:
    root = Path(dataset_root)
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        need_cols = ["index", "episode_index", label_col]
        for c in need_cols:
            if c not in df.columns:
                raise KeyError(f"{f} missing column: {c}")
        dfs.append(df[need_cols])

    out = pd.concat(dfs, ignore_index=True).sort_values("index").reset_index(drop=True)
    out["is_failure_data"] = out[label_col].astype(int)
    out["label"] = 1 - out["is_failure_data"]  # 1=success/non-failure, 0=failure
    return out


def sample_eval_df(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    if max_per_class is None or max_per_class <= 0:
        return df.copy()

    parts = []
    for label in [0, 1]:
        sub = df[df["label"] == label]
        n = min(max_per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def to_image_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    # HWC -> CHW
    if x.ndim == 3 and x.shape[-1] in [1, 3]:
        x = x.permute(2, 0, 1)

    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()

    return x


@torch.no_grad()
def evaluate(model, dataset, eval_df, batch_size: int, device: str, threshold: float):
    model_image_keys = list(model.image_keys)
    print("[INFO] model.image_keys =", model_image_keys)

    # The saved reward classifier may store image keys as:
    #   observation_images_top
    # while LeRobotDataset returns:
    #   observation.images.top
    #
    # So we map model keys back to dataset keys only for reading data.
    def resolve_dataset_key(model_key: str, sample_item: dict) -> str:
        candidates = [
            model_key,
            model_key.replace("observation_images_", "observation.images."),
            model_key.replace("observation.image.", "observation.images."),
        ]

        for c in candidates:
            if c in sample_item:
                return c

        raise KeyError(
            f"Cannot map model image key '{model_key}' to dataset item keys. "
            f"Candidates={candidates}. "
            f"Available dataset keys={list(sample_item.keys())}"
        )

    # Use one sample to build stable key mapping.
    sample_index = int(eval_df.iloc[0]["index"])
    sample_item = dataset[sample_index]
    dataset_image_keys = [resolve_dataset_key(k, sample_item) for k in model_image_keys]

    print("[INFO] dataset_image_keys =", dataset_image_keys)

    rows = []

    for start_idx in tqdm(range(0, len(eval_df), batch_size), desc="Evaluating"):
        batch_df = eval_df.iloc[start_idx:start_idx + batch_size]

        items = [dataset[int(idx)] for idx in batch_df["index"].tolist()]

        images = []
        for dataset_key in dataset_image_keys:
            xs = torch.stack([to_image_tensor(item[dataset_key]) for item in items], dim=0).to(device)
            images.append(xs)

        outputs = model.predict(images)
        logits = outputs.logits.squeeze(-1)
        probs = torch.sigmoid(logits).detach().cpu()

        for i, (_, r) in enumerate(batch_df.iterrows()):
            prob = float(probs[i].item())
            pred = int(prob >= threshold)
            label = int(r["label"])

            rows.append({
                "index": int(r["index"]),
                "episode_index": int(r["episode_index"]),
                "is_failure_data": int(r["is_failure_data"]),
                "label": label,
                "prob_success": prob,
                "pred": pred,
                "correct": int(pred == label),
            })

    return pd.DataFrame(rows)


def print_metrics(result: pd.DataFrame, threshold: float):
    y = result["label"].astype(int)
    p = result["pred"].astype(int)

    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(1, len(result))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    tpr = recall
    tnr = tn / max(1, tn + fp)
    bal_acc = 0.5 * (tpr + tnr)

    success = result[result["label"] == 1]["prob_success"]
    failure = result[result["label"] == 0]["prob_success"]

    print("\n================ Offline Evaluation ================")
    print(f"num_samples: {len(result)}")
    print(f"threshold:   {threshold:.3f}")
    print(f"accuracy:    {acc:.4f}")
    print(f"balanced_acc:{bal_acc:.4f}")
    print(f"precision:   {precision:.4f}   positive=success")
    print(f"recall:      {recall:.4f}   positive=success")
    print(f"f1:          {f1:.4f}")
    print("")
    print("Confusion matrix:")
    print("              pred_failure(0)  pred_success(1)")
    print(f"true_failure(0)   {tn:8d}        {fp:8d}")
    print(f"true_success(1)   {fn:8d}        {tp:8d}")
    print("")
    print(f"failure prob_success mean/std: {failure.mean():.4f} / {failure.std():.4f}")
    print(f"success prob_success mean/std: {success.mean():.4f} / {success.std():.4f}")

    # Threshold sweep.
    print("\nThreshold sweep:")
    best = None
    for th in [i / 10 for i in range(1, 10)]:
        pred = (result["prob_success"] >= th).astype(int)
        tp2 = int(((pred == 1) & (y == 1)).sum())
        tn2 = int(((pred == 0) & (y == 0)).sum())
        fp2 = int(((pred == 1) & (y == 0)).sum())
        fn2 = int(((pred == 0) & (y == 1)).sum())
        acc2 = (tp2 + tn2) / max(1, len(result))
        tpr2 = tp2 / max(1, tp2 + fn2)
        tnr2 = tn2 / max(1, tn2 + fp2)
        bal2 = 0.5 * (tpr2 + tnr2)
        print(f"  th={th:.1f} acc={acc2:.4f} bal_acc={bal2:.4f} tp={tp2} tn={tn2} fp={fp2} fn={fn2}")
        if best is None or bal2 > best[0]:
            best = (bal2, th)

    if best:
        print(f"\nBest threshold by balanced_acc: th={best[1]:.2f}, balanced_acc={best[0]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30")
    parser.add_argument("--run-dir", default=None, help="Training run dir. If omitted, use latest run under hil-serl.")
    parser.add_argument("--base-output-dir", default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl")
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--config-dir", default="/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/hil-serl/configs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-per-class", type=int, default=500, help="<=0 means evaluate all frames")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode ids to evaluate, e.g. 17,18,19,20",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.run_dir is None:
        run_dir = find_latest_run(args.base_output_dir)
    else:
        run_dir = Path(args.run_dir)

    model_dir = find_model_dir(str(run_dir))

    if args.config_json is None:
        try:
            config_json = find_latest_config(args.config_dir)
        except Exception:
            config_json = None
    else:
        config_json = Path(args.config_json)

    print("[INFO] run_dir    =", run_dir)
    print("[INFO] model_dir  =", model_dir)
    print("[INFO] config_json=", config_json)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    model = load_model(model_dir, config_json, device=device)

    df = load_label_table(args.dataset_root)

    if args.episodes is not None:
        episode_ids = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
        df = df[df["episode_index"].isin(episode_ids)].reset_index(drop=True)
        print("[INFO] evaluating only episodes:", episode_ids)
        print("[INFO] num samples after episode filter:", len(df))
        if len(df) == 0:
            raise RuntimeError("No samples left after episode filtering.")
    print("[INFO] full label counts:")
    print(df["label"].value_counts().sort_index().rename(index={0: "failure_label_0", 1: "success_label_1"}))

    eval_df = sample_eval_df(df, args.max_per_class, args.seed)
    print("[INFO] eval label counts:")
    print(eval_df["label"].value_counts().sort_index().rename(index={0: "failure_label_0", 1: "success_label_1"}))

    dataset = LeRobotDataset(
        repo_id="arx_bimanual_0611_1511",
        root=args.dataset_root,
    )

    result = evaluate(
        model=model,
        dataset=dataset,
        eval_df=eval_df,
        batch_size=args.batch_size,
        device=device,
        threshold=args.threshold,
    )

    print_metrics(result, args.threshold)

    if args.out_csv is None:
        out_csv = run_dir / "offline_eval_predictions.csv"
    else:
        out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved predictions to: {out_csv}")


if __name__ == "__main__":
    main()
