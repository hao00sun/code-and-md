#!/usr/bin/env python
"""Serve a trained LeRobot MultiTaskDiT flow-matching policy.

This entry point reuses the shared MultiTaskDiT socket/WebSocket protocol from
dit_diffusion_server.py, but defaults to the local flow-matching training layout
under /media/wu/data/SUN_ht/dit/runs.
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dit_diffusion_server import (
    decode_wire_value,
    infer_action,
    infer_action_chunk,
    load_pre_post_processors,
    recv_msg,
    run_ws_server,
    send_msg,
    set_hf_env,
)


DEFAULT_RUNS_ROOT = Path("/media/wu/data/SUN_ht/dit/runs")
DEFAULT_HF_CACHE = "/media/wu/data/SUN_ht/dit/cache/huggingface"
REQUIRED_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")


def find_latest_flow_matching_checkpoint(runs_root: Path = DEFAULT_RUNS_ROOT) -> Path | None:
    candidates = sorted(
        runs_root.glob("multi_task_dit_flow_matching_arx_*/checkpoints/*/pretrained_model"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in REQUIRED_FILES):
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    latest_checkpoint = find_latest_flow_matching_checkpoint()
    default_policy_path = str(latest_checkpoint) if latest_checkpoint else ""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_path",
        type=str,
        default=default_policy_path,
        help="Path to a trained MultiTaskDiT flow-matching pretrained_model directory.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--transport", choices=["openpi_ws", "tcp_pickle"], default="openpi_ws")
    parser.add_argument("--action_mode", choices=["chunk", "single"], default="chunk")
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=50,
        help="Return this many actions in chunk mode. The model predicts 24 actions; longer chunks are padded.",
    )
    parser.add_argument("--hf_cache", type=str, default=DEFAULT_HF_CACHE)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_initial_metadata", action="store_true")
    parser.add_argument("--print_action", action="store_true")
    return parser.parse_args()


def load_flow_matching_policy(args: argparse.Namespace):
    if not args.policy_path:
        raise FileNotFoundError(
            "No flow-matching checkpoint found. Pass --policy_path explicitly, for example "
            "/media/wu/data/SUN_ht/dit/runs/<RUN_NAME>/checkpoints/050000/pretrained_model"
        )

    policy_dir = Path(args.policy_path).expanduser()
    missing_files = [filename for filename in REQUIRED_FILES if not (policy_dir / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Incomplete MultiTaskDiT checkpoint at {policy_dir}: missing {missing_files}")

    from lerobot.policies.multi_task_dit import MultiTaskDiTPolicy

    device = torch.device(args.device)
    policy_path = str(policy_dir)
    print(f"[SERVER] Loading MultiTaskDiT flow-matching policy from: {policy_path}")
    policy = MultiTaskDiTPolicy.from_pretrained(policy_path, device=str(device), local_files_only=args.offline)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    if policy.config.objective != "flow_matching":
        raise ValueError(f"Expected objective='flow_matching', got {policy.config.objective!r}")

    preprocessor, postprocessor = load_pre_post_processors(policy, policy_path, device)
    return policy, preprocessor, postprocessor, policy_path, device


def run_tcp_pickle_server(args, policy, preprocessor, postprocessor, policy_path, device) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print(f"[SERVER] Listening on {args.host}:{args.port} transport={args.transport}")
    while True:
        conn, addr = server.accept()
        conn.settimeout(30.0)
        print(f"[SERVER] Connected by {addr}")
        if hasattr(policy, "reset"):
            policy.reset()
        try:
            step = 0
            while True:
                import time
                import numpy as np

                t0 = time.time()
                payload = decode_wire_value(recv_msg(conn))
                if args.action_mode == "chunk":
                    action_result = infer_action_chunk(
                        policy, preprocessor, postprocessor, payload, device, args.chunk_len
                    )
                else:
                    action_result = infer_action(policy, preprocessor, postprocessor, payload, device)

                reply = {
                    "ok": True,
                    "action": action_result.astype(float).tolist(),
                    "actions": action_result.astype(float).tolist(),
                    "action_mode": args.action_mode,
                    "server_step": int(step),
                    "latency_s": float(time.time() - t0),
                }
                send_msg(conn, reply)
                if args.print_action or step % 10 == 0:
                    print(
                        f"[SERVER] step={step} latency={reply['latency_s']:.3f}s "
                        f"mode={args.action_mode} shape={np.asarray(action_result).shape}"
                    )
                step += 1
        except (ConnectionError, TimeoutError, socket.timeout):
            print("[SERVER] client disconnected")
        except KeyboardInterrupt:
            print("[SERVER] keyboard interrupt")
            break
        except Exception as exc:
            import traceback

            print(f"[SERVER] error at step={step}: {repr(exc)}")
            traceback.print_exc()
            try:
                send_msg(conn, {"ok": False, "error": repr(exc)})
            except Exception:
                pass
        finally:
            conn.close()

    server.close()


def main() -> None:
    args = parse_args()
    set_hf_env(args)
    policy, preprocessor, postprocessor, policy_path, device = load_flow_matching_policy(args)

    print("[SERVER] MultiTaskDiT flow-matching policy loaded.")
    print("[SERVER] policy_path:", policy_path)
    print("[SERVER] input_features:", policy.config.input_features)
    print("[SERVER] output_features:", policy.config.output_features)
    print("[SERVER] action_mode:", args.action_mode)
    print("[SERVER] chunk_len:", args.chunk_len)

    if args.transport == "openpi_ws":
        asyncio.run(run_ws_server(args, policy, preprocessor, postprocessor, policy_path, device))
        return

    run_tcp_pickle_server(args, policy, preprocessor, postprocessor, policy_path, device)


if __name__ == "__main__":
    main()
