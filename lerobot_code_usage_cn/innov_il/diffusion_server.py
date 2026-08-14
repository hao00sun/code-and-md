"""Serve a LeRobot Diffusion Policy while a remote client executes actions.

The wire protocol intentionally matches ``pi05_server.py``:

- ``openpi_ws``: msgpack-numpy messages over WebSocket.
- ``tcp_pickle``: a 4-byte length prefix followed by a pickle payload.

The client sends one observation containing ``qpos``/``state`` and the three
camera images. The server returns either one action or a complete action chunk.
"""

import argparse
import asyncio
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

# Allow this script to reuse the protocol helpers in the repository-root
# pi05_server.py regardless of the caller's current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi05_server import (
    build_observation,
    extract_action,
    extract_action_chunk,
    load_pre_post_processors,
    recv_msg,
    send_msg,
)


DEFAULT_POLICY_PATH = (
    "/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/innov/innov_il/deffusion/"
    "runs/diffusion_innov_20260618_105502/checkpoints/005000/pretrained_model"
)

_WARNED_IMAGE_SHAPES = set()


def align_observation_images(policy, obs):
    """Match client camera frames to the image shapes used during training.

    The original INNOV front camera is 848x480, while both wrist cameras are
    640x480. The training dataset center-cropped the front image to 640x480.
    This function reproduces that operation before DiffusionPolicy stacks the
    three camera tensors.
    """
    for key, feature in policy.config.image_features.items():
        image = obs[key]
        target_height, target_width = feature.shape[-2:]
        source_height, source_width = image.shape[-2:]

        if (source_height, source_width) == (target_height, target_width):
            continue

        warning_key = (key, source_height, source_width, target_height, target_width)
        if warning_key not in _WARNED_IMAGE_SHAPES:
            print(
                f"[SERVER] adapting {key}: "
                f"{source_width}x{source_height} -> {target_width}x{target_height}"
            )
            _WARNED_IMAGE_SHAPES.add(warning_key)

        target_ratio = target_width / target_height
        source_ratio = source_width / source_height

        # Center-crop to the training aspect ratio. For the original front
        # camera this is exactly 848x480 -> centered 640x480.
        if source_ratio > target_ratio:
            crop_width = max(1, round(source_height * target_ratio))
            left = max(0, (source_width - crop_width) // 2)
            image = image[..., :, left : left + crop_width]
        elif source_ratio < target_ratio:
            crop_height = max(1, round(source_width / target_ratio))
            top = max(0, (source_height - crop_height) // 2)
            image = image[..., top : top + crop_height, :]

        if image.shape[-2:] != (target_height, target_width):
            image = F.interpolate(
                image,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        obs[key] = image

    return obs


def limit_action_delta(actions, current_state, max_action_delta):
    """Limit each absolute action relative to the preceding joint position."""
    actions = np.asarray(actions, dtype=np.float32)
    if max_action_delta <= 0:
        return actions

    limited = actions.copy()
    previous = np.asarray(current_state, dtype=np.float32).reshape(-1)
    is_single = limited.ndim == 1
    sequence = limited.reshape(1, -1) if is_single else limited

    for index in range(sequence.shape[0]):
        delta = np.clip(
            sequence[index] - previous,
            -max_action_delta,
            max_action_delta,
        )
        sequence[index] = previous + delta
        previous = sequence[index]

    return sequence[0] if is_single else sequence


def infer_single_action(
    policy,
    preprocessor,
    postprocessor,
    payload,
    device,
    max_action_delta,
):
    """Update history and return the next action from the policy action queue."""
    obs = build_observation(payload, device)
    obs = align_observation_images(policy, obs)
    current_state = obs["observation.state"][0].detach().cpu().numpy()
    generated_new_chunk = len(policy._queues["action"]) == 0  # noqa: SLF001
    with torch.inference_mode():
        processed_obs = preprocessor(obs)
        action = policy.select_action(processed_obs)
        action = postprocessor(action)
    raw_action = extract_action(action)
    limited_action = limit_action_delta(raw_action, current_state, max_action_delta)
    diagnostics = build_action_diagnostics(
        current_state,
        raw_action,
        limited_action,
        max_action_delta,
        generated_new_chunk=generated_new_chunk,
    )
    return limited_action, diagnostics


def infer_action_chunk(
    policy,
    preprocessor,
    postprocessor,
    payload,
    device,
    max_action_delta,
):
    """Generate a chunk through DiffusionPolicy's native queue management."""
    from lerobot.utils.constants import ACTION

    obs = build_observation(payload, device)
    obs = align_observation_images(policy, obs)
    current_state = obs["observation.state"][0].detach().cpu().numpy()
    with torch.inference_mode():
        processed_obs = preprocessor(obs)

        # select_action performs the camera stacking, initializes/updates the
        # n_obs_steps history, generates a fresh chunk, and returns its first
        # action. The remaining actions are kept in the policy action queue.
        first_action = policy.select_action(processed_obs)
        remaining_actions = list(policy._queues[ACTION])  # noqa: SLF001
        actions = torch.stack([first_action, *remaining_actions], dim=1)

        # A chunk-mode client executes all returned actions itself. Clear only
        # the server-side action queue so the next request uses its new
        # observation and generates a fresh chunk; keep the observation history.
        policy._queues[ACTION].clear()  # noqa: SLF001
        actions = postprocessor(actions)

    actions = extract_action_chunk(actions)
    expected_steps = policy.config.n_action_steps
    if actions.shape[0] < expected_steps:
        padding = np.repeat(actions[-1:], expected_steps - actions.shape[0], axis=0)
        actions = np.concatenate([actions, padding], axis=0)
    actions = actions[:expected_steps]
    limited_actions = limit_action_delta(actions, current_state, max_action_delta)
    diagnostics = build_action_diagnostics(
        current_state,
        actions[0],
        limited_actions[0],
        max_action_delta,
        generated_new_chunk=True,
    )
    diagnostics["raw_chunk_max_step_delta"] = float(
        np.max(np.abs(np.diff(np.vstack([current_state, actions]), axis=0)))
    )
    diagnostics["limited_chunk_max_step_delta"] = float(
        np.max(np.abs(np.diff(np.vstack([current_state, limited_actions]), axis=0)))
    )
    return limited_actions, diagnostics


def build_action_diagnostics(
    current_state,
    raw_action,
    limited_action,
    max_action_delta,
    generated_new_chunk,
):
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    limited_action = np.asarray(limited_action, dtype=np.float32).reshape(-1)
    raw_delta = raw_action - current_state
    limited_delta = limited_action - current_state
    clipped_mask = np.abs(raw_delta - limited_delta) > 1e-6

    return {
        "current_qpos": current_state,
        "raw_action": raw_action,
        "limited_action": limited_action,
        "raw_delta": raw_delta,
        "limited_delta": limited_delta,
        "raw_max_abs_delta": float(np.max(np.abs(raw_delta))),
        "limited_max_abs_delta": float(np.max(np.abs(limited_delta))),
        "clipped_joint_indices": np.flatnonzero(clipped_mask).astype(np.int64),
        "max_action_delta": float(max_action_delta),
        "generated_new_chunk": bool(generated_new_chunk),
    }


def infer(
    policy,
    preprocessor,
    postprocessor,
    payload,
    device,
    action_mode,
    max_action_delta,
):
    if action_mode == "chunk":
        return infer_action_chunk(
            policy,
            preprocessor,
            postprocessor,
            payload,
            device,
            max_action_delta,
        )
    return infer_single_action(
        policy,
        preprocessor,
        postprocessor,
        payload,
        device,
        max_action_delta,
    )


def build_metadata(policy, policy_path, action_mode):
    return {
        "model": "lerobot_diffusion",
        "policy_path": policy_path,
        "action_mode": action_mode,
        "n_obs_steps": int(policy.config.n_obs_steps),
        "horizon": int(policy.config.horizon),
        "n_action_steps": int(policy.config.n_action_steps),
        "num_inference_steps": int(policy.config.num_inference_steps),
        "input_features": {key: tuple(value.shape) for key, value in policy.config.input_features.items()},
        "output_features": {key: tuple(value.shape) for key, value in policy.config.output_features.items()},
    }


def make_reply(action_result, action_mode, step, started_at):
    result = np.asarray(action_result, dtype=np.float32)
    return {
        "ok": True,
        "actions": result,
        "action": result,
        "action_mode": action_mode,
        "server_step": int(step),
        "latency_s": float(time.time() - started_at),
    }


def log_result(step, reply, diagnostics, request_interval_s, print_action):
    if not print_action and step % 10 != 0:
        return

    result = np.asarray(reply["action"])
    if reply["action_mode"] == "chunk":
        description = f"actions_shape={result.shape} first_action={result[0]}"
    else:
        description = f"action={result}"
    print(
        f"[SERVER] step={step} request_interval_s={request_interval_s:.4f} "
        f"latency={reply['latency_s']:.3f}s "
        f"mode={reply['action_mode']} {description}"
    )
    print(
        f"[SERVER] diagnostic step={step} "
        f"new_chunk={diagnostics['generated_new_chunk']} "
        f"raw_max_abs_delta={diagnostics['raw_max_abs_delta']:.6f} "
        f"limited_max_abs_delta={diagnostics['limited_max_abs_delta']:.6f} "
        f"clipped_joints={diagnostics['clipped_joint_indices'].tolist()}"
    )
    print(f"[SERVER] current_qpos={diagnostics['current_qpos']}")
    print(f"[SERVER] raw_action={diagnostics['raw_action']}")
    print(f"[SERVER] raw_delta={diagnostics['raw_delta']}")
    print(f"[SERVER] limited_action={diagnostics['limited_action']}")
    print(f"[SERVER] limited_delta={diagnostics['limited_delta']}")
    if "raw_chunk_max_step_delta" in diagnostics:
        print(
            f"[SERVER] chunk_delta raw_max={diagnostics['raw_chunk_max_step_delta']:.6f} "
            f"limited_max={diagnostics['limited_chunk_max_step_delta']:.6f}"
        )


async def run_websocket_server(args, policy, preprocessor, postprocessor, policy_path, device):
    import msgpack_numpy as msgpack
    import websockets

    state = {"step": 0, "last_request_time": None}
    inference_lock = asyncio.Lock()

    async def handler(websocket):
        address = getattr(websocket, "remote_address", None)
        print(f"[SERVER] websocket connected: {address}")
        policy.reset()

        if not args.no_initial_metadata:
            metadata = build_metadata(policy, policy_path, args.action_mode)
            await websocket.send(msgpack.packb(metadata, use_bin_type=True))

        try:
            async for message in websocket:
                if isinstance(message, str):
                    message = message.encode("utf-8")
                request = msgpack.unpackb(message, raw=False)
                method = request.get("method", "infer")

                if method == "reset":
                    async with inference_lock:
                        policy.reset()
                    reply = {"ok": True}
                elif method in {"get_server_metadata", "get_metadata", "metadata"}:
                    reply = build_metadata(policy, policy_path, args.action_mode)
                elif method == "infer":
                    started_at = time.time()
                    last_request_time = state["last_request_time"]
                    request_interval_s = (
                        0.0 if last_request_time is None else started_at - last_request_time
                    )
                    state["last_request_time"] = started_at
                    async with inference_lock:
                        result, diagnostics = infer(
                            policy,
                            preprocessor,
                            postprocessor,
                            request,
                            device,
                            args.action_mode,
                            args.max_action_delta,
                        )
                        step = state["step"]
                        state["step"] += 1
                    reply = make_reply(result, args.action_mode, step, started_at)
                    log_result(
                        step,
                        reply,
                        diagnostics,
                        request_interval_s,
                        args.print_action,
                    )
                else:
                    reply = {"ok": False, "error": f"unknown method: {method}"}

                await websocket.send(msgpack.packb(reply, use_bin_type=True))
        except websockets.exceptions.ConnectionClosed as error:
            print(
                f"[SERVER] websocket disconnected: {address} "
                f"code={error.code} reason={error.reason}"
            )
        except Exception as error:
            print(f"[SERVER] websocket error: {error!r}")
            traceback.print_exc()
            try:
                await websocket.send(
                    msgpack.packb({"ok": False, "error": repr(error)}, use_bin_type=True)
                )
            except Exception:
                pass

    print(f"[SERVER] Listening on {args.host}:{args.port} transport=openpi_ws")
    async with websockets.serve(
        handler,
        args.host,
        args.port,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ):
        await asyncio.Future()


def run_tcp_server(args, policy, preprocessor, postprocessor, device):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[SERVER] Listening on {args.host}:{args.port} transport=tcp_pickle")

    try:
        while True:
            connection, address = server.accept()
            connection.settimeout(30.0)
            policy.reset()
            print(f"[SERVER] connected: {address}; policy history reset")

            step = 0
            last_request_time = None
            try:
                while True:
                    payload = recv_msg(connection)
                    started_at = time.time()
                    request_interval_s = (
                        0.0 if last_request_time is None else started_at - last_request_time
                    )
                    last_request_time = started_at
                    result, diagnostics = infer(
                        policy,
                        preprocessor,
                        postprocessor,
                        payload,
                        device,
                        args.action_mode,
                        args.max_action_delta,
                    )
                    reply = make_reply(result, args.action_mode, step, started_at)

                    # Lists avoid NumPy 1.x/2.x pickle compatibility problems.
                    reply["action"] = np.asarray(reply["action"]).astype(float).tolist()
                    reply["actions"] = reply["action"]
                    send_msg(connection, reply)
                    log_result(
                        step,
                        reply,
                        diagnostics,
                        request_interval_s,
                        args.print_action,
                    )
                    step += 1
            except (ConnectionError, TimeoutError, socket.timeout):
                print("[SERVER] client disconnected or timed out")
            except KeyboardInterrupt:
                raise
            except Exception as error:
                print(f"[SERVER] error at step={step}: {error!r}")
                traceback.print_exc()
                try:
                    send_msg(connection, {"ok": False, "error": repr(error)})
                except Exception:
                    pass
            finally:
                connection.close()
    finally:
        server.close()


def configure_environment():
    cache_base = "/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/cache/huggingface"
    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("HF_HUB_CACHE", f"{cache_base}/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{cache_base}/transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for proxy_key in [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ]:
        os.environ.pop(proxy_key, None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transport", choices=["openpi_ws", "tcp_pickle"], default="openpi_ws")
    parser.add_argument(
        "--action_mode",
        choices=["chunk", "single"],
        default="single",
        help=(
            "chunk returns n_action_steps actions per request; single should be called at the robot "
            "control frequency so the 10-frame observation history remains meaningful."
        ),
    )
    parser.add_argument("--no_initial_metadata", action="store_true")
    parser.add_argument("--print_action", action="store_true")
    parser.add_argument(
        "--max_action_delta",
        type=float,
        default=0.05,
        help=(
            "Maximum absolute change per joint for each returned action. "
            "Set to 0 to disable limiting."
        ),
    )
    return parser.parse_args()


def load_diffusion_policy(policy_path, device):
    """Load a trained checkpoint without downloading redundant ImageNet weights."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy

    config = PreTrainedConfig.from_pretrained(policy_path)
    if not isinstance(config, DiffusionConfig):
        raise TypeError(
            f"Expected a DiffusionConfig at {policy_path}, got {type(config).__name__}"
        )
    config.device = str(device)

    if config.pretrained_backbone_weights is not None:
        print(
            "[SERVER] Skipping backbone initialization weights "
            f"{config.pretrained_backbone_weights!r}; checkpoint weights will be loaded instead"
        )
        config.pretrained_backbone_weights = None

    return DiffusionPolicy.from_pretrained(
        policy_path,
        config=config,
    )


def main():
    configure_environment()
    args = parse_args()
    device = torch.device(args.device)
    policy_path = str(Path(args.policy_path).expanduser())

    print(f"[SERVER] Loading Diffusion Policy from: {policy_path}")
    policy = load_diffusion_policy(policy_path, device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = load_pre_post_processors(policy, policy_path, device)

    print("[SERVER] Diffusion Policy loaded")
    print("[SERVER] input_features:", policy.config.input_features)
    print("[SERVER] output_features:", policy.config.output_features)
    print(
        f"[SERVER] n_obs_steps={policy.config.n_obs_steps} "
        f"horizon={policy.config.horizon} "
        f"n_action_steps={policy.config.n_action_steps}"
    )
    print("[SERVER] action_mode:", args.action_mode)
    print("[SERVER] max_action_delta:", args.max_action_delta)

    if args.transport == "openpi_ws":
        asyncio.run(
            run_websocket_server(
                args,
                policy,
                preprocessor,
                postprocessor,
                policy_path,
                device,
            )
        )
    else:
        run_tcp_server(args, policy, preprocessor, postprocessor, device)


if __name__ == "__main__":
    main()
