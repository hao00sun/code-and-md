"""Serve a left-arm-only MultiTaskDiT Flow Matching Policy.

The wire protocol intentionally matches ``pi05_server.py``:

- ``openpi_ws``: msgpack-numpy messages over WebSocket.
- ``tcp_pickle``: a 4-byte length prefix followed by a pickle payload.

The client may keep sending its existing 14-dimensional state and three camera
images. Only state dimensions 0:7, the front camera, and the left wrist camera
are passed to the model. The 7-dimensional model action is joined with the
current right-arm state before returning a 14-dimensional action to the client.
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
    "/media/wu/data/SUN_ht/innov/inference_models/"
    "flow_matching_left_arm_latest"
)

_WARNED_IMAGE_SHAPES = set()
LEFT_ARM_SLICE = slice(0, 7)
RIGHT_ARM_SLICE = slice(7, 14)


def build_left_arm_observation(payload, device):
    """Adapt the existing client payload to the left-arm-only policy inputs."""
    full_obs = build_observation(payload, device)
    full_state = full_obs["observation.state"][0].detach().cpu().numpy()
    if full_state.shape[0] != 14:
        raise ValueError(f"Expected client state dim 14, got {full_state.shape[0]}")

    model_obs = {
        "observation.state": full_obs["observation.state"][:, LEFT_ARM_SLICE],
        "observation.images.front": full_obs["observation.images.front"],
        "observation.images.left_wrist": full_obs["observation.images.left_wrist"],
        "task": full_obs["task"],
    }
    return model_obs, full_state


def join_left_actions_with_right_state(left_actions, full_state):
    """Return client-compatible 14D actions while holding the right arm still."""
    left_actions = np.asarray(left_actions, dtype=np.float32)
    full_state = np.asarray(full_state, dtype=np.float32).reshape(-1)
    if left_actions.shape[-1] != LEFT_ARM_SLICE.stop:
        raise ValueError(f"Expected left action dim 7, got {left_actions.shape[-1]}")

    if left_actions.ndim == 1:
        return np.concatenate([left_actions, full_state[RIGHT_ARM_SLICE]], axis=0)

    right_actions = np.broadcast_to(
        full_state[RIGHT_ARM_SLICE],
        (left_actions.shape[0], RIGHT_ARM_SLICE.stop - RIGHT_ARM_SLICE.start),
    )
    return np.concatenate([left_actions, right_actions], axis=-1)


def align_observation_images(policy, obs):
    """Match client camera frames to the image shapes used during training.

    The original INNOV front camera is 848x480, while both wrist cameras are
    640x480. The training dataset center-cropped the front image to 640x480.
    This function reproduces that operation before the policy stacks the
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


def get_feature_bounds(processor, feature_key):
    """Extract feature min/max from saved normalizer or unnormalizer stats."""
    for step in getattr(processor, "steps", []):
        tensor_stats = getattr(step, "_tensor_stats", None)
        if not tensor_stats or feature_key not in tensor_stats:
            continue

        feature_stats = tensor_stats[feature_key]
        min_val = feature_stats.get("min")
        max_val = feature_stats.get("max")
        if min_val is None or max_val is None:
            continue

        return (
            min_val.detach().cpu().numpy().astype(np.float32).reshape(-1),
            max_val.detach().cpu().numpy().astype(np.float32).reshape(-1),
        )

    return None, None


def build_state_range_diagnostics(current_state, state_min, state_max):
    """Compare the current physical state with the training dataset range."""
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if state_min is None or state_max is None:
        return {
            "state_bounds_available": False,
            "state_out_of_range_indices": np.array([], dtype=np.int64),
        }

    state_min = np.asarray(state_min, dtype=np.float32).reshape(-1)
    state_max = np.asarray(state_max, dtype=np.float32).reshape(-1)
    if current_state.shape != state_min.shape or current_state.shape != state_max.shape:
        return {
            "state_bounds_available": False,
            "state_out_of_range_indices": np.array([], dtype=np.int64),
            "state_bounds_shape_error": (
                f"state={current_state.shape} min={state_min.shape} max={state_max.shape}"
            ),
        }

    below = current_state < state_min
    above = current_state > state_max
    out_of_range = below | above
    distance = np.where(below, state_min - current_state, np.where(above, current_state - state_max, 0))
    return {
        "state_bounds_available": True,
        "state_min": state_min,
        "state_max": state_max,
        "state_below_range_indices": np.flatnonzero(below).astype(np.int64),
        "state_above_range_indices": np.flatnonzero(above).astype(np.int64),
        "state_out_of_range_indices": np.flatnonzero(out_of_range).astype(np.int64),
        "state_max_out_of_range_distance": float(distance.max(initial=0.0)),
    }


def apply_action_safety(
    actions,
    current_state,
    *,
    freeze_right_arm,
    clip_to_dataset_action_range,
    action_min,
    action_max,
):
    """Apply deployment safety constraints before delta limiting."""
    safe_actions = np.asarray(actions, dtype=np.float32).copy()
    is_single = safe_actions.ndim == 1
    sequence = safe_actions.reshape(1, -1) if is_single else safe_actions
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    safety = {
        "freeze_right_arm": bool(freeze_right_arm),
        "clip_to_dataset_action_range": bool(clip_to_dataset_action_range),
        "dataset_clip_applied": False,
        "right_arm_frozen": False,
    }

    if clip_to_dataset_action_range:
        if action_min is None or action_max is None:
            print("[SERVER] warning: requested action range clipping but action stats are unavailable")
        else:
            before_clip = sequence.copy()
            sequence[:] = np.clip(sequence, action_min, action_max)
            safety["dataset_clip_applied"] = bool(np.any(np.abs(sequence - before_clip) > 1e-6))

    if freeze_right_arm:
        if current_state.shape[0] < RIGHT_ARM_SLICE.stop or sequence.shape[-1] < RIGHT_ARM_SLICE.stop:
            print(
                "[SERVER] warning: requested right arm freeze but action/state dim is "
                f"action={sequence.shape[-1]} state={current_state.shape[0]}"
            )
        else:
            sequence[:, RIGHT_ARM_SLICE] = current_state[RIGHT_ARM_SLICE]
            safety["right_arm_frozen"] = True

    return (sequence[0] if is_single else sequence), safety


def infer_single_action(
    policy,
    preprocessor,
    postprocessor,
    payload,
    device,
    max_action_delta,
    freeze_right_arm,
    clip_to_dataset_action_range,
    action_min,
    action_max,
    state_min,
    state_max,
):
    """Update history and return the next action from the policy action queue."""
    obs, full_state = build_left_arm_observation(payload, device)
    obs = align_observation_images(policy, obs)
    current_state = full_state[LEFT_ARM_SLICE]
    generated_new_chunk = len(policy._queues["action"]) == 0  # noqa: SLF001
    with torch.inference_mode():
        processed_obs = preprocessor(obs)
        action = policy.select_action(processed_obs)
        normalized_action = extract_action(action)
        action = postprocessor(action)
    raw_action = extract_action(action)
    safe_action, safety = apply_action_safety(
        raw_action,
        current_state,
        freeze_right_arm=False,
        clip_to_dataset_action_range=clip_to_dataset_action_range,
        action_min=action_min,
        action_max=action_max,
    )
    limited_action = limit_action_delta(safe_action, current_state, max_action_delta)
    diagnostics = build_action_diagnostics(
        current_state,
        raw_action,
        limited_action,
        max_action_delta,
        generated_new_chunk=generated_new_chunk,
    )
    diagnostics["safe_action"] = np.asarray(safe_action, dtype=np.float32)
    diagnostics["normalized_action"] = np.asarray(normalized_action, dtype=np.float32)
    diagnostics.update(build_state_range_diagnostics(current_state, state_min, state_max))
    diagnostics.update(safety)
    diagnostics["full_current_qpos"] = full_state
    diagnostics["right_arm_held"] = full_state[RIGHT_ARM_SLICE]
    return join_left_actions_with_right_state(limited_action, full_state), diagnostics


def infer_action_chunk(
    policy,
    preprocessor,
    postprocessor,
    payload,
    device,
    max_action_delta,
    freeze_right_arm,
    clip_to_dataset_action_range,
    action_min,
    action_max,
    state_min,
    state_max,
):
    """Generate a chunk through MultiTaskDiTPolicy's native queue management."""
    from lerobot.utils.constants import ACTION

    obs, full_state = build_left_arm_observation(payload, device)
    obs = align_observation_images(policy, obs)
    current_state = full_state[LEFT_ARM_SLICE]
    with torch.inference_mode():
        processed_obs = preprocessor(obs)

        # select_action performs the camera stacking, initializes/updates the
        # n_obs_steps history, generates a fresh chunk, and returns its first
        # action. The remaining actions are kept in the policy action queue.
        first_action = policy.select_action(processed_obs)
        remaining_actions = list(policy._queues[ACTION])  # noqa: SLF001
        actions = torch.stack([first_action, *remaining_actions], dim=1)
        normalized_actions = extract_action_chunk(actions)

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
    safe_actions, safety = apply_action_safety(
        actions,
        current_state,
        freeze_right_arm=False,
        clip_to_dataset_action_range=clip_to_dataset_action_range,
        action_min=action_min,
        action_max=action_max,
    )
    limited_actions = limit_action_delta(safe_actions, current_state, max_action_delta)
    diagnostics = build_action_diagnostics(
        current_state,
        actions[0],
        limited_actions[0],
        max_action_delta,
        generated_new_chunk=True,
    )
    diagnostics["safe_action"] = np.asarray(safe_actions[0], dtype=np.float32)
    diagnostics["normalized_action"] = np.asarray(normalized_actions[0], dtype=np.float32)
    diagnostics["normalized_action_min"] = float(np.min(normalized_actions))
    diagnostics["normalized_action_max"] = float(np.max(normalized_actions))
    diagnostics["normalized_action_max_abs"] = float(np.max(np.abs(normalized_actions)))
    diagnostics.update(build_state_range_diagnostics(current_state, state_min, state_max))
    diagnostics.update(safety)
    diagnostics["full_current_qpos"] = full_state
    diagnostics["right_arm_held"] = full_state[RIGHT_ARM_SLICE]
    diagnostics["raw_chunk_max_step_delta"] = float(
        np.max(np.abs(np.diff(np.vstack([current_state, actions]), axis=0)))
    )
    diagnostics["safe_chunk_max_step_delta"] = float(
        np.max(np.abs(np.diff(np.vstack([current_state, safe_actions]), axis=0)))
    )
    diagnostics["limited_chunk_max_step_delta"] = float(
        np.max(np.abs(np.diff(np.vstack([current_state, limited_actions]), axis=0)))
    )
    return join_left_actions_with_right_state(limited_actions, full_state), diagnostics


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
    freeze_right_arm,
    clip_to_dataset_action_range,
    action_min,
    action_max,
    state_min,
    state_max,
):
    if action_mode == "chunk":
        return infer_action_chunk(
            policy,
            preprocessor,
            postprocessor,
            payload,
            device,
            max_action_delta,
            freeze_right_arm,
            clip_to_dataset_action_range,
            action_min,
            action_max,
            state_min,
            state_max,
        )
    return infer_single_action(
        policy,
        preprocessor,
        postprocessor,
        payload,
        device,
        max_action_delta,
        freeze_right_arm,
        clip_to_dataset_action_range,
        action_min,
        action_max,
        state_min,
        state_max,
    )


def build_metadata(policy, policy_path, action_mode):
    inference_steps = getattr(
        policy.config,
        "num_integration_steps",
        getattr(policy.config, "num_inference_steps", None),
    )
    return {
        "model": "lerobot_multi_task_dit_flow_matching_left_arm",
        "policy_path": policy_path,
        "action_mode": action_mode,
        "client_action_dim": 14,
        "model_action_dim": 7,
        "right_arm_mode": "hold_current_state",
        "clip_to_dataset_action_range": bool(
            getattr(policy, "_server_clip_to_dataset_action_range", False)
        ),
        "objective": getattr(policy.config, "objective", None),
        "n_obs_steps": int(policy.config.n_obs_steps),
        "horizon": int(policy.config.horizon),
        "n_action_steps": int(policy.config.n_action_steps),
        "num_integration_steps": None if inference_steps is None else int(inference_steps),
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
    print(f"[SERVER] full_current_qpos={diagnostics['full_current_qpos']}")
    print(f"[SERVER] right_arm_held={diagnostics['right_arm_held']}")
    normalized_action = np.asarray(diagnostics["normalized_action"])
    normalized_min = float(diagnostics.get("normalized_action_min", normalized_action.min()))
    normalized_max = float(diagnostics.get("normalized_action_max", normalized_action.max()))
    normalized_max_abs = float(
        diagnostics.get("normalized_action_max_abs", np.max(np.abs(normalized_action)))
    )
    print(
        f"[SERVER] normalized_action range=[{normalized_min:.6f}, {normalized_max:.6f}] "
        f"max_abs={normalized_max_abs:.6f} first={normalized_action}"
    )
    if diagnostics.get("state_bounds_available"):
        out_of_range = diagnostics["state_out_of_range_indices"].tolist()
        print(
            f"[SERVER] state_range out_of_range={out_of_range} "
            f"below={diagnostics['state_below_range_indices'].tolist()} "
            f"above={diagnostics['state_above_range_indices'].tolist()} "
            f"max_distance={diagnostics['state_max_out_of_range_distance']:.6f}"
        )
        if out_of_range:
            print(f"[SERVER] state_min={diagnostics['state_min']}")
            print(f"[SERVER] state_max={diagnostics['state_max']}")
    else:
        print(
            "[SERVER] state_range unavailable "
            f"{diagnostics.get('state_bounds_shape_error', '')}"
        )
    print(f"[SERVER] raw_action={diagnostics['raw_action']}")
    print(f"[SERVER] raw_delta={diagnostics['raw_delta']}")
    if "safe_action" in diagnostics:
        print(
            f"[SERVER] safety right_arm_mode=hold_current_state "
            f"clip_to_dataset_action_range={diagnostics.get('clip_to_dataset_action_range')} "
            f"dataset_clip_applied={diagnostics.get('dataset_clip_applied')}"
        )
        print(f"[SERVER] safe_action={diagnostics['safe_action']}")
    print(f"[SERVER] limited_action={diagnostics['limited_action']}")
    print(f"[SERVER] limited_delta={diagnostics['limited_delta']}")
    if "raw_chunk_max_step_delta" in diagnostics:
        print(
            f"[SERVER] chunk_delta raw_max={diagnostics['raw_chunk_max_step_delta']:.6f} "
            f"safe_max={diagnostics['safe_chunk_max_step_delta']:.6f} "
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
                            args.freeze_right_arm,
                            args.clip_to_dataset_action_range,
                            args.action_min,
                            args.action_max,
                            args.state_min,
                            args.state_max,
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
                        args.freeze_right_arm,
                        args.clip_to_dataset_action_range,
                        args.action_min,
                        args.action_max,
                        args.state_min,
                        args.state_max,
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
    cache_base = "/media/wu/data/SUN_ht/innov/cache/huggingface"
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
        default="chunk",
        help=(
            "chunk returns n_action_steps actions per request; single should be called at the robot "
            "control frequency so the observation history remains meaningful."
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
    parser.add_argument(
        "--freeze_right_arm",
        dest="freeze_right_arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--clip_to_dataset_action_range",
        dest="clip_to_dataset_action_range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip unnormalized model actions to the min/max action range observed in the training dataset.",
    )
    return parser.parse_args()


def load_flow_matching_policy(policy_path, device):
    """Load a trained MultiTaskDiT Flow Matching checkpoint from local files."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.multi_task_dit import MultiTaskDiTConfig, MultiTaskDiTPolicy

    config = PreTrainedConfig.from_pretrained(policy_path)
    if not isinstance(config, MultiTaskDiTConfig):
        raise TypeError(
            f"Expected a MultiTaskDiTConfig at {policy_path}, got {type(config).__name__}"
        )
    if config.objective != "flow_matching":
        raise TypeError(
            f"Expected objective='flow_matching' at {policy_path}, got {config.objective!r}"
        )
    state_shape = config.input_features["observation.state"].shape
    action_shape = config.output_features["action"].shape
    expected_images = {
        "observation.images.front",
        "observation.images.left_wrist",
    }
    if state_shape != (7,) or action_shape != (7,):
        raise TypeError(
            f"Expected left-arm state/action shapes (7,), got state={state_shape} action={action_shape}"
        )
    if set(config.image_features) != expected_images:
        raise TypeError(
            f"Expected left-arm cameras {sorted(expected_images)}, "
            f"got {sorted(config.image_features)}"
        )
    config.device = str(device)

    return MultiTaskDiTPolicy.from_pretrained(
        policy_path,
        config=config,
    )


def main():
    configure_environment()
    args = parse_args()
    device = torch.device(args.device)
    policy_path = str(Path(args.policy_path).expanduser())

    print(f"[SERVER] Loading left-arm Flow Matching Policy from: {policy_path}")
    policy = load_flow_matching_policy(policy_path, device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = load_pre_post_processors(policy, policy_path, device)
    args.action_min, args.action_max = get_feature_bounds(postprocessor, "action")
    args.state_min, args.state_max = get_feature_bounds(preprocessor, "observation.state")
    policy._server_clip_to_dataset_action_range = args.clip_to_dataset_action_range  # noqa: SLF001

    print("[SERVER] left-arm Flow Matching Policy loaded")
    print("[SERVER] input_features:", policy.config.input_features)
    print("[SERVER] output_features:", policy.config.output_features)
    print(
        f"[SERVER] n_obs_steps={policy.config.n_obs_steps} "
        f"horizon={policy.config.horizon} "
        f"n_action_steps={policy.config.n_action_steps} "
        f"num_integration_steps={policy.config.num_integration_steps}"
    )
    print("[SERVER] action_mode:", args.action_mode)
    print("[SERVER] max_action_delta:", args.max_action_delta)
    print("[SERVER] client/model action dims: 14/7")
    print("[SERVER] right arm mode: hold current client state")
    print("[SERVER] clip_to_dataset_action_range:", args.clip_to_dataset_action_range)
    if args.action_min is None or args.action_max is None:
        print("[SERVER] action bounds: unavailable")
    else:
        print("[SERVER] action_min:", args.action_min)
        print("[SERVER] action_max:", args.action_max)
    if args.state_min is None or args.state_max is None:
        print("[SERVER] state bounds: unavailable")
    else:
        print("[SERVER] state_min:", args.state_min)
        print("[SERVER] state_max:", args.state_max)

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
