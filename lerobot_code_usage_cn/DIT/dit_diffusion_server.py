#!/usr/bin/env python
"""Serve a trained LeRobot MultiTaskDiT policy.

The wire protocol matches pi05_server.py: MessagePack over WebSocket by default,
or a length-prefixed pickle TCP mode. Requests may contain top-level observation
fields or an "obs" object.
"""

import argparse
import asyncio
import os
import pickle
import socket
import struct
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch


_WARNED_STATE_DICT = False
_WARNED_STATE_PAD = False


def _wire_key(mapping, name):
    if name in mapping:
        return mapping[name]
    return mapping.get(name.encode("utf-8"))


def decode_wire_value(value):
    if isinstance(value, dict):
        ndarray_marker = _wire_key(value, "__ndarray__")
        dtype_value = _wire_key(value, "dtype")
        shape_value = _wire_key(value, "shape")
        data_value = _wire_key(value, "data")

        if ndarray_marker is not None and dtype_value is not None and shape_value is not None:
            if isinstance(dtype_value, bytes):
                dtype_value = dtype_value.decode("utf-8")
            dtype = np.dtype(dtype_value)
            shape = tuple(int(dim) for dim in shape_value)
            if isinstance(data_value, (bytes, bytearray, memoryview)):
                array = np.frombuffer(data_value, dtype=dtype).copy()
            else:
                array = np.asarray(decode_wire_value(data_value), dtype=dtype)
            expected_size = int(np.prod(shape, dtype=np.int64))
            if array.size != expected_size:
                raise ValueError(
                    f"Invalid encoded ndarray: shape={shape} expects {expected_size} values, "
                    f"but data contains {array.size}"
                )
            return array.reshape(shape)

        decoded = {}
        for key, item in value.items():
            if isinstance(key, bytes):
                try:
                    key = key.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            decoded[key] = decode_wire_value(item)
        return decoded

    if isinstance(value, list):
        return [decode_wire_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(decode_wire_value(item) for item in value)
    return value


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_exact(sock, n):
    chunks = []
    received = 0
    while received < n:
        chunk = sock.recv(n - received)
        if not chunk:
            raise ConnectionError("socket connection closed")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_msg(sock):
    header = recv_exact(sock, 4)
    msg_len = struct.unpack("!I", header)[0]
    return pickle.loads(recv_exact(sock, msg_len))


def image_to_tensor(img, device):
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"bad image shape: {img.shape}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))

    if img.dtype == np.uint8:
        x = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True, order="C")).to(device)
        x = x.permute(2, 0, 1).float() / 255.0
    else:
        img = np.asarray(img, dtype=np.float32)
        if float(np.nanmax(img)) > 1.5:
            img = np.clip(img, 0.0, 255.0) / 255.0
        else:
            img = np.clip(img, 0.0, 1.0)
        x = torch.from_numpy(np.array(img, dtype=np.float32, copy=True, order="C")).to(device)
        x = x.permute(2, 0, 1).float()
    return x.unsqueeze(0)


def state_to_vector(state_value):
    global _WARNED_STATE_DICT
    if not isinstance(state_value, dict):
        return np.asarray(state_value, dtype=np.float32).reshape(-1)

    should_log = not _WARNED_STATE_DICT
    if should_log:
        print(f"[SERVER] state is dict, keys={sorted(state_value.keys())}")
        _WARNED_STATE_DICT = True

    preferred_keys = [
        "left",
        "right",
        "left_arm",
        "right_arm",
        "left_joint",
        "right_joint",
        "left_joints",
        "right_joints",
        "left_qpos",
        "right_qpos",
        "qpos",
        "joint_positions",
        "joints",
        "state",
    ]
    parts = []
    used = set()
    for key in preferred_keys:
        if key in state_value:
            try:
                arr = np.asarray(state_value[key], dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size:
                parts.append(arr)
                used.add(key)

    if not parts:
        for key in sorted(state_value.keys()):
            value = state_value[key]
            if isinstance(value, dict):
                continue
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size:
                parts.append(arr)
                used.add(key)

    if not parts:
        raise ValueError(f"Could not extract numeric state from dict keys: {sorted(state_value.keys())}")

    if should_log:
        print(f"[SERVER] state vector built from keys={sorted(used)} dim={sum(p.size for p in parts)}")
    return np.concatenate(parts, axis=0)


def get_first_present(mapping, keys: Iterable[str]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def build_observation(payload, device):
    global _WARNED_STATE_PAD
    if "obs" in payload:
        payload = payload["obs"]

    state_value = get_first_present(payload, ["qpos", "state", "observation.state"])
    if state_value is None:
        raise KeyError(f"Missing qpos/state/observation.state in payload keys: {sorted(payload.keys())}")
    qpos = state_to_vector(state_value)
    if qpos.shape[0] < 14:
        if not _WARNED_STATE_PAD:
            print(f"[SERVER] warning: received state dim {qpos.shape[0]}, padding to 14")
            _WARNED_STATE_PAD = True
        qpos = np.pad(qpos, (0, 14 - qpos.shape[0]), mode="constant")
    qpos = qpos[:14]

    task = payload.get("task", payload.get("prompt", "pick and place"))

    images = payload.get("images", {})
    top = get_first_present(payload, ["observation.images.top", "top", "observation.images.front", "front"])
    left_hand = get_first_present(
        payload,
        [
            "observation.images.left_hand",
            "left_hand",
            "observation.images.left",
            "left",
            "observation.images.left_wrist",
            "left_wrist",
        ],
    )
    right_hand = get_first_present(
        payload,
        [
            "observation.images.right_hand",
            "right_hand",
            "observation.images.right",
            "right",
            "observation.images.right_wrist",
            "right_wrist",
        ],
    )
    if top is None:
        top = images.get("top", images.get("front"))
    if left_hand is None:
        left_hand = images.get("left_hand", images.get("left", images.get("left_wrist")))
    if right_hand is None:
        right_hand = images.get("right_hand", images.get("right", images.get("right_wrist")))

    missing = [
        name
        for name, value in {
            "observation.images.top/top/front": top,
            "observation.images.left_hand/left_hand/left/left_wrist": left_hand,
            "observation.images.right_hand/right_hand/right/right_wrist": right_hand,
        }.items()
        if value is None
    ]
    if missing:
        image_keys = sorted(images.keys()) if isinstance(images, dict) else []
        raise KeyError(f"Missing image keys: {missing}. Got payload={sorted(payload.keys())}, images={image_keys}")

    return {
        "observation.state": torch.from_numpy(np.array(qpos, dtype=np.float32, copy=True))
        .to(device)
        .unsqueeze(0),
        "observation.images.top": image_to_tensor(top, device),
        "observation.images.left_hand": image_to_tensor(left_hand, device),
        "observation.images.right_hand": image_to_tensor(right_hand, device),
        "task": [task],
    }


def extract_action(action_obj, action_dim=14):
    if isinstance(action_obj, dict):
        if "action" in action_obj:
            action_obj = action_obj["action"]
        elif "ACTION" in action_obj:
            action_obj = action_obj["ACTION"]
        else:
            raise KeyError(f"Cannot find action key in: {action_obj.keys()}")

    if torch.is_tensor(action_obj):
        action_obj = action_obj.detach().float().cpu().numpy()

    action = np.asarray(action_obj, dtype=np.float32)
    if action.ndim == 1:
        pass
    elif action.shape[-1] == action_dim:
        action = action.reshape(-1, action_dim)[0]
    else:
        action = action.reshape(-1)
    if action.shape[0] < action_dim:
        raise ValueError(f"Expected action dim >= {action_dim}, got {action.shape}")
    return action[:action_dim].astype(np.float32)


def extract_action_chunk(action_obj, action_dim=14):
    if isinstance(action_obj, dict):
        if "action" in action_obj:
            action_obj = action_obj["action"]
        elif "actions" in action_obj:
            action_obj = action_obj["actions"]
        elif "ACTION" in action_obj:
            action_obj = action_obj["ACTION"]
        else:
            raise KeyError(f"Cannot find action chunk key in: {action_obj.keys()}")

    if torch.is_tensor(action_obj):
        action_obj = action_obj.detach().float().cpu().numpy()

    actions = np.asarray(action_obj, dtype=np.float32)
    if actions.ndim == 3:
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[-1] < action_dim:
        raise ValueError(f"Expected action chunk dim >= {action_dim}, got {actions.shape}")
    return actions[:, :action_dim].astype(np.float32)


def load_pre_post_processors(policy, policy_path, device):
    from lerobot.policies.factory import make_pre_post_processors

    try:
        return make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    except TypeError:
        return make_pre_post_processors(
            policy.config,
            policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )


def infer_action(policy, preprocessor, postprocessor, payload, device):
    obs = build_observation(payload, device)
    with torch.inference_mode():
        processed_obs = preprocessor(obs)
        action_obj = policy.select_action(processed_obs)
        action_obj = postprocessor(action_obj)
    return extract_action(action_obj, policy.config.action_feature.shape[0])


def infer_action_chunk(policy, preprocessor, postprocessor, payload, device, min_chunk_len):
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import ACTION

    obs = build_observation(payload, device)
    with torch.inference_mode():
        processed_obs = preprocessor(obs)
        if isinstance(processed_obs, dict) and ACTION in processed_obs:
            processed_obs = dict(processed_obs)
            processed_obs.pop(ACTION)
        if hasattr(policy, "_prepare_batch") and hasattr(policy, "_queues"):
            processed_obs = policy._prepare_batch(processed_obs)
            policy._queues = populate_queues(policy._queues, processed_obs, exclude_keys=[ACTION])
        action_obj = policy.predict_action_chunk(processed_obs)
        action_obj = postprocessor(action_obj)
    actions = extract_action_chunk(action_obj, policy.config.action_feature.shape[0])
    if actions.shape[0] < min_chunk_len:
        pad = np.repeat(actions[-1:], min_chunk_len - actions.shape[0], axis=0)
        actions = np.concatenate([actions, pad], axis=0)
    return actions[:min_chunk_len]


def build_metadata(policy, policy_path):
    return {
        "model": "lerobot_multi_task_dit",
        "policy_path": policy_path,
        "input_features": {k: tuple(v.shape) for k, v in policy.config.input_features.items()},
        "output_features": {k: tuple(v.shape) for k, v in policy.config.output_features.items()},
        "objective": policy.config.objective,
    }


async def run_ws_server(args, policy, preprocessor, postprocessor, policy_path, device):
    import msgpack
    import msgpack_numpy as m
    import websockets

    state = {"step": 0}
    infer_lock = asyncio.Lock()

    async def handler(websocket):
        addr = getattr(websocket, "remote_address", None)
        print(f"[SERVER] websocket connected: {addr}")
        if hasattr(policy, "reset"):
            policy.reset()
        if not args.no_initial_metadata:
            await websocket.send(msgpack.packb(build_metadata(policy, policy_path), use_bin_type=True))

        try:
            async for message in websocket:
                t0 = time.time()
                if isinstance(message, str):
                    message = message.encode("utf-8")
                request = decode_wire_value(m.unpackb(message, raw=False))
                method = request.get("method", "infer")

                if method == "reset":
                    async with infer_lock:
                        if hasattr(policy, "reset"):
                            policy.reset()
                    reply = {"ok": True}
                elif method in {"get_server_metadata", "get_metadata", "metadata"}:
                    reply = build_metadata(policy, policy_path)
                elif method == "infer":
                    async with infer_lock:
                        if args.action_mode == "chunk":
                            action_result = infer_action_chunk(
                                policy, preprocessor, postprocessor, request, device, args.chunk_len
                            )
                        else:
                            action_result = infer_action(policy, preprocessor, postprocessor, request, device)
                        step = state["step"]
                        state["step"] += 1

                    action_payload = action_result.astype(np.float32).tolist()
                    reply = {
                        "actions": action_payload,
                        "action": action_payload,
                        "action_mode": args.action_mode,
                        "server_step": int(step),
                        "latency_s": float(time.time() - t0),
                    }
                    if args.print_action or step % 10 == 0:
                        print(
                            f"[SERVER] step={step} latency={reply['latency_s']:.3f}s "
                            f"mode={args.action_mode} shape={np.asarray(action_result).shape}"
                        )
                else:
                    reply = {"error": f"unknown method: {method}"}

                await websocket.send(msgpack.packb(reply, use_bin_type=True))
        except websockets.exceptions.ConnectionClosed as exc:
            print(f"[SERVER] websocket disconnected: {addr} code={exc.code} reason={exc.reason}")
        except Exception as exc:
            import traceback

            print(f"[SERVER] websocket error: {repr(exc)}")
            traceback.print_exc()
            try:
                await websocket.send(msgpack.packb({"error": repr(exc)}, use_bin_type=True))
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


def set_hf_env(args):
    hf_cache_base = args.hf_cache
    os.environ["HF_HOME"] = hf_cache_base
    os.environ["HF_HUB_CACHE"] = f"{hf_cache_base}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{hf_cache_base}/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{hf_cache_base}/datasets"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    for proxy_key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        os.environ.pop(proxy_key, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_path",
        type=str,
        default="/media/wu/data/SUN_ht/dit/runs/latest/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--transport", choices=["openpi_ws", "tcp_pickle"], default="openpi_ws")
    parser.add_argument("--action_mode", choices=["chunk", "single"], default="chunk")
    parser.add_argument("--chunk_len", type=int, default=50)
    parser.add_argument("--hf_cache", type=str, default="/media/wu/data/SUN_ht/dit/cache/huggingface")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_initial_metadata", action="store_true")
    parser.add_argument("--print_action", action="store_true")
    args = parser.parse_args()

    set_hf_env(args)

    device = torch.device(args.device)
    policy_dir = Path(args.policy_path).expanduser()
    required_files = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    missing_files = [name for name in required_files if not (policy_dir / name).is_file()]
    if missing_files:
        parser.error(f"Incomplete MultiTaskDiT checkpoint at {policy_dir}: missing {missing_files}")

    from lerobot.policies.multi_task_dit import MultiTaskDiTPolicy

    policy_path = str(policy_dir)
    print(f"[SERVER] Loading MultiTaskDiT from: {policy_path}")
    policy = MultiTaskDiTPolicy.from_pretrained(policy_path, device=str(device), local_files_only=args.offline)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = load_pre_post_processors(policy, policy_path, device)

    print("[SERVER] MultiTaskDiT loaded.")
    print("[SERVER] input_features:", policy.config.input_features)
    print("[SERVER] output_features:", policy.config.output_features)
    print("[SERVER] action_mode:", args.action_mode)

    if args.transport == "openpi_ws":
        asyncio.run(run_ws_server(args, policy, preprocessor, postprocessor, policy_path, device))
        return

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


if __name__ == "__main__":
    main()
