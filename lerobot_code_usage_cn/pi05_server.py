import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import pickle
import socket
import struct
import subprocess
import time
from pathlib import Path

# Pin the offline Hugging Face cache before importing Torch/LeRobot/Transformers.
# Do not use setdefault here: an activated shell may contain stale cache paths.
HF_CACHE_BASE = "/data/SUN_ht/pi/cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_BASE
os.environ["HF_HUB_CACHE"] = f"{HF_CACHE_BASE}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_BASE}/transformers"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch


_WARNED_STATE_PAD = False
_WARNED_STATE_DICT = False


ACTION_NAMES = [
    "left_joint1.pos",
    "left_joint2.pos",
    "left_joint3.pos",
    "left_joint4.pos",
    "left_joint5.pos",
    "left_joint6.pos",
    "left_gripper.pos",
    "right_joint1.pos",
    "right_joint2.pos",
    "right_joint3.pos",
    "right_joint4.pos",
    "right_joint5.pos",
    "right_joint6.pos",
    "right_gripper.pos",
]

ACTION_NAME_TO_DIM = {}
for _idx, _name in enumerate(ACTION_NAMES):
    ACTION_NAME_TO_DIM[_name] = _idx
    ACTION_NAME_TO_DIM[_name.removesuffix(".pos")] = _idx
    ACTION_NAME_TO_DIM[_name.replace(".pos", "")] = _idx


def get_host_addresses(host):
    """Return addresses clients can try for a bound server host."""
    if host not in {"0.0.0.0", "::", ""}:
        return [host]

    addresses = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127."):
                continue
            if ip not in addresses:
                addresses.append(ip)
    except OSError:
        pass

    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=1)  # noqa: S607
        for ip in output.split():
            if ":" in ip or ip.startswith("127."):
                continue
            if ip not in addresses:
                addresses.append(ip)
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if not ip.startswith("127.") and ip not in addresses:
                addresses.append(ip)
    except OSError:
        pass

    return addresses or ["127.0.0.1"]


def print_server_addresses(host, port, transport):
    protocol = "ws" if transport == "openpi_ws" else "tcp"
    print(f"[SERVER] Listening on {host}:{port} transport={transport}")
    for ip in get_host_addresses(host):
        print(f"[SERVER] Client URL: {protocol}://{ip}:{port}")


def resolve_policy_dir(policy_path):
    """Accept either a pretrained_model dir or a checkpoint step dir."""
    raw_path = Path(policy_path).expanduser()
    candidates = [raw_path]

    if raw_path.name == "pretrained_model":
        candidates.append(raw_path.parent)
    elif raw_path.name.isdigit() and len(raw_path.name) < 6:
        padded = raw_path.with_name(f"{int(raw_path.name):06d}")
        candidates.extend([raw_path / "pretrained_model", padded / "pretrained_model", padded])
    else:
        candidates.append(raw_path / "pretrained_model")

    for candidate in candidates:
        if candidate.is_dir() and candidate.name == "pretrained_model":
            return candidate
        pretrained_dir = candidate / "pretrained_model"
        if pretrained_dir.is_dir():
            return pretrained_dir

    return raw_path


def _wire_key(mapping, name):
    """Read a protocol field encoded as either a str key or a bytes key."""
    if name in mapping:
        return mapping[name]
    return mapping.get(name.encode("utf-8"))


def decode_wire_value(value):
    """Recursively restore NumPy arrays from the client's legacy wire format."""
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
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


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
    data = recv_exact(sock, msg_len)
    return pickle.loads(data)


def recv_ws_frame(sock):
    header = recv_exact(sock, 2)
    byte1, byte2 = header
    opcode = byte1 & 0x0F
    masked = bool(byte2 & 0x80)
    length = byte2 & 0x7F

    if length == 126:
        length = struct.unpack("!H", recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", recv_exact(sock, 8))[0]

    mask = recv_exact(sock, 4) if masked else b""
    payload = recv_exact(sock, length) if length else b""
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def send_ws_frame(sock, payload, opcode=2):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
        opcode = 1
    length = len(payload)
    header = bytearray([0x80 | opcode])
    if length < 126:
        header.append(length)
    elif length < 2**16:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    sock.sendall(bytes(header) + payload)


def websocket_handshake(conn):
    request = b""
    while b"\r\n\r\n" not in request:
        request += recv_exact(conn, 1)
        if len(request) > 65536:
            raise ConnectionError("websocket handshake too large")

    headers = {}
    lines = request.decode("utf-8", errors="replace").split("\r\n")
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    ws_key = headers.get("sec-websocket-key")
    if not ws_key:
        raise ConnectionError("missing Sec-WebSocket-Key")

    accept_src = ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    accept = base64.b64encode(hashlib.sha1(accept_src.encode("ascii")).digest()).decode("ascii")
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
        "\r\n"
    )
    conn.sendall(response.encode("ascii"))


def load_pre_post_processors(policy, policy_path, device):
    try:
        from lerobot.policies.factory import make_pre_post_processors
    except Exception:
        from lerobot.policies import make_pre_post_processors

    attempts = [
        lambda: make_pre_post_processors(
            policy.config,
            policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        ),
        lambda: make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        ),
        lambda: make_pre_post_processors(policy.config, policy_path),
        lambda: make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_path,
        ),
    ]

    last_err = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_err = e

    raise RuntimeError("Failed to create LeRobot pre/post processors") from last_err


def image_to_tensor(img, device):
    """
    input: H W C uint8
    output: 1 C H W float32, 0~1
    """
    img = np.asarray(img)

    if img.ndim != 3:
        raise ValueError(f"bad image shape: {img.shape}")

    chw_input = img.shape[0] == 3 and img.shape[-1] != 3
    if chw_input:
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
    x = x.unsqueeze(0)
    return x


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


def get_first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def extract_qpos_from_payload(payload):
    global _WARNED_STATE_PAD
    if "obs" in payload:
        payload = payload["obs"]

    state_value = get_first_present(payload, ["qpos", "state", "observation.state"])
    if state_value is None:
        raise KeyError(f"Missing qpos/state/observation.state in payload keys: {sorted(payload.keys())}")
    qpos = state_to_vector(state_value)
    if qpos.shape[0] < 14:
        if not _WARNED_STATE_PAD:
            print(f"[SERVER] warning: received state dim {qpos.shape[0]}, padding to 14 for PI05")
            _WARNED_STATE_PAD = True
        qpos = np.pad(qpos, (0, 14 - qpos.shape[0]), mode="constant")
    qpos = qpos[:14]
    return qpos.astype(np.float32)


def build_observation(payload, device):
    if "obs" in payload:
        payload = payload["obs"]

    qpos = extract_qpos_from_payload(payload)

    task = payload.get("task", payload.get("prompt", "Place the camera into the box."))

    images = payload.get("images", {})
    top = get_first_present(payload, ["observation.images.top", "top", "observation.images.front", "front"])
    left_wrist = get_first_present(
        payload, ["observation.images.left_hand", "left_hand", "observation.images.left_wrist", "left_wrist"]
    )
    right_wrist = get_first_present(
        payload, ["observation.images.right_hand", "right_hand", "observation.images.right_wrist", "right_wrist"]
    )
    if top is None:
        top = images.get("top", images.get("front"))
    if left_wrist is None:
        left_wrist = images.get("left_wrist", images.get("left_hand"))
    if right_wrist is None:
        right_wrist = images.get("right_wrist", images.get("right_hand"))
    missing = [
        name
        for name, value in {
            "observation.images.top/top/front": top,
            "observation.images.left_hand/left_hand/left_wrist": left_wrist,
            "observation.images.right_hand/right_hand/right_wrist": right_wrist,
        }.items()
        if value is None
    ]
    if missing:
        image_keys = sorted(images.keys()) if isinstance(images, dict) else []
        raise KeyError(f"Missing image keys: {missing}. Got payload={sorted(payload.keys())}, images={image_keys}")

    top_tensor = image_to_tensor(top, device)
    left_wrist_tensor = image_to_tensor(left_wrist, device)
    right_wrist_tensor = image_to_tensor(right_wrist, device)

    obs = {
        "observation.state": torch.from_numpy(np.array(qpos, dtype=np.float32, copy=True))
        .to(device)
        .unsqueeze(0),

        # Support both naming conventions used by local checkpoints:
        # ARX-style top/left_hand/right_hand and INNOV-style front/left_wrist/right_wrist.
        "observation.images.top": top_tensor,
        "observation.images.front": top_tensor,
        "observation.images.left_hand": left_wrist_tensor,
        "observation.images.left_wrist": left_wrist_tensor,
        "observation.images.right_hand": right_wrist_tensor,
        "observation.images.right_wrist": right_wrist_tensor,

        "task": [task],
    }

    return obs


def parse_active_action_dims(active_arms, active_action_dims):
    """Return sorted active action dimensions.

    ``active_action_dims`` can contain integer dims or names such as
    ``left_joint1``, ``left_joint1.pos``, ``left_gripper``, separated by commas.
    If provided, it takes precedence over ``active_arms``.
    """
    spec = (active_action_dims or "").strip()
    if spec:
        dims = set()
        for raw_token in spec.split(","):
            token = raw_token.strip()
            if not token:
                continue
            if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
                dim = int(token)
                if dim < 0:
                    dim += len(ACTION_NAMES)
            else:
                if token not in ACTION_NAME_TO_DIM:
                    valid = ", ".join(ACTION_NAMES)
                    raise ValueError(f"Unknown action dim/name {token!r}. Valid names: {valid}")
                dim = ACTION_NAME_TO_DIM[token]
            if dim < 0 or dim >= len(ACTION_NAMES):
                raise ValueError(f"Action dim out of range: {raw_token!r} -> {dim}")
            dims.add(dim)
        return sorted(dims)

    arms = (active_arms or "both").strip().lower()
    if arms in {"both", "all"}:
        return list(range(14))
    if arms == "left":
        return list(range(0, 7))
    if arms == "right":
        return list(range(7, 14))
    if arms in {"none", "freeze", "frozen"}:
        return []
    raise ValueError("--active_arms must be one of: both/all, left, right, none")


def apply_active_action_dims(actions, payload, active_dims):
    return apply_active_action_dims_with_reference(actions, payload, active_dims, frozen_reference=None)


def parse_action_reference_values(values):
    raw = (values or "").strip()
    if not raw:
        return None
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != 14:
        raise ValueError(f"--frozen_action_values expects 14 comma-separated floats, got {len(parts)}")
    return np.asarray([float(item) for item in parts], dtype=np.float32)


def load_dataset_action_reference(dataset_root, source):
    """Load a 14-dim action reference from a LeRobot v3 dataset."""
    if source == "current_state":
        return None

    root = Path(dataset_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"--frozen_action_dataset does not exist: {root}")

    import pandas as pd

    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    actions = []
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file, columns=["action"])
        actions.append(np.stack(df["action"].to_numpy()).astype(np.float32))
    all_actions = np.concatenate(actions, axis=0)
    if all_actions.shape[1] < 14:
        raise ValueError(f"Expected dataset action dim >= 14, got {all_actions.shape}")
    all_actions = all_actions[:, :14]

    if source == "dataset_mean":
        return all_actions.mean(axis=0).astype(np.float32)
    if source == "dataset_median":
        return np.median(all_actions, axis=0).astype(np.float32)
    if source == "dataset_first":
        return all_actions[0].astype(np.float32)
    raise ValueError(
        "--frozen_action_source must be one of: current_state, dataset_mean, dataset_median, dataset_first, values"
    )


def apply_active_action_dims_with_reference(actions, payload, active_dims, frozen_reference):
    """Keep policy output only on active dims; frozen dims hold a reference pose."""
    active_dims = sorted(set(active_dims))
    if len(active_dims) == 14:
        return actions

    reference = frozen_reference
    if reference is None:
        reference = extract_qpos_from_payload(payload)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    if reference.shape[0] < 14:
        raise ValueError(f"Frozen action reference must have dim >= 14, got {reference.shape}")
    reference = reference[:14]

    masked = np.array(actions, dtype=np.float32, copy=True)
    frozen_dims = [dim for dim in range(14) if dim not in active_dims]
    if masked.ndim == 1:
        masked[frozen_dims] = reference[frozen_dims]
    else:
        masked[:, frozen_dims] = reference[frozen_dims]
    return masked.astype(np.float32)


def extract_action(action_obj):
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

    # Compatible with [14], [1,14], [50,14], [1,50,14]
    if action.ndim == 1:
        pass
    elif action.shape[-1] == 14:
        action = action.reshape(-1, 14)[0]
    else:
        action = action.reshape(-1)

    action = action.reshape(-1)

    if action.shape[0] < 14:
        raise ValueError(f"Expected action dim >= 14, got {action.shape}")

    return action[:14].astype(np.float32)


def extract_action_chunk(action_obj):
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

    if actions.shape[-1] < 14:
        raise ValueError(f"Expected action chunk dim >= 14, got {actions.shape}")

    return actions[:, :14].astype(np.float32)


def infer_action(policy, preprocessor, postprocessor, payload, device, active_dims, frozen_reference):
    obs = build_observation(payload, device)
    with torch.inference_mode():
        processed_obs = preprocessor(obs)
        action_obj = policy.select_action(processed_obs)
        action_obj = postprocessor(action_obj)
    action = extract_action(action_obj)
    return apply_active_action_dims_with_reference(action, payload, active_dims, frozen_reference)


def infer_action_chunk(policy, preprocessor, postprocessor, payload, device, active_dims, frozen_reference):
    obs = build_observation(payload, device)
    with torch.inference_mode():
        processed_obs = preprocessor(obs)
        action_obj = policy.predict_action_chunk(processed_obs)
        action_obj = postprocessor(action_obj)
    actions = extract_action_chunk(action_obj)
    if actions.shape[0] < 50:
        pad = np.repeat(actions[-1:], 50 - actions.shape[0], axis=0)
        actions = np.concatenate([actions, pad], axis=0)
    actions = actions[:50]
    return apply_active_action_dims_with_reference(actions, payload, active_dims, frozen_reference)


def build_metadata(policy, policy_path, active_dims=None, frozen_reference=None, frozen_action_source="current_state"):
    active_dims = list(range(14)) if active_dims is None else list(active_dims)
    frozen_dims = [dim for dim in range(14) if dim not in active_dims]
    return {
        "model": "lerobot_pi05",
        "policy_path": policy_path,
        "input_features": {k: tuple(v.shape) for k, v in policy.config.input_features.items()},
        "output_features": {k: tuple(v.shape) for k, v in policy.config.output_features.items()},
        "active_action_dims": active_dims,
        "active_action_names": [ACTION_NAMES[dim] for dim in active_dims],
        "frozen_action_dims": frozen_dims,
        "frozen_action_names": [ACTION_NAMES[dim] for dim in frozen_dims],
        "frozen_action_source": frozen_action_source,
        "frozen_action_reference": None if frozen_reference is None else frozen_reference.astype(float).tolist(),
    }


def write_action_log(args, step, action_result, active_dims, frozen_reference):
    action_log = getattr(args, "_action_log_file", None)
    if action_log is None:
        return

    action_arr = np.asarray(action_result, dtype=np.float32)
    active_dims = list(active_dims)
    frozen_dims = [dim for dim in range(14) if dim not in active_dims]
    entry = {
        "time": time.time(),
        "server_step": int(step),
        "transport": args.transport,
        "action_mode": args.action_mode,
        "action_shape": list(action_arr.shape),
        "active_action_dims": active_dims,
        "active_action_names": [ACTION_NAMES[dim] for dim in active_dims],
        "frozen_action_dims": frozen_dims,
        "frozen_action_names": [ACTION_NAMES[dim] for dim in frozen_dims],
        "frozen_action_source": args.frozen_action_source,
        "frozen_action_reference": None if frozen_reference is None else frozen_reference.astype(float).tolist(),
        "actions": action_arr.astype(float).tolist(),
    }
    action_log.write(json.dumps(entry) + "\n")
    action_log.flush()


async def run_openpi_ws_server(
    args, policy, preprocessor, postprocessor, policy_path, device, active_dims, frozen_reference
):
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
            print("[SERVER] policy reset for new websocket client")
        if not args.no_initial_metadata:
            await websocket.send(
                msgpack.packb(
                    build_metadata(policy, policy_path, active_dims, frozen_reference, args.frozen_action_source),
                    use_bin_type=True,
                )
            )
            print("[SERVER] sent initial metadata")

        try:
            async for message in websocket:
                t0 = time.time()
                if isinstance(message, str):
                    message = message.encode("utf-8")

                request = decode_wire_value(m.unpackb(message, raw=False))
                method = request.get("method", "infer")
                if args.print_action:
                    print(f"[SERVER] websocket method={method} keys={sorted(request.keys())}")

                if method == "reset":
                    async with infer_lock:
                        if hasattr(policy, "reset"):
                            policy.reset()
                    reply = {"ok": True}
                elif method in {"get_server_metadata", "get_metadata", "metadata"}:
                    reply = build_metadata(policy, policy_path, active_dims, frozen_reference, args.frozen_action_source)
                elif method == "infer":
                    async with infer_lock:
                        if args.action_mode == "chunk":
                            action_result = infer_action_chunk(
                                policy, preprocessor, postprocessor, request, device, active_dims, frozen_reference
                            )
                        else:
                            action_result = infer_action(
                                policy, preprocessor, postprocessor, request, device, active_dims, frozen_reference
                            )
                        step = state["step"]
                        state["step"] += 1

                    action_result = action_result.astype(np.float32)
                    # Keep WebSocket responses on plain MessagePack types. Some clients
                    # use a different msgpack-numpy version/patch mode and decode ndarray
                    # extension objects as 0-D scalars. A nested list is version-neutral;
                    # clients can recover float32 with np.asarray(actions, dtype=np.float32).
                    action_payload = action_result.tolist()
                    reply = {
                        "actions": action_payload,
                        "action": action_payload,
                        "action_mode": args.action_mode,
                        "server_step": int(step),
                        "latency_s": float(time.time() - t0),
                    }
                    write_action_log(args, step, action_result, active_dims, frozen_reference)
                    if args.print_action or step % 10 == 0:
                        if args.action_mode == "chunk":
                            action_desc = (
                                f"actions_shape={action_result.shape} "
                                f"first_action={action_result[0]}"
                            )
                        else:
                            action_desc = f"action={action_result}"
                        print(
                            f"[SERVER] step={step} "
                            f"latency={reply['latency_s']:.3f}s "
                            f"mode={args.action_mode} "
                            f"{action_desc}"
                        )
                else:
                    reply = {"error": f"unknown method: {method}"}

                await websocket.send(msgpack.packb(reply, use_bin_type=True))
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[SERVER] websocket disconnected: {addr} code={e.code} reason={e.reason}")
        except Exception as e:
            import traceback

            print(f"[SERVER] websocket error: {repr(e)}")
            traceback.print_exc()
            try:
                await websocket.send(msgpack.packb({"error": repr(e)}, use_bin_type=True))
            except Exception:
                pass

    ping_timeout = None if args.client_timeout_s <= 0 else args.client_timeout_s
    ping_interval = None if args.client_timeout_s <= 0 else min(20, max(1, args.client_timeout_s // 2))

    print_server_addresses(args.host, args.port, "openpi_ws")
    async with websockets.serve(
        handler,
        args.host,
        args.port,
        max_size=None,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    ):
        await asyncio.Future()


def main():
    # Port scanners or TCP clients hitting the WebSocket endpoint can trigger noisy
    # "opening handshake failed" tracebacks in websockets. Keep server output focused.
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

    for proxy_key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        os.environ.pop(proxy_key, None)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_path",
        type=str,
        default=(
            "/mnt/bigdata/SUN_ht/runs/"
            "pi05_arx_0723_1401_2026-07-24_10-05-11/checkpoints/004000/pretrained_model"
        ),
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--transport", choices=["openpi_ws", "tcp_pickle"], default="openpi_ws")
    parser.add_argument(
        "--action_mode",
        choices=["chunk", "single"],
        default="chunk",
        help="Return a 50-step action chunk or a single 14-dim action for each inference request.",
    )
    parser.add_argument("--no_initial_metadata", action="store_true")
    parser.add_argument("--print_action", action="store_true")
    parser.add_argument(
        "--action_log_path",
        default="",
        help="Optional JSONL path for every action returned by the server.",
    )
    parser.add_argument(
        "--active_arms",
        type=str,
        default="both",
        help=(
            "Which arm action dims are allowed to use PI05 output when --active_action_dims is empty. "
            "Use both/all, left, right, or none. Frozen dims are overwritten by current qpos/state."
        ),
    )
    parser.add_argument(
        "--active_action_dims",
        type=str,
        default="",
        help=(
            "Comma-separated action dims or names to keep from PI05 output, e.g. "
            "'0,1,2,6' or 'left_joint1,left_joint2,left_gripper'. "
            "All other dims are overwritten by current qpos/state. Overrides --active_arms when non-empty."
        ),
    )
    parser.add_argument(
        "--frozen_action_source",
        choices=["current_state", "dataset_mean", "dataset_median", "dataset_first", "values"],
        default="current_state",
        help=(
            "Reference used for frozen action dims. current_state uses client qpos/state. "
            "dataset_* loads a 14-dim reference from --frozen_action_dataset. "
            "values uses --frozen_action_values."
        ),
    )
    parser.add_argument(
        "--frozen_action_dataset",
        type=str,
        default="/data/SUN_ht/datasets/arx_0723_1401",
        help="LeRobot dataset root used when --frozen_action_source is dataset_mean/median/first.",
    )
    parser.add_argument(
        "--frozen_action_values",
        type=str,
        default="",
        help="14 comma-separated floats used when --frozen_action_source=values.",
    )
    parser.add_argument(
        "--client_timeout_s",
        type=float,
        default=300.0,
        help=(
            "Seconds to wait before considering an idle client dead. "
            "For tcp_pickle this is the socket read timeout; for openpi_ws this is the ping timeout. "
            "Set to 0 to disable timeout/ping keepalive."
        ),
    )
    args = parser.parse_args()
    active_dims = parse_active_action_dims(args.active_arms, args.active_action_dims)
    frozen_dims = [dim for dim in range(14) if dim not in active_dims]
    if args.frozen_action_source == "values":
        frozen_reference = parse_action_reference_values(args.frozen_action_values)
    else:
        frozen_reference = load_dataset_action_reference(args.frozen_action_dataset, args.frozen_action_source)

    device = torch.device(args.device)

    from lerobot.policies.pi05 import PI05Policy

    policy_dir = resolve_policy_dir(args.policy_path)
    required_files = ("config.json", "model.safetensors", "policy_preprocessor.json")
    missing_files = [name for name in required_files if not (policy_dir / name).is_file()]
    if missing_files:
        parser.error(f"Incomplete PI05 checkpoint at {policy_dir}: missing {missing_files}")
    policy_path = str(policy_dir)
    print(f"[SERVER] Loading PI05 from: {policy_path}")

    policy = PI05Policy.from_pretrained(policy_path, device=str(device))
    policy.eval()

    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = load_pre_post_processors(policy, policy_path, device)

    print("[SERVER] PI05 loaded.")
    print("[SERVER] input_features:", policy.config.input_features)
    print("[SERVER] output_features:", policy.config.output_features)
    print("[SERVER] action_mode:", args.action_mode)
    print(
        "[SERVER] active action dims:",
        active_dims,
        [ACTION_NAMES[dim] for dim in active_dims],
    )
    print(
        "[SERVER] frozen action dims:",
        frozen_dims,
        [ACTION_NAMES[dim] for dim in frozen_dims],
    )
    print("[SERVER] frozen action source:", args.frozen_action_source)
    if frozen_reference is None:
        print("[SERVER] frozen action reference: client current qpos/state")
    else:
        print("[SERVER] frozen action reference:", frozen_reference.tolist())
    args._action_log_file = None
    if args.action_log_path:
        action_log_path = Path(args.action_log_path).expanduser()
        action_log_path.parent.mkdir(parents=True, exist_ok=True)
        args._action_log_file = action_log_path.open("a", buffering=1)
        print("[SERVER] action log:", action_log_path)

    if args.transport == "openpi_ws":
        asyncio.run(
            run_openpi_ws_server(
                args, policy, preprocessor, postprocessor, policy_path, device, active_dims, frozen_reference
            )
        )
        return

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print_server_addresses(args.host, args.port, args.transport)

    while True:
        conn, addr = server.accept()
        if args.client_timeout_s > 0:
            conn.settimeout(args.client_timeout_s)
        else:
            conn.settimeout(None)
        print(f"[SERVER] Connected by {addr}")
        if hasattr(policy, "reset"):
            policy.reset()
            print("[SERVER] policy reset for new client connection")
        try:
            step = 0
            while True:
                t0 = time.time()

                payload = decode_wire_value(recv_msg(conn))

                if args.action_mode == "chunk":
                    action_result = infer_action_chunk(
                        policy, preprocessor, postprocessor, payload, device, active_dims, frozen_reference
                    )
                else:
                    action_result = infer_action(
                        policy, preprocessor, postprocessor, payload, device, active_dims, frozen_reference
                    )

                reply = {
                    "ok": True,
                    # Convert numpy array to plain Python list to avoid numpy 1.x / 2.x pickle incompatibility.
                    "action": action_result.astype(float).tolist(),
                    "actions": action_result.astype(float).tolist(),
                    "action_mode": args.action_mode,
                    "server_step": int(step),
                    "latency_s": float(time.time() - t0),
                }
                write_action_log(args, step, action_result, active_dims, frozen_reference)

                send_msg(conn, reply)

                if args.print_action or step % 10 == 0:
                    if args.action_mode == "chunk":
                        action_desc = (
                            f"actions_shape={action_result.shape} "
                            f"first_action={action_result[0]}"
                        )
                    else:
                        action_desc = f"action={action_result}"
                    print(
                        f"[SERVER] step={step} "
                        f"latency={reply['latency_s']:.3f}s "
                        f"mode={args.action_mode} "
                        f"{action_desc}"
                    )

                step += 1

        except ConnectionError:
            print("[SERVER] client disconnected")
        except TimeoutError:
            print("[SERVER] client idle timeout, closing connection")
        except socket.timeout:
            print("[SERVER] client idle timeout, closing connection")
        except KeyboardInterrupt:
            print("[SERVER] keyboard interrupt")
            break
        except Exception as e:
            import traceback
            print(f"[SERVER] error at step={step}: {repr(e)}")
            traceback.print_exc()
            try:
                send_msg(conn, {"ok": False, "error": repr(e)})
            except Exception:
                pass
        finally:
            conn.close()

    server.close()


if __name__ == "__main__":
    main()
