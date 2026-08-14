#!/usr/bin/env python
"""Run PI05 locally on the Innov bimanual arm without a policy server."""

from __future__ import annotations

import argparse
import select
import threading
import json
import os
import sys
import termios
import time
import tty
from pathlib import Path


ROBODEPLOY_SRC = Path("/data/SUN_ht/roboploy/robodeploy/src")
if str(ROBODEPLOY_SRC) not in sys.path:
    sys.path.insert(0, str(ROBODEPLOY_SRC))

HF_CACHE_BASE = "/data/SUN_ht/pi/cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_BASE
os.environ["HF_HUB_CACHE"] = f"{HF_CACHE_BASE}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_BASE}/transformers"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for proxy_key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(proxy_key, None)

import numpy as np
import torch


DEFAULT_CAMERAS = {
    "front": {
        "type": "intelrealsense",
        "serial_number_or_name": "935422072733",
        "width": 848,
        "height": 480,
        "fps": 30,
    },
    "left_wrist": {
        "type": "intelrealsense",
        "serial_number_or_name": "409122273564",
        "width": 640,
        "height": 480,
        "fps": 30,
    },
    "right_wrist": {
        "type": "intelrealsense",
        "serial_number_or_name": "409122273228",
        "width": 640,
        "height": 480,
        "fps": 30,
    },
}

DEFAULT_TOKENIZER = (
    "/data/SUN_ht/pi/cache/huggingface/hub/models--google--paligemma-3b-pt-224/"
    "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
)


def resolve_policy_dir(policy_path: str) -> Path:
    raw_path = Path(policy_path).expanduser()
    candidates = [raw_path]
    if raw_path.name == "pretrained_model":
        candidates.append(raw_path.parent)
    elif raw_path.name.isdigit():
        candidates.extend([raw_path / "pretrained_model", raw_path.with_name(f"{int(raw_path.name):06d}")])
    else:
        candidates.append(raw_path / "pretrained_model")

    for candidate in candidates:
        if candidate.is_dir() and candidate.name == "pretrained_model":
            return candidate
        pretrained_dir = candidate / "pretrained_model"
        if pretrained_dir.is_dir():
            return pretrained_dir
    return raw_path


def make_camera_configs(raw: str):
    from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    specs = json.loads(raw)
    cameras = {}
    for name, spec in specs.items():
        cam_type = spec.get("type", "intelrealsense")
        if cam_type != "intelrealsense":
            raise ValueError(f"Only intelrealsense cameras are wired in this script, got {name}={cam_type}")
        cameras[name] = RealSenseCameraConfig(
            serial_number_or_name=spec["serial_number_or_name"],
            width=spec.get("width"),
            height=spec.get("height"),
            fps=spec.get("fps"),
        )
    return cameras


def image_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got {img.shape}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype == np.uint8:
        x = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True, order="C")).to(device)
        x = x.permute(2, 0, 1).float() / 255.0
    else:
        img = np.asarray(img, dtype=np.float32)
        if float(np.nanmax(img)) > 1.5:
            img = np.clip(img, 0.0, 255.0) / 255.0
        x = torch.from_numpy(np.array(img, dtype=np.float32, copy=True, order="C")).to(device)
        x = x.permute(2, 0, 1).float()
    return x.unsqueeze(0)


def stack_front_cameras_for_policy(obs: dict, expected_front_shape: tuple[int, ...] | None = None) -> dict:
    """Return a policy observation with front + front_1 stacked as front when present.

    record_body_teaching.py uses the same convention for four-camera setups:
    front and front_1 are vertically concatenated and the policy receives the
    result as observation.images.front.
    """
    if "front" not in obs or "front_1" not in obs:
        return obs

    front = np.asarray(obs["front"])
    front_1 = np.asarray(obs["front_1"])
    if front.ndim != 3 or front_1.ndim != 3:
        raise ValueError(f"Expected front/front_1 images to be HWC/CHW, got {front.shape} and {front_1.shape}")

    if front.shape[0] == 3 and front.shape[-1] != 3:
        front = np.transpose(front, (1, 2, 0))
    if front_1.shape[0] == 3 and front_1.shape[-1] != 3:
        front_1 = np.transpose(front_1, (1, 2, 0))

    if front.shape[1] != front_1.shape[1] or front.shape[2] != front_1.shape[2]:
        raise ValueError(f"Cannot stack front/front_1 with shapes {front.shape} and {front_1.shape}")

    stacked = np.concatenate([front, front_1], axis=0)
    if expected_front_shape:
        # Policy image feature shape is CHW. Only warn when the model clearly
        # expects a different front height/width; image processors may still
        # resize downstream.
        if len(expected_front_shape) == 3:
            _, expected_h, expected_w = expected_front_shape
            if expected_h not in (None, stacked.shape[0]) or expected_w not in (None, stacked.shape[1]):
                print(
                    "[WARN] stacked front shape does not match policy feature: "
                    f"stacked={stacked.shape}, expected CHW={expected_front_shape}"
                )

    policy_obs = dict(obs)
    policy_obs["front"] = stacked
    policy_obs.pop("front_1", None)
    return policy_obs


def camera_for_feature(feature_key: str) -> str:
    suffix = feature_key.rsplit(".", 1)[-1]
    aliases = {
        "front": "front",
        "front_1": "front_1",
        "top": "front",
        "base_0_rgb": "front",
        "left_wrist": "left_wrist",
        "left_hand": "left_wrist",
        "left_wrist_0_rgb": "left_wrist",
        "right_wrist": "right_wrist",
        "right_hand": "right_wrist",
        "right_wrist_0_rgb": "right_wrist",
    }
    if suffix not in aliases:
        raise KeyError(f"No camera alias for policy image feature: {feature_key}")
    return aliases[suffix]


def get_feature_shape(feature) -> tuple[int, ...]:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    return tuple(shape or ())


def policy_expects_front_1(input_features: dict) -> bool:
    return "observation.images.front_1" in input_features


def policy_expects_stacked_front(input_features: dict) -> bool:
    if policy_expects_front_1(input_features):
        return False
    front_shape = get_feature_shape(input_features.get("observation.images.front"))
    if len(front_shape) != 3:
        return False
    _channels, height, _width = front_shape
    return height >= 900


def build_policy_batch(policy, obs: dict, action_features: dict[str, type], device: torch.device, task: str) -> dict:
    input_features = policy.config.input_features
    front_shape = get_feature_shape(input_features.get("observation.images.front"))
    policy_obs = stack_front_cameras_for_policy(obs, front_shape) if policy_expects_stacked_front(input_features) else obs
    state_keys = list(action_features.keys())
    state = np.array([policy_obs.get(key, 0.0) for key in state_keys], dtype=np.float32)

    batch = {"task": [task]}
    if "observation.state" in input_features:
        state_dim = get_feature_shape(input_features["observation.state"])[0]
        if state.size < state_dim:
            state = np.pad(state, (0, state_dim - state.size), mode="constant")
        batch["observation.state"] = torch.from_numpy(state[:state_dim]).to(device).unsqueeze(0)

    image_features = [key for key in input_features if key.startswith("observation.images.")]
    for feature_key in image_features:
        camera_name = camera_for_feature(feature_key)
        if camera_name not in policy_obs:
            raise KeyError(f"Camera '{camera_name}' missing from robot observation for {feature_key}")
        batch[feature_key] = image_to_tensor(policy_obs[camera_name], device)

    return batch


def required_camera_names(policy) -> list[str]:
    names = []
    input_features = policy.config.input_features
    for feature_key in input_features:
        if feature_key.startswith("observation.images."):
            camera_name = camera_for_feature(feature_key)
            if camera_name not in names:
                names.append(camera_name)
    if policy_expects_stacked_front(input_features) and "front_1" not in names:
        names.append("front_1")
    return names


def extract_action_chunk(action_obj) -> np.ndarray:
    if isinstance(action_obj, dict):
        action_obj = action_obj.get("action", action_obj.get("actions", action_obj.get("ACTION")))
    if torch.is_tensor(action_obj):
        action_obj = action_obj.detach().float().cpu().numpy()
    actions = np.asarray(action_obj, dtype=np.float32)
    if actions.ndim == 3:
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)
    return actions


def action_keys_for_arm(action_keys: list[str], arm: str) -> list[str]:
    prefix = f"{arm}_"
    return [key for key in action_keys if key.startswith(prefix)]


def infer_policy_action_arms(policy_action_dim: int, robot_action_dim: int, policy_action_arms: str) -> str:
    if policy_action_arms != "auto":
        return policy_action_arms
    if policy_action_dim == robot_action_dim:
        return "both"
    if robot_action_dim == 14 and policy_action_dim == 7:
        return "right"
    raise ValueError(
        "Cannot infer how to map policy actions to robot actions: "
        f"policy_action_dim={policy_action_dim}, robot_action_dim={robot_action_dim}. "
        "Set --policy_action_arms to left, right, or both."
    )


def adapt_action_chunk_to_robot(
    actions: np.ndarray,
    action_features: dict[str, type],
    reference_action: dict[str, float],
    policy_action_arms: str,
) -> np.ndarray:
    """Adapt policy action chunks to the robot action dimension.

    A 14-dim policy chunk is kept as-is. A 7-dim chunk can drive one arm while
    the other arm is held at the reference pose from current state/previous action.
    """
    action_keys = list(action_features.keys())
    robot_action_dim = len(action_keys)
    policy_action_dim = int(actions.shape[-1])
    arm_mode = infer_policy_action_arms(policy_action_dim, robot_action_dim, policy_action_arms)

    if arm_mode == "both":
        if policy_action_dim < robot_action_dim:
            raise ValueError(f"Policy action dim {policy_action_dim} < robot action dim {robot_action_dim}")
        return actions[:, :robot_action_dim]

    if arm_mode not in {"left", "right"}:
        raise ValueError(f"Unsupported --policy_action_arms={policy_action_arms!r}")

    target_keys = action_keys_for_arm(action_keys, arm_mode)
    if policy_action_dim < len(target_keys):
        raise ValueError(
            f"Policy action dim {policy_action_dim} is too small for {arm_mode} arm "
            f"({len(target_keys)} dims required)"
        )
    if not target_keys:
        raise ValueError(f"Robot action features do not contain {arm_mode} arm keys: {action_keys}")

    reference_np = action_dict_to_np(reference_action, action_keys)
    adapted = np.repeat(reference_np[None, :], actions.shape[0], axis=0).astype(np.float32)
    key_to_idx = {key: idx for idx, key in enumerate(action_keys)}
    for src_idx, key in enumerate(target_keys):
        adapted[:, key_to_idx[key]] = actions[:, src_idx]
    return adapted


def action_to_dict(action_np: np.ndarray, action_features: dict[str, type]) -> dict[str, float]:
    keys = list(action_features.keys())
    if len(action_np) != len(keys):
        raise ValueError(f"Action dim mismatch: policy={len(action_np)}, robot={len(keys)}")
    return {key: float(action_np[i]) for i, key in enumerate(keys)}


def parse_optional_action_vector(raw: str, action_dim: int, name: str) -> np.ndarray | None:
    raw = raw.strip()
    if not raw:
        return None
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) == 1:
        return np.full(action_dim, values[0], dtype=np.float32)
    if len(values) != action_dim:
        raise ValueError(f"{name} must contain either 1 value or {action_dim} comma-separated values, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def action_dict_to_np(action: dict[str, float], action_keys: list[str]) -> np.ndarray:
    return np.asarray([float(action[k]) for k in action_keys], dtype=np.float32)


def observation_action_reference(obs: dict, action_keys: list[str]) -> dict[str, float]:
    return {key: float(obs.get(key, 0.0)) for key in action_keys}


def filter_and_limit_action(
    raw_action: dict[str, float],
    reference_action: dict[str, float],
    action_keys: list[str],
    filter_alpha: float,
    max_delta: float,
    clip_min: np.ndarray | None,
    clip_max: np.ndarray | None,
) -> dict[str, float]:
    """Low-pass filter and clamp one policy action before it can be sent to hardware."""
    raw_np = action_dict_to_np(raw_action, action_keys)
    ref_np = action_dict_to_np(reference_action, action_keys)

    filtered_np = raw_np
    if filter_alpha < 1.0:
        filtered_np = filter_alpha * raw_np + (1.0 - filter_alpha) * ref_np

    if max_delta > 0:
        filtered_np = np.clip(filtered_np, ref_np - max_delta, ref_np + max_delta)

    if clip_min is not None:
        filtered_np = np.maximum(filtered_np, clip_min)
    if clip_max is not None:
        filtered_np = np.minimum(filtered_np, clip_max)

    return {key: float(filtered_np[i]) for i, key in enumerate(action_keys)}


class CameraPreview:
    def __init__(self, enabled: bool, window_name: str, camera_names: list[str], image_width: int):
        self.enabled = enabled
        self.window_name = window_name
        self.camera_names = camera_names
        self.image_width = image_width
        self._cv2 = None
        self._active = False

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            import cv2

            self._cv2 = cv2
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._active = True
            print("[INFO] Camera preview enabled. Focus preview window: 'z'=home, 'q'=close preview.")
        except Exception as exc:  # noqa: BLE001 - preview should not block robot control.
            print(f"[WARN] Camera preview disabled: {exc}")
            self.enabled = False
            self._active = False
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        if self._active and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self._active = False

    def _format_frame(self, frame: np.ndarray, name: str) -> np.ndarray:
        cv2 = self._cv2
        img = np.asarray(frame)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim != 3:
            img = np.zeros((240, 320, 3), dtype=np.uint8)

        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            if float(np.nanmax(img)) <= 1.5:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.shape[-1] == 3:
            # Most RealSense wrappers return RGB; OpenCV display expects BGR.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]
        target_w = max(64, self.image_width)
        target_h = max(48, int(h * target_w / max(1, w)))
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.putText(img, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return img

    def update(self, obs: dict) -> str | None:
        if not self.enabled or not self._active or self._cv2 is None:
            return None
        try:
            frames = []
            for name in self.camera_names:
                if name in obs:
                    frames.append(self._format_frame(obs[name], name))
            if not frames:
                return None
            max_h = max(frame.shape[0] for frame in frames)
            padded = []
            for frame in frames:
                if frame.shape[0] < max_h:
                    pad = np.zeros((max_h - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                    frame = np.vstack([frame, pad])
                padded.append(frame)
            canvas = np.hstack(padded)
            self._cv2.imshow(self.window_name, canvas)
            key = self._cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Camera preview closed by user.")
                self.close()
                return "q"
            if key == ord("z"):
                return "z"
        except Exception as exc:  # noqa: BLE001 - disable flaky GUI without stopping inference.
            print(f"[WARN] Camera preview disabled after error: {exc}")
            self.close()
            self.enabled = False
        return None


def read_camera_with_retry(camera, camera_name: str, timeout_ms: int, retries: int) -> np.ndarray:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return camera.read(timeout_ms=timeout_ms)
        except Exception as exc:  # noqa: BLE001 - hardware retries need the original exception.
            last_error = exc
            print(f"[WARN] camera {camera_name} read failed ({attempt}/{retries}): {exc}")
            time.sleep(0.1)
    raise RuntimeError(f"Camera {camera_name} failed after {retries} retries") from last_error


def get_observation_sync_cameras(robot, timeout_ms: int, retries: int) -> dict:
    """Read robot state and cameras, using synchronous RealSense reads for reliability."""
    obs = {}

    # BiInnovArmV1Robot state path, kept local to avoid async camera failures
    # inside robot.get_observation() from hiding which camera failed.
    if hasattr(robot, "left_arm") and hasattr(robot, "right_arm"):
        left_pos = robot.left_arm.get_current_joint_angles()
        left_gripper = robot.left_arm.get_current_gripper_angles()
        right_pos = robot.right_arm.get_current_joint_angles()
        right_gripper = robot.right_arm.get_current_gripper_angles()

        for i in range(6):
            obs[f"left_joint{i + 1}.pos"] = float(left_pos[i])
            obs[f"right_joint{i + 1}.pos"] = float(right_pos[i])
        obs["left_gripper.pos"] = float(left_gripper)
        obs["right_gripper.pos"] = float(right_gripper)
    else:
        obs.update(robot.get_observation())

    for camera_name, camera in robot.cameras.items():
        obs[camera_name] = read_camera_with_retry(camera, camera_name, timeout_ms, retries)

    return obs


def load_policy(policy_path: Path, device: torch.device, tokenizer_path: str):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(str(policy_path), device=str(device))
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        str(policy_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": tokenizer_path},
            "device_processor": {"device": str(device)},
        },
    )
    return policy, preprocessor, postprocessor


def send_smoothed_action_interruptible(
    robot,
    prev_action: dict[str, float] | None,
    new_action: dict[str, float],
    action_features: dict[str, type],
    max_step: float,
    home_listener,
    dt: float = 0.005,
) -> bool:
    """Send one policy action, interrupting before any sub-step if z was pressed."""
    if home_listener.home_requested.is_set():
        return False

    if prev_action is None or max_step <= 0:
        robot.send_action(new_action)
        return True

    keys = list(action_features.keys())
    prev_np = np.array([prev_action.get(k, 0.0) for k in keys], dtype=np.float64)
    new_np = np.array([new_action.get(k, 0.0) for k in keys], dtype=np.float64)
    max_disp = float(abs(new_np - prev_np).max())

    if max_disp <= max_step:
        if home_listener.home_requested.is_set():
            return False
        robot.send_action(new_action)
        return True

    steps = int(np.ceil(max_disp / max_step))
    for i in range(1, steps):
        if home_listener.home_requested.is_set():
            return False
        t_val = i / steps
        t_smoothed = (1 - np.cos(t_val * np.pi)) / 2
        interp = prev_np * (1 - t_smoothed) + new_np * t_smoothed
        action = {key: float(interp[j]) for j, key in enumerate(keys)}
        robot.send_action(action)
        time.sleep(dt)

    if home_listener.home_requested.is_set():
        return False
    robot.send_action(new_action)
    return True


class KeyboardHomeListener:
    """Background key listener that only raises events; robot writes stay in main thread."""

    def __init__(self, home_key: str = "z"):
        self.home_key = home_key.lower()
        self.home_requested = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_termios = None
        self._enabled = False

    def __enter__(self):
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; hotkey home is disabled.")
            return self

        fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._drain_stdin()
        self._enabled = True
        self._thread = threading.Thread(target=self._run, name="keyboard-home-listener", daemon=True)
        self._thread.start()
        print(f"[INFO] Hotkey: press '{self.home_key}' to stop policy actions and move to zero/home pose.")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._old_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
        return False

    def _drain_stdin(self) -> None:
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char = sys.stdin.read(1)
                    if char.lower() == self.home_key:
                        self.home_requested.set()
                        print(f"\n[INFO] Hotkey '{self.home_key}' received: policy action output will be paused.")
            except Exception as exc:  # noqa: BLE001 - keep the robot loop alive if keyboard fails.
                print(f"[WARN] keyboard listener stopped: {exc}")
                return

    def consume_home_request(self) -> bool:
        if not self.home_requested.is_set():
            return False
        self.home_requested.clear()
        return True


def selected_arms(robot, arms: str):
    if arms == "none":
        return []
    if arms == "left":
        return [("left", robot.left_arm)]
    if arms == "right":
        return [("right", robot.right_arm)]
    if arms == "both":
        return [("left", robot.left_arm), ("right", robot.right_arm)]
    raise ValueError(f"Unsupported arms={arms!r}")


def close_robot_serials(robot) -> None:
    for arm_name in ["left_arm", "right_arm"]:
        arm = getattr(robot, arm_name, None)
        if arm is None:
            continue
        if hasattr(arm, "disable"):
            try:
                arm.disable()
            except Exception as exc:  # noqa: BLE001 - best-effort hardware cleanup
                print(f"[WARN] failed to disable {arm_name}: {exc}")
        if hasattr(arm, "close_serial"):
            arm.close_serial()


def parse_home_joints(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError(f"--home_joints must contain 6 comma-separated values, got {len(values)}: {raw!r}")
    return values


def configure_arm_for_position_control(arm) -> None:
    arm.enable()
    time.sleep(0.1)
    arm.type = "follower"
    arm.set_pos_vel_mode()
    time.sleep(0.1)
    arm.enable()


def read_arm_state(arm) -> tuple[list[float], float]:
    joints = [float(v) for v in arm.get_current_joint_angles()]
    gripper = float(arm.get_current_gripper_angles())
    return joints, gripper


def move_arm_to_home(
    arm,
    name: str,
    target_joints: list[float],
    target_gripper: float | None,
    duration_s: float,
    fps: int,
    joint_velocity: float,
    gripper_velocity: float,
) -> None:
    configure_arm_for_position_control(arm)
    start_joints, start_gripper = read_arm_state(arm)
    print(
        f"[INFO] [{name}] home start joints={[round(v, 4) for v in start_joints]} "
        f"gripper={round(start_gripper, 4)}"
    )
    print(
        f"[INFO] [{name}] home target joints={[round(v, 4) for v in target_joints]} "
        f"gripper={target_gripper if target_gripper is not None else 'keep-current'}"
    )

    steps = max(1, int(duration_s * fps))
    period_s = 1.0 / max(1, fps)
    for step_i in range(1, steps + 1):
        alpha = step_i / steps
        joints = [(1.0 - alpha) * start_joints[i] + alpha * target_joints[i] for i in range(6)]
        arm.set_joint_angles(joints, joint_velocity)
        if target_gripper is not None:
            gripper = (1.0 - alpha) * start_gripper + alpha * target_gripper
            arm.set_gripper_angles(gripper_angle=gripper, v=gripper_velocity, tau_limit=0.1)
        time.sleep(period_s)

    arm.set_joint_angles(target_joints, joint_velocity)
    if target_gripper is not None:
        arm.set_gripper_angles(gripper_angle=target_gripper, v=gripper_velocity, tau_limit=0.1)
    time.sleep(0.2)
    end_joints, end_gripper = read_arm_state(arm)
    print(
        f"[INFO] [{name}] home done joints={[round(v, 4) for v in end_joints]} "
        f"gripper={round(end_gripper, 4)}"
    )


def maybe_move_home(
    robot,
    home_arms: str,
    home_joints: list[float],
    home_gripper: float | None,
    home_duration_s: float,
    home_fps: int,
    home_joint_velocity: float,
    home_gripper_velocity: float,
    home_no_confirm: bool,
) -> None:
    arms = selected_arms(robot, home_arms)
    if not arms:
        return

    selected = ", ".join(name for name, _ in arms)
    print("[WARN] Move-to-zero/home requested.")
    print("[WARN] This sends position-control commands to move selected arm(s) to the target zero pose.")
    print("[WARN] Make sure the workspace is clear and the current pose can safely move to zero.")
    print(f"[WARN] Selected arm(s): {selected}")
    print(f"[WARN] Target joints: {[round(v, 4) for v in home_joints]}")
    print(f"[WARN] Target gripper: {home_gripper if home_gripper is not None else 'keep-current'}")
    if not home_no_confirm:
        confirm = input("Type ZERO and press Enter to move to zero pose, or anything else to abort: ")
        if confirm != "ZERO":
            raise RuntimeError("Move-to-zero aborted by user.")

    for name, arm in arms:
        move_arm_to_home(
            arm,
            name,
            home_joints,
            home_gripper,
            home_duration_s,
            home_fps,
            home_joint_velocity,
            home_gripper_velocity,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True, help="Path to checkpoint/pretrained_model.")
    parser.add_argument("--left_port", default="/dev/ttyACM1")
    parser.add_argument("--right_port", default="/dev/ttyACM0")
    parser.add_argument("--cameras", default=json.dumps(DEFAULT_CAMERAS))
    parser.add_argument("--task", default="Move the cup to the center of the table.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration_s", type=float, default=120.0)
    parser.add_argument("--warmup_rounds", type=int, default=3)
    parser.add_argument("--action_smooth_max_step", type=float, default=0.05)
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--camera_timeout_ms", type=int, default=1000)
    parser.add_argument("--camera_retries", type=int, default=5)
    parser.add_argument("--show_cameras", type=int, default=1, help="Show live front/left/right camera preview window.")
    parser.add_argument("--camera_preview_width", type=int, default=420, help="Preview width for each camera panel.")
    parser.add_argument("--camera_preview_window", default="Innov PI05 cameras")
    parser.add_argument("--action_log_path", default="", help="Write every policy action to this JSONL file.")
    parser.add_argument("--print_action_every", type=int, default=30, help="Print one action summary every N steps.")
    parser.add_argument("--print_action", action="store_true")
    parser.add_argument(
        "--action_filter_alpha",
        type=float,
        default=0.5,
        help="EMA filter alpha for policy actions. 1 disables filtering; lower values are smoother.",
    )
    parser.add_argument(
        "--action_max_delta",
        type=float,
        default=0.05,
        help="Max per-dimension action change per control step relative to the last sent/current action. <=0 disables.",
    )
    parser.add_argument(
        "--action_clip_min",
        default="",
        help="Optional absolute min clamp. Empty disables; provide 1 value or action_dim comma-separated values.",
    )
    parser.add_argument(
        "--action_clip_max",
        default="",
        help="Optional absolute max clamp. Empty disables; provide 1 value or action_dim comma-separated values.",
    )
    parser.add_argument(
        "--policy_action_arms",
        choices=["auto", "left", "right", "both"],
        default="auto",
        help=(
            "How to map policy action dims to the bimanual robot. auto keeps full-dim actions as both arms "
            "and maps 7-dim actions to the right arm while holding the other arm at the reference pose."
        ),
    )
    parser.add_argument(
        "--zero_arms",
        choices=["none", "left", "right", "both"],
        default="both",
        help="Arm(s) to move to zero/home pose when pressing z. Use 'none' to disable the hotkey.",
    )
    parser.add_argument("--zero_only", action="store_true", help="Move to zero/home pose immediately and exit without loading/running policy.")
    parser.add_argument("--zero_no_confirm", action="store_true", help="Skip interactive ZERO confirmation for move-to-zero.")
    parser.add_argument("--home_joints", default="0,0,0,0,0,0", help="Comma-separated 6 joint target for zero/home pose.")
    parser.add_argument(
        "--home_gripper",
        default="1.0",
        help="Gripper target for zero/home pose. Use 'keep' to keep current gripper position.",
    )
    parser.add_argument("--home_duration_s", type=float, default=5.0)
    parser.add_argument("--home_fps", type=int, default=30)
    parser.add_argument("--home_joint_velocity", type=float, default=1.0)
    parser.add_argument("--home_gripper_velocity", type=float, default=1.0)
    args = parser.parse_args()

    if not 0.0 < args.action_filter_alpha <= 1.0:
        raise ValueError(f"--action_filter_alpha must be in (0, 1], got {args.action_filter_alpha}")

    from robodeploy.robots.lerobot_robot_my_arm.bi_innov_arm_v1 import BiInnovArmV1Robot
    from robodeploy.robots.lerobot_robot_my_arm.config_innov_arm import BiInnovArmV1Config

    device = torch.device(args.device)
    policy_path = resolve_policy_dir(args.policy_path)

    print(f"[INFO] policy    = {policy_path}")
    print(f"[INFO] tokenizer = {args.tokenizer_path}")
    print(f"[INFO] robot     = bi_innov_arm_v1 left={args.left_port} right={args.right_port}")
    print(f"[INFO] task      = {args.task}")
    print(f"[INFO] zero_arms = {args.zero_arms} (hotkey z move-to-zero arms)")

    camera_configs = make_camera_configs(args.cameras)
    robot = BiInnovArmV1Robot(
        BiInnovArmV1Config(
            left_port=args.left_port,
            right_port=args.right_port,
            mode="control",
            cameras=camera_configs,
        )
    )

    if not args.zero_only:
        for required in ["config.json", "model.safetensors", "policy_preprocessor.json"]:
            if not (policy_path / required).is_file():
                close_robot_serials(robot)
                raise FileNotFoundError(f"Missing {required} in {policy_path}")

    home_joints = parse_home_joints(args.home_joints)
    home_gripper = None if args.home_gripper.strip().lower() == "keep" else float(args.home_gripper)

    if args.zero_only and args.zero_arms != "none":
        try:
            maybe_move_home(
                robot,
                args.zero_arms,
                home_joints,
                home_gripper,
                args.home_duration_s,
                args.home_fps,
                args.home_joint_velocity,
                args.home_gripper_velocity,
                args.zero_no_confirm,
            )
        finally:
            if args.zero_only:
                close_robot_serials(robot)
        if args.zero_only:
            print("[INFO] zero_only requested; exiting before policy load/control.")
            return
    elif args.zero_only:
        print("[INFO] zero_only requested but zero_arms=none; exiting without moving.")
        close_robot_serials(robot)
        return

    for required in ["config.json", "model.safetensors", "policy_preprocessor.json"]:
        if not (policy_path / required).is_file():
            close_robot_serials(robot)
            raise FileNotFoundError(f"Missing {required} in {policy_path}")

    policy, preprocessor, postprocessor = load_policy(policy_path, device, args.tokenizer_path)
    required_cameras = required_camera_names(policy)
    missing_cameras = [name for name in required_cameras if name not in camera_configs]
    if missing_cameras:
        raise ValueError(
            "Camera config is missing required policy cameras: "
            f"{missing_cameras}. Configured cameras: {sorted(camera_configs.keys())}. "
            "Use INNOV_CAMERA_CONFIG='{\"front\":...,\"left_wrist\":...,\"right_wrist\":...}' "
            "or unset stale CAMERA_CONFIG/INNOV_CAMERA_CONFIG variables."
        )

    robot.connect()
    action_features = robot.action_features
    action_dim = len(action_features)
    action_clip_min = parse_optional_action_vector(args.action_clip_min, action_dim, "--action_clip_min")
    action_clip_max = parse_optional_action_vector(args.action_clip_max, action_dim, "--action_clip_max")
    print(f"[INFO] cameras   = {list(robot.cameras.keys())}")
    print(f"[INFO] required  = {required_cameras}")
    print(
        "[INFO] action_filter/smoothing/limit = "
        f"alpha={args.action_filter_alpha} max_delta={args.action_max_delta} "
        f"smooth_max_step={args.action_smooth_max_step} "
        f"clip_min={'off' if action_clip_min is None else action_clip_min.tolist()} "
        f"clip_max={'off' if action_clip_max is None else action_clip_max.tolist()}"
    )
    print(f"[INFO] policy_action_arms = {args.policy_action_arms}")

    try:
        with CameraPreview(
            enabled=bool(args.show_cameras),
            window_name=args.camera_preview_window,
            camera_names=list(robot.cameras.keys()),
            image_width=args.camera_preview_width,
        ) as pre_start_preview:
            print("[INFO] Camera preflight...")
            pre_start_obs = get_observation_sync_cameras(robot, args.camera_timeout_ms, args.camera_retries)
            for camera_name in robot.cameras:
                frame = pre_start_obs[camera_name]
                print(f"[INFO] camera {camera_name}: frame={frame.shape} dtype={frame.dtype}")
            pre_start_preview.update(pre_start_obs)

            print("[INFO] Connected. Camera preview is shown before policy control.")
            print("[INFO] Press Enter in this terminal to start local policy control, Ctrl+C to stop.")
            print("[INFO] Focus preview window and press 'q' only closes preview, not policy.")
            while True:
                if select.select([sys.stdin], [], [], 0.03)[0]:
                    sys.stdin.readline()
                    break
                pre_start_obs = get_observation_sync_cameras(robot, args.camera_timeout_ms, args.camera_retries)
                pre_start_preview.update(pre_start_obs)

        print("[INFO] After start, press 'z' at any time to stop policy actions and move to zero/home pose.")

        period_s = 1.0 / args.fps
        end_t = time.monotonic() + args.duration_s
        actions = np.zeros((0, action_dim), dtype=np.float32)
        action_index = 0
        prev_action = None
        step = 0
        action_keys = list(action_features.keys())
        action_log = None
        if args.action_log_path:
            action_log_path = Path(args.action_log_path).expanduser()
            action_log_path.parent.mkdir(parents=True, exist_ok=True)
            action_log = action_log_path.open("a", buffering=1)
            print(f"[INFO] action_log = {action_log_path}")

        def run_hotkey_home(reason: str) -> bool:
            nonlocal actions, action_index, prev_action
            if args.zero_arms == "none":
                print("[WARN] hotkey z requested, but zero_arms=none; ignoring.")
                return False
            actions = np.zeros((0, action_dim), dtype=np.float32)
            action_index = 0
            prev_action = None
            print(f"[SAFETY] HOMING requested during {reason}.")
            print("[SAFETY] Policy action buffer cleared; no more policy actions will be sent.")
            maybe_move_home(
                robot,
                args.zero_arms,
                home_joints,
                home_gripper,
                args.home_duration_s,
                args.home_fps,
                args.home_joint_velocity,
                args.home_gripper_velocity,
                home_no_confirm=True,
            )
            if action_log is not None:
                action_log.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "step": step,
                            "event": "hotkey_home",
                            "reason": reason,
                            "zero_arms": args.zero_arms,
                            "home_joints": home_joints,
                            "home_gripper": home_gripper,
                        }
                    )
                    + "\n"
                )
            print("[SAFETY] HOMING complete. Exiting policy control loop; restart the script to run inference again.")
            return True

        with (
            KeyboardHomeListener(home_key="z") as home_listener,
            CameraPreview(
                enabled=bool(args.show_cameras),
                window_name=args.camera_preview_window,
                camera_names=list(robot.cameras.keys()),
                image_width=args.camera_preview_width,
            ) as camera_preview,
        ):
            for i in range(args.warmup_rounds):
                if home_listener.consume_home_request() and run_hotkey_home("warmup"):
                    return
                obs = get_observation_sync_cameras(robot, args.camera_timeout_ms, args.camera_retries)
                if camera_preview.update(obs) == "z":
                    home_listener.home_requested.set()
                if i == 0:
                    print(f"[INFO] observation keys = {sorted(obs.keys())}")
                if home_listener.consume_home_request() and run_hotkey_home("warmup"):
                    return
                batch = build_policy_batch(policy, obs, action_features, device, args.task)
                with torch.inference_mode():
                    raw_chunk = extract_action_chunk(postprocessor(policy.predict_action_chunk(preprocessor(batch))))
                    reference_action = prev_action if prev_action is not None else observation_action_reference(obs, action_keys)
                    chunk = adapt_action_chunk_to_robot(
                        raw_chunk,
                        action_features,
                        reference_action,
                        args.policy_action_arms,
                    )
                if home_listener.consume_home_request() and run_hotkey_home("warmup"):
                    return
                print(
                    f"[INFO] warmup {i + 1}/{args.warmup_rounds}: "
                    f"policy_chunk={raw_chunk.shape} robot_chunk={chunk.shape}"
                )

            while time.monotonic() < end_t:
                loop_t = time.monotonic()
                if home_listener.consume_home_request() and run_hotkey_home("control-loop-start"):
                    break

                obs = get_observation_sync_cameras(robot, args.camera_timeout_ms, args.camera_retries)
                if camera_preview.update(obs) == "z":
                    home_listener.home_requested.set()
                if home_listener.consume_home_request() and run_hotkey_home("after-observation"):
                    break

                if action_index >= len(actions):
                    batch = build_policy_batch(policy, obs, action_features, device, args.task)
                    with torch.inference_mode():
                        raw_actions = extract_action_chunk(postprocessor(policy.predict_action_chunk(preprocessor(batch))))
                        reference_action = (
                            prev_action if prev_action is not None else observation_action_reference(obs, action_keys)
                        )
                        actions = adapt_action_chunk_to_robot(
                            raw_actions,
                            action_features,
                            reference_action,
                            args.policy_action_arms,
                        )
                    action_index = 0
                    if home_listener.consume_home_request() and run_hotkey_home("after-policy-inference"):
                        break
                    if args.print_action:
                        first_action = {
                            key: float(actions[0][idx])
                            for idx, key in enumerate(action_keys)
                        }
                        print(
                            f"[ACTION_CHUNK] step={step} "
                            f"policy_shape={raw_actions.shape} robot_shape={actions.shape} first={first_action}"
                        )

                model_action_np = actions[action_index].copy()
                raw_action = action_to_dict(model_action_np, action_features)
                action_index += 1
                reference_action = prev_action if prev_action is not None else observation_action_reference(obs, action_keys)
                action = filter_and_limit_action(
                    raw_action,
                    reference_action,
                    action_keys,
                    args.action_filter_alpha,
                    args.action_max_delta,
                    action_clip_min,
                    action_clip_max,
                )

                sent = send_smoothed_action_interruptible(
                    robot,
                    prev_action,
                    action,
                    action_features,
                    args.action_smooth_max_step,
                    home_listener,
                )
                if not sent:
                    if run_hotkey_home("before-policy-action-send"):
                        break
                if home_listener.consume_home_request():
                    if run_hotkey_home("after-policy-action-send"):
                        break
                prev_action = action

                log_entry = {
                    "time": time.time(),
                    "step": step,
                    "chunk_action_index": action_index - 1,
                    "action_keys": action_keys,
                    "model_action": model_action_np.astype(float).tolist(),
                    "raw_action": raw_action,
                    "sent_action": action,
                    "action_filter_alpha": args.action_filter_alpha,
                    "action_max_delta": args.action_max_delta,
                    "action_smooth_max_step": args.action_smooth_max_step,
                }
                if action_log is not None:
                    action_log.write(json.dumps(log_entry) + "\n")

                if args.print_action and (args.print_action_every <= 1 or step % args.print_action_every == 0):
                    print(
                        "[ACTION] "
                        f"step={step} chunk_i={action_index - 1} "
                        f"left={[round(action[k], 4) for k in action_keys if k.startswith('left_')]} "
                        f"right={[round(action[k], 4) for k in action_keys if k.startswith('right_')]}"
                    )
                step += 1

                sleep_s = period_s - (time.monotonic() - loop_t)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    finally:
        if "action_log" in locals() and action_log is not None:
            action_log.close()
        robot.disconnect()
        print("[INFO] disconnected")


if __name__ == "__main__":
    main()
