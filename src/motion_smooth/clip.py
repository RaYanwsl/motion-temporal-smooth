"""A jittered walk, and a loader for an external WHAM-style npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _rest() -> np.ndarray:
    pose = np.zeros((24, 3))
    pose[0] = (0.0, 0.0, 0.95)
    pose[12] = (0.0, 0.02, 1.50)
    pose[15] = (0.0, 0.02, 1.68)
    pose[7] = (-0.11, 0.04, 0.02)
    pose[10] = (-0.12, 0.08, 0.0)
    pose[8] = (0.11, 0.04, 0.12)
    pose[11] = (0.12, 0.08, 0.10)
    pose[16] = (-0.18, 0.0, 1.42)
    pose[17] = (0.18, 0.0, 1.42)
    pose[20] = (-0.55, 0.0, 1.30)
    pose[21] = (0.55, 0.0, 1.30)
    return pose


def synthetic_clip(frames: int = 24, seed: int = 3) -> dict:
    """Upright walk, then a tilted camera, high-frequency noise, and foot slide."""

    rng = np.random.default_rng(seed)
    clean = np.repeat(_rest()[None], frames, axis=0)
    for index in range(frames):
        clean[index, :, 1] += index * 0.03
        if index % 8 < 4:
            clean[index, 7, 2] = 0.0
            clean[index, 10, 2] = 0.0
            clean[index, 7, 1] = _rest()[7, 1] + (index // 8) * 0.24
            clean[index, 10, 1] = _rest()[10, 1] + (index // 8) * 0.24
            clean[index, 8, 2] = 0.12
            clean[index, 11, 2] = 0.10
        else:
            clean[index, 8, 2] = 0.0
            clean[index, 11, 2] = 0.0
            clean[index, 8, 1] = _rest()[8, 1] + (index // 8) * 0.24
            clean[index, 11, 1] = _rest()[11, 1] + (index // 8) * 0.24
            clean[index, 7, 2] = 0.12
            clean[index, 10, 2] = 0.10
    tilt = np.radians(18.0)
    rotation = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    camera = clean @ rotation.T
    camera += rng.normal(0, 0.012, size=camera.shape)
    camera[:, FOOT_SLIDE, 1] += 0.02 * np.sin(np.arange(frames) * 1.7)[:, None]
    return {
        "clean_world": clean,
        "camera": camera,
        "ground_normal_camera": rotation @ np.array([0.0, 0.0, 1.0]),
    }


FOOT_SLIDE = np.array([7, 8, 10, 11])


def load_motion(path: Path) -> np.ndarray:
    """Read joints (T, J, 3), or poses (T, J, 3) plus transl (T, 3)."""

    with np.load(path) as data:
        if "joints" in data:
            joints = np.asarray(data["joints"], dtype=np.float64)
        elif "poses" in data and "transl" in data:
            joints = np.asarray(data["poses"], dtype=np.float64) + np.asarray(data["transl"], dtype=np.float64)[:, None, :]
        else:
            raise ValueError("npz 需要 joints，或 poses 和 transl")
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"动作数组必须是 (T, J, 3)，收到 {joints.shape}")
    return joints
