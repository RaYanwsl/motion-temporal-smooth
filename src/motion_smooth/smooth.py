"""Sliding average on joint positions and the root."""

from __future__ import annotations

import numpy as np


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("平滑窗口必须是正奇数")
    if values.shape[0] < 2:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius),) + ((0, 0),) * (values.ndim - 1), mode="edge")
    kernel = np.ones(window) / window
    flat = padded.reshape(padded.shape[0], -1)
    smoothed = np.vstack([np.convolve(flat[:, column], kernel, mode="valid") for column in range(flat.shape[1])]).T
    return smoothed.reshape(values.shape)


def smooth_trajectory(joints: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth every joint, then replace the root with its own smoothed path.

    Joints are (T, J, 3). The root is joint 0. Smoothing joints in the camera
    frame already smooths the root; the second pass keeps the root from being
    pulled by a noisy limb average.
    """

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("关节轨迹必须是 (T, J, 3)")
    smoothed = moving_average(joints, window)
    smoothed[:, 0] = moving_average(joints[:, 0], window)
    return smoothed
