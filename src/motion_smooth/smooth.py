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
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    total = cumulative[window - 1 :].copy()
    total[1:] -= cumulative[: total.shape[0] - 1]
    return (total / window).reshape(values.shape).astype(values.dtype, copy=False)


def smooth_trajectory(joints: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth every joint, including the root. Joints are (T, J, 3)."""

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("关节轨迹必须是 (T, J, 3)")
    return moving_average(joints, window)
