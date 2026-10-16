"""Jitter, foot skate, and heading, before and after smoothing."""

from __future__ import annotations

import numpy as np

from motion_smooth.contact import FOOTS

PELVIS = 0
NECK = 12


def jitter_mm(joints: np.ndarray) -> float:
    """Root-mean-square of the second difference, in millimeters."""

    if joints.shape[0] < 3:
        return 0.0
    second = joints[2:] - 2 * joints[1:-1] + joints[:-2]
    return float(np.sqrt(np.mean(second**2)) * 1000.0)


def foot_skate_mm(joints: np.ndarray, height_m: float = 0.05) -> float:
    if joints.shape[0] < 2:
        return 0.0
    travel = []
    for index in range(1, joints.shape[0]):
        for joint in FOOTS:
            previous = joints[index - 1, joint]
            current = joints[index, joint]
            if previous[2] <= height_m and current[2] <= height_m:
                travel.append(float(np.linalg.norm(current[:2] - previous[:2])))
    if not travel:
        return 0.0
    return float(np.mean(travel) * 1000.0)


def heading_jump_deg(joints: np.ndarray) -> float:
    if joints.shape[0] < 2:
        return 0.0
    jumps = []
    previous = None
    for frame in joints:
        heading = frame[NECK, :2] - frame[PELVIS, :2]
        if previous is not None and np.linalg.norm(previous) > 1e-8 and np.linalg.norm(heading) > 1e-8:
            cosine = float(np.clip(np.dot(previous, heading) / (np.linalg.norm(previous) * np.linalg.norm(heading)), -1, 1))
            jumps.append(float(np.degrees(np.arccos(cosine))))
        if np.linalg.norm(heading) > 1e-8:
            previous = heading
    if not jumps:
        return 0.0
    return float(np.mean(jumps))


def summarize(joints: np.ndarray) -> dict[str, float]:
    return {
        "jitter": jitter_mm(joints),
        "foot_skate": foot_skate_mm(joints),
        "heading_jump": heading_jump_deg(joints),
    }
