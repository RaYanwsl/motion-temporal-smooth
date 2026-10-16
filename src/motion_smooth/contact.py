"""Foot contact and a shared ground plane. Z is up after alignment."""

from __future__ import annotations

import numpy as np

FOOTS = (7, 8, 10, 11)


def align_to_ground(joints: np.ndarray, normal: np.ndarray | None = None) -> np.ndarray:
    """Rotate a ground normal onto +Z and put the lowest foot on z=0.

    `normal` is the camera-frame ground normal. When it is omitted the clip is
    assumed to already use an upright frame, and only the height offset is removed.
    """

    aligned = np.asarray(joints, dtype=np.float64).copy()
    if normal is not None:
        source = np.asarray(normal, dtype=np.float64)
        source = source / np.linalg.norm(source)
        target = np.array([0.0, 0.0, 1.0])
        axis = np.cross(source, target)
        sine = np.linalg.norm(axis)
        cosine = float(np.dot(source, target))
        if sine > 1e-8:
            axis = axis / sine
            cross = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            rotation = np.eye(3) + cross * sine + cross @ cross * (1 - cosine)
            aligned = aligned @ rotation.T
        elif cosine < 0:
            aligned[:, 2] *= -1
    floor = float(np.min(aligned[:, FOOTS, 2]))
    aligned[:, :, 2] -= floor
    return aligned


def apply_foot_contact(joints: np.ndarray, height_m: float = 0.05) -> np.ndarray:
    """Hold a foot still while it stays near the ground.

    The first contact frame keeps its horizontal position. Later contact frames
    reuse it and sit on z=0, so a planted foot stops skating.
    """

    held = np.asarray(joints, dtype=np.float64).copy()
    contacts: dict[int, np.ndarray] = {}
    for index in range(held.shape[0]):
        for joint in FOOTS:
            if held[index, joint, 2] <= height_m:
                if joint not in contacts:
                    contacts[joint] = held[index, joint, :2].copy()
                held[index, joint, :2] = contacts[joint]
                held[index, joint, 2] = 0.0
            else:
                contacts.pop(joint, None)
    return held
