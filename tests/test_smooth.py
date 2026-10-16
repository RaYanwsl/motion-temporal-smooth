from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_smooth.clip import load_motion, synthetic_clip
from motion_smooth.contact import align_to_ground
from motion_smooth.pipeline import process, run


def test_ground_alignment_makes_the_normal_vertical():
    clip = synthetic_clip()
    upright = align_to_ground(clip["camera"], clip["ground_normal_camera"])
    floor = upright[:, [7, 8, 10, 11], 2].min()
    assert abs(floor) < 1e-6
    neck = upright[:, 12, 2].mean()
    pelvis = upright[:, 0, 2].mean()
    assert neck > pelvis + 0.3


def test_smoothing_cuts_jitter_and_contact_cuts_skate():
    clip = synthetic_clip()
    result = process(clip["camera"], clip["ground_normal_camera"])
    assert result["after"]["jitter"] < result["before"]["jitter"] * 0.55
    assert result["after"]["foot_skate"] < result["before"]["foot_skate"] * 0.35
    assert result["after"]["heading_jump"] < result["before"]["heading_jump"]


def test_wham_style_npz_roundtrip(tmp_path: Path):
    clip = synthetic_clip(frames=8)
    path = tmp_path / "wham.npz"
    poses = clip["clean_world"] - clip["clean_world"][:, 0:1, :]
    transl = clip["clean_world"][:, 0, :]
    np.savez(path, poses=poses, transl=transl)
    loaded = load_motion(path)
    assert loaded.shape == (8, 24, 3)
    assert np.allclose(loaded, clip["clean_world"])


def test_report_and_svg(tmp_path: Path):
    summary = run(tmp_path)
    text = "\n".join(summary["conclusion"])
    assert "抖动" in text
    assert "触地" in text
    assert (tmp_path / "skeleton.svg").exists()
    assert (tmp_path / "motion.npz").exists()
