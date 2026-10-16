"""Smooth, pin the feet, and write a small skeleton sheet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motion_smooth.clip import synthetic_clip
from motion_smooth.contact import align_to_ground, apply_foot_contact
from motion_smooth.metrics import summarize
from motion_smooth.smooth import smooth_trajectory

EDGES = (
    (0, 12),
    (12, 15),
    (0, 7),
    (7, 10),
    (0, 8),
    (8, 11),
    (12, 16),
    (16, 20),
    (12, 17),
    (17, 21),
)


def process(joints: np.ndarray, ground_normal: np.ndarray | None, window: int = 5) -> dict:
    upright = align_to_ground(joints, ground_normal)
    smoothed = smooth_trajectory(upright, window)
    contacted = apply_foot_contact(smoothed)
    return {
        "upright": upright,
        "smoothed": contacted,
        "before": summarize(upright),
        "after": summarize(contacted),
    }


def run(out_dir: Path) -> dict:
    clip = synthetic_clip()
    result = process(clip["camera"], clip["ground_normal_camera"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_svg(out_dir / "skeleton.svg", result["upright"], result["smoothed"])
    summary = {
        "before": result["before"],
        "after": result["after"],
        "frames": int(result["smoothed"].shape[0]),
        "conclusion": _conclusion(result["before"], result["after"]),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown(summary), encoding="utf-8")
    np.savez(out_dir / "motion.npz", joints=result["smoothed"].astype(np.float32))
    return summary


def _conclusion(before: dict, after: dict) -> list[str]:
    return [
        (
            f"关节抖动从 {before['jitter']:.1f} mm 降到 {after['jitter']:.1f} mm。"
            "滑动平均作用在每个关节点和根节点上，高频噪声先被压掉。"
        ),
        (
            f"触地脚的水平滑动从 {before['foot_skate']:.1f} mm/帧 降到 {after['foot_skate']:.1f} mm/帧。"
            "脚还在地面附近时，水平位置锁定在落地的那一帧。"
        ),
        (
            f"朝向跳变从 {before['heading_jump']:.2f}° 降到 {after['heading_jump']:.2f}°。"
            "相机里的地面法向先转到竖直向上，尺度和朝向才在同一个世界系里比较。"
        ),
    ]


def _markdown(summary: dict) -> str:
    before = summary["before"]
    after = summary["after"]
    lines = [
        "# 视频人体运动的时序平滑",
        "",
        "输入是相机系里的关节轨迹。先把地面法向转到 +Z，再平滑关节和根节点，最后用触地约束按住脚。",
        "",
        "| | 抖动 mm | 脚滑 mm/帧 | 朝向跳变 ° |",
        "| --- | ---: | ---: | ---: |",
        f"| 平滑前 | {before['jitter']:.2f} | {before['foot_skate']:.2f} | {before['heading_jump']:.2f} |",
        f"| 平滑后 | {after['jitter']:.2f} | {after['foot_skate']:.2f} | {after['heading_jump']:.2f} |",
        "",
        "## 结论",
        "",
    ]
    lines.extend(f"- {line}" for line in summary["conclusion"])
    lines.append("")
    return "\n".join(lines)


def _project(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(-point[2])


def _write_svg(path: Path, before: np.ndarray, after: np.ndarray) -> None:
    frames = (0, before.shape[0] // 2, before.shape[0] - 1)
    width = 180
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * 6}" height="220" viewBox="0 0 {width * 6} 220">',
        '<rect width="100%" height="100%" fill="#f4f1ea"/>',
    ]
    for column, (label, clip) in enumerate((("before", before), ("after", after))):
        for row, frame in enumerate(frames):
            origin_x = (column * 3 + row) * width + 90
            origin_y = 180
            parts.append(f'<text x="{origin_x - 40}" y="24" font-size="14" fill="#243126">{label} #{frame}</text>')
            joints = clip[frame]
            for start, end in EDGES:
                x1, y1 = _project(joints[start])
                x2, y2 = _project(joints[end])
                parts.append(
                    f'<line x1="{origin_x + x1 * 70:.1f}" y1="{origin_y + y1 * 70:.1f}" '
                    f'x2="{origin_x + x2 * 70:.1f}" y2="{origin_y + y2 * 70:.1f}" stroke="#245c38" stroke-width="2"/>'
                )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
