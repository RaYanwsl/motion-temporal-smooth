# 视频人体运动估计与时序平滑

相机系里的关节轨迹先对齐到地面：地面法向转到 +Z，最低的脚落到 z=0。然后对每个关节点和根节点做滑动平均，再把还挨着地面的脚锁定在落地时的水平位置。

仓库里的片段是一段加了高频噪声和脚滑的行走，用来把抖动、脚滑和朝向跳变这三项分开看。WHAM 或其他视频估计器如果导出了 `joints`，或者 `poses` 加 `transl`，可以用 `load_motion` 走同一条链路。这里没有模型权重。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m motion_smooth --out outputs/report
pytest -q
```

`outputs/report/skeleton.svg` 是平滑前后的骨架，`motion.npz` 是平滑后的轨迹。
