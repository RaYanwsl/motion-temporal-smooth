import argparse
from pathlib import Path

from motion_smooth.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="平滑视频人体关节轨迹")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.out)
    print("\n".join(summary["conclusion"]))
    print(f"report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
