"""Generate figures for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render figures from statistics")
    parser.add_argument(
        "--stats_dir",
        default="data/statistics",
        help="Directory with stats JSON",
    )
    parser.add_argument(
        "--out_dir",
        default="docs/figures",
        help="Directory to store generated figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[todo] Draw figures using stats from {args.stats_dir}")


if __name__ == "__main__":
    main()
