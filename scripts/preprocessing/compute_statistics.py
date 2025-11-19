"""Compute dataset statistics for reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument(
        "--planner_path",
        default="data/processed/planner/train.jsonl",
        help="Planner dataset JSONL",
    )
    parser.add_argument(
        "--controller_path",
        default="data/processed/controller/train.jsonl",
        help="Controller dataset JSONL",
    )
    parser.add_argument(
        "--out_dir",
        default="data/statistics",
        help="Destination folder for summary JSON",
    )
    args = parser.parse_args()

    planner_samples = list(load_jsonl(Path(args.planner_path)))
    controller_samples = list(load_jsonl(Path(args.controller_path)))

    stats = {
        "planner_examples": len(planner_samples),
        "controller_examples": len(controller_samples),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] Stats saved to {out_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
