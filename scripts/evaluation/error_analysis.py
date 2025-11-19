"""Simple error analysis stub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze failed trajectories")
    parser.add_argument(
        "failed_dir",
        default="data/trajectories/failed",
        nargs="?",
        help="Folder with failed JSON",
    )
    args = parser.parse_args()

    for file in Path(args.failed_dir).glob("*.json"):
        episode = json.loads(file.read_text(encoding="utf-8"))
        if not episode.get("success"):
            print(f"- {file.name}: {episode.get('summary', 'No summary')}")


if __name__ == "__main__":
    main()
