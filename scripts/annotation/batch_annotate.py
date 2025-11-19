"""Batch annotator orchestrating Gemini calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.annotation.gemini_annotator import annotate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch annotate raw episodes")
    parser.add_argument("input_dir", help="Folder with raw episodes")
    parser.add_argument("output_dir", help="Folder to store annotated episodes")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for episode_file in in_dir.glob("*.json"):
        labeled = annotate_episode(json.loads(episode_file.read_text(encoding="utf-8")))
        target = out_dir / f"{episode_file.stem}_labeled.json"
        target.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {episode_file.name} -> {target.name}")


if __name__ == "__main__":
    main()
