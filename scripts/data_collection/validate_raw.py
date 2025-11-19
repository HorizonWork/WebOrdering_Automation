"""Simple validator for raw episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_episode(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid json: {exc}")
        return errors
    required_keys = ["episode_id", "goal", "steps"]
    for key in required_keys:
        if key not in data:
            errors.append(f"missing key: {key}")
    if not isinstance(data.get("steps"), list):
        errors.append("steps must be a list")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw episode JSON")
    parser.add_argument("root", default="data/raw", nargs="?", help="Root folder e.g. data/raw/shopee/episodes")
    args = parser.parse_args()

    root = Path(args.root)
    total = 0
    failed = 0
    for episode_file in root.rglob("*.json"):
        total += 1
        errs = validate_episode(episode_file)
        if errs:
            failed += 1
            print(f"[warn] {episode_file}: {'; '.join(errs)}")
    print(f"Checked {total} files, {failed} failed validation.")


if __name__ == "__main__":
    main()
