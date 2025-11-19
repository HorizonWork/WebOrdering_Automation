"""Validate annotated episodes for schema correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = [
    ("planner", ["next_plan_step"]),
    ("controller", ["chosen_action"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Gemini annotations")
    parser.add_argument("input", help="Annotated episode JSON or folder")
    args = parser.parse_args()

    target = Path(args.input)
    files = [target] if target.is_file() else sorted(target.glob("*.json"))
    failures = 0
    for file_path in files:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        labels = data.get("teacher_labels") or {}
        missing = []
        for section, fields in REQUIRED_FIELDS:
            bucket = labels.get(section) or {}
            for field in fields:
                if field not in bucket:
                    missing.append(f"{section}.{field}")
        if missing:
            failures += 1
            print(f"[warn] {file_path}: missing {', '.join(missing)}")
    print(f"Validation done. {failures} / {len(files)} files missing fields.")


if __name__ == "__main__":
    main()
