"""
Verify labeled episodes against raw episodes.

Checks:
- Labeled file exists for each raw episode (optional; here we inspect labeled dir only).
- Step count and step ids match between labeled and raw.
- Required fields exist in labeled steps (action, thought).
- Thought is non-empty and not left as "(human teleop)" placeholder.

Usage:
python -m scripts.annotation.verify_labels --raw data/manual/shopee/compacted_episodes --labeled data/manual/shopee/labeled_episodes [--delete-bad]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


PLACEHOLDER_THOUGHTS = {"(human teleop)", "(human teleop)", "(human teleop)".lower()}


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_bad_thought(thought: str) -> bool:
    if not thought or not thought.strip():
        return True
    if thought.strip().lower() in PLACEHOLDER_THOUGHTS:
        return True
    return False


def verify_episode(raw_path: Path, labeled_path: Path) -> List[str]:
    issues: List[str] = []
    try:
        raw = load_json(raw_path)
    except Exception as e:
        return [f"Cannot read raw: {e}"]

    try:
        labeled = load_json(labeled_path)
    except Exception as e:
        return [f"Cannot read labeled: {e}"]

    raw_steps = raw.get("steps", [])
    lab_steps = labeled.get("steps", [])

    if len(raw_steps) != len(lab_steps):
        issues.append(f"Step count mismatch raw={len(raw_steps)} labeled={len(lab_steps)}")

    # Align by step id
    raw_map = {s.get("step"): s for s in raw_steps}
    lab_map = {s.get("step"): s for s in lab_steps}

    for step_id, raw_step in raw_map.items():
        if step_id not in lab_map:
            issues.append(f"Missing step {step_id} in labeled")
            continue
        lab_step = lab_map[step_id]
        action = lab_step.get("action")
        skill = None
        if not action:
            issues.append(f"Step {step_id} missing action")
        else:
            skill = action.get("skill")
            if not skill:
                issues.append(f"Step {step_id} action missing skill")
        thought = lab_step.get("thought", "")
        if is_bad_thought(thought):
            issues.append(f"Step {step_id} missing/placeholder thought")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify labeled episodes against raw episodes")
    parser.add_argument("--raw", required=True, help="Folder with raw/compacted episodes")
    parser.add_argument("--raw_alt", help="Optional fallback folder (e.g., episodes vs compacted_episodes)")
    parser.add_argument("--labeled", required=True, help="Folder with labeled episodes")
    parser.add_argument("--delete-bad", action="store_true", help="Delete labeled files with issues")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    raw_alt = Path(args.raw_alt) if args.raw_alt else None
    labeled_dir = Path(args.labeled)

    labeled_files = sorted(labeled_dir.glob("*_labeled.json"))
    if not labeled_files:
        print("[error] No labeled files found.")
        return

    bad_files: List[Path] = []

    for lab_file in labeled_files:
        base_name = lab_file.name.replace("_labeled", "")
        raw_file = raw_dir / base_name
        if not raw_file.exists() and raw_alt:
            raw_file = raw_alt / base_name
        if not raw_file.exists():
            print(f"[warn] Raw file not found for {lab_file.name}")
            bad_files.append(lab_file)
            continue

        issues = verify_episode(raw_file, lab_file)
        if issues:
            bad_files.append(lab_file)
            print(f"[bad] {lab_file.name}:")
            for iss in issues:
                print(f"  - {iss}")

    if not bad_files:
        print("[ok] All labeled episodes passed basic checks.")
        return

    print(f"[summary] {len(bad_files)} files have issues.")
    if args.delete_bad:
        for f in bad_files:
            try:
                f.unlink()
                print(f"[deleted] {f.name}")
            except Exception as e:
                print(f"[error] Failed to delete {f.name}: {e}")
        # If we deleted them, still exit non-zero to signal problems were found
        raise SystemExit(1)
    else:
        print("Run again with --delete-bad to remove the bad labeled files.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
