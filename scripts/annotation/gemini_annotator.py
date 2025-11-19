"""Gemini-based annotator skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def annotate_episode(raw_episode: Dict[str, Any]) -> Dict[str, Any]:
    """Stub: attach planner/controller teacher labels."""
    raw_episode.setdefault("teacher_labels", {})
    return raw_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a raw trajectory with Gemini labels")
    parser.add_argument("input", help="Path to raw episode JSON")
    parser.add_argument("output", help="Destination JSON path")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    labeled = annotate_episode(data)
    Path(args.output).write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Annotated episode saved to {args.output}")


if __name__ == "__main__":
    main()
