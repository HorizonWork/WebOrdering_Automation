"""Manual review helper for annotated data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_REVIEW_PATH = Path("data/annotation_quality/manual_review_samples.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample annotations for manual QA")
    parser.add_argument("input", help="Annotated dataset path")
    parser.add_argument("--samples", type=int, default=20, help="How many samples to export")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_REVIEW_PATH),
        help="Where to dump sampled annotations",
    )
    args = parser.parse_args()

    entries = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    subset = entries[: args.samples]
    Path(args.output).write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Wrote {len(subset)} samples to {args.output}")


if __name__ == "__main__":
    main()
