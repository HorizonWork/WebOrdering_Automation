"""Dataset/model metric computation helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from results JSON")
    parser.add_argument("results", help="Path to results.json")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
