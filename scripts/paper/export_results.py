"""Export artifacts for paper submission."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle paper results")
    parser.add_argument("--source", default="results", help="Folder with results")
    parser.add_argument("--dest", default="paper_export", help="Destination folder")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[todo] Copy files from {args.source} to {args.dest}")


if __name__ == "__main__":
    main()
