"""Ablation runner placeholder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", help="Path to ablation config", required=False)
    args = parser.parse_args()
    print("[todo] Implement ablation logic", args)


if __name__ == "__main__":
    main()
