"""Unified entry point for evaluating trained models."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation suite for planner/controller")
    parser.add_argument(
        "--benchmark_script",
        default="scripts/evaluation/run_benchmark.py",
        help="Path to benchmark runner",
    )
    args, extra = parser.parse_known_args()

    cmd = [sys.executable, args.benchmark_script, *extra]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
