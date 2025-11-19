"""Generate LaTeX tables from experiment results."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce tables for the paper")
    parser.add_argument("--results", default="results/summary.json", help="Metrics JSON")
    args = parser.parse_args()
    print(f"[todo] Convert {args.results} into LaTeX tables")


if __name__ == "__main__":
    main()
