"""
Controller dataset builder wrapper.

This delegates to `build_planner_dataset.main` but sets the default dataset
flag to `controller`, so CLI parity is preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocessing.build_planner_dataset import main as _build_main  # noqa: E402


def main() -> None:
    _build_main(default_dataset="controller")


if __name__ == "__main__":
    main()
