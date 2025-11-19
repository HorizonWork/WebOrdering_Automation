"""Project root path helper for tests and scripts.

This small helper mirrors ``tests/path_setup.py`` so that modules can simply
``import path_setup`` without worrying about package-relative paths.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _locate_project_root() -> Path:
    """Locate the project root by walking up until ``pyproject.toml`` is found."""
    current = Path(__file__).resolve().parent
    while True:
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            raise RuntimeError("Cannot locate project root (pyproject.toml not found)")
        current = current.parent


PROJECT_ROOT = _locate_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


__all__ = ["PROJECT_ROOT"]

