"""Central configuration for the WOA agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer from environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the orchestrator and subsystems."""

    max_steps: int = field(default_factory=lambda: _env_int("AGENT_MAX_STEPS", 25))
    headless: bool = field(default_factory=lambda: _env_bool("AGENT_HEADLESS", True))
    browser_type: str = field(default_factory=lambda: os.getenv("AGENT_BROWSER", "chromium"))
    viewport_width: int = field(default_factory=lambda: _env_int("AGENT_VIEWPORT_WIDTH", 1280))
    viewport_height: int = field(default_factory=lambda: _env_int("AGENT_VIEWPORT_HEIGHT", 720))
    log_level: str = field(default_factory=lambda: os.getenv("AGENT_LOG_LEVEL", "INFO"))
    data_dir: str = field(default_factory=lambda: os.getenv("AGENT_DATA_DIR", "data"))
    device: str = field(default_factory=lambda: os.getenv("AGENT_DEVICE", "cuda" if _env_bool("CUDA_AVAILABLE", False) else "cpu"))

    @property
    def viewport(self) -> Dict[str, int]:
        """Return viewport configuration for Playwright."""
        return {"width": self.viewport_width, "height": self.viewport_height}


settings = Settings()

__all__ = ["Settings", "settings"]

