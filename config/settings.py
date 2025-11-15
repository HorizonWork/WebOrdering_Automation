"""Central configuration for the WOA agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


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
    viewport_width: int = field(default_factory=lambda: _env_int("AGENT_VIEWPORT_WIDTH", 1920))
    viewport_height: int = field(default_factory=lambda: _env_int("AGENT_VIEWPORT_HEIGHT", 1080))
    log_level: str = field(default_factory=lambda: os.getenv("AGENT_LOG_LEVEL", "INFO"))
    data_dir: str = field(default_factory=lambda: os.getenv("AGENT_DATA_DIR", "data"))
    device: str = field(default_factory=lambda: os.getenv("AGENT_DEVICE", "cuda" if _env_bool("CUDA_AVAILABLE", False) else "cpu"))
    enable_vision: bool = field(default_factory=lambda: _env_bool("ENABLE_VISION", True))
    yolo_model_path: str = field(default_factory=lambda: os.getenv("YOLO_MODEL_PATH", "checkpoints/yolov8_bestm.pt"))
    florence_base_model: str = field(default_factory=lambda: os.getenv("FLORENCE_BASE_MODEL", "microsoft/Florence-2-base"))
    florence_adapter_path: str = field(default_factory=lambda: os.getenv("FLORENCE_ADAPTER_PATH", "checkpoints/florence"))
    omnipasser_email: str = field(default_factory=lambda: os.getenv("OMNIPASSER_EMAIL", ""))
    omnipasser_password: str = field(default_factory=lambda: os.getenv("OMNIPASSER_PASSWORD", ""))
    
    # Chrome Profile settings
    use_chrome_profile: bool = field(default_factory=lambda: _env_bool("USE_CHROME_PROFILE", False))
    chrome_executable_path: str = field(default_factory=lambda: os.getenv("CHROME_EXECUTABLE_PATH", ""))
    chrome_profile_directory: str = "Profile 18"

    @property
    def viewport(self) -> Dict[str, int]:
        """Return viewport configuration for Playwright."""
        return {"width": self.viewport_width, "height": self.viewport_height}
    
    @property
    def browser_config(self) -> Dict:
        """Return browser configuration for BrowserManager."""
        config = {
            "browser_type": self.browser_type,
            "headless": self.headless,
            "viewport": self.viewport,
            "use_chrome_profile": self.use_chrome_profile,
        }
        
        if self.chrome_executable_path:
            config["chrome_executable_path"] = self.chrome_executable_path
        
        if self.chrome_profile_directory:
            config["chrome_profile_directory"] = self.chrome_profile_directory
        
        return config


settings = Settings()

__all__ = ["Settings", "settings"]

