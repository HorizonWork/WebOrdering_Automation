"""
Utils Package - Utility functions and helpers
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from .logger import get_logger, setup_logging
from .metrics import MetricsTracker
from .validators import validate_url, validate_selector, validate_action

__all__ = [
    'get_logger',
    'setup_logging',
    'MetricsTracker',
    'validate_url',
    'validate_selector',
    'validate_action'
]
