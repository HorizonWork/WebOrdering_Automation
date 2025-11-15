"""
Execution Package - Action Execution Layer
Manages browser interactions and skill execution
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from .browser_manager import BrowserManager
from .skill_executor import SkillExecutor

__all__ = [
    'BrowserManager',
    'SkillExecutor'
]
