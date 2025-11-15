"""
Base Skill - Abstract base class for all skills
"""

import sys
from pathlib import Path
from abc import ABC
from typing import Dict

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseSkill(ABC):
    """
    Abstract base class for skills.
    
    **Result Format**:
        {
            'status': 'success' | 'error',
            'message': str,
            'data': dict (optional)
        }
    """
    
    def __init__(self, name: str):
        """
        Initialize skill.
        
        Args:
            name: Skill name
        """
        self.name = name
        logger.debug(f"{self.name} skill initialized")
    
    def _success(self, message: str = "Success", data: Dict = None) -> Dict:
        """Create success result"""
        return {
            'status': 'success',
            'message': message,
            'data': data or {}
        }
    
    def _error(self, message: str) -> Dict:
        """Create error result"""
        return {
            'status': 'error',
            'message': message,
            'data': {}
        }
