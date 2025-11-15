"""
Validation Skills - Check element states
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.skills.base_skill import BaseSkill


class ValidationSkills(BaseSkill):
    """Validation skill implementations"""
    
    def __init__(self):
        super().__init__("Validation")
    
    async def check_exists(self, page, selector: str, **kwargs):
        """Check if element exists"""
        try:
            element = await page.query_selector(selector)
            exists = element is not None
            return self._success(f"Checked {selector}", {'exists': exists})
        except Exception as e:
            return self._error(f"Check exists failed: {str(e)}")
    
    async def check_visible(self, page, selector: str, timeout: int = 5000, **kwargs):
        """Check if element is visible"""
        try:
            element = await page.query_selector(selector)
            if element:
                visible = await element.is_visible()
                return self._success(f"Checked {selector}", {'visible': visible})
            else:
                return self._success(f"Element not found", {'visible': False})
        except Exception as e:
            return self._error(f"Check visible failed: {str(e)}")
    
    async def check_enabled(self, page, selector: str, **kwargs):
        """Check if element is enabled"""
        try:
            element = await page.query_selector(selector)
            if element:
                enabled = await element.is_enabled()
                return self._success(f"Checked {selector}", {'enabled': enabled})
            else:
                return self._success(f"Element not found", {'enabled': False})
        except Exception as e:
            return self._error(f"Check enabled failed: {str(e)}")
