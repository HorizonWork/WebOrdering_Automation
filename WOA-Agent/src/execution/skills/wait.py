"""
Wait Skills - Wait for conditions
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.skills.base_skill import BaseSkill


class WaitSkills(BaseSkill):
    """Wait skill implementations"""
    
    def __init__(self):
        super().__init__("Wait")
    
    async def wait_for(self, page, duration: float = 1.0, **kwargs):
        """Wait for duration (seconds)"""
        try:
            await asyncio.sleep(duration)
            return self._success(f"Waited {duration}s")
        except Exception as e:
            return self._error(f"Wait failed: {str(e)}")
    
    async def wait_for_selector(self, page, selector: str, timeout: int = 30000, state: str = "visible", **kwargs):
        """Wait for selector"""
        try:
            await page.wait_for_selector(selector, timeout=timeout, state=state)
            return self._success(f"Selector {selector} is {state}")
        except Exception as e:
            return self._error(f"Wait for selector failed: {str(e)}")
    
    async def wait_for_navigation(self, page, timeout: int = 30000, wait_until: str = "networkidle", **kwargs):
        """Wait for navigation"""
        try:
            await page.wait_for_load_state(wait_until, timeout=timeout)
            return self._success(f"Navigation complete ({wait_until})")
        except Exception as e:
            return self._error(f"Wait for navigation failed: {str(e)}")
