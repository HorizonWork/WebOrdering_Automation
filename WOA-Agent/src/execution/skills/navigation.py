"""
Navigation Skills - Page navigation operations
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.skills.base_skill import BaseSkill


class NavigationSkills(BaseSkill):
    """Navigation skill implementations"""
    
    def __init__(self):
        super().__init__("Navigation")
    
    async def goto(self, page, url: str, wait_until: str = "networkidle", timeout: int = 30000):
        """Navigate to URL"""
        try:
            await page.goto(url, wait_until=wait_until, timeout=timeout)
            return self._success(f"Navigated to {url}", {'url': page.url})
        except Exception as e:
            return self._error(f"Navigation failed: {str(e)}")
    
    async def back(self, page, **kwargs):
        """Go back"""
        try:
            await page.go_back()
            return self._success("Went back", {'url': page.url})
        except Exception as e:
            return self._error(f"Back failed: {str(e)}")
    
    async def forward(self, page, **kwargs):
        """Go forward"""
        try:
            await page.go_forward()
            return self._success("Went forward", {'url': page.url})
        except Exception as e:
            return self._error(f"Forward failed: {str(e)}")
    
    async def reload(self, page, **kwargs):
        """Reload page"""
        try:
            await page.reload()
            return self._success("Page reloaded")
        except Exception as e:
            return self._error(f"Reload failed: {str(e)}")
    
    async def close(self, page, **kwargs):
        """Close page"""
        try:
            await page.close()
            return self._success("Page closed")
        except Exception as e:
            return self._error(f"Close failed: {str(e)}")
