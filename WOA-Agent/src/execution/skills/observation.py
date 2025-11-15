"""
Observation Skills - Get page information
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.skills.base_skill import BaseSkill


class ObservationSkills(BaseSkill):
    """Observation skill implementations"""
    
    def __init__(self):
        super().__init__("Observation")
    
    async def screenshot(self, page, path: str = None, full_page: bool = False, **kwargs):
        """Take screenshot"""
        try:
            screenshot = await page.screenshot(path=path, full_page=full_page)
            return self._success("Screenshot taken", {'screenshot': screenshot, 'path': path})
        except Exception as e:
            return self._error(f"Screenshot failed: {str(e)}")
    
    async def get_dom(self, page, **kwargs):
        """Get page HTML"""
        try:
            html = await page.content()
            return self._success("Got DOM", {'dom': html, 'size': len(html)})
        except Exception as e:
            return self._error(f"Get DOM failed: {str(e)}")
    
    async def get_text(self, page, selector: str, timeout: int = 5000, **kwargs):
        """Get element text"""
        try:
            text = await page.inner_text(selector, timeout=timeout)
            return self._success(f"Got text from {selector}", {'text': text})
        except Exception as e:
            return self._error(f"Get text failed: {str(e)}")
    
    async def get_url(self, page, **kwargs):
        """Get current URL"""
        try:
            url = page.url
            return self._success("Got URL", {'url': url})
        except Exception as e:
            return self._error(f"Get URL failed: {str(e)}")
    
    async def get_title(self, page, **kwargs):
        """Get page title"""
        try:
            title = await page.title()
            return self._success("Got title", {'title': title})
        except Exception as e:
            return self._error(f"Get title failed: {str(e)}")
