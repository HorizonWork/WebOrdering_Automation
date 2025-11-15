"""
Browser Manager - Playwright browser lifecycle management
Handles browser initialization, page creation, and cleanup
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import asyncio

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class BrowserManager:
    """
    Manages Playwright browser lifecycle.
    
    **Responsibilities**:
        - Launch browser (Chrome/Firefox/Safari)
        - Create browser contexts
        - Manage pages
        - Handle cleanup
        - Configure browser options
    
    **Features**:
        - Headless/headed mode
        - Custom user agent
        - Viewport configuration
        - Screenshot support
        - Multiple contexts
    """
    
    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport: Optional[Dict] = None
    ):
        """
        Initialize browser manager.
        
        Args:
            browser_type: Browser type (chromium, firefox, webkit)
            headless: Run in headless mode
            viewport: Viewport size {width, height}
        """
        self.browser_type = browser_type
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.pages = []
        
        logger.info(f"BrowserManager initialized ({browser_type}, headless={headless})")
    
    async def launch(self):
        """Launch browser"""
        if self.browser:
            logger.warning("Browser already launched")
            return
        
        logger.info("🚀 Launching browser...")
        
        self.playwright = await async_playwright().start()
        
        # Select browser
        if self.browser_type == "chromium":
            browser_launcher = self.playwright.chromium
        elif self.browser_type == "firefox":
            browser_launcher = self.playwright.firefox
        elif self.browser_type == "webkit":
            browser_launcher = self.playwright.webkit
        else:
            raise ValueError(f"Unknown browser type: {self.browser_type}")
        
        # Launch browser
        self.browser = await browser_launcher.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        
        # Create context
        self.context = await self.browser.new_context(
            viewport=self.viewport,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        logger.info("✅ Browser launched successfully")
    
    async def new_page(self) -> Page:
        """Create new page"""
        if not self.context:
            await self.launch()
        
        page = await self.context.new_page()
        self.pages.append(page)
        
        logger.info(f"📄 New page created (total: {len(self.pages)})")
        
        return page
    
    async def close_page(self, page: Page):
        """Close specific page"""
        if page in self.pages:
            await page.close()
            self.pages.remove(page)
            logger.info(f"📄 Page closed (remaining: {len(self.pages)})")
    
    async def close(self):
        """Close browser and cleanup"""
        logger.info("🔒 Closing browser...")
        
        # Close all pages
        for page in self.pages[:]:
            try:
                await page.close()
            except:
                pass
        
        self.pages = []
        
        # Close context
        if self.context:
            await self.context.close()
            self.context = None
        
        # Close browser
        if self.browser:
            await self.browser.close()
            self.browser = None
        
        # Stop playwright
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        
        logger.info("✅ Browser closed")
    
    async def get_html(self, page: Page) -> str:
        """Get page HTML"""
        return await page.content()
    
    async def screenshot(
        self,
        page: Page,
        path: Optional[str] = None,
        full_page: bool = False
    ) -> bytes:
        """Take screenshot"""
        screenshot_bytes = await page.screenshot(
            path=path,
            full_page=full_page
        )
        
        if path:
            logger.info(f"📸 Screenshot saved to {path}")
        
        return screenshot_bytes
    
    async def wait_for_load(
        self,
        page: Page,
        wait_until: str = "networkidle",
        timeout: int = 30000
    ):
        """Wait for page load"""
        try:
            await page.wait_for_load_state(wait_until, timeout=timeout)
            logger.debug(f"✓ Page loaded ({wait_until})")
        except Exception as e:
            logger.warning(f"Wait for load timeout: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.browser:
                asyncio.create_task(self.close())
        except:
            pass


# Test
async def test_browser_manager():
    """Test browser manager"""
    print("=" * 70)
    print("BrowserManager - Test")
    print("=" * 70 + "\n")
    
    manager = BrowserManager(headless=False)
    
    try:
        # Launch
        await manager.launch()
        print("✓ Browser launched\n")
        
        # Create page
        page = await manager.new_page()
        print("✓ Page created\n")
        
        # Navigate
        await page.goto("https://example.com")
        await manager.wait_for_load(page)
        print("✓ Navigated to example.com\n")
        
        # Get HTML
        html = await manager.get_html(page)
        print(f"✓ HTML length: {len(html)} chars\n")
        
        # Screenshot
        screenshot = await manager.screenshot(page)
        print(f"✓ Screenshot: {len(screenshot)} bytes\n")
        
        # Wait a bit
        await asyncio.sleep(2)
        
    finally:
        await manager.close()
        print("✓ Browser closed\n")
    
    print("=" * 70)
    print("✅ Test Completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_browser_manager())
