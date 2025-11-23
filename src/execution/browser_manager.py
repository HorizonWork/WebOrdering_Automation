"""
Browser Manager - Playwright browser lifecycle management
Handles browser initialization, page creation, and cleanup
Supports Chrome profiles for persistent sessions (User Data Dir)
"""

import sys
import asyncio
import platform
from pathlib import Path
from typing import Optional, Dict, List

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
        - Manage pages with Chrome profile support
        - Handle cleanup
    
    **Key Features**:
        - **Standard Mode**: Incognito, fresh session every time.
        - **Persistent Mode (User Data Dir)**: Keeps cookies, login state (Crucial for Shopee/Lazada).
    """
    
    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport: Optional[Dict] = None,
        use_chrome_profile: bool = False,
        chrome_executable_path: Optional[str] = None,
        chrome_profile_directory: Optional[str] = None,
        user_data_dir: Optional[str] = None  # <--- New: Custom profile path
    ):
        """
        Initialize browser manager.
        
        Args:
            browser_type: 'chromium', 'firefox', or 'webkit'.
            headless: Run without UI (Note: False is better for anti-bot).
            viewport: Window size.
            use_chrome_profile: (Legacy) Use system default profile.
            chrome_executable_path: Path to chrome.exe (Auto-detected if None).
            chrome_profile_directory: (Legacy) Specific profile folder name.
            user_data_dir: Full path to a custom Chrome User Data directory.
        """
        self.browser_type = browser_type
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        
        # Profile configurations
        self.user_data_dir = user_data_dir
        self.use_chrome_profile = use_chrome_profile
        self.chrome_executable_path = chrome_executable_path
        self.chrome_profile_directory = chrome_profile_directory
        
        # If user_data_dir is provided, we implicitly enable profile mode
        if self.user_data_dir:
            self.use_chrome_profile = True

        # State
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.pages: List[Page] = []
        
        logger.info(
            f"BrowserManager initialized [Headless={self.headless}, "
            f"UserDataDir={self.user_data_dir or 'None'}]"
        )

    def _get_chrome_executable_path(self) -> str:
        """
        Auto-detect Chrome executable path.
        Required for launching persistent contexts properly.
        """
        # 1. Use provided path if exists
        if self.chrome_executable_path and Path(self.chrome_executable_path).exists():
            return self.chrome_executable_path
        
        # 2. Auto-detect based on OS
        system = platform.system()
        possible_paths = []
        
        if system == "Windows":
            possible_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe"
            ]
        elif system == "Darwin":  # macOS
            possible_paths = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
        else:  # Linux
            possible_paths = ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.debug(f"Found Chrome executable: {path}")
                return str(path)
        
        logger.warning("Chrome executable not found. Playwright will use bundled Chromium.")
        return ""
    
    async def launch(self):
        """
        Launch the browser instance.
        Dispatches to specific launch method based on configuration.
        """
        if self.browser or self.context:
            logger.debug("Browser already launched.")
            return
        
        logger.info("🚀 Launching browser...")
        self.playwright = await async_playwright().start()
        
        # MODE 1: Custom User Data Dir (Priority for Agent)
        if self.user_data_dir:
            await self._launch_persistent_custom()
            
        # MODE 2: System Default Profile (Legacy)
        elif self.use_chrome_profile:
            # Not recommended for Agents due to clutter, but supported
            logger.warning("Using System Default Profile is not recommended for automation.")
            await self._launch_standard() # Fallback for now or implement if needed
            
        # MODE 3: Standard Incognito (Clean Slate)
        else:
            await self._launch_standard()
            
    async def _launch_persistent_custom(self):
        """Launch with a specific custom user data directory."""
        logger.info(f"🌐 Launching Persistent Context: {self.user_data_dir}")
        
        # Arguments to hide automation
        args = [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--password-store=basic", # Avoid system keyring popups on Linux
            f"--window-size={self.viewport['width']},{self.viewport['height']}"
        ]
        
        executable = self._get_chrome_executable_path()
        
        try:
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir,
                headless=self.headless,
                executable_path=executable if executable else None,
                viewport=self.viewport,
                args=args,
                # device_scale_factor=1,
            )
            
            # Manage pages
            self.pages = list(self.context.pages)
            if not self.pages:
                page = await self.context.new_page()
                self.pages.append(page)
                
            logger.info(f"yes Persistent context ready with {len(self.pages)} page(s).")
            
        except Exception as e:
            error_msg = str(e)
            if "Target page, context or browser has been closed" in error_msg:
                logger.critical("no Browser crash! Maybe the profile is in use by another Chrome window?")
            raise e

    async def _launch_standard(self):
        """Launch standard browser (Incognito/Ephemeral)."""
        logger.info("🌐 Launching Standard Browser (Incognito)...")
        
        browser_launcher = getattr(self.playwright, self.browser_type)
        
        self.browser = await browser_launcher.launch(
            headless=self.headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        self.context = await self.browser.new_context(
            viewport=self.viewport,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        logger.info("yes Standard browser launched.")
    
    async def new_page(self) -> Page:
        """
        Get a page instance.
        In Persistent mode, tries to reuse the existing tab to maintain session.
        """
        if not self.context:
            await self.launch()
        
        # Optimization for Persistent Mode: Reuse the first tab
        # Opening too many tabs in a profile might confuse the agent or consume RAM
        if (self.user_data_dir or self.use_chrome_profile) and self.pages:
            page = self.pages[0]
            try:
                await page.bring_to_front()
                logger.debug("♻️ Reusing existing page in persistent context")
                return page
            except Exception:
                # If the page was closed externally, remove it and create new
                self.pages.remove(page)

        # Create new page
        page = await self.context.new_page()
        self.pages.append(page)
        logger.info(f"📄 New page created (Total: {len(self.pages)})")
        return page
    
    async def close_page(self, page: Page):
        """Close a specific page."""
        if page in self.pages:
            # In persistent mode, try to keep at least one page open so context doesn't die
            if (self.user_data_dir) and len(self.pages) == 1:
                logger.debug("⚠️ Keeping last page open for persistent context.")
                return

            try:
                await page.close()
            except Exception:
                pass
            self.pages.remove(page)
            logger.info(f"📄 Page closed (Remaining: {len(self.pages)})")
    
    async def close(self):
        """Close browser and release all resources."""
        logger.info("🔒 Closing browser resources...")
        
        # Close persistent context (this closes the browser window)
        if self.context:
            try:
                await self.context.close()
            except Exception as e:
                logger.warning(f"Error closing context: {e}")
            self.context = None
            
        # Close standard browser instance
        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self.browser = None
            
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
            
        self.pages = []
        logger.info("yes Browser shutdown complete.")

    async def get_html(self, page: Page) -> str:
        """Safe get HTML content."""
        try:
            return await page.content()
        except Exception as e:
            logger.error(f"Failed to get HTML: {e}")
            return ""
    
    async def screenshot(self, page: Page, path: Optional[str] = None) -> bytes:
        """Take screenshot safely."""
        try:
            return await page.screenshot(path=path)
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
            return b""

    async def wait_for_load(self, page: Page, timeout: int = 30000):
        """Wait for page load state."""
        try:
            # domcontentloaded is faster and usually enough for scraping
            await page.wait_for_load_state("domcontentloaded", timeout=timeout)
        except Exception as e:
            logger.warning(f"Wait for load warning: {e}")


# ------------------------------------------------------------------------------
# Self-Test
# ------------------------------------------------------------------------------
async def test_browser_manager():
    print("=" * 70)
    print("BrowserManager - Test")
    print("=" * 70 + "\n")
    
    # 1. Test Standard
    print("[Test] Standard Incognito Mode")
    manager = BrowserManager(headless=True)
    await manager.launch()
    page = await manager.new_page()
    await page.goto("https://example.com")
    print(f"Page Title: {await page.title()}")
    await manager.close()
    print("Standard Test Passed.\n")

    # 2. Test Custom Profile (Mock)
    # Uncomment to test real profile (Need valid path)
    """
    print("[Test] Custom Profile Mode")
    profile_path = "./chrome_profile_test" # Ensure this exists or is valid
    manager_p = BrowserManager(headless=False, user_data_dir=profile_path)
    try:
        await manager_p.launch()
        page_p = await manager_p.new_page()
        await page_p.goto("https://shopee.vn")
        print("Navigated to Shopee with Profile.")
        await asyncio.sleep(5)
    finally:
        await manager_p.close()
    """
    print("Test Completed.")

if __name__ == "__main__":
    asyncio.run(test_browser_manager())