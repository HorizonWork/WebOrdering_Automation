"""
Browser Manager - Playwright browser lifecycle management
Handles browser initialization, page creation, and cleanup
Supports Chrome profiles for persistent sessions
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import asyncio
import platform

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from playwright.async_api import async_playwright, Browser, BrowserContext, Page # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from config.settings import settings  # noqa: E402

logger = get_logger(__name__)


class BrowserManager:
    """
    Manages Playwright browser lifecycle.
    
    **Responsibilities**:
        - Launch browser (Chrome/Firefox/Safari)
        - Create browser contexts
        - Manage pages with Chrome profile support
        - Handle cleanup
        - Configure browser options
    
    **Features**:
        - Headless/headed mode
        - Chrome profile support (persistent context)
        - Custom user agent
        - Viewport configuration
        - Screenshot support
        - Multiple contexts
    """
    
    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport: Optional[Dict] = None,
        use_chrome_profile: bool = False,
        chrome_executable_path: Optional[str] = None,
        chrome_profile_directory: Optional[str] = None
    ):
        """
        Initialize browser manager.
        
        Args:
            browser_type: Browser type (chromium, firefox, webkit)
            headless: Run in headless mode
            viewport: Viewport size {width, height}
            use_chrome_profile: Use existing Chrome profile
            chrome_executable_path: Path to Chrome executable
            chrome_profile_directory: Profile directory name (e.g., "Profile 18")
        """
        self.browser_type = browser_type
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.use_chrome_profile = use_chrome_profile
        self.chrome_executable_path = chrome_executable_path
        self.chrome_profile_directory = chrome_profile_directory
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.pages = []
        
        logger.info(
            f"BrowserManager initialized ({browser_type}, "
            f"headless={headless}, chrome_profile={use_chrome_profile})"
        )
    
    def _get_chrome_user_data_dir(self) -> str:
        """
        Get Chrome User Data directory based on OS.
        
        Returns:
            Path to Chrome User Data directory
        """
        system = platform.system()
        
        if system == "Windows":
            user_data_dir = Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
        elif system == "Darwin":  # macOS
            user_data_dir = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
        else:  # Linux
            user_data_dir = Path.home() / ".config" / "google-chrome"
        
        if not user_data_dir.exists():
            raise FileNotFoundError(
                f"Chrome User Data directory not found: {user_data_dir}\n"
                f"Please make sure Chrome is installed."
            )
        
        logger.info(f"Chrome User Data directory: {user_data_dir}")
        return str(user_data_dir)
    
    def _get_chrome_executable_path(self) -> str:
        """
        Auto-detect Chrome executable path.
        
        Returns:
            Path to Chrome executable
        """
        if self.chrome_executable_path:
            if Path(self.chrome_executable_path).exists():
                return self.chrome_executable_path
            else:
                logger.warning(
                    f"Provided Chrome path not found: {self.chrome_executable_path}"
                )
        
        system = platform.system()
        
        if system == "Windows":
            possible_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe"
            ]
        elif system == "Darwin":  # macOS
            possible_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            ]
        else:  # Linux
            possible_paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable"
            ]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.info(f"✅ Found Chrome at: {path}")
                return str(path)
        
        raise FileNotFoundError(
            "Chrome executable not found. Please install Chrome or specify "
            "chrome_executable_path"
        )
    
    async def launch(self):
        """Launch browser"""
        if self.browser or self.context:
            logger.warning("Browser already launched")
            return
        
        logger.info("🚀 Launching browser...")
        
        self.playwright = await async_playwright().start()
        
        if self.use_chrome_profile:
            # Launch Chrome with persistent context (profile)
            await self._launch_with_profile()
        else:
            # Launch standard browser
            await self._launch_standard()
    
    async def _launch_with_profile(self):
        """Launch Chrome with existing profile"""
        logger.info("🌐 Launching Chrome with profile...")

        base_user_data_dir = Path(self._get_chrome_user_data_dir())
        if self.chrome_profile_directory:
            profile_path = base_user_data_dir / self.chrome_profile_directory
            logger.info(f"📁 Using Chrome profile directory: {profile_path}")
        else:
            profile_path = base_user_data_dir / "Default"
            logger.info("📁 No profile specified, defaulting to 'Default'")

        if not profile_path.exists():
            available_profiles = [
                p.name for p in base_user_data_dir.iterdir()
                if p.is_dir() and (p.name.startswith("Profile") or p.name == "Default")
            ]
            raise FileNotFoundError(
                f"Profile directory not found: {profile_path}\n"
                f"Available profiles in {base_user_data_dir}:\n" +
                "\n".join([f"  - {p}" for p in available_profiles])
            )

        # Get Chrome executable
        chrome_path = self._get_chrome_executable_path()

        # Build args (no explicit --profile-directory when pointing directly to the profile path)
        window_arg = f"--window-size={self.viewport['width']},{self.viewport['height']}"
        args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--disable-session-crashed-bubble",
            "--disable-restore-session-state",
            "--no-first-run",
            "--no-default-browser-check",
            window_arg,
        ]

        # Launch persistent context
        try:
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=self.headless,
                executable_path=chrome_path,
                viewport=self.viewport,
                args=args,
                timeout=60000  # 60 seconds timeout
            )
        except Exception as e:
            error_msg = str(e)
            if "Target page, context or browser has been closed" in error_msg or "User data directory is already in use" in error_msg:
                raise RuntimeError(
                    "❌ Cannot launch Chrome profile - profile is already in use!\n"
                    "Please close all Chrome windows before running with Chrome profile.\n"
                    f"Profile: {self.chrome_profile_directory}\n"
                    f"User Data Dir: {profile_path}"
                ) from e
            raise
        
        # Get or create first page
        if self.context.pages:
            self.pages = list(self.context.pages)
            logger.info(f"✅ Chrome profile loaded with {len(self.pages)} existing page(s)")
        else:
            page = await self.context.new_page()
            self.pages.append(page)
            logger.info("✅ Chrome profile loaded with new page")
    
    async def _launch_standard(self):
        """Launch standard browser without profile"""
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
                '--disable-dev-shm-usage',
                f"--window-size={self.viewport['width']},{self.viewport['height']}"
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
        
        # Close browser (if not using persistent context)
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
            await page.wait_for_load_state(wait_until, timeout=timeout)  # type: ignore
            logger.debug(f"✓ Page loaded ({wait_until})")
        except Exception as e:
            logger.warning(f"Wait for load timeout: {e}")
    
# Test
async def test_browser_manager():
    """Test browser manager"""
    print("=" * 70)
    print("BrowserManager - Test")
    print("=" * 70 + "\n")
    
    # Test 1: Standard browser
    print("Test 1: Standard Chromium Browser")
    print("-" * 70)
    manager = BrowserManager(headless=False)
    
    try:
        await manager.launch()
        print("✓ Browser launched\n")
        
        page = await manager.new_page()
        print("✓ Page created\n")
        
        await page.goto("https://example.com")
        await manager.wait_for_load(page)
        print("✓ Navigated to example.com\n")
        
        html = await manager.get_html(page)
        print(f"✓ HTML length: {len(html)} chars\n")
        
        await asyncio.sleep(2)
        
    finally:
        await manager.close()
        print("✓ Browser closed\n")
    
    # Test 2: Chrome profile (if available)
    print("\nTest 2: Chrome Profile")
    print("-" * 70)
    print("⚠️  IMPORTANT: Close all Chrome windows before running this test!")
    print("-" * 70)
    
    try:
        manager_profile = BrowserManager(
            headless=False,
            use_chrome_profile=True,
            chrome_executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            chrome_profile_directory="Profile 18"
        )
        
        await manager_profile.launch()
        print("✓ Chrome profile loaded\n")
        
        page = manager_profile.pages[0] if manager_profile.pages else await manager_profile.new_page()
        
        await page.goto("https://shopee.vn")
        await manager_profile.wait_for_load(page)
        print("✓ Navigated to Shopee\n")
        
        # Check if logged in
        try:
            await asyncio.sleep(2)
            title = await page.title()
            print(f"✓ Page title: {title}\n")
        except Exception:
            pass
        
        await asyncio.sleep(3)
        await manager_profile.close()
        print("✓ Chrome profile closed\n")
        
    except RuntimeError as e:
        print(f"\n❌ {e}\n")
    except FileNotFoundError as e:
        print(f"\n⚠️  Chrome profile test skipped: {e}\n")
    except Exception as e:
        print(f"\n⚠️  Chrome profile test failed: {e}\n")
    
    print("=" * 70)
    print("✅ Test Completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_browser_manager())
