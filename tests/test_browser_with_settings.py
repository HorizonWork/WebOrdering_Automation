"""
Quick test to verify BrowserManager with settings.py config
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.browser_manager import BrowserManager
from config.settings import settings


async def test_with_settings():
    """Test BrowserManager using config from settings.py"""
    print("=" * 70)
    print("BrowserManager Test - Using settings.py config")
    print("=" * 70)
    
    print("\n📋 Current Configuration:")
    print("-" * 70)
    print(f"  Browser Type: {settings.browser_type}")
    print(f"  Headless: {settings.headless}")
    print(f"  Viewport: {settings.viewport}")
    print(f"  Use Chrome Profile: {settings.use_chrome_profile}")
    if settings.use_chrome_profile:
        print(f"  Chrome Executable: {settings.chrome_executable_path}")
        print(f"  Chrome Profile: {settings.chrome_profile_directory}")
    print()
    
    # Create BrowserManager with settings
    manager = BrowserManager(**settings.browser_config)
    
    try:
        print("🚀 Launching browser...\n")
        await manager.launch()
        
        # Get or create page
        if manager.pages:
            page = manager.pages[0]
            print(f"yes Using existing page ({len(manager.pages)} page(s) available)\n")
        else:
            page = await manager.new_page()
            print("yes Created new page\n")
        
        # Navigate to test page
        print("📄 Navigating to Shopee...\n")
        await page.goto("https://shopee.vn")
        await manager.wait_for_load(page)
        
        title = await page.title()
        print(f"yes Page Title: {title}")
        
        # Check login status (if using profile)
        if settings.use_chrome_profile:
            try:
                await asyncio.sleep(2)
                user_element = await page.query_selector('[class*="navbar__username"]')
                if user_element:
                    username = await user_element.inner_text()
                    print(f"yes Logged in as: {username}")
                else:
                    print("ℹ Not logged in")
            except Exception:
                print("ℹ Could not check login status")
        
        print("\n⏳ Keeping browser open for 5 seconds...")
        await asyncio.sleep(5)
        
        print("\nyes Test completed successfully!")
        
    except RuntimeError as e:
        print(f"\nno Error: {e}\n")
        if "already in use" in str(e):
            print("💡 Close all Chrome windows and try again\n")
        return False
        
    except Exception as e:
        print(f"\nno Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\n🔒 Closing browser...")
        await manager.close()
        print("yes Browser closed\n")
    
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_with_settings())
    sys.exit(0 if success else 1)
