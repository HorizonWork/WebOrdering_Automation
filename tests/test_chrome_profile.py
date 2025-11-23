"""
Test Chrome Profile with BrowserManager
Run this test when Chrome is NOT running
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.browser_manager import BrowserManager

async def test_chrome_profile():
    """Test Chrome profile"""
    print("=" * 70)
    print("Chrome Profile Test")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Make sure all Chrome windows are closed!\n")
    print("-" * 70)
    
    # Configure Chrome profile
    manager = BrowserManager(
        headless=False,
        use_chrome_profile=True,
        chrome_executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        chrome_profile_directory="Profile 18"  # Change this to your profile
    )
    
    try:
        print("🚀 Launching Chrome with your profile...\n")
        await manager.launch()
        
        # Get page
        page = manager.pages[0] if manager.pages else await manager.new_page()
        print(f"yes Chrome profile loaded with {len(manager.pages)} page(s)\n")
        
        # Test 1: Navigate to Shopee
        print("Test 1: Navigate to Shopee")
        print("-" * 70)
        await page.goto("https://shopee.vn")
        await manager.wait_for_load(page)
        
        await asyncio.sleep(2)
        title = await page.title()
        print(f"yes Page title: {title}")
        
        # Check if logged in
        try:
            # Check for user profile
            user_element = await page.query_selector('[class*="navbar__username"]')
            if user_element:
                username = await user_element.inner_text()
                print(f"yes Logged in as: {username}")
            else:
                print("ℹ Not logged in yet")
        except Exception as e:
            print(f"ℹ Could not check login status: {e}")
        
        print()
        
        # Test 2: New tab
        print("Test 2: Open new tab")
        print("-" * 70)
        page2 = await manager.new_page()
        await page2.goto("https://www.lazada.vn")
        await manager.wait_for_load(page2)
        print(f"yes Opened new tab (total: {len(manager.pages)} tabs)\n")
        
        # Test 3: Screenshot
        print("Test 3: Take screenshot")
        print("-" * 70)
        screenshot_path = ROOT_DIR / "temp" / "chrome_profile_test.png"
        screenshot_path.parent.mkdir(exist_ok=True)
        await manager.screenshot(page, path=str(screenshot_path))
        print(f"yes Screenshot saved to: {screenshot_path}\n")
        
        # Keep browser open to verify
        print("⏳ Keeping browser open for 10 seconds...")
        print("   You can verify your profile is loaded correctly")
        print("   (Check bookmarks, extensions, login status, etc.)")
        await asyncio.sleep(10)
        
    except RuntimeError as e:
        print(f"\nno Error: {e}\n")
        print("💡 Solution:")
        print("   1. Close all Chrome windows")
        print("   2. Run this test again\n")
        return False
        
    except FileNotFoundError as e:
        print(f"\nno Error: {e}\n")
        print("💡 Solution:")
        print("   1. Check Chrome installation path")
        print("   2. Verify profile directory name")
        print("   3. Use chrome://version/ to find your profile path\n")
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
    print("yes Chrome Profile Test Completed Successfully!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_chrome_profile())
    sys.exit(0 if success else 1)
