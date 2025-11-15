"""
Quick Execution Test - Fast validation of execution layer
Tests basic functionality without Chrome profile conflicts
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.browser_manager import BrowserManager
from src.execution.skill_executor import SkillExecutor


async def quick_test():
    """Quick test of execution layer"""
    print("=" * 70)
    print("QUICK EXECUTION TEST")
    print("=" * 70)
    print("\n⚡ Running fast validation tests...\n")
    
    # Use standard browser (no profile)
    manager = BrowserManager(headless=False, use_chrome_profile=False)
    executor = SkillExecutor()
    
    try:
        print("1️⃣ Testing BrowserManager...")
        print("-" * 70)
        
        # Launch browser
        await manager.launch()
        print("✅ Browser launched")
        
        # Create page
        page = await manager.new_page()
        print("✅ Page created")
        
        # Navigate
        await page.goto("https://example.com")
        await manager.wait_for_load(page)
        print(f"✅ Navigated to: {page.url}")
        
        # Screenshot
        temp_dir = ROOT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        await manager.screenshot(page, path=str(temp_dir / "quick_test.png"))
        print(f"✅ Screenshot saved")
        
        print("\n2️⃣ Testing SkillExecutor...")
        print("-" * 70)
        
        # Test goto skill
        await executor.execute(
            page,
            {"skill": "goto", "params": {"url": "https://google.com"}}
        )
        await asyncio.sleep(2)
        print(f"✅ goto skill works: {page.url}")
        
        # Test extract skill
        title = await executor.execute(
            page,
            {"skill": "extract", "params": {"selector": "title"}}
        )
        print(f"✅ extract skill works: {title}")
        
        # Test wait skill
        await executor.execute(
            page,
            {"skill": "wait", "params": {"selector": "body"}}
        )
        print("✅ wait skill works")
        
        # Test scroll skill
        await executor.execute(
            page,
            {"skill": "scroll", "params": {"direction": "down"}}
        )
        print("✅ scroll skill works")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n📊 Results:")
        print("  • BrowserManager: ✅ Working")
        print("  • SkillExecutor: ✅ Working")
        print("  • Skills tested: goto, extract, wait, scroll")
        print("\n🎉 Execution layer is functioning correctly!")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\n🔒 Closing browser...")
        await manager.close()
        print("✅ Cleanup complete\n")


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
