"""
Step-by-Step Execution Test
Tests each component individually to identify issues
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_imports():
    """Test 1: Can import all components"""
    print("=" * 70)
    print("TEST 1: Import Components")
    print("=" * 70)
    
    try:
        print("Importing BrowserManager...")
        from src.execution.browser_manager import BrowserManager
        print("✅ BrowserManager imported")
        
        print("Importing SkillExecutor...")
        from src.execution.skill_executor import SkillExecutor
        print("✅ SkillExecutor imported")
        
        print("Importing Skills...")
        from src.execution.skills import (
            NavigationSkills,
            InteractionSkills,
            ObservationSkills,
            ValidationSkills,
            WaitSkills
        )
        print("✅ All Skills imported")
        
        print("\n✅ TEST 1 PASSED: All imports successful\n")
        return True, (BrowserManager, SkillExecutor)
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False, None


async def test_browser_manager(BrowserManager):
    """Test 2: BrowserManager basic functionality"""
    print("=" * 70)
    print("TEST 2: BrowserManager Functionality")
    print("=" * 70)
    
    manager = None
    try:
        print("Creating BrowserManager...")
        manager = BrowserManager(headless=False, use_chrome_profile=False)
        print("✅ BrowserManager created")
        
        print("Launching browser...")
        await manager.launch()
        print("✅ Browser launched")
        
        print("Creating page...")
        page = await manager.new_page()
        print("✅ Page created")
        
        print("Navigating to example.com...")
        await page.goto("https://example.com", timeout=30000)
        print(f"✅ Navigation successful: {page.url}")
        
        print("Getting page title...")
        title = await page.title()
        print(f"✅ Page title: {title}")
        
        print("\n✅ TEST 2 PASSED: BrowserManager works\n")
        return True, manager, page
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        
        if manager:
            await manager.close()
        
        return False, None, None


async def test_skill_executor(SkillExecutor, page):
    """Test 3: SkillExecutor basic functionality"""
    print("=" * 70)
    print("TEST 3: SkillExecutor Functionality")
    print("=" * 70)
    
    try:
        print("Creating SkillExecutor...")
        executor = SkillExecutor()
        print("✅ SkillExecutor created")
        
        print(f"Available skills: {len(executor.get_available_skills())}")
        print(f"Skills: {', '.join(executor.get_available_skills()[:10])}...")
        
        print("\nTesting 'goto' skill...")
        result = await executor.execute(
            page,
            {"skill": "goto", "params": {"url": "https://google.com"}}
        )
        print(f"✅ goto result: {result}")
        
        await asyncio.sleep(2)
        print(f"Current URL: {page.url}")
        
        print("\nTesting 'get_url' skill...")
        result = await executor.execute(
            page,
            {"skill": "get_url", "params": {}}
        )
        print(f"✅ get_url result: {result}")
        
        print("\nTesting 'get_title' skill...")
        result = await executor.execute(
            page,
            {"skill": "get_title", "params": {}}
        )
        print(f"✅ get_title result: {result}")
        
        print("\n✅ TEST 3 PASSED: SkillExecutor works\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


async def test_interaction_skills(executor, page):
    """Test 4: Interaction skills"""
    print("=" * 70)
    print("TEST 4: Interaction Skills")
    print("=" * 70)
    
    try:
        # Make sure we're on a page with inputs
        print("Navigating to Google...")
        await executor.execute(
            page,
            {"skill": "goto", "params": {"url": "https://google.com"}}
        )
        await asyncio.sleep(2)
        
        print("Testing 'wait_for_selector' skill...")
        result = await executor.execute(
            page,
            {"skill": "wait_for_selector", "params": {"selector": "textarea[name='q']"}}
        )
        print(f"✅ wait_for_selector result: {result}")
        
        print("Testing 'type' skill...")
        result = await executor.execute(
            page,
            {"skill": "type", "params": {"selector": "textarea[name='q']", "text": "playwright"}}
        )
        print(f"✅ type result: {result}")
        
        await asyncio.sleep(1)
        
        print("\n✅ TEST 4 PASSED: Interaction skills work\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n🧪 EXECUTION LAYER VALIDATION")
    print("=" * 70)
    print("Testing each component step-by-step\n")
    
    # Test 1: Imports
    success, components = test_imports()
    if not success:
        print("\n❌ ABORTED: Cannot import components\n")
        return False
    
    BrowserManager, SkillExecutor = components
    
    # Test 2: BrowserManager
    success, manager, page = await test_browser_manager(BrowserManager)
    if not success:
        print("\n❌ ABORTED: BrowserManager not working\n")
        return False
    
    try:
        # Test 3: SkillExecutor
        success = await test_skill_executor(SkillExecutor, page)
        if not success:
            print("\n❌ SkillExecutor has issues\n")
            return False
        
        # Test 4: Interaction Skills
        executor = SkillExecutor()
        success = await test_interaction_skills(executor, page)
        if not success:
            print("\n⚠️  Some interaction skills have issues\n")
        
        # Final summary
        print("=" * 70)
        print("🎉 EXECUTION LAYER VALIDATION COMPLETE")
        print("=" * 70)
        print("\n📊 Summary:")
        print("  ✅ Imports: Working")
        print("  ✅ BrowserManager: Working")
        print("  ✅ SkillExecutor: Working")
        print(f"  {'✅' if success else '⚠️ '} Interaction Skills: {'Working' if success else 'Partial'}")
        print("\n💡 Next Steps:")
        print("  1. Run full test suite: python tests/unit/test_execution_suite.py")
        print("  2. Test with real workflows")
        print("  3. Test Chrome profile if needed")
        print("\n" + "=" * 70 + "\n")
        
        return True
        
    finally:
        print("\n🔒 Cleanup...")
        if manager:
            await manager.close()
        print("✅ Done\n")


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user\n")
        sys.exit(1)
