import asyncio

import pytest

from src.execution.browser_manager import BrowserManager
from src.execution.skill_executor import SkillExecutor


@pytest.mark.asyncio
async def test_execution_layer():
    print("Testing Execution Layer...")
    manager = BrowserManager(headless=False)
    page = await manager.new_page()
    
    executor = SkillExecutor()

    try:
        # 1. Test 'goto' - action phải là dict với skill và params
        print("Testing 'goto' skill...")
        await executor.execute(page, {"skill": "goto", "params": {"url": "https://google.com"}})
        await asyncio.sleep(3)  # Chờ trang load
        
        # Debug: Kiểm tra xem trang có load không
        print(f"Current URL: {page.url}")
        print(f"Page title: {await page.title()}")
        
        # Chụp screenshot để debug
        await page.screenshot(path="debug_after_goto.png")
        print("Screenshot saved: debug_after_goto.png")

        # 2. Test 'type' và 'click'
        print("\nTesting 'type' and 'click'...")
        
        # Thử nhiều selector khác nhau cho Google
        selectors_to_try = [
            "textarea[name='q']",
            "input[name='q']",
            "textarea[title='Search']",
            "input[type='search']",
        ]
        
        search_input = None
        found_selector = None
        for selector in selectors_to_try:
            try:
                print(f"Trying selector: {selector}")
                if selector.startswith("//"):
                    search_input = await page.wait_for_selector(f"xpath={selector}", timeout=2000)
                else:
                    search_input = await page.wait_for_selector(selector, timeout=2000)
                found_selector = selector
                print(f"yes Found with selector: {selector}")
                break
            except Exception:
                print(f"no Not found: {selector}")
                continue
        
        if not search_input:
            print("⚠️ Could not find search input! Printing page content...")
            content = await page.content()
            print(content[:500])  
            await page.screenshot(path="debug_no_input.png")
            return
        
        # Type vào search box
        await executor.execute(page, {"skill": "type", "params": {"selector": found_selector, "text": "laptop"}})
        await asyncio.sleep(1)
        
        # Click vào nút tìm kiếm Google
        await executor.execute(page, {"skill": "click", "params": {"selector": "input[name='btnK']"}})
        
        await asyncio.sleep(5)  # Chờ lâu hơn để xem kết quả tìm kiếm
        print("Execution Test Successful!")

    except Exception as e:
        print(f"EXECUTION LAYER FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(test_execution_layer())
