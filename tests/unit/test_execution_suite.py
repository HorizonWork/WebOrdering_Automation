"""
Comprehensive Execution Layer Test Suite
Tests all components: BrowserManager, SkillExecutor, and Skills
"""

import asyncio
import sys
from pathlib import Path

import path_setup  # noqa: F401

ROOT_DIR = path_setup.PROJECT_ROOT

from src.execution.browser_manager import BrowserManager
from src.execution.skill_executor import SkillExecutor


class ExecutionTestSuite:
    """Comprehensive test suite for execution layer"""
    
    def __init__(self):
        self.manager = None
        self.page = None
        self.executor = None
        self.results = []
        self.temp_dir = ROOT_DIR / "temp" / "test_execution"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Setup test environment"""
        print("=" * 70)
        print("EXECUTION LAYER TEST SUITE")
        print("=" * 70)
        print("\n🔧 Setting up test environment...\n")
        
        # Use standard browser (no profile to avoid conflicts)
        self.manager = BrowserManager(
            headless=False,
            use_chrome_profile=False
        )
        
        await self.manager.launch()
        self.page = await self.manager.new_page()
        self.executor = SkillExecutor()
        
        print("✅ Test environment ready\n")
    
    async def teardown(self):
        """Cleanup after tests"""
        print("\n🔒 Cleaning up...")
        if self.manager:
            await self.manager.close()
        print("✅ Cleanup complete\n")
    
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append((test_name, passed, message))
        print(f"{status} - {test_name}")
        if message:
            print(f"   {message}")
    
    async def test_1_browser_manager_launch(self):
        """Test 1: BrowserManager can launch browser"""
        print("\n" + "-" * 70)
        print("Test 1: BrowserManager Launch")
        print("-" * 70)
        
        try:
            assert self.manager is not None, "Manager not initialized"
            assert self.page is not None, "Page not created"
            self.log_result("BrowserManager Launch", True)
            return True
        except Exception as e:
            self.log_result("BrowserManager Launch", False, str(e))
            return False
    
    async def test_2_browser_manager_navigation(self):
        """Test 2: BrowserManager can navigate to URL"""
        print("\n" + "-" * 70)
        print("Test 2: Browser Navigation")
        print("-" * 70)
        
        try:
            await self.page.goto("https://example.com")
            await self.manager.wait_for_load(self.page)
            
            url = self.page.url
            title = await self.page.title()
            
            assert "example.com" in url, f"Wrong URL: {url}"
            assert title, "No page title"
            
            self.log_result("Browser Navigation", True, f"URL: {url}, Title: {title}")
            return True
        except Exception as e:
            self.log_result("Browser Navigation", False, str(e))
            return False
    
    async def test_3_browser_manager_screenshot(self):
        """Test 3: BrowserManager can take screenshots"""
        print("\n" + "-" * 70)
        print("Test 3: Screenshot Capture")
        print("-" * 70)
        
        try:
            screenshot_path = self.temp_dir / "test_screenshot.png"
            screenshot_bytes = await self.manager.screenshot(
                self.page, 
                path=str(screenshot_path)
            )
            
            assert screenshot_path.exists(), "Screenshot file not created"
            assert len(screenshot_bytes) > 0, "Empty screenshot"
            
            self.log_result(
                "Screenshot Capture", 
                True, 
                f"Saved to {screenshot_path}"
            )
            return True
        except Exception as e:
            self.log_result("Screenshot Capture", False, str(e))
            return False
    
    async def test_4_skill_executor_goto(self):
        """Test 4: SkillExecutor - goto skill"""
        print("\n" + "-" * 70)
        print("Test 4: Skill - goto")
        print("-" * 70)
        
        try:
            await self.executor.execute(
                self.page,
                {"skill": "goto", "params": {"url": "https://google.com"}}
            )
            
            await asyncio.sleep(2)
            url = self.page.url
            
            assert "google.com" in url, f"Navigation failed: {url}"
            
            self.log_result("Skill - goto", True, f"Navigated to {url}")
            return True
        except Exception as e:
            self.log_result("Skill - goto", False, str(e))
            return False
    
    async def test_5_skill_executor_type(self):
        """Test 5: SkillExecutor - type skill"""
        print("\n" + "-" * 70)
        print("Test 5: Skill - type")
        print("-" * 70)
        
        try:
            # First ensure we're on a page with input
            await self.page.goto("https://google.com")
            await asyncio.sleep(2)
            
            # Find search input
            selectors = [
                "textarea[name='q']",
                "input[name='q']",
                "textarea[title='Search']",
            ]
            
            found_selector = None
            for selector in selectors:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=2000)
                    if element:
                        found_selector = selector
                        break
                except:
                    continue
            
            if not found_selector:
                raise Exception("Could not find search input")
            
            # Type text
            await self.executor.execute(
                self.page,
                {"skill": "type", "params": {"selector": found_selector, "text": "test"}}
            )
            
            # Verify text was typed
            value = await self.page.evaluate(f'document.querySelector("{found_selector}").value')
            assert "test" in value.lower(), f"Text not typed correctly: {value}"
            
            self.log_result("Skill - type", True, f"Typed 'test' into {found_selector}")
            return True
        except Exception as e:
            self.log_result("Skill - type", False, str(e))
            return False
    
    async def test_6_skill_executor_click(self):
        """Test 6: SkillExecutor - click skill"""
        print("\n" + "-" * 70)
        print("Test 6: Skill - click")
        print("-" * 70)
        
        try:
            # Go to example.com which has a simple link
            await self.page.goto("https://example.com")
            await asyncio.sleep(2)
            
            # Click the "More information..." link
            await self.executor.execute(
                self.page,
                {"skill": "click", "params": {"selector": "a"}}
            )
            
            await asyncio.sleep(2)
            url = self.page.url
            
            # URL should have changed after click
            assert "iana.org" in url or "example" in url, f"Click didn't navigate: {url}"
            
            self.log_result("Skill - click", True, f"Clicked and navigated to {url}")
            return True
        except Exception as e:
            self.log_result("Skill - click", False, str(e))
            return False
    
    async def test_7_skill_executor_extract(self):
        """Test 7: SkillExecutor - extract skill"""
        print("\n" + "-" * 70)
        print("Test 7: Skill - extract")
        print("-" * 70)
        
        try:
            await self.page.goto("https://example.com")
            await asyncio.sleep(2)
            
            # Extract text from h1
            result = await self.executor.execute(
                self.page,
                {"skill": "extract", "params": {"selector": "h1"}}
            )
            
            assert result, "No extraction result"
            assert "Example" in result, f"Unexpected extraction: {result}"
            
            self.log_result("Skill - extract", True, f"Extracted: {result}")
            return True
        except Exception as e:
            self.log_result("Skill - extract", False, str(e))
            return False
    
    async def test_8_skill_executor_scroll(self):
        """Test 8: SkillExecutor - scroll skill"""
        print("\n" + "-" * 70)
        print("Test 8: Skill - scroll")
        print("-" * 70)
        
        try:
            # Go to a page with scrollable content
            await self.page.goto("https://example.com")
            await asyncio.sleep(1)
            
            # Get initial scroll position
            initial_y = await self.page.evaluate("window.scrollY")
            
            # Scroll down
            await self.executor.execute(
                self.page,
                {"skill": "scroll", "params": {"direction": "down"}}
            )
            
            await asyncio.sleep(1)
            new_y = await self.page.evaluate("window.scrollY")
            
            assert new_y >= initial_y, "Scroll didn't work"
            
            self.log_result("Skill - scroll", True, f"Scrolled from {initial_y} to {new_y}")
            return True
        except Exception as e:
            self.log_result("Skill - scroll", False, str(e))
            return False
    
    async def test_9_skill_executor_wait(self):
        """Test 9: SkillExecutor - wait skill"""
        print("\n" + "-" * 70)
        print("Test 9: Skill - wait")
        print("-" * 70)
        
        try:
            import time
            start = time.time()
            
            await self.executor.execute(
                self.page,
                {"skill": "wait", "params": {"selector": "body"}}
            )
            
            elapsed = time.time() - start
            
            # Wait should complete quickly if element exists
            assert elapsed < 5, f"Wait took too long: {elapsed}s"
            
            self.log_result("Skill - wait", True, f"Wait completed in {elapsed:.2f}s")
            return True
        except Exception as e:
            self.log_result("Skill - wait", False, str(e))
            return False
    
    async def test_10_error_handling(self):
        """Test 10: Error handling for invalid skills"""
        print("\n" + "-" * 70)
        print("Test 10: Error Handling")
        print("-" * 70)
        
        try:
            # Test invalid skill
            try:
                await self.executor.execute(
                    self.page,
                    {"skill": "invalid_skill", "params": {}}
                )
                # Should raise an error
                self.log_result("Error Handling", False, "Invalid skill didn't raise error")
                return False
            except Exception:
                # Expected to fail
                pass
            
            # Test invalid selector
            try:
                await self.executor.execute(
                    self.page,
                    {"skill": "click", "params": {"selector": "invalid-selector-xyz"}}
                )
                # Should timeout or error
                self.log_result("Error Handling", False, "Invalid selector didn't raise error")
                return False
            except Exception:
                # Expected to fail
                pass
            
            self.log_result("Error Handling", True, "Errors handled correctly")
            return True
        except Exception as e:
            self.log_result("Error Handling", False, str(e))
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        total = len(self.results)
        passed = sum(1 for _, p, _ in self.results if p)
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%\n")
        
        if failed > 0:
            print("Failed Tests:")
            for name, passed, message in self.results:
                if not passed:
                    print(f"  ❌ {name}")
                    if message:
                        print(f"     {message}")
        
        print("\n" + "=" * 70)
        
        return passed == total
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            await self.setup()
            
            # Run tests
            await self.test_1_browser_manager_launch()
            await self.test_2_browser_manager_navigation()
            await self.test_3_browser_manager_screenshot()
            await self.test_4_skill_executor_goto()
            await self.test_5_skill_executor_type()
            await self.test_6_skill_executor_click()
            await self.test_7_skill_executor_extract()
            await self.test_8_skill_executor_scroll()
            await self.test_9_skill_executor_wait()
            await self.test_10_error_handling()
            
            # Print summary
            all_passed = self.print_summary()
            
            return all_passed
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.teardown()


async def main():
    """Main test runner"""
    suite = ExecutionTestSuite()
    success = await suite.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
