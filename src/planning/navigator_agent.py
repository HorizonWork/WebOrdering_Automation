"""
Navigator Agent - Browser Navigation Management
Coordinates skill execution and handles navigation-specific logic
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.skill_executor import SkillExecutor
from src.planning.change_observer import ChangeObserver
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NavigationResult:
    """Result of navigation operation"""
    success: bool
    action: Dict
    result: Dict
    changes: Dict
    error: Optional[str] = None


class NavigatorAgent:
    """
    Navigator agent for browser navigation and skill execution.
    
    **Responsibilities**:
        - Execute navigation skills (goto, click, type)
        - Observe changes after actions
        - Detect navigation failures
        - Suggest recovery actions
        - Handle timeouts and retries
    
    **Features**:
        - Change observation integration
        - Automatic retry on failure (3 attempts)
        - Backtracking on errors
        - Navigation state tracking
    """
    
    def __init__(
        self,
        skill_executor: Optional[SkillExecutor] = None,
        max_retries: int = 3,
        enable_change_observer: bool = True
    ):
        """
        Initialize navigator agent.
        
        Args:
            skill_executor: Skill executor (creates new if None)
            max_retries: Max retry attempts on failure
            enable_change_observer: Enable DOM change observation
        """
        self.skill_executor = skill_executor or SkillExecutor()
        self.max_retries = max_retries
        self.enable_change_observer = enable_change_observer
        
        if self.enable_change_observer:
            self.change_observer = ChangeObserver()
        
        self.navigation_history: List[str] = []
        
        logger.info(f"NavigatorAgent initialized (max_retries={max_retries}, change_observer={enable_change_observer})")
    
    async def execute_with_observation(
        self,
        page,
        action: Dict,
        expect_change: str = 'any'
    ) -> NavigationResult:
        """
        Execute action with change observation.
        
        Args:
            page: Playwright page
            action: Action to execute
            expect_change: Expected change level ('any', 'major', 'moderate')
            
        Returns:
            NavigationResult with success status and changes
        """
        skill = action.get('skill')
        
        logger.info(f"🚀 Executing: {skill}({action.get('params')})")
        
        # Start observing (if enabled)
        if self.enable_change_observer:
            await self.change_observer.start_observing(page)
        
        # Execute action
        result = await self.skill_executor.execute(page, action)
        
        # Get changes (if enabled)
        changes = {}
        if self.enable_change_observer:
            change_list = await self.change_observer.get_changes(page)
            changes = self.change_observer.analyze_changes(change_list)
            
            # Log changes
            logger.info(f"📊 Changes: {changes['status']} - {changes['message']}")
        
        # Determine success
        action_success = result.get('status') == 'success'
        change_success = True
        
        if self.enable_change_observer and expect_change != 'none':
            change_success = self.change_observer.did_action_succeed(expect_change)
        
        overall_success = action_success and change_success
        
        # Track navigation
        if skill == 'goto' and overall_success:
            url = action['params'].get('url', '')
            self.navigation_history.append(url)
        
        return NavigationResult(
            success=overall_success,
            action=action,
            result=result,
            changes=changes,
            error=result.get('message') if not action_success else None
        )
    
    async def execute_with_retry(
        self,
        page,
        action: Dict,
        expect_change: str = 'any'
    ) -> NavigationResult:
        """
        Execute action with automatic retry on failure.
        
        Args:
            page: Playwright page
            action: Action to execute
            expect_change: Expected change level
            
        Returns:
            NavigationResult
        """
        last_result = None
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"🔄 Attempt {attempt}/{self.max_retries}")
            
            result = await self.execute_with_observation(page, action, expect_change)
            
            if result.success:
                logger.info(f"✅ Action succeeded on attempt {attempt}")
                return result
            
            last_result = result
            logger.warning(f"❌ Attempt {attempt} failed: {result.error}")
            
            # Wait before retry
            if attempt < self.max_retries:
                import asyncio
                await asyncio.sleep(1)
        
        logger.error(f"❌ All {self.max_retries} attempts failed")
        return last_result
    
    async def navigate_to(
        self,
        page,
        url: str,
        wait_until: str = 'networkidle'
    ) -> NavigationResult:
        """
        Navigate to URL with observation.
        
        Args:
            page: Playwright page
            url: Target URL
            wait_until: Wait condition
            
        Returns:
            NavigationResult
        """
        action = {
            'skill': 'goto',
            'params': {
                'url': url,
                'wait_until': wait_until
            }
        }
        
        return await self.execute_with_retry(page, action, expect_change='major')
    
    async def click_and_wait(
        self,
        page,
        selector: str,
        expect_navigation: bool = False
    ) -> NavigationResult:
        """
        Click element and wait for changes.
        
        Args:
            page: Playwright page
            selector: Element selector
            expect_navigation: Whether click causes navigation
            
        Returns:
            NavigationResult
        """
        action = {
            'skill': 'click',
            'params': {'selector': selector}
        }
        
        expect_change = 'major' if expect_navigation else 'any'
        return await self.execute_with_retry(page, action, expect_change=expect_change)
    
    async def type_and_submit(
        self,
        page,
        input_selector: str,
        text: str,
        submit_selector: Optional[str] = None
    ) -> NavigationResult:
        """
        Type text and optionally submit.
        
        Args:
            page: Playwright page
            input_selector: Input field selector
            text: Text to type
            submit_selector: Submit button selector (optional)
            
        Returns:
            NavigationResult
        """
        # Type text
        type_action = {
            'skill': 'type',
            'params': {
                'selector': input_selector,
                'text': text
            }
        }
        
        result = await self.execute_with_observation(page, type_action, expect_change='minor')
        
        if not result.success:
            return result
        
        # Submit if selector provided
        if submit_selector:
            submit_action = {
                'skill': 'click',
                'params': {'selector': submit_selector}
            }
            
            result = await self.execute_with_retry(page, submit_action, expect_change='major')
        
        return result
    
    def can_go_back(self) -> bool:
        """Check if can navigate back"""
        return len(self.navigation_history) > 1
    
    async def go_back(self, page) -> NavigationResult:
        """Navigate back in history"""
        if not self.can_go_back():
            return NavigationResult(
                success=False,
                action={'skill': 'back'},
                result={'status': 'error', 'message': 'No history to go back'},
                changes={},
                error='No navigation history'
            )
        
        action = {'skill': 'back', 'params': {}}
        
        result = await self.execute_with_observation(page, action, expect_change='major')
        
        if result.success:
            self.navigation_history.pop()
        
        return result
    
    def get_navigation_history(self) -> List[str]:
        """Get navigation history"""
        return self.navigation_history.copy()
    
    def reset(self):
        """Reset navigator state"""
        self.navigation_history = []
        if self.enable_change_observer:
            self.change_observer.reset()
        logger.info("NavigatorAgent reset")


# Test & Example
async def test_navigator_agent():
    """Test navigator agent"""
    from src.execution.browser_manager import BrowserManager
    
    print("=" * 70)
    print("NavigatorAgent - Test")
    print("=" * 70 + "\n")
    
    # Initialize
    navigator = NavigatorAgent()
    manager = BrowserManager(headless=False)
    
    try:
        # Create page
        page = await manager.new_page()
        print("✓ Page created\n")
        
        # Test 1: Navigate
        print("🔄 Test 1: Navigate to URL")
        result = await navigator.navigate_to(page, "https://example.com")
        print(f"   Success: {result.success}")
        print(f"   Changes: {result.changes.get('status', 'N/A')}\n")
        
        # Test 2: Click (if link exists)
        print("🔄 Test 2: Click link")
        result = await navigator.click_and_wait(page, "a", expect_navigation=True)
        print(f"   Success: {result.success}")
        print(f"   Changes: {result.changes.get('status', 'N/A')}\n")
        
        # Test 3: Navigation history
        print("📜 Navigation History:")
        for i, url in enumerate(navigator.get_navigation_history(), 1):
            print(f"   {i}. {url}")
        
        print("\n✅ Tests completed!")
        
    finally:
        await manager.close()
        print("\n✓ Cleanup complete")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_navigator_agent())
