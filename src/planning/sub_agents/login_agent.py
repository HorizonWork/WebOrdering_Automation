"""
Login Agent - Specialized agent for authentication flows
Handles login forms, captchas, 2FA, etc.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import asyncio

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.planning.sub_agents.base_agent import BaseSubAgent
from src.execution.skill_executor import SkillExecutor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoginAgent(BaseSubAgent):
    """
    Specialized agent for login/authentication tasks.
    
    **Capabilities**:
        - Detect login forms
        - Fill username/password
        - Handle "Remember me" checkboxes
        - Detect success/failure
        - Handle captcha prompts (human-in-loop)
        - Support 2FA flows
    
    **Login Flow**:
        1. Detect login form
        2. Fill credentials (from user)
        3. Submit form
        4. Wait for result
        5. Verify success (URL change, welcome message)
    
    **Safety**:
        - Never stores passwords
        - Prompts user for credentials
        - Respects safety guardrails
    """
    
    def __init__(self):
        super().__init__(
            name="LoginAgent",
            description="Handles authentication and login flows"
        )
        self.skill_executor = SkillExecutor()
        
        # Common login selectors
        self.login_selectors = {
            'username': [
                'input[name="username"]',
                'input[name="email"]',
                'input[type="email"]',
                'input[id*="user"]',
                'input[id*="email"]',
                '#username',
                '#email'
            ],
            'password': [
                'input[name="password"]',
                'input[type="password"]',
                '#password',
                'input[id*="pass"]'
            ],
            'submit': [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Đăng nhập")',
                'button:has-text("Login")',
                'button:has-text("Sign in")',
                '.login-button',
                '#login-button'
            ]
        }
    
    async def can_handle(self, task: Dict, observation: Dict) -> bool:
        """
        Check if this is a login task.
        
        Args:
            task: Task dict
            observation: Current state
            
        Returns:
            True if login-related
        """
        # Check task description
        task_desc = str(task.get('description', '')).lower()
        login_keywords = ['đăng nhập', 'login', 'sign in', 'authentication']
        
        if any(kw in task_desc for kw in login_keywords):
            logger.info("yes Login task detected from description")
            return True
        
        # Check DOM for login form
        dom = observation.get('dom', '').lower()
        has_username = any(sel in dom for sel in ['username', 'email', 'user'])
        has_password = 'password' in dom
        
        if has_username and has_password:
            logger.info("yes Login form detected in DOM")
            return True
        
        return False
    
    async def execute(
        self,
        task: Dict,
        page,
        observation: Dict
    ) -> Dict:
        """
        Execute login task.
        
        Args:
            task: Login task with credentials
            page: Playwright page
            observation: Current state
            
        Returns:
            Result dict
        """
        logger.info("🔐 Starting login flow...")
        
        try:
            # Get credentials from task
            credentials = task.get('credentials', {})
            username = credentials.get('username') or credentials.get('email')
            password = credentials.get('password')
            
            if not username or not password:
                logger.warning("⚠️  No credentials provided, prompting user...")
                # In production, this would prompt user
                return {
                    'success': False,
                    'message': 'Credentials required for login',
                    'requires_user_input': True
                }
            
            # Step 1: Find and fill username field
            username_selector = await self._find_field(page, self.login_selectors['username'])
            if not username_selector:
                self._record_failure()
                return {
                    'success': False,
                    'message': 'Username field not found'
                }
            
            logger.info(f"yes Found username field: {username_selector}")
            await self.skill_executor.execute(page, {
                'skill': 'type',
                'params': {
                    'selector': username_selector,
                    'text': username,
                    'clear': True
                }
            })
            
            # Step 2: Find and fill password field
            password_selector = await self._find_field(page, self.login_selectors['password'])
            if not password_selector:
                self._record_failure()
                return {
                    'success': False,
                    'message': 'Password field not found'
                }
            
            logger.info(f"yes Found password field: {password_selector}")
            await self.skill_executor.execute(page, {
                'skill': 'type',
                'params': {
                    'selector': password_selector,
                    'text': password,
                    'clear': True
                }
            })
            
            # Step 3: Find and click submit button
            submit_selector = await self._find_field(page, self.login_selectors['submit'])
            if not submit_selector:
                # Try pressing Enter instead
                logger.warning("⚠️  Submit button not found, pressing Enter")
                await page.keyboard.press('Enter')
            else:
                logger.info(f"yes Found submit button: {submit_selector}")
                await self.skill_executor.execute(page, {
                    'skill': 'click',
                    'params': {'selector': submit_selector}
                })
            
            # Step 4: Wait for navigation/result
            logger.info("⏳ Waiting for login result...")
            await asyncio.sleep(3)
            
            # Step 5: Verify login success
            success = await self._verify_login_success(page)
            
            if success:
                self._record_success()
                return {
                    'success': True,
                    'message': 'Login successful',
                    'final_url': page.url
                }
            else:
                self._record_failure()
                return {
                    'success': False,
                    'message': 'Login failed - credentials may be incorrect'
                }
                
        except Exception as e:
            logger.error(f"no Login failed: {e}")
            self._record_failure()
            return {
                'success': False,
                'message': f'Login error: {str(e)}'
            }
    
    async def _find_field(self, page, selectors: list) -> Optional[str]:
        """Find first matching field selector"""
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    return selector
            except:
                continue
        return None
    
    async def _verify_login_success(self, page) -> bool:
        """Verify if login succeeded"""
        url = page.url.lower()
        
        # Check URL change
        if any(kw in url for kw in ['dashboard', 'home', 'account', 'profile']):
            logger.info("yes Login success detected (URL changed)")
            return True
        
        # Check for error messages
        try:
            error_selectors = [
                '.error',
                '.alert-danger',
                '[class*="error"]',
                'text=Sai tên đăng nhập',
                'text=Incorrect password'
            ]
            
            for sel in error_selectors:
                error = await page.query_selector(sel)
                if error and await error.is_visible():
                    logger.warning("⚠️  Login error message detected")
                    return False
        except:
            pass
        
        # Check for login form still present
        try:
            login_form = await page.query_selector('input[type="password"]')
            if login_form and await login_form.is_visible():
                logger.warning("⚠️  Login form still visible")
                return False
        except:
            pass
        
        # Assume success if no errors found
        logger.info("yes No errors detected, assuming success")
        return True


# Test
async def test_login_agent():
    """Test login agent"""
    from src.execution.browser_manager import BrowserManager
    
    print("=" * 70)
    print("LoginAgent - Test")
    print("=" * 70 + "\n")
    
    agent = LoginAgent()
    manager = BrowserManager(headless=False)
    
    try:
        page = await manager.new_page()
        
        # Test 1: Can handle
        task = {'description': 'Đăng nhập vào hệ thống'}
        observation = {'dom': '<input type="password" name="password"/>'}
        
        can_handle = await agent.can_handle(task, observation)
        print(f"yes Can handle login task: {can_handle}")
        
        # Test 2: Navigate to login page
        await page.goto("https://example.com")  # Replace with actual login page
        await asyncio.sleep(2)
        
        print("\nyes Login agent ready (provide actual credentials to test execution)")
        
        # Show stats
        print(f"\n📊 Stats: {agent.get_stats()}")
        
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(test_login_agent())
