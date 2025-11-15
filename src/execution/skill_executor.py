"""
Skill Executor - Executes skills on web pages
Dispatches actions to appropriate skill implementations
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.skills import (
    NavigationSkills,
    InteractionSkills,
    ObservationSkills,
    ValidationSkills,
    WaitSkills
)
from src.execution.omni_passer import OmniPasser
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SkillExecutor:
    """
    Executes skills on browser pages.
    
    **Available Skills**:
        Navigation: goto, back, forward, reload, close
        Interaction: click, type, select, hover, press
        Observation: screenshot, get_dom, get_text, get_url
        Validation: check_exists, check_visible, check_enabled
        Wait: wait_for, wait_for_selector, wait_for_navigation
    
    **Execution Flow**:
        1. Receive action dict {skill, params}
        2. Dispatch to appropriate skill handler
        3. Execute with error handling
        4. Return result dict {status, message, data}
    """
    
    def __init__(self):
        """Initialize skill executor"""
        # Initialize skill handlers
        self.navigation = NavigationSkills()
        self.interaction = InteractionSkills()
        self.observation = ObservationSkills()
        self.validation = ValidationSkills()
        self.wait = WaitSkills()
        self.omnipasser = OmniPasser()
        
        # Map skill names to handlers
        self.skill_map = {
            # Navigation
            'goto': self.navigation.goto,
            'back': self.navigation.back,
            'forward': self.navigation.forward,
            'reload': self.navigation.reload,
            'close': self.navigation.close,
            
            # Interaction
            'click': self.interaction.click,
            'type': self.interaction.type,
            'select': self.interaction.select,
            'hover': self.interaction.hover,
            'press': self.interaction.press,
            'fill': self.interaction.fill,
            
            # Observation
            'screenshot': self.observation.screenshot,
            'get_dom': self.observation.get_dom,
            'get_text': self.observation.get_text,
            'get_url': self.observation.get_url,
            'get_title': self.observation.get_title,
            
            # Validation
            'check_exists': self.validation.check_exists,
            'check_visible': self.validation.check_visible,
            'check_enabled': self.validation.check_enabled,
            
            # Wait
            'wait_for': self.wait.wait_for,
            'wait_for_selector': self.wait.wait_for_selector,
            'wait_for_navigation': self.wait.wait_for_navigation,
            
            # Special
            'complete': self._complete,
            'omni_login': self.omnipasser.login,
        }
        
        logger.info(f"SkillExecutor initialized ({len(self.skill_map)} skills)")
    
    async def execute(self, page, action: Dict) -> Dict:
        """
        Execute action on page.
        
        Args:
            page: Playwright page
            action: Action dict {skill, params}
            
        Returns:
            Result dict {status, message, data}
        """
        skill_name = action.get('skill')
        params = action.get('params', {})
        
        if not skill_name:
            return {
                'status': 'error',
                'message': 'No skill specified'
            }
        
        # Get skill handler
        skill_handler = self.skill_map.get(skill_name)
        
        if not skill_handler:
            logger.error(f"Unknown skill: {skill_name}")
            return {
                'status': 'error',
                'message': f'Unknown skill: {skill_name}'
            }
        
        # Execute skill
        try:
            logger.debug(f"⚡ Executing: {skill_name}({params})")
            
            result = await skill_handler(page, **params)
            
            logger.debug(f"✓ {skill_name}: {result.get('status')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Skill execution failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_available_skills(self) -> List[str]:
        """Get list of available skills"""
        return list(self.skill_map.keys())
    
    async def _complete(self, page, **kwargs) -> Dict:
        """Complete task (special skill)"""
        message = kwargs.get('message', 'Task completed')
        
        logger.info(f"🏁 {message}")
        
        return {
            'status': 'success',
            'message': message,
            'data': {'completed': True}
        }


# Test
async def test_skill_executor():
    """Test skill executor"""
    from src.execution.browser_manager import BrowserManager
    import asyncio
    
    print("=" * 70)
    print("SkillExecutor - Test")
    print("=" * 70 + "\n")
    
    manager = BrowserManager(headless=False)
    executor = SkillExecutor()
    
    try:
        # Launch browser
        page = await manager.new_page()
        
        # Test 1: Navigation
        print("Test 1: Navigation")
        print("-" * 40)
        result = await executor.execute(page, {
            'skill': 'goto',
            'params': {'url': 'https://example.com'}
        })
        print(f"✓ goto: {result['status']}\n")
        
        await asyncio.sleep(1)
        
        # Test 2: Observation
        print("Test 2: Observation")
        print("-" * 40)
        result = await executor.execute(page, {
            'skill': 'get_title',
            'params': {}
        })
        print(f"✓ get_title: {result['data']['title']}\n")
        
        # Test 3: Check exists
        print("Test 3: Validation")
        print("-" * 40)
        result = await executor.execute(page, {
            'skill': 'check_exists',
            'params': {'selector': 'h1'}
        })
        print(f"✓ check_exists: {result['data']['exists']}\n")
        
        # Test 4: Get skills
        print("Test 4: Available Skills")
        print("-" * 40)
        skills = executor.get_available_skills()
        print(f"Available skills: {len(skills)}")
        for skill in sorted(skills)[:10]:
            print(f"  - {skill}")
        print(f"  ... and {len(skills)-10} more\n")
        
    finally:
        await manager.close()
    
    print("=" * 70)
    print("✅ Test Completed!")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_skill_executor())
