"""
Form Agent - Specialized agent for form filling
Handles contact forms, registration, surveys, etc.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import asyncio

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.planning.sub_agents.base_agent import BaseSubAgent
from src.execution.skill_executor import SkillExecutor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FormAgent(BaseSubAgent):
    """
    Specialized agent for form filling.
    
    **Capabilities**:
        - Detect form fields (text, email, phone, select, checkbox)
        - Fill fields with provided data
        - Validate input formats
        - Handle required fields
        - Submit forms
        - Detect errors
    
    **Form Types**:
        - Contact forms
        - Registration forms
        - Survey forms
        - Shipping/address forms
        - Profile update forms
    
    **Flow**:
        1. Detect form fields
        2. Map data to fields
        3. Fill required fields first
        4. Validate inputs
        5. Submit form
        6. Check for errors
    """
    
    def __init__(self):
        super().__init__(
            name="FormAgent",
            description="Handles form detection and filling"
        )
        self.skill_executor = SkillExecutor()
        
        # Field type mapping
        self.field_types = {
            'text': ['text', 'search'],
            'email': ['email'],
            'phone': ['tel', 'phone'],
            'password': ['password'],
            'number': ['number'],
            'date': ['date'],
            'select': ['select-one', 'select-multiple'],
            'checkbox': ['checkbox'],
            'radio': ['radio'],
            'textarea': ['textarea']
        }
    
    async def can_handle(self, task: Dict, observation: Dict) -> bool:
        """Check if this is a form-filling task"""
        task_desc = str(task.get('description', '')).lower()
        form_keywords = [
            'điền', 'fill', 'form', 'đăng ký', 'register',
            'liên hệ', 'contact', 'thông tin', 'information'
        ]
        
        if any(kw in task_desc for kw in form_keywords):
            logger.info("✓ Form task detected from description")
            return True
        
        # Check for form elements
        dom = observation.get('dom', '').lower()
        has_form = '<form' in dom or 'type="submit"' in dom
        has_inputs = dom.count('input') >= 2
        
        if has_form and has_inputs:
            logger.info("✓ Form detected in DOM")
            return True
        
        return False
    
    async def execute(
        self,
        task: Dict,
        page,
        observation: Dict
    ) -> Dict:
        """
        Execute form filling task.
        
        Args:
            task: Form task with {form_data, submit}
            page: Playwright page
            observation: Current state
            
        Returns:
            Result dict
        """
        logger.info("📝 Starting form filling...")
        
        try:
            form_data = task.get('form_data', {})
            should_submit = task.get('submit', True)
            
            if not form_data:
                logger.warning("⚠️  No form data provided")
                return {
                    'success': False,
                    'message': 'Form data required'
                }
            
            logger.info(f"   Fields to fill: {list(form_data.keys())}")
            
            # Step 1: Detect form fields
            logger.info("🔍 Detecting form fields...")
            fields = await self._detect_form_fields(page)
            logger.info(f"✓ Found {len(fields)} form fields")
            
            # Step 2: Fill fields
            logger.info("📝 Filling form fields...")
            filled_count = 0
            
            for field_name, field_value in form_data.items():
                # Find matching field
                field = self._find_matching_field(fields, field_name)
                
                if field:
                    success = await self._fill_field(page, field, field_value)
                    if success:
                        filled_count += 1
                        logger.info(f"   ✓ Filled: {field_name}")
                    else:
                        logger.warning(f"   ✗ Failed: {field_name}")
                else:
                    logger.warning(f"   ? Field not found: {field_name}")
            
            logger.info(f"✓ Filled {filled_count}/{len(form_data)} fields")
            
            # Step 3: Submit form (if requested)
            if should_submit:
                logger.info("📤 Submitting form...")
                submit_success = await self._submit_form(page)
                
                if submit_success:
                    await asyncio.sleep(2)
                    
                    # Check for errors
                    errors = await self._check_form_errors(page)
                    
                    if errors:
                        logger.warning(f"⚠️  Form errors detected: {errors}")
                        self._record_failure()
                        return {
                            'success': False,
                            'message': 'Form submitted with errors',
                            'errors': errors,
                            'filled_fields': filled_count
                        }
            
            self._record_success()
            return {
                'success': True,
                'message': f'Form filled successfully ({filled_count} fields)',
                'filled_fields': filled_count,
                'total_fields': len(form_data),
                'submitted': should_submit
            }
            
        except Exception as e:
            logger.error(f"❌ Form filling failed: {e}")
            self._record_failure()
            return {
                'success': False,
                'message': f'Form error: {str(e)}'
            }
    
    async def _detect_form_fields(self, page) -> List[Dict]:
        """Detect all form fields on page"""
        fields = []
        
        # Get all input elements
        inputs = await page.query_selector_all('input, select, textarea')
        
        for idx, element in enumerate(inputs):
            try:
                # Get attributes
                tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                field_type = await element.get_attribute('type') or 'text'
                field_name = await element.get_attribute('name') or f'field_{idx}'
                field_id = await element.get_attribute('id')
                placeholder = await element.get_attribute('placeholder')
                required = await element.get_attribute('required') is not None
                
                # Check visibility
                visible = await element.is_visible()
                
                if visible:
                    fields.append({
                        'index': idx,
                        'tag': tag_name,
                        'type': field_type,
                        'name': field_name,
                        'id': field_id,
                        'placeholder': placeholder,
                        'required': required,
                        'selector': f'#{field_id}' if field_id else f'[name="{field_name}"]'
                    })
            except:
                continue
        
        return fields
    
    def _find_matching_field(self, fields: List[Dict], field_name: str) -> Optional[Dict]:
        """Find field matching the given name"""
        field_name_lower = field_name.lower()
        
        # Exact match
        for field in fields:
            if field['name'].lower() == field_name_lower:
                return field
        
        # Partial match (name, id, placeholder)
        for field in fields:
            if (field_name_lower in field['name'].lower() or
                (field['id'] and field_name_lower in field['id'].lower()) or
                (field['placeholder'] and field_name_lower in field['placeholder'].lower())):
                return field
        
        return None
    
    async def _fill_field(self, page, field: Dict, value: str) -> bool:
        """Fill a single field"""
        try:
            selector = field['selector']
            field_type = field['type']
            
            if field['tag'] == 'select':
                # Dropdown
                await self.skill_executor.execute(page, {
                    'skill': 'select',
                    'params': {
                        'selector': selector,
                        'value': value
                    }
                })
            elif field_type == 'checkbox':
                # Checkbox
                if value in [True, 'true', '1', 'yes']:
                    await self.skill_executor.execute(page, {
                        'skill': 'click',
                        'params': {'selector': selector}
                    })
            elif field['tag'] == 'textarea' or field_type in ['text', 'email', 'tel', 'number']:
                # Text input
                await self.skill_executor.execute(page, {
                    'skill': 'type',
                    'params': {
                        'selector': selector,
                        'text': str(value),
                        'clear': True
                    }
                })
            else:
                # Default: type
                await self.skill_executor.execute(page, {
                    'skill': 'type',
                    'params': {
                        'selector': selector,
                        'text': str(value)
                    }
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fill field {field['name']}: {e}")
            return False
    
    async def _submit_form(self, page) -> bool:
        """Submit the form"""
        # Try to find submit button
        submit_selectors = [
            'button[type="submit"]',
            'input[type="submit"]',
            'button:has-text("Submit")',
            'button:has-text("Gửi")',
            'button:has-text("Đăng ký")',
            'button:has-text("Register")',
            '.submit-button',
            '#submit'
        ]
        
        for selector in submit_selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    logger.info(f"✓ Clicked submit button: {selector}")
                    return True
            except:
                continue
        
        # Fallback: Press Enter
        logger.info("   Pressing Enter to submit")
        await page.keyboard.press('Enter')
        return True
    
    async def _check_form_errors(self, page) -> List[str]:
        """Check for form validation errors"""
        errors = []
        
        error_selectors = [
            '.error',
            '.error-message',
            '.alert-danger',
            '.form-error',
            '[class*="error"]',
            '[role="alert"]'
        ]
        
        for selector in error_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for elem in elements:
                    if await elem.is_visible():
                        text = await elem.inner_text()
                        if text.strip():
                            errors.append(text.strip())
            except:
                continue
        
        return errors


# Test
async def test_form_agent():
    """Test form agent"""
    print("=" * 70)
    print("FormAgent - Test")
    print("=" * 70 + "\n")
    
    agent = FormAgent()
    
    # Test can_handle
    task = {'description': 'Điền form liên hệ'}
    observation = {'dom': '<form><input name="email"/><input type="submit"/></form>'}
    
    can_handle = await agent.can_handle(task, observation)
    print(f"✓ Can handle form task: {can_handle}")
    
    print(f"\n📊 Stats: {agent.get_stats()}")
    print("\n✅ Form agent ready")


if __name__ == "__main__":
    asyncio.run(test_form_agent())
