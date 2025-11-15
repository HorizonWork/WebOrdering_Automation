"""
Payment Agent - Specialized agent for checkout/payment flows
Handles cart, checkout forms, payment methods, etc.
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


class PaymentAgent(BaseSubAgent):
    """
    Specialized agent for payment/checkout tasks.
    
    **Capabilities**:
        - Add items to cart
        - Fill checkout forms (address, phone)
        - Select payment method
        - Review order before confirmation
        - Require user confirmation for payment
    
    **Safety**:
        - ALWAYS requires user confirmation before payment
        - Never enters credit card details
        - Respects payment threshold from guardrails
        - Human-in-loop for final step
    
    **Flow**:
        1. Navigate to cart
        2. Review items & total
        3. Fill shipping info
        4. Select payment method (COD preferred)
        5. **STOP** - Prompt user confirmation
        6. User confirms → Complete order
    """
    
    def __init__(self):
        super().__init__(
            name="PaymentAgent",
            description="Handles checkout and payment flows (with human confirmation)"
        )
        self.skill_executor = SkillExecutor()
        
        # Payment/checkout selectors
        self.selectors = {
            'cart_button': [
                '.cart-button',
                '[href*="cart"]',
                'button:has-text("Giỏ hàng")',
                'button:has-text("Cart")'
            ],
            'checkout_button': [
                '.checkout-button',
                'button:has-text("Thanh toán")',
                'button:has-text("Checkout")',
                'button:has-text("Đặt hàng")'
            ],
            'address_input': [
                'input[name*="address"]',
                'textarea[name*="address"]',
                '#address'
            ],
            'phone_input': [
                'input[name*="phone"]',
                'input[type="tel"]',
                '#phone'
            ],
            'payment_method': [
                'input[name="payment_method"]',
                '.payment-method',
                '[data-payment-method]'
            ]
        }
    
    async def can_handle(self, task: Dict, observation: Dict) -> bool:
        """Check if this is a payment/checkout task"""
        task_desc = str(task.get('description', '')).lower()
        payment_keywords = [
            'thanh toán', 'checkout', 'payment', 'đặt hàng',
            'mua', 'buy', 'cart', 'giỏ hàng'
        ]
        
        if any(kw in task_desc for kw in payment_keywords):
            logger.info("✓ Payment task detected")
            return True
        
        # Check DOM
        dom = observation.get('dom', '').lower()
        if any(kw in dom for kw in ['checkout', 'payment', 'cart', 'thanh toán']):
            logger.info("✓ Payment page detected")
            return True
        
        return False
    
    async def execute(
        self,
        task: Dict,
        page,
        observation: Dict
    ) -> Dict:
        """
        Execute payment flow.
        
        Args:
            task: Payment task with order details
            page: Playwright page
            observation: Current state
            
        Returns:
            Result dict
        """
        logger.info("💳 Starting payment flow...")
        
        try:
            order_info = task.get('order_info', {})
            
            # Step 1: Navigate to cart (if not already there)
            current_url = page.url.lower()
            if 'cart' not in current_url and 'checkout' not in current_url:
                logger.info("📦 Navigating to cart...")
                cart_selector = await self._find_field(page, self.selectors['cart_button'])
                if cart_selector:
                    await self.skill_executor.execute(page, {
                        'skill': 'click',
                        'params': {'selector': cart_selector}
                    })
                    await asyncio.sleep(2)
            
            # Step 2: Review cart items
            logger.info("📋 Reviewing cart items...")
            cart_summary = await self._get_cart_summary(page)
            logger.info(f"   Items: {cart_summary.get('items', 0)}")
            logger.info(f"   Total: {cart_summary.get('total', 'N/A')}")
            
            # Step 3: Click checkout
            logger.info("🛒 Proceeding to checkout...")
            checkout_selector = await self._find_field(page, self.selectors['checkout_button'])
            if checkout_selector:
                await self.skill_executor.execute(page, {
                    'skill': 'click',
                    'params': {'selector': checkout_selector}
                })
                await asyncio.sleep(2)
            
            # Step 4: Fill shipping info
            logger.info("📝 Filling shipping information...")
            shipping_info = order_info.get('shipping', {})
            
            # Address
            address_selector = await self._find_field(page, self.selectors['address_input'])
            if address_selector and shipping_info.get('address'):
                await self.skill_executor.execute(page, {
                    'skill': 'type',
                    'params': {
                        'selector': address_selector,
                        'text': shipping_info['address']
                    }
                })
            
            # Phone
            phone_selector = await self._find_field(page, self.selectors['phone_input'])
            if phone_selector and shipping_info.get('phone'):
                await self.skill_executor.execute(page, {
                    'skill': 'type',
                    'params': {
                        'selector': phone_selector,
                        'text': shipping_info['phone']
                    }
                })
            
            # Step 5: Select payment method (COD preferred)
            logger.info("💰 Selecting payment method...")
            payment_method = order_info.get('payment_method', 'cod')
            await self._select_payment_method(page, payment_method)
            
            # Step 6: **STOP HERE** - Require user confirmation
            logger.warning("⚠️  PAYMENT REQUIRES USER CONFIRMATION")
            logger.warning("⚠️  Agent will NOT complete payment automatically")
            
            self._record_success()
            return {
                'success': True,
                'message': 'Ready for payment - USER CONFIRMATION REQUIRED',
                'requires_user_confirmation': True,
                'cart_summary': cart_summary,
                'action_needed': 'User must review and confirm order manually'
            }
            
        except Exception as e:
            logger.error(f"❌ Payment flow failed: {e}")
            self._record_failure()
            return {
                'success': False,
                'message': f'Payment error: {str(e)}'
            }
    
    async def _find_field(self, page, selectors: list) -> Optional[str]:
        """Find first matching field"""
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    return selector
            except:
                continue
        return None
    
    async def _get_cart_summary(self, page) -> Dict:
        """Get cart summary"""
        try:
            # Try to get item count
            items_text = await page.inner_text('.cart-items, .item-count')
            
            # Try to get total
            total_text = await page.inner_text('.total, .cart-total, .order-total')
            
            return {
                'items': items_text or 'Unknown',
                'total': total_text or 'Unknown'
            }
        except:
            return {'items': 'Unknown', 'total': 'Unknown'}
    
    async def _select_payment_method(self, page, method: str):
        """Select payment method"""
        method_map = {
            'cod': ['COD', 'Thanh toán khi nhận hàng', 'Cash on delivery'],
            'bank': ['Chuyển khoản', 'Bank transfer'],
            'card': ['Thẻ tín dụng', 'Credit card']
        }
        
        keywords = method_map.get(method, method_map['cod'])
        
        for keyword in keywords:
            try:
                selector = f'text="{keyword}"'
                element = await page.query_selector(selector)
                if element:
                    await element.click()
                    logger.info(f"✓ Selected payment method: {keyword}")
                    return
            except:
                continue
        
        logger.warning("⚠️  Could not select payment method")


# Test
async def test_payment_agent():
    """Test payment agent"""
    print("=" * 70)
    print("PaymentAgent - Test")
    print("=" * 70 + "\n")
    
    agent = PaymentAgent()
    
    # Test can_handle
    task = {'description': 'Thanh toán đơn hàng'}
    observation = {'dom': '<button>Checkout</button>'}
    
    can_handle = await agent.can_handle(task, observation)
    print(f"✓ Can handle payment task: {can_handle}")
    
    print(f"\n📊 Stats: {agent.get_stats()}")
    print("\n✅ Payment agent ready")


if __name__ == "__main__":
    asyncio.run(test_payment_agent())
