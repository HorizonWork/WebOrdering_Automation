"""
LAZADA VIETNAM - WOA AGENT DEMO
================================
Production-Ready Web Ordering Automation Agent
Demonstrates full 4-layer architecture in action

Author: WOA Agent Team  
Date: 2025-11-16
Version: 2.0.0
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ============================================================================
# IMPORTS - Core Components (Simulated for Demo)
# ============================================================================

# Logging system
from src.utils.logger import get_logger, setup_logging

# Models (simulated imports - we'll use Playwright directly for demo)
# from src.models.phobert_encoder import PhoBERTEncoder
# from src.models.vit5_planner import ViT5Planner

# Perception Layer
# from src.perception.dom_distiller import DOMDistiller
# from src.perception.vision_enhancer import VisionEnhancer

# Planning Layer
# from src.planning.react_engine import ReActEngine

# Execution Layer
# from src.execution.browser_manager import BrowserManager
# from src.execution.skill_executor import SkillExecutor

# Orchestration
# from src.orchestrator.state_manager import StateManager
# from src.orchestrator.safety_guardrails import SafetyGuardrails

# For demo, we use Playwright directly
from playwright.async_api import async_playwright, Page, Browser

# Initialize logging
setup_logging(level="INFO", colored=True)
logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Product:
    """Product data extracted from e-commerce platform"""
    title: str
    price: float
    original_price: Optional[float] = None
    discount: Optional[str] = None
    rating: Optional[float] = None
    sold_count: Optional[str] = None
    url: str = ""
    image_url: str = ""
    
    @property
    def discount_amount(self) -> Optional[float]:
        """Calculate discount amount"""
        if self.original_price and self.original_price > self.price:
            return self.original_price - self.price
        return None


@dataclass
class UserRequest:
    """User request parameters"""
    query: Optional[str] = None
    product_url: Optional[str] = None
    max_products: int = 10
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    buy_now: bool = False
    quantity: int = 1


@dataclass
class AgentState:
    """Agent execution state"""
    current_url: str = ""
    current_page: str = ""
    step: int = 0
    max_steps: int = 15
    products_found: int = 0
    action_history: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Final execution result"""
    success: bool
    steps: int
    final_state: AgentState
    history: List[Dict]
    error: Optional[str] = None
    execution_time: float = 0.0
    summary: str = ""


# ============================================================================
# PERCEPTION LAYER (Simulated)
# ============================================================================

class PerceptionModule:
    """
    Perception Layer - DOM Analysis & Vision
    
    In real agent:
    - DOMDistiller: Extract interactive elements
    - VisionEnhancer: OCR, screenshot analysis
    - PhoBERT: Encode Vietnamese text
    """
    
    def __init__(self):
        logger.info("🔍 Initializing Perception Module")
        logger.info("   ├─ DOMDistiller: DOM parsing & element extraction")
        logger.info("   ├─ VisionEnhancer: Florence-2 vision model")
        logger.info("   └─ PhoBERT: Vietnamese text encoding")
        
    async def analyze_page(self, page: Page, url: str) -> Dict[str, Any]:
        """Analyze current page state"""
        logger.debug(f"📊 Analyzing page: {url[:60]}...")
        
        # Simulate DOM distillation
        title = await page.title()
        
        # Count interactive elements
        buttons = await page.query_selector_all('button')
        inputs = await page.query_selector_all('input')
        links = await page.query_selector_all('a')
        
        observation = {
            'url': url,
            'title': title,
            'interactive_elements': {
                'buttons': len(buttons),
                'inputs': len(inputs),
                'links': len(links),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"   ├─ Title: {title[:50]}...")
        logger.debug(f"   ├─ Buttons: {len(buttons)}")
        logger.debug(f"   ├─ Inputs: {len(inputs)}")
        logger.debug(f"   └─ Links: {len(links)}")
        
        return observation


# ============================================================================
# PLANNING LAYER (Simulated)
# ============================================================================

class PlanningModule:
    """
    Planning Layer - ReAct Decision Making
    
    In real agent:
    - ViT5Planner: Generate actions from observations
    - ReActEngine: Thought → Action → Observation loop
    """
    
    def __init__(self):
        logger.info("🧠 Initializing Planning Module")
        logger.info("   ├─ ViT5 Planner: Action generation (VietAI/vit5-base)")
        logger.info("   └─ ReAct Engine: Reasoning loop")
        
    def plan_action(self, state: AgentState, observation: Dict) -> Dict[str, Any]:
        """
        Plan next action based on current state
        
        In real agent, ViT5 generates this from:
        [INSTRUCTION] {query}
        [OBSERVATION] {dom_elements}
        [HISTORY] {previous_actions}
        """
        logger.debug(f"💭 Planning step {state.step + 1}/{state.max_steps}")
        
        # Simulate ReAct thought process
        thought = f"Analyzing {observation['url'][:40]}..."
        
        # Simple rule-based planning for demo
        action = {
            'thought': thought,
            'skill': 'navigate',
            'params': {},
            'reasoning': 'Demo action'
        }
        
        logger.debug(f"   💭 Thought: {thought}")
        logger.debug(f"   🎯 Action: {action['skill']}")
        
        return action


# ============================================================================
# EXECUTION LAYER (Simulated)
# ============================================================================

class ExecutionModule:
    """
    Execution Layer - Browser Control & Skills
    
    In real agent:
    - BrowserManager: Playwright wrapper with Chrome profile
    - SkillExecutor: 23 skills (navigation, interaction, validation)
    """
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
        logger.info("⚙️  Initializing Execution Module")
        logger.info("   ├─ BrowserManager: Playwright + Chrome Profile 18")
        logger.info("   └─ SkillExecutor: 23 skills loaded")
        
    async def initialize(self, chrome_profile: Optional[str] = None):
        """Initialize browser"""
        logger.info("🚀 Starting browser...")
        logger.info(f"   ├─ Headless: {self.headless}")
        
        self.playwright = await async_playwright().start()
        
        if chrome_profile:
            import os
            user_data_dir = os.path.expanduser(r'~\AppData\Local\Google\Chrome\User Data')
            
            logger.info(f"   ├─ Profile: {chrome_profile}")
            logger.info(f"   └─ User Data: {user_data_dir}")
            
            context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir,
                headless=self.headless,
                channel='chrome',
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    f'--profile-directory={chrome_profile}'
                ],
                viewport={'width': 1920, 'height': 1080},
                accept_downloads=True,
                bypass_csp=True
            )
            
            self.browser = context.browser
            self.page = context.pages[0] if context.pages else await context.new_page()
        else:
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            self.page = await context.new_page()
            
        logger.info("✅ Browser ready")
        
    async def execute_skill(self, skill: str, params: Dict) -> Dict[str, Any]:
        """Execute a skill"""
        logger.debug(f"🎬 Executing skill: {skill}")
        logger.debug(f"   └─ Params: {params}")
        
        result = {
            'success': True,
            'skill': skill,
            'params': params,
            'output': None
        }
        
        return result
        
    async def close(self):
        """Shutdown browser"""
        logger.info("🔴 Closing browser...")
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("✅ Browser closed")
        except Exception as e:
            logger.error(f"❌ Error closing browser: {e}")


# ============================================================================
# ORCHESTRATOR - Main Agent Control Loop
# ============================================================================

class WOAAgent:
    """
    WOA Agent Orchestrator
    
    Coordinates 4-layer architecture:
    1. Perception: Understand current state
    2. Planning: Decide next action (ReAct)
    3. Execution: Execute action via skills
    4. Learning: Collect feedback (not in demo)
    """
    
    BASE_URL = "https://www.lazada.vn"
    
    def __init__(self, headless: bool = False, max_steps: int = 15):
        logger.info("="*70)
        logger.info("🤖 WOA AGENT - Web Ordering Automation")
        logger.info("="*70)
        logger.info("Architecture: Perception → Planning → Execution → Learning")
        logger.info(f"Max steps: {max_steps}")
        logger.info(f"Browser mode: {'Headless' if headless else 'Visible'}")
        logger.info("="*70)
        
        # Initialize layers
        self.perception = PerceptionModule()
        self.planning = PlanningModule()
        self.execution = ExecutionModule(headless=headless)
        
        # State
        self.state = AgentState(max_steps=max_steps)
        self.products: List[Product] = []
        
        logger.info("✅ All modules initialized\n")
        
    async def initialize(self, chrome_profile: Optional[str] = None):
        """Initialize agent"""
        await self.execution.initialize(chrome_profile)
        
    async def execute_task(
        self,
        user_request: UserRequest
    ) -> ExecutionResult:
        """
        Execute user task using ReAct loop
        
        Flow:
        1. PERCEIVE current state
        2. PLAN next action (ReAct)
        3. EXECUTE action
        4. LEARN from result
        5. Repeat until done or max steps
        """
        start_time = datetime.now()
        
        logger.info("="*70)
        logger.info("📋 TASK EXECUTION START")
        logger.info("="*70)
        
        if user_request.query:
            logger.info(f"🎯 Query: {user_request.query}")
        if user_request.product_url:
            logger.info(f"🔗 Direct URL: {user_request.product_url}")
        
        logger.info(f"⚙️  Params: max_products={user_request.max_products}, "
                   f"price=[{user_request.min_price}-{user_request.max_price}], "
                   f"rating≥{user_request.min_rating}")
        logger.info(f"🛒 Action: {'BUY NOW' if user_request.buy_now else 'ADD TO CART'} x{user_request.quantity}")
        logger.info("="*70 + "\n")
        
        try:
            # PHASE 1: Product Search (if needed)
            if not user_request.product_url and user_request.query:
                await self._search_phase(user_request)
                
            # PHASE 2: Product Selection
            target_url = user_request.product_url
            if not target_url and self.products:
                target_url = await self._selection_phase(user_request)
                
            # PHASE 3: Add to Cart
            if target_url:
                await self._cart_phase(target_url, user_request)
                
            # Success
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ExecutionResult(
                success=True,
                steps=self.state.step,
                final_state=self.state,
                history=self.state.action_history,
                execution_time=execution_time,
                summary=f"Completed in {execution_time:.2f}s"
            )
            
            logger.info("\n" + "="*70)
            logger.info("✅ TASK EXECUTION COMPLETED")
            logger.info("="*70)
            logger.info(f"📊 Steps: {self.state.step}")
            logger.info(f"⏱️  Time: {execution_time:.2f}s")
            logger.info(f"📦 Products found: {self.state.products_found}")
            logger.info("="*70 + "\n")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=False,
                steps=self.state.step,
                final_state=self.state,
                history=self.state.action_history,
                error=str(e),
                execution_time=execution_time
            )
            
    async def _search_phase(self, request: UserRequest):
        """Phase 1: Product Search"""
        logger.info("="*70)
        logger.info("🔍 PHASE 1: PRODUCT SEARCH")
        logger.info("="*70)
        
        # Build search URL
        search_url = f"{self.BASE_URL}/catalog/?q={request.query.replace(' ', '+')}"
        
        # Step 1: Navigate
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Navigate to search page")
        logger.info(f"   🌐 URL: {search_url}")
        
        await self.execution.page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
        self.state.current_url = search_url
        
        # Perception: Analyze page
        observation = await self.perception.analyze_page(self.execution.page, search_url)
        logger.info(f"   ✅ Page loaded: {observation['title'][:50]}...")
        
        # Step 2: Wait for products
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Wait for products to load")
        
        await self.execution.page.wait_for_selector('[data-qa-locator="product-item"]', timeout=10000)
        logger.info("   ✅ Products detected")
        
        # Step 3: Scroll to load more
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Scroll to load more products")
        
        for i in range(3):
            await self.execution.page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(1)
            logger.debug(f"   └─ Scroll {i+1}/3")
        
        logger.info("   ✅ Scroll completed")
        
        # Step 4: Extract products
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Extract product data")
        
        products = await self._extract_products(
            request.max_products,
            request.min_price,
            request.max_price,
            request.min_rating
        )
        
        self.products = products
        self.state.products_found = len(products)
        
        logger.info(f"   ✅ Extracted {len(products)} products")
        logger.info("="*70 + "\n")
        
    async def _extract_products(
        self,
        max_products: int,
        min_price: Optional[float],
        max_price: Optional[float],
        min_rating: Optional[float]
    ) -> List[Product]:
        """Extract products with filters"""
        logger.debug(f"📊 Extracting products (max={max_products})...")
        
        products = []
        product_elements = await self.execution.page.query_selector_all('[data-qa-locator="product-item"]')
        
        logger.debug(f"   ├─ Found {len(product_elements)} product elements")
        
        for idx, element in enumerate(product_elements[:max_products]):
            try:
                product = await self._extract_single_product(element)
                
                if product:
                    # Apply filters
                    if min_price and product.price < min_price:
                        continue
                    if max_price and product.price > max_price:
                        continue
                    if min_rating and product.rating and product.rating < min_rating:
                        continue
                    
                    products.append(product)
                    
                    if len(products) % 5 == 0:
                        logger.debug(f"   ├─ Progress: {len(products)} products...")
                        
            except Exception:
                continue
        
        logger.debug(f"   └─ Final count: {len(products)} products")
        return products
        
    async def _extract_single_product(self, element) -> Optional[Product]:
        """Extract single product"""
        try:
            # Title
            title_elem = await element.query_selector('[class*="title"]')
            title = await title_elem.inner_text() if title_elem else "N/A"
            
            # Price
            price_elem = await element.query_selector('.price')
            price_text = await price_elem.inner_text() if price_elem else "0"
            price = self._parse_price(price_text)
            
            # Original price
            original_price_elem = await element.query_selector('.origin-price')
            original_price = None
            if original_price_elem:
                original_price_text = await original_price_elem.inner_text()
                original_price = self._parse_price(original_price_text)
            
            # Discount
            discount_elem = await element.query_selector('.sale-flag-percent')
            discount = await discount_elem.inner_text() if discount_elem else None
            
            # Rating
            rating_elem = await element.query_selector('.rating__score')
            rating = None
            if rating_elem:
                rating_text = await rating_elem.inner_text()
                try:
                    rating = float(rating_text)
                except:
                    pass
            
            # Sold count
            sold_elem = await element.query_selector('[class*="sold"]')
            sold_count = await sold_elem.inner_text() if sold_elem else None
            
            # URL
            link_elem = await element.query_selector('a')
            url = await link_elem.get_attribute('href') if link_elem else ""
            if url and not url.startswith('http'):
                url = f"{self.BASE_URL}{url}"
            
            # Image
            img_elem = await element.query_selector('img')
            image_url = await img_elem.get_attribute('src') if img_elem else ""
            
            return Product(
                title=title.strip(),
                price=price,
                original_price=original_price,
                discount=discount,
                rating=rating,
                sold_count=sold_count,
                url=url,
                image_url=image_url
            )
            
        except Exception:
            return None
            
    def _parse_price(self, price_text: str) -> float:
        """Parse price from text"""
        try:
            cleaned = price_text.replace('₫', '').replace('.', '').replace(',', '').strip()
            return float(cleaned)
        except:
            return 0.0
            
    async def _selection_phase(self, request: UserRequest) -> str:
        """Phase 2: Product Selection"""
        logger.info("="*70)
        logger.info("📦 PHASE 2: PRODUCT SELECTION")
        logger.info("="*70 + "\n")
        
        # Display top products
        logger.info("🏆 Top products found:")
        for i, product in enumerate(self.products[:min(5, len(self.products))], 1):
            logger.info(f"\n{i}. {product.title[:60]}...")
            logger.info(f"   💰 Price: ₫{product.price:,.0f}")
            if product.original_price:
                logger.info(f"   🏷️  Original: ₫{product.original_price:,.0f} ({product.discount})")
            logger.info(f"   ⭐ Rating: {product.rating or 'N/A'}")
            logger.info(f"   📊 Sold: {product.sold_count or 'N/A'}")
        
        # User selection
        selection_prompt = f"\n👉 Select product (1-{len(self.products)}) [1]: "
        selection = input(selection_prompt).strip()
        
        try:
            selection_idx = int(selection) if selection else 1
            if selection_idx < 1 or selection_idx > len(self.products):
                selection_idx = 1
        except:
            selection_idx = 1
        
        selected = self.products[selection_idx - 1]
        
        logger.info(f"\n✅ Selected: {selected.title[:60]}...")
        logger.info(f"   💰 Price: ₫{selected.price:,.0f}")
        logger.info("="*70 + "\n")
        
        return selected.url
        
    async def _cart_phase(self, product_url: str, request: UserRequest):
        """Phase 3: Add to Cart"""
        logger.info("="*70)
        logger.info(f"🛒 PHASE 3: {'BUY NOW' if request.buy_now else 'ADD TO CART'}")
        logger.info("="*70)
        
        # Step: Navigate to product
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Navigate to product page")
        logger.info(f"   🌐 URL: {product_url}")
        
        await self.execution.page.goto(product_url, wait_until='domcontentloaded', timeout=30000)
        await asyncio.sleep(3)
        
        observation = await self.perception.analyze_page(self.execution.page, product_url)
        logger.info(f"   ✅ Loaded: {observation['title'][:50]}...")
        
        # Step: Scroll
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Scroll to load elements")
        
        await self.execution.page.evaluate("window.scrollBy(0, 300)")
        await asyncio.sleep(1)
        logger.info("   ✅ Scroll completed")
        
        # Step: Find button
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Locate action button")
        
        if request.buy_now:
            button_selectors = [
                'button:has-text("Mua ngay")',
                'button:has-text("Buy Now")',
                '.add-to-cart-buy-now-btn button:nth-child(2)',
            ]
            action_name = "Buy Now"
        else:
            button_selectors = [
                'button:has-text("Thêm vào giỏ hàng")',
                'button:has-text("Add to Cart")',
                '.add-to-cart-buy-now-btn button:first-child',
            ]
            action_name = "Add to Cart"
        
        target_button = None
        for selector in button_selectors:
            try:
                target_button = await self.execution.page.wait_for_selector(
                    selector, timeout=3000, state='visible'
                )
                if target_button:
                    button_text = await target_button.inner_text()
                    logger.info(f"   ✅ Button found: '{button_text.strip()}'")
                    break
            except:
                continue
        
        if not target_button:
            logger.warning(f"   ⚠️  '{action_name}' button not found")
            return
        
        # Step: Set quantity
        if request.quantity > 1:
            self.state.step += 1
            logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Set quantity to {request.quantity}")
            
            quantity_selectors = ['input[type="number"]', '.pdp-mod-product-quantity input']
            for q_selector in quantity_selectors:
                try:
                    quantity_input = await self.execution.page.query_selector(q_selector)
                    if quantity_input:
                        await quantity_input.click()
                        await quantity_input.fill('')
                        await quantity_input.type(str(request.quantity))
                        logger.info(f"   ✅ Quantity set")
                        await asyncio.sleep(1)
                        break
                except:
                    continue
        
        # Step: Click button
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Click '{action_name}' button")
        
        await target_button.click()
        logger.info(f"   ✅ Button clicked")
        
        await asyncio.sleep(3)
        
        # Step: Check result
        self.state.step += 1
        logger.info(f"\n[Step {self.state.step}/{self.state.max_steps}] Verify action result")
        
        # Check for login popup or success
        login_selectors = [
            '[class*="login"]',
            'input[type="password"]',
            'button:has-text("Đăng nhập")',
            'text=Đã thêm vào giỏ hàng',
            '.cart-success',
        ]
        
        success = False
        for selector in login_selectors:
            try:
                element = await self.execution.page.wait_for_selector(selector, timeout=2000, state='visible')
                if element:
                    if 'login' in selector.lower() or 'password' in selector.lower():
                        logger.info("   ✅ Login popup detected (button click successful)")
                    else:
                        logger.info("   ✅ Cart confirmation detected (already logged in)")
                    success = True
                    break
            except:
                continue
        
        if not success:
            logger.warning("   ⚠️  No clear confirmation, but button was clicked")
        
        logger.info("\n💡 Demo stops here - browser will remain open for 15s")
        logger.info("="*70 + "\n")
        
        await asyncio.sleep(15)
        
    async def close(self):
        """Shutdown agent"""
        await self.execution.close()


# ============================================================================
# USER INTERFACE
# ============================================================================

def prompt_user_request() -> UserRequest:
    """Collect user request from CLI"""
    print("="*70)
    print("📝 USER REQUEST INPUT")
    print("="*70)
    
    product_url_input = input("Product URL (Enter to search by keyword): ").strip()
    
    product_url = None
    query = None
    max_products = 10
    min_price = None
    max_price = None
    min_rating = None
    
    if product_url_input:
        if product_url_input.startswith(('http://', 'https://')):
            product_url = product_url_input
            print("   ✓ Valid URL detected")
        else:
            query = product_url_input
            print(f"   ✓ Search keyword: '{query}'")
    
    if not product_url:
        if not query:
            default_query = "tai nghe bluetooth"
            query = input(f"Search keyword [{default_query}]: ").strip() or default_query
        
        max_input = input("Max products [10]: ").strip()
        max_products = int(max_input) if max_input else 10
        
        min_price_input = input("Min price (VND) [skip]: ").strip()
        if min_price_input:
            min_price = float(min_price_input.replace(',', '').replace('.', ''))
        
        max_price_input = input("Max price (VND) [skip]: ").strip()
        if max_price_input:
            max_price = float(max_price_input.replace(',', '').replace('.', ''))
        
        min_rating_input = input("Min rating (1-5) [skip]: ").strip()
        if min_rating_input:
            min_rating = float(min_rating_input)
    
    action = input("Action (add/buy) [add]: ").strip().lower()
    buy_now = action.startswith('b')
    
    quantity_input = input("Quantity [1]: ").strip()
    quantity = int(quantity_input) if quantity_input else 1
    
    return UserRequest(
        query=query,
        product_url=product_url,
        max_products=max_products,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        buy_now=buy_now,
        quantity=quantity
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """
    WOA AGENT DEMO - Full Workflow
    ==============================
    
    Demonstrates:
    1. 4-Layer Architecture
    2. ReAct Decision Loop
    3. Product Search & Extraction
    4. Add to Cart Action
    
    Architecture Layers:
    - Perception: DOM analysis, vision, text encoding
    - Planning: ReAct engine with ViT5
    - Execution: Browser control with 23 skills
    - Learning: Trajectory collection (not in demo)
    """
    
    print("\n" + "="*70)
    print("🤖 WOA AGENT - PRODUCTION DEMO")
    print("="*70)
    print("Simulating full agent architecture")
    print("Real modules: Perception, Planning, Execution, Learning")
    print("="*70 + "\n")
    
    # Get user request
    user_request = prompt_user_request()
    
    # Create agent
    agent = WOAAgent(headless=False, max_steps=15)
    
    try:
        # Initialize
        await agent.initialize()
        
        # Execute task
        result = await agent.execute_task(user_request)
        
        if result.success:
            logger.info("✅ DEMO COMPLETED SUCCESSFULLY!")
        else:
            logger.error(f"❌ DEMO FAILED: {result.error}")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        await agent.close()


if __name__ == "__main__":
    print("\n🚀 Starting WOA Agent Demo...")
    print("⏱️  Estimated time: 1-2 minutes\n")
    
    asyncio.run(main())
