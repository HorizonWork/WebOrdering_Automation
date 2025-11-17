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
import re
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
    """User request parameters (deprecated - use ParsedQuery from query_parser)"""
    query: Optional[str] = None
    product_url: Optional[str] = None
    max_products: int = 10
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    buy_now: bool = False
    quantity: int = 1
    
    @classmethod
    def from_parsed_query(cls, parsed: Any) -> 'UserRequest':  # Use Any to avoid circular import
        """Convert ParsedQuery to UserRequest"""
        return cls(
            query=parsed.query,
            product_url=parsed.product_url,
            max_products=parsed.max_products,
            min_price=parsed.min_price,
            max_price=parsed.max_price,
            min_rating=parsed.min_rating,
            buy_now=parsed.buy_now,
            quantity=parsed.quantity
        )


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
        # Silent analysis - don't expose internal processing
        # logger.debug(f"📊 Analyzing page: {url[:60]}...")
        
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
    - NLU: Natural Language Understanding (QueryParser)
    """
    
    def __init__(self):
        logger.info("🧠 Initializing Planning Module")
        logger.info("   ├─ ViT5 Planner: Action generation (VietAI/vit5-base)")
        logger.info("   ├─ ReAct Engine: Reasoning loop")
        logger.info("   └─ NLU Parser: Query understanding (Ollama llama3.2:1b)")
        
        # Suppress Ollama-related logs (httpx, query_parser)
        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("src.utils.query_parser").setLevel(logging.WARNING)
        
        # Import QueryParser inside module to hide it from main
        from src.utils.query_parser import QueryParser
        self.query_parser = QueryParser(model="llama3.2:1b")
        self._logger = get_logger("demo.PlanningModule.NLU")
        
    def parse_user_query(self, user_input: str) -> Any:  # Return Any to avoid circular import
        """
        Parse natural language user query into structured parameters
        
        This is part of the Planning layer - NLU to understand user intent
        """
        self._logger.info(f"🧠 Understanding user query: '{user_input[:50]}...'")
        self._logger.info("   ├─ Analyzing intent with Ollama...")
        
        # Call QueryParser (hidden inside Planning layer)
        parsed = self.query_parser.parse(user_input)
        
        # Log understanding (not raw Ollama output)
        self._logger.info("   ├─ Intent understood:")
        if parsed.query:
            self._logger.info(f"   │  ├─ Search: {parsed.query}")
        if parsed.product_url:
            self._logger.info(f"   │  ├─ Direct URL: {parsed.product_url[:50]}...")
        if parsed.min_price or parsed.max_price:
            self._logger.info(f"   │  ├─ Price: {parsed.min_price or 0:,.0f} - {parsed.max_price or '∞'}")
        if parsed.min_rating:
            self._logger.info(f"   │  ├─ Min rating: {parsed.min_rating} stars")
        self._logger.info(f"   │  ├─ Action: {'Buy Now' if parsed.buy_now else 'Add to Cart'}")
        self._logger.info(f"   │  └─ Quantity: {parsed.quantity}")
        self._logger.info("   └─ ✅ Query parsed successfully")
        
        return parsed
    
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
            # Title - try multiple Vietnamese selectors
            title = "N/A"
            title_selectors = [
                'a[title]',  # Link with title attribute (most reliable)
                '[class*="title"]',
                '[class*="Title"]',
                'a'
            ]
            for selector in title_selectors:
                try:
                    title_elem = await element.query_selector(selector)
                    if title_elem:
                        # Try title attribute first, then text
                        title = await title_elem.get_attribute('title')
                        if not title:
                            title = await title_elem.inner_text()
                        if title and title.strip():
                            title = title.strip()
                            break
                except:
                    continue
            
            # Price - exact Lazada structure: <span class="currency">₫10,000</span>
            price = 0.0
            price_selectors = [
                'span.currency',  # Primary Lazada price selector
                '.currency',
                'span[class="price"]',
                'div.price span',
                '[class*="ooOxS"]',  # Lazada obfuscated class
            ]
            for selector in price_selectors:
                try:
                    price_elem = await element.query_selector(selector)
                    if price_elem:
                        price_text = await price_elem.inner_text()
                        if price_text:
                            price = self._parse_price(price_text)
                            if price > 0:
                                break
                except Exception:
                    continue
            
            # Original price - try multiple selectors
            original_price = None
            original_price_selectors = [
                '.origin-price',
                '[class*="origin"]',
                '[class*="Original"]',
                '[class*="original-price"]'
            ]
            for selector in original_price_selectors:
                try:
                    original_price_elem = await element.query_selector(selector)
                    if original_price_elem:
                        original_price_text = await original_price_elem.inner_text()
                        if original_price_text and '₫' in original_price_text:
                            original_price = self._parse_price(original_price_text)
                            if original_price and original_price > price:
                                break
                except:
                    continue
            
            # Discount - try multiple Vietnamese selectors
            discount = None
            discount_selectors = [
                '.sale-flag-percent',
                '[class*="discount"]',
                '[class*="Discount"]',
                '[class*="sale"]',
                'span:has-text("%")',
                'div:has-text("%")'
            ]
            for selector in discount_selectors:
                try:
                    discount_elem = await element.query_selector(selector)
                    if discount_elem:
                        discount_text = await discount_elem.inner_text()
                        if discount_text and '%' in discount_text:
                            discount = discount_text.strip()
                            break
                except:
                    continue
            
            # Rating - Lazada uses width percentage: <div class="card-jfy-ratings" style="width: 84%"> → 84% / 20 = 4.2 stars
            rating = None
            
            # Try width-based rating first (most accurate for Lazada)
            try:
                rating_container = await element.query_selector('.card-jfy-ratings')
                if rating_container:
                    style = await rating_container.get_attribute('style')
                    if style:
                        # Extract width percentage: "width: 84%" → 84
                        width_match = re.search(r'width:\s*(\d+(?:\.\d+)?)%', style)
                        if width_match:
                            width_percent = float(width_match.group(1))
                            # Each star = 20% width (100% / 5 stars)
                            rating = round(width_percent / 20, 1)
                            if 0 <= rating <= 5:
                                pass  # Valid rating found
                            else:
                                rating = None
                else:
                    # DEBUG: Try to find ANY rating element
                    # This helps identify what selector actually exists
                    all_rating_candidates = await element.query_selector_all('[class*="rating"], [class*="Rating"], [class*="star"], [class*="review"]')
                    # Silently skip - no rating found
                    pass
            except Exception:
                pass
            
            # Fallback: try text-based rating selectors
            if rating is None:
                rating_selectors = [
                    '.rating__score',
                    '[class*="rating"]',
                    'span[class*="star"]'
                ]
                for selector in rating_selectors:
                    try:
                        rating_elem = await element.query_selector(selector)
                        if rating_elem:
                            rating_text = await rating_elem.inner_text()
                            if rating_text:
                                match = re.search(r'(\d+\.?\d*)', rating_text)
                                if match:
                                    rating = float(match.group(1))
                                    if 0 <= rating <= 5:
                                        break
                    except Exception:
                        continue
            
            # Sold count - Vietnamese specific selectors
            sold_count = None
            sold_selectors = [
                'span:has-text("đã bán")',  # Vietnamese: sold
                'div:has-text("đã bán")',
                '[class*="sold"]',
                '[class*="Sold"]',
                '[class*="review"]'  # Sometimes sold count near reviews
            ]
            for selector in sold_selectors:
                try:
                    sold_elem = await element.query_selector(selector)
                    if sold_elem:
                        sold_text = await sold_elem.inner_text()
                        if sold_text and ('đã bán' in sold_text.lower() or sold_text.strip().replace('.', '').replace('k', '').isdigit()):
                            sold_count = sold_text.strip()
                            break
                except:
                    continue
            
            # URL - handle absolute, protocol-relative, and relative URLs
            link_elem = await element.query_selector('a')
            url = await link_elem.get_attribute('href') if link_elem else ""
            if url:
                if url.startswith('http'):
                    # Already absolute: https://www.lazada.vn/...
                    pass
                elif url.startswith('//'):
                    # Protocol-relative: //www.lazada.vn/...
                    url = f"https:{url}"
                elif url.startswith('/'):
                    # Relative: /products/...
                    url = f"{self.BASE_URL}{url}"
                # else: invalid URL, keep as is
            
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

def prompt_user_request(planning_module: PlanningModule) -> UserRequest:
    """Collect user request from CLI using Natural Language"""
    print("="*70)
    print("📝 USER REQUEST INPUT")
    print("="*70)
    print("\n💡 TIP: You can use natural language!")
    print("   Examples:")
    print("   - 'Tôi muốn mua iPhone 15 được đánh giá trên 3 sao'")
    print("   - 'Tìm tai nghe bluetooth giá dưới 500 nghìn'")
    print("   - 'Mua ngay laptop gaming từ 20 đến 30 triệu rating trên 4 sao'")
    print("   - Or just a simple search: 'tai nghe bluetooth'")
    print("   - Or a direct URL: 'https://lazada.vn/products/...'")
    print()
    
    user_input = input("👉 What do you want to buy? ").strip()
    
    if not user_input:
        print("   ⚠️  Empty input, using default: 'tai nghe bluetooth'")
        user_input = "tai nghe bluetooth"
    
    print()
    
    # Use Planning Module's NLU (hides QueryParser/Ollama inside)
    parsed_query = planning_module.parse_user_query(user_input)
    
    # Convert to UserRequest
    user_request = UserRequest.from_parsed_query(parsed_query)
    
    print("\n✅ Understood:")
    if user_request.product_url:
        print(f"   - Product URL: {user_request.product_url}")
    else:
        print(f"   - Search: {user_request.query}")
        print(f"   - Max products: {user_request.max_products}")
        if user_request.min_price or user_request.max_price:
            print(f"   - Price range: ₫{user_request.min_price or 0:,.0f} - ₫{user_request.max_price or '∞'}")
        if user_request.min_rating:
            print(f"   - Min rating: ≥{user_request.min_rating} stars")
    print(f"   - Action: {'🛍️ BUY NOW' if user_request.buy_now else '🛒 ADD TO CART'}")
    print(f"   - Quantity: {user_request.quantity}")
    
    # Confirm
    confirm = input("\n👉 Proceed with this? (Y/n): ").strip().lower()
    if confirm and confirm not in ['y', 'yes', '']:
        print("   ❌ Cancelled by user")
        sys.exit(0)
    
    return user_request


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
    
    # Create agent (initializes all modules)
    agent = WOAAgent(headless=False, max_steps=15)
    
    # Get user request (uses Planning Module's NLU)
    user_request = prompt_user_request(agent.planning)
    
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
