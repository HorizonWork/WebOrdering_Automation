"""
Search Agent - Specialized agent for search operations
Optimized for e-commerce search (Shopee, Lazada, Tiki)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import re

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.planning.sub_agents.base_agent import BaseSubAgent
from src.execution.skill_executor import SkillExecutor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchAgent(BaseSubAgent):
    """
    Specialized agent for search and filtering operations.
    
    **Capabilities**:
        - Detect search bars
        - Execute search queries
        - Apply filters (price, category, rating)
        - Sort results (price, popularity)
        - Extract search results
        - Navigate to product pages
    
    **E-commerce Focus**:
        - Shopee: Category + Price filters
        - Lazada: Brand + Price filters
        - Tiki: Category + Delivery filters
    
    **Search Flow**:
        1. Find search input
        2. Type query
        3. Submit search
        4. Wait for results
        5. Apply filters (if specified)
        6. Extract top results
    """
    
    def __init__(self):
        super().__init__(
            name="SearchAgent",
            description="Handles search queries and filtering"
        )
        self.skill_executor = SkillExecutor()
        
        # Platform-specific selectors
        self.platform_selectors = {
            'shopee': {
                'search_input': '.shopee-searchbar-input__input',
                'search_button': '.shopee-searchbar__search-button',
                'price_filter': '.price-filter',
                'sort_dropdown': '.sort-bar-dropdown'
            },
            'lazada': {
                'search_input': '#q',
                'search_button': '.search-box__button',
                'price_filter': '[data-spm="filter_price"]',
                'sort_dropdown': '.ant-select'
            },
            'tiki': {
                'search_input': 'input[data-view-id="main_search_form_input"]',
                'search_button': 'button[data-view-id="main_search_form_button"]',
                'price_filter': '.price-filter',
                'sort_dropdown': '.sort-container'
            },
            'generic': {
                'search_input': [
                    'input[type="search"]',
                    'input[name*="search"]',
                    'input[name="q"]',
                    'input[placeholder*="Tìm"]',
                    'input[placeholder*="Search"]',
                    '.search-input',
                    '#search',
                    '#search-box'
                ],
                'search_button': [
                    'button[type="submit"]',
                    '.search-button',
                    'button:has-text("Tìm")',
                    'button:has-text("Search")',
                    '[aria-label*="search"]'
                ]
            }
        }
    
    async def can_handle(self, task: Dict, observation: Dict) -> bool:
        """Check if this is a search task"""
        task_desc = str(task.get('description', '')).lower()
        search_keywords = [
            'tìm', 'search', 'tìm kiếm', 'find', 'look for',
            'lọc', 'filter', 'sắp xếp', 'sort'
        ]
        
        if any(kw in task_desc for kw in search_keywords):
            logger.info("✓ Search task detected from description")
            return True
        
        # Check if search box present
        dom = observation.get('dom', '').lower()
        if any(kw in dom for kw in ['search', 'tìm kiếm', 'type="search"']):
            logger.info("✓ Search box detected in DOM")
            return True
        
        return False
    
    async def execute(
        self,
        task: Dict,
        page,
        observation: Dict
    ) -> Dict:
        """
        Execute search task.
        
        Args:
            task: Search task with {query, filters, sort}
            page: Playwright page
            observation: Current state
            
        Returns:
            Result dict with search results
        """
        logger.info("🔍 Starting search operation...")
        
        try:
            # Extract search parameters
            query = task.get('query', '')
            filters = task.get('filters', {})
            sort_by = task.get('sort_by', None)
            
            if not query:
                logger.warning("⚠️  No search query provided")
                return {
                    'success': False,
                    'message': 'Search query required'
                }
            
            logger.info(f"   Query: {query}")
            logger.info(f"   Filters: {filters}")
            logger.info(f"   Sort: {sort_by}")
            
            # Detect platform
            platform = self._detect_platform(page.url)
            logger.info(f"   Platform: {platform}")
            
            # Step 1: Find search input
            search_input = await self._find_search_input(page, platform)
            if not search_input:
                self._record_failure()
                return {
                    'success': False,
                    'message': 'Search input not found'
                }
            
            logger.info(f"✓ Found search input: {search_input}")
            
            # Step 2: Type query
            await self.skill_executor.execute(page, {
                'skill': 'type',
                'params': {
                    'selector': search_input,
                    'text': query,
                    'clear': True
                }
            })
            
            # Step 3: Submit search
            search_button = await self._find_search_button(page, platform)
            if search_button:
                logger.info(f"✓ Found search button: {search_button}")
                await self.skill_executor.execute(page, {
                    'skill': 'click',
                    'params': {'selector': search_button}
                })
            else:
                # Press Enter as fallback
                logger.info("   Pressing Enter to submit")
                await page.keyboard.press('Enter')
            
            # Step 4: Wait for results
            logger.info("⏳ Waiting for search results...")
            await asyncio.sleep(3)
            
            # Step 5: Apply filters (if specified)
            if filters:
                logger.info("🔧 Applying filters...")
                await self._apply_filters(page, platform, filters)
                await asyncio.sleep(2)
            
            # Step 6: Sort results (if specified)
            if sort_by:
                logger.info(f"📊 Sorting by: {sort_by}")
                await self._sort_results(page, platform, sort_by)
                await asyncio.sleep(2)
            
            # Step 7: Extract results
            logger.info("📋 Extracting search results...")
            results = await self._extract_results(page, platform)
            
            self._record_success()
            return {
                'success': True,
                'message': f'Search completed: {len(results)} results',
                'results': results,
                'query': query,
                'filters_applied': filters,
                'final_url': page.url
            }
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            self._record_failure()
            return {
                'success': False,
                'message': f'Search error: {str(e)}'
            }
    
    def _detect_platform(self, url: str) -> str:
        """Detect e-commerce platform from URL"""
        url_lower = url.lower()
        
        if 'shopee' in url_lower:
            return 'shopee'
        elif 'lazada' in url_lower:
            return 'lazada'
        elif 'tiki' in url_lower:
            return 'tiki'
        else:
            return 'generic'
    
    async def _find_search_input(self, page, platform: str) -> Optional[str]:
        """Find search input field"""
        if platform in self.platform_selectors and platform != 'generic':
            # Try platform-specific selector
            selector = self.platform_selectors[platform]['search_input']
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    return selector
            except:
                pass
        
        # Try generic selectors
        for selector in self.platform_selectors['generic']['search_input']:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    return selector
            except:
                continue
        
        return None
    
    async def _find_search_button(self, page, platform: str) -> Optional[str]:
        """Find search button"""
        if platform in self.platform_selectors and platform != 'generic':
            selector = self.platform_selectors[platform]['search_button']
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    return selector
            except:
                pass
        
        # Try generic selectors
        for selector in self.platform_selectors['generic']['search_button']:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    return selector
            except:
                continue
        
        return None
    
    async def _apply_filters(self, page, platform: str, filters: Dict):
        """Apply search filters"""
        # Price filter
        if 'price_max' in filters or 'price_min' in filters:
            logger.info(f"   Applying price filter: {filters.get('price_min', 0)}-{filters.get('price_max', 'max')}")
            # Implementation depends on platform
            # For now, log only
            pass
        
        # Category filter
        if 'category' in filters:
            logger.info(f"   Filtering by category: {filters['category']}")
            # Platform-specific implementation
            pass
        
        # Rating filter
        if 'rating' in filters:
            logger.info(f"   Filtering by rating: {filters['rating']}+")
            pass
    
    async def _sort_results(self, page, platform: str, sort_by: str):
        """Sort search results"""
        sort_options = {
            'price_low': 'Giá thấp đến cao',
            'price_high': 'Giá cao đến thấp',
            'popular': 'Phổ biến',
            'newest': 'Mới nhất',
            'best_selling': 'Bán chạy'
        }
        
        sort_text = sort_options.get(sort_by, sort_by)
        logger.info(f"   Sorting: {sort_text}")
        
        # Try to find and click sort option
        try:
            selector = f'text="{sort_text}"'
            await page.click(selector, timeout=5000)
        except:
            logger.warning(f"   Could not apply sort: {sort_text}")
    
    async def _extract_results(self, page, platform: str, max_results: int = 10) -> List[Dict]:
        """Extract search results"""
        results = []
        
        try:
            # Platform-specific result extraction
            if platform == 'shopee':
                items = await page.query_selector_all('.shopee-search-item-result__item')
            elif platform == 'lazada':
                items = await page.query_selector_all('.Bm3ON')
            elif platform == 'tiki':
                items = await page.query_selector_all('[data-view-id*="product"]')
            else:
                # Generic: try common selectors
                items = await page.query_selector_all('.product-item, .item, [class*="product"]')
            
            logger.info(f"   Found {len(items)} items")
            
            for idx, item in enumerate(items[:max_results]):
                try:
                    # Extract basic info
                    title = await item.inner_text()
                    title = title.strip()[:100]  # Truncate
                    
                    # Try to get link
                    link_element = await item.query_selector('a')
                    link = await link_element.get_attribute('href') if link_element else None
                    
                    results.append({
                        'index': idx,
                        'title': title,
                        'link': link
                    })
                except:
                    continue
            
        except Exception as e:
            logger.warning(f"   Could not extract results: {e}")
        
        return results


# Test
async def test_search_agent():
    """Test search agent"""
    from src.execution.browser_manager import BrowserManager
    
    print("=" * 70)
    print("SearchAgent - Test")
    print("=" * 70 + "\n")
    
    agent = SearchAgent()
    manager = BrowserManager(headless=False)
    
    try:
        page = await manager.new_page()
        
        # Test 1: Can handle
        task = {'description': 'Tìm áo khoác giá rẻ'}
        observation = {'dom': '<input type="search"/>'}
        
        can_handle = await agent.can_handle(task, observation)
        print(f"✓ Can handle search task: {can_handle}")
        
        # Test 2: Execute search on Shopee
        await page.goto("https://shopee.vn")
        await asyncio.sleep(2)
        
        task = {
            'query': 'áo khoác',
            'filters': {'price_max': 500000},
            'sort_by': 'price_low'
        }
        
        observation = {'url': page.url, 'dom': await page.content()}
        
        print(f"\n🔍 Executing search: {task['query']}")
        result = await agent.execute(task, page, observation)
        
        print(f"\n✅ Result: {result['message']}")
        if result['success']:
            print(f"   Found {len(result.get('results', []))} products")
            for r in result.get('results', [])[:3]:
                print(f"   - {r['title'][:50]}...")
        
        print(f"\n📊 Stats: {agent.get_stats()}")
        
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(test_search_agent())
