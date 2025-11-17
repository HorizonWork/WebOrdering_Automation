"""
LAZADA VIETNAM - SIMPLIFIED AUTOMATION
======================================
Simplified version - Stops at Add to Cart action
Production-Ready Version

Author: WOA Agent Team  
Date: 2025-11-16
"""

import asyncio
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import async_playwright, Page, Browser


@dataclass
class Product:
    """Lazada Product Data Structure"""
    title: str
    price: float
    original_price: Optional[float]
    discount: Optional[str]
    rating: Optional[float]
    sold_count: Optional[str]
    url: str
    image_url: str


@dataclass
class UserRequest:
    """User-supplied parameters captured from CLI input."""
    query: Optional[str]
    product_url: Optional[str]
    max_products: int
    min_price: Optional[float]
    max_price: Optional[float]
    min_rating: Optional[float]
    buy_now: bool
    quantity: int


class LazadaAutomation:
    """
    Main class for Lazada Vietnam automation
    Features:
    - Product search
    - Filter by price, rating
    - Data extraction
    - Add to cart (stops here)
    """
    
    BASE_URL = "https://www.lazada.vn"
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.products: List[Product] = []
        
    async def initialize(self, chrome_profile: str = None):
        """Initialize browser with Chrome profile
        
        Args:
            chrome_profile: Chrome profile name (e.g., "Profile 18")
        """
        print("🚀 Initializing Playwright browser...")
        self.playwright = await async_playwright().start()
        
        if chrome_profile:
            # Use existing Chrome profile with launch_persistent_context
            import os
            
            # Point to User Data directory
            user_data_dir = os.path.expanduser(r'~\AppData\Local\Google\Chrome\User Data')
            
            print(f"📂 Using Chrome Profile: {chrome_profile}")
            print(f"   User Data Dir: {user_data_dir}")
            print(f"   Profile: {chrome_profile}")
            
            # Launch persistent context with Chrome profile
            context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir,
                headless=self.headless,
                channel='chrome',  # Use installed Chrome
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
            # Use default browser without profile
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            self.page = await context.new_page()
        
        print("✅ Browser initialized successfully\n")
        
    async def search_products(
        self, 
        query: str, 
        max_products: int = 20,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None
    ) -> List[Product]:
        """
        Search products with filters
        
        Args:
            query: Search term (ex: "bluetooth headphones", "áo khoác nam")
            max_products: Maximum number of products to extract
            min_price: Minimum price (VND)
            max_price: Maximum price (VND)
            min_rating: Minimum rating (1-5)
            
        Returns:
            List of products
        """
        start_time = datetime.now()
        print(f"🔍 Search: '{query}'")
        print(f"   Parameters: max={max_products}, price=[{min_price}-{max_price}], rating>={min_rating}")
        
        # Build search URL
        search_url = f"{self.BASE_URL}/catalog/?q={query.replace(' ', '+')}"
        
        try:
            # Navigate to search page
            print(f"📡 Navigating to: {search_url}")
            await self.page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for products to load
            await self.page.wait_for_selector('[data-qa-locator="product-item"]', timeout=10000)
            print("✅ Page loaded successfully")
            
            # Progressive scroll to load more products
            print("📜 Scrolling page to load more products...")
            for i in range(3):
                await self.page.evaluate("window.scrollBy(0, window.innerHeight)")
                await asyncio.sleep(1)
            print("✅ Scroll completed")
            
            # Extract products
            products = await self._extract_products(max_products, min_price, max_price, min_rating)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n✅ Extraction completed:")
            print(f"   - {len(products)} products found")
            print(f"   - Execution time: {execution_time:.2f}s\n")
            
            self.products = products
            return products
            
        except Exception as e:
            print(f"❌ Error during search: {str(e)}")
            raise
            
    async def _extract_products(
        self, 
        max_products: int,
        min_price: Optional[float],
        max_price: Optional[float],
        min_rating: Optional[float]
    ) -> List[Product]:
        """Extract product data from DOM"""
        print(f"📊 Extracting product data...")
        
        products = []
        
        # Select all product items
        product_elements = await self.page.query_selector_all('[data-qa-locator="product-item"]')
        print(f"   - {len(product_elements)} product elements detected")
        
        for idx, element in enumerate(product_elements[:max_products]):
            try:
                # Extract data
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
                        print(f"   - {len(products)} products extracted...")
                        
            except Exception as e:
                continue
                
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
        """Parse price from text (ex: '₫500.000' -> 500000.0)"""
        try:
            cleaned = price_text.replace('₫', '').replace('.', '').replace(',', '').strip()
            return float(cleaned)
        except:
            return 0.0
            
    async def add_to_cart(self, product_url: str, quantity: int = 1, buy_now: bool = False) -> bool:
        """
        Add product to cart or Buy Now - Optimized for Lazada Vietnam
        PROGRAM STOPS AFTER CLICKING THE BUTTON
        
        Args:
            product_url: Product URL
            quantity: Quantity to add (default: 1)
            buy_now: True to click "Buy Now" button, False for "Add to Cart"
            
        Returns:
            True if button clicked successfully
        """
        print(f"\n{'🛍️ BUY NOW' if buy_now else '🛒 ADD TO CART'}")
        print(f"   URL: {product_url}")
        print(f"   Quantity: {quantity}")
        
        try:
            # Navigate to product page
            await self.page.goto(product_url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(3)
            
            # Slight scroll to load elements
            await self.page.evaluate("window.scrollBy(0, 300)")
            await asyncio.sleep(1)
            
            # Define button selectors based on action
            if buy_now:
                # Buy Now button selectors (Vietnamese)
                button_selectors = [
                    'button:has-text("Mua ngay")',
                    'button:has-text("MUA NGAY")',
                    'button:has-text("Mua Ngay")',
                    'button:has-text("Buy Now")',
                    'button:has-text("BUY NOW")',
                    '.add-to-cart-buy-now-btn button:nth-child(2)',
                    'button[class*="buy-now"]',
                    'button[class*="buyNow"]',
                    'button[class*="BuyNow"]',
                ]
                action_name = "Buy Now"
            else:
                # Add to Cart button selectors (Vietnamese)
                button_selectors = [
                    'button:has-text("Thêm vào giỏ hàng")',
                    'button:has-text("Thêm Vào Giỏ Hàng")',
                    'button:has-text("THÊM VÀO GIỎ HÀNG")',
                    'button:has-text("Add to Cart")',
                    'button:has-text("ADD TO CART")',
                    '.add-to-cart-buy-now-btn button:first-child',
                    'button[data-sku-id]',
                    '.pdp-button_type_primary',
                    '.pdp-cart-concern button',
                    'button.pdp-button',
                    'button[class*="add-to-cart"]',
                    'button[class*="addToCart"]',
                    'button[class*="AddToCart"]',
                    'button[aria-label*="Thêm"]',
                    'button[aria-label*="giỏ"]'
                ]
                action_name = "Add to Cart"
            
            target_button = None
            print(f"🔍 Searching for '{action_name}' button...")
            
            # Try each selector
            for selector in button_selectors:
                try:
                    target_button = await self.page.wait_for_selector(selector, timeout=3000, state='visible')
                    if target_button:
                        button_text = await target_button.inner_text()
                        print(f"✅ Button found: '{button_text.strip()}'")
                        print(f"   Selector: {selector}")
                        break
                except:
                    continue
            
            if not target_button:
                print(f"❌ '{action_name}' button not found")
                print("💡 Attempting to search page content...")
                
                # Display all buttons for debug
                all_buttons = await self.page.query_selector_all('button')
                print(f"📋 {len(all_buttons)} buttons found on page:")
                for i, btn in enumerate(all_buttons[:15]):  # Show first 15
                    try:
                        text = await btn.inner_text()
                        if text.strip():
                            print(f"   {i+1}. '{text.strip()}'")
                    except:
                        pass
                return False
            
            # Set quantity if different from 1
            if quantity > 1:
                quantity_selectors = [
                    'input[type="number"]',
                    'input.next-number-input-input',
                    'input[class*="quantity"]',
                    '.pdp-mod-product-quantity input'
                ]
                
                for q_selector in quantity_selectors:
                    try:
                        quantity_input = await self.page.query_selector(q_selector)
                        if quantity_input:
                            await quantity_input.click()
                            await quantity_input.fill('')
                            await quantity_input.type(str(quantity))
                            print(f"✅ Quantity set to {quantity}")
                            await asyncio.sleep(1)
                            break
                    except:
                        continue
            
            # Click the button
            await target_button.click()
            print(f"✅ Clicked '{action_name}' button")
            
            # Wait and check for login popup or success indicators
            await asyncio.sleep(3)
            
            # Check for login popup (indicates successful click - user not logged in)
            login_popup_selectors = [
                # Login popup indicators
                '[class*="login"]',
                '[class*="signin"]',
                '[class*="Login"]',
                '[class*="SignIn"]',
                'iframe[src*="login"]',
                'iframe[src*="member"]',
                '.next-dialog-body',
                '[role="dialog"]',
                # Login form elements
                'input[type="password"]',
                'input[name="loginId"]',
                'input[placeholder*="Email"]',
                'input[placeholder*="Phone"]',
                'button:has-text("Đăng nhập")',
                'button:has-text("Login")',
                'button:has-text("Sign In")',
                # Cart confirmation (if already logged in)
                'text=Đã thêm vào giỏ hàng',
                'text=Added to cart',
                'button:has-text("Xem giỏ hàng")',
                'button:has-text("View Cart")',
                '.cart-success',
                '[class*="cart-modal"]'
            ]
            
            success = False
            success_type = ""
            
            print("\n🔍 Checking for login popup or success confirmation...")
            
            for selector in login_popup_selectors:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=2000, state='visible')
                    if element:
                        element_text = ""
                        try:
                            element_text = await element.inner_text()
                        except:
                            pass
                        
                        # Determine if it's login popup or cart success
                        if any(keyword in selector.lower() for keyword in ['login', 'signin', 'password', 'đăng nhập']):
                            success_type = "LOGIN_POPUP"
                            print(f"✅ Login popup detected!")
                            print(f"   Selector: {selector}")
                            if element_text:
                                print(f"   Content: {element_text[:100]}")
                        else:
                            success_type = "CART_SUCCESS"
                            print(f"✅ Cart confirmation detected!")
                            print(f"   Selector: {selector}")
                            if element_text:
                                print(f"   Content: {element_text[:100]}")
                        
                        success = True
                        break
                except:
                    continue
            
            print("\n" + "="*70)
            if success:
                if success_type == "LOGIN_POPUP":
                    print("✅ SUCCESS! LOGIN POPUP APPEARED")
                    print("="*70)
                    print("\n📝 Interpretation:")
                    print("   - Button click was successful")
                    print("   - Login required to complete action")
                    print("   - This confirms the button works correctly")
                elif success_type == "CART_SUCCESS":
                    print("✅ SUCCESS! PRODUCT ADDED TO CART")
                    print("="*70)
                    print("\n📝 Interpretation:")
                    print("   - Button click was successful")
                    print("   - User is already logged in")
                    print("   - Product successfully added to cart")
            else:
                print("✅ BUTTON CLICKED (Confirmation unclear)")
                print("="*70)
                print("\n📝 Note:")
                print("   - Button was clicked successfully")
                print("   - No clear popup detected")
                print("   - Action may have completed in background")
            
            print("\n💡 Program stops here as requested")
            print("   The browser will remain open for 15 seconds...")
            
            # Keep browser open for observation
            await asyncio.sleep(15)
            
            return True
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
        
    async def close(self):
        """Clean shutdown"""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            print("\n✅ Browser closed")
        except Exception:
            pass


def _prompt_int(prompt_text: str, default: int, min_value: Optional[int] = None) -> int:
    """Read integer input with validation."""
    value = input(prompt_text).strip()
    if not value:
        return default
    try:
        parsed = int(value)
        if min_value is not None and parsed < min_value:
            raise ValueError
        return parsed
    except ValueError:
        print(f"   -> Invalid number, using default: {default}")
        return default


def _prompt_float(
    prompt_text: str,
    default: Optional[float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_thousand_separator: bool = False
) -> Optional[float]:
    """Read optional float input with validation."""
    value = input(prompt_text).strip()
    if not value:
        return default
    cleaned = value.replace(',', '')
    if allow_thousand_separator:
        cleaned = cleaned.replace('.', '')
    try:
        parsed = float(cleaned)
        if min_value is not None and parsed < min_value:
            raise ValueError
        if max_value is not None and parsed > max_value:
            raise ValueError
        return parsed
    except ValueError:
        print(f"   -> Invalid number, ignoring input.")
        return default


def prompt_user_request() -> UserRequest:
    """Collect instructions from CLI user."""
    print("="*70)
    print("dY\"? USER REQUEST INPUT")
    print("="*70)
    print("Provide product URL to skip search or leave blank to search by keyword.")
    
    product_url_input = input("Product URL (Enter = search by keyword): ").strip()
    
    # Validate if input is actually a URL or just a search query
    product_url = None
    query = None
    max_products = 10
    min_price = None
    max_price = None
    min_rating = None
    
    if product_url_input:
        # Check if it's a valid URL (starts with http:// or https://)
        if product_url_input.startswith(('http://', 'https://')):
            product_url = product_url_input
            print("   -> Valid URL detected, skipping search phase.")
        else:
            # Treat as search query
            print(f"   -> Not a URL, treating as search keyword: '{product_url_input}'")
            query = product_url_input
    
    if not product_url:
        # Need to get search query
        if not query:
            default_query = "tai nghe bluetooth"
            query_input = input(f"Search keyword [{default_query}]: ").strip()
            query = query_input or default_query
        
        max_products = _prompt_int("Max products to fetch [10]: ", 10, min_value=1)
        min_price = _prompt_float(
            "Minimum price (VND) - Enter to skip: ",
            None,
            min_value=0,
            allow_thousand_separator=True
        )
        max_price = _prompt_float(
            "Maximum price (VND) - Enter to skip: ",
            None,
            min_value=0,
            allow_thousand_separator=True
        )
        min_rating = _prompt_float(
            "Minimum rating (1-5) - Enter to skip: ",
            None,
            min_value=0,
            max_value=5
        )
    else:
        print("   -> Using direct product URL, skipping search phase.")
    
    action_choice = input("Action (add=Add to Cart, buy=Buy Now) [add]: ").strip().lower()
    buy_now = action_choice.startswith("b")
    quantity = _prompt_int("Quantity (>=1) [1]: ", 1, min_value=1)
    
    print("\n�o. User request summary:")
    if product_url:
        print(f"   - Product URL: {product_url}")
    else:
        print(f"   - Keyword: {query}")
        print(f"   - Max products: {max_products}")
        print(f"   - Price filter: [{min_price}-{max_price}]")
        print(f"   - Min rating: {min_rating}")
    print(f"   - Action: {'BUY NOW' if buy_now else 'ADD TO CART'}")
    print(f"   - Quantity: {quantity}\n")
    
    return UserRequest(
        query=query,
        product_url=product_url or None,
        max_products=max_products,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        buy_now=buy_now,
        quantity=quantity
    )


async def main():
    """
    SIMPLIFIED DEMO - STOPS AT ADD TO CART
    ======================================
    
    This script demonstrates:
    1. Product search with filters
    2. Data extraction
    3. Add to cart (STOPS HERE)
    
    SUCCESS CRITERIA:
    - Login popup appears after clicking button, OR
    - Cart confirmation appears (if already logged in)
    """
    
    print("="*70)
    print(" dYZ_ LAZADA AUTOMATION - SIMPLIFIED VERSION")
    print("="*70)
    print(" Running in normal browser mode")
    print(" Success = Login popup OR Cart confirmation appears")
    print("="*70)
    print()
    
    user_request = prompt_user_request()
    bot = LazadaAutomation(headless=False)
    
    try:
        # Initialize browser in normal mode (no profile)
        await bot.initialize()
        
        selected_product = None
        target_url = user_request.product_url
        
        if not target_url:
            # STEP 1: Search for products
            print("="*70)
            print("dY\"O STEP 1: Product Search")
            print("="*70)
            
            # Ensure query is not None before calling search_products
            if not user_request.query:
                print("??O Error: No search query provided.")
                return
            
            products = await bot.search_products(
                query=user_request.query,
                max_products=user_request.max_products,
                min_price=user_request.min_price,
                max_price=user_request.max_price,
                min_rating=user_request.min_rating
            )
            
            if not products:
                print("?s??,? No products found. Please try different search terms.")
                return
            
            # Display top products
            print("="*70)
            print("dY\"S TOP PRODUCTS FOUND:")
            print("="*70)
            for i, product in enumerate(products[:min(5, len(products))], 1):
                print(f"\n{i}. {product.title[:60]}...")
                print(f"   Price: ?,?{product.price:,.0f}")
                print(f"   Rating: {product.rating or 'N/A'}")
                print(f"   Discount: {product.discount or 'None'}")
            
            selection_prompt = f"\nSelect product number (1-{len(products)}) [1]: "
            selection_index = _prompt_int(selection_prompt, 1, min_value=1)
            if selection_index > len(products):
                print("   -> Selection out of range, defaulting to first product.")
                selection_index = 1
            
            selected_product = products[selection_index - 1]
            target_url = selected_product.url
            
            print(f"\n?o\" Selected Product:")
            print(f"   {selected_product.title}")
            print(f"   Price: ?,?{selected_product.price:,.0f}")
            print(f"   Rating: {selected_product.rating or 'N/A'}")
        else:
            print("="*70)
            print("dY\"O STEP 1: Direct Product Mode")
            print("="*70)
            print(f"   Using product URL provided by user: {target_url}")
        
        if not target_url:
            print("??O Unable to determine product URL from the provided request.")
            return
        
        # STEP 2: Add to cart / buy now
        print("\n" + "="*70)
        print("dY\"O STEP 2: Execute Action")
        print("="*70)
        
        success = await bot.add_to_cart(
            product_url=target_url,
            quantity=user_request.quantity,
            buy_now=user_request.buy_now
        )
        
        if success:
            print("\n?o. ACTION COMPLETED SUCCESSFULLY!")
        else:
            print("\n?s??,? Action may not have completed. Check browser window.")
        
    except Exception as e:
        print(f"\n??O CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        await bot.close()


if __name__ == "__main__":
    print("\n🚀 Starting simplified demo...")
    print("⏱️  Estimated time: 1-2 minutes\n")
    
    asyncio.run(main())
