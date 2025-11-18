"""
UI Detector - Detect UI components and patterns
Identifies common UI elements (search boxes, forms, buttons, etc.)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger # noqa: E402

logger = get_logger(__name__)


class UIDetector:
    """
    Detects common UI components and patterns.
    
    **Capabilities**:
        - Detect search boxes
        - Detect login forms
        - Detect navigation menus
        - Detect product listings
        - Detect buttons by function
    
    **Use Cases**:
        - Guide skill selection
        - Identify page type
        - Find interaction points
        - Validate page state
    """
    
    def __init__(self):
        """Initialize UI detector"""
        # UI patterns
        self.patterns = {
            'search_box': {
                'selectors': [
                    'input[type="search"]',
                    'input[name*="search"]',
                    'input[name*="q"]',
                    'input[placeholder*="Tìm"]',
                    'input[placeholder*="Search"]'
                ],
                'keywords': ['search', 'tìm', 'tìm kiếm']
            },
            'login_form': {
                'indicators': [
                    ('input[type="password"]', True),
                    ('input[name*="user"]', False),
                    ('input[name*="email"]', False)
                ],
                'keywords': ['login', 'sign in', 'đăng nhập']
            },
            'product_listing': {
                'selectors': [
                    '.product-item',
                    '[data-product-id]',
                    '.item-card'
                ],
                'min_count': 3
            },
            'navigation': {
                'selectors': [
                    'nav',
                    '.nav',
                    '.menu',
                    '[role="navigation"]'
                ]
            }
        }
        
        logger.info("UIDetector initialized")
    
    def detect_all(self, html: str) -> Dict:
        """
        Detect all UI components.
        
        Args:
            html: HTML string
            
        Returns:
            Detection results
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        results = {
            'search_box': self.detect_search_box(soup),
            'login_form': self.detect_login_form(soup),
            'product_listing': self.detect_product_listing(soup),
            'navigation': self.detect_navigation(soup),
            'page_type': None
        }
        
        # Infer page type
        results['page_type'] = self._infer_page_type(results)
        
        logger.info(f"✓ UI detection complete: {results['page_type']}")
        
        return results
    
    def detect_search_box(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Detect search box"""
        pattern = self.patterns['search_box']
        
        # Try selectors
        for selector in pattern['selectors']:
            elements = soup.select(selector)
            for elem in elements:
                if elem and self._is_visible_heuristic(elem):
                    return {
                        'found': True,
                        'selector': selector,
                        'element': {
                            'tag': elem.name,
                            'id': elem.get('id'),
                            'name': elem.get('name'),
                            'placeholder': elem.get('placeholder')
                        }
                    }
        
        return {'found': False}
    
    def detect_login_form(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Detect login form"""
        # pattern = self.patterns['login_form']
        
        # Must have password field
        password_input = soup.select('input[type="password"]')
        if not password_input:
            return {'found': False}
        
        # Look for username/email field nearby
        form = password_input[0].find_parent('form')
        
        if form:
            # Find username field
            username_fields = form.select('input[type="text"], input[type="email"], input[name*="user"], input[name*="email"]')
            
            if username_fields:
                return {
                    'found': True,
                    'form_id': form.get('id'),
                    'username_field': username_fields[0].get('name'),
                    'password_field': password_input[0].get('name'),
                    'submit_button': bool(form.select('button[type="submit"], input[type="submit"]'))
                }
        
        return {'found': False}
    
    def detect_product_listing(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Detect product listing page"""
        pattern = self.patterns['product_listing']
        
        # Try to find product items
        for selector in pattern['selectors']:
            items = soup.select(selector)
            
            if len(items) >= pattern['min_count']:
                return {
                    'found': True,
                    'selector': selector,
                    'count': len(items),
                    'sample_item': {
                        'text': items[0].get_text(strip=True)[:100]
                    }
                }
        
        return {'found': False}
    
    def detect_navigation(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Detect navigation menu"""
        pattern = self.patterns['navigation']
        
        for selector in pattern['selectors']:
            nav = soup.select_one(selector)
            
            if nav:
                # Count links
                links = nav.select('a')
                
                return {
                    'found': True,
                    'selector': selector,
                    'links_count': len(links),
                    'sample_links': [
                        {'text': link.get_text(strip=True), 'href': link.get('href')}
                        for link in links[:5]
                    ]
                }
        
        return {'found': False}
    
    def _is_visible_heuristic(self, element) -> bool:
        """Heuristic to check if element is visible"""
        # Check style attribute
        style = element.get('style', '')
        if 'display:none' in style or 'display: none' in style:
            return False
        
        if 'visibility:hidden' in style or 'visibility: hidden' in style:
            return False
        
        # Check common hidden classes
        classes = element.get('class', [])
        hidden_keywords = ['hidden', 'hide', 'd-none']
        
        for cls in classes:
            if any(kw in str(cls).lower() for kw in hidden_keywords):
                return False
        
        return True
    
    def _infer_page_type(self, detection_results: Dict) -> str:
        """Infer page type from detection results"""
        if detection_results['login_form']['found']:
            return 'login_page'
        elif detection_results['product_listing']['found']:
            return 'product_listing'
        elif detection_results['search_box']['found']:
            return 'search_page'
        else:
            return 'generic'
    
    def find_action_buttons(self, html: str, action_type: str = 'submit') -> List[Dict]:
        """
        Find buttons by action type.
        
        Args:
            html: HTML string
            action_type: Button type (submit, add_to_cart, checkout, etc.)
            
        Returns:
            List of button elements
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Action keywords
        keywords = {
            'submit': ['submit', 'gửi', 'send', 'search', 'tìm'],
            'add_to_cart': ['add to cart', 'thêm vào giỏ', 'mua ngay'],
            'checkout': ['checkout', 'thanh toán', 'đặt hàng'],
            'login': ['login', 'sign in', 'đăng nhập']
        }
        
        action_keywords = keywords.get(action_type, [])
        
        # Find buttons
        buttons = []
        for button in soup.select('button, input[type="submit"], input[type="button"], a.button'):
            text = button.get_text(strip=True).lower()
            
            # Check text matches
            if any(kw in text for kw in action_keywords):
                buttons.append({
                    'tag': button.name,
                    'text': button.get_text(strip=True),
                    'id': button.get('id'),
                    'class': button.get('class'),
                    'type': button.get('type')
                })
        
        logger.info(f"✓ Found {len(buttons)} '{action_type}' buttons")
        return buttons


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("UIDetector - Test")
    print("=" * 70 + "\n")
    
    # Test HTML
    test_html = """
    <html>
    <body>
        <nav class="menu">
            <a href="/">Home</a>
            <a href="/products">Products</a>
            <a href="/about">About</a>
        </nav>
        
        <div class="search-bar">
            <input type="search" name="q" placeholder="Tìm kiếm sản phẩm" />
            <button type="submit">Tìm</button>
        </div>
        
        <div class="products">
            <div class="product-item">Product 1</div>
            <div class="product-item">Product 2</div>
            <div class="product-item">Product 3</div>
            <div class="product-item">Product 4</div>
        </div>
        
        <form id="login-form">
            <input type="text" name="username" placeholder="Username" />
            <input type="password" name="password" placeholder="Password" />
            <button type="submit">Đăng nhập</button>
        </form>
    </body>
    </html>
    """
    
    detector = UIDetector()
    
    # Test 1: Detect all
    print("Test 1: Detect All UI Components")
    print("-" * 40)
    
    results = detector.detect_all(test_html)
    
    print(f"Page type: {results['page_type']}")
    print(f"Search box: {results['search_box']['found']}")
    print(f"Login form: {results['login_form']['found']}")
    print(f"Product listing: {results['product_listing']['found']} ({results['product_listing'].get('count', 0)} items)")
    print(f"Navigation: {results['navigation']['found']}\n")
    
    # Test 2: Find action buttons
    print("\nTest 2: Find Action Buttons")
    print("-" * 40)
    
    submit_buttons = detector.find_action_buttons(test_html, 'submit')
    print(f"Submit buttons: {len(submit_buttons)}")
    for btn in submit_buttons:
        print(f"  - {btn['text']} ({btn['tag']})")
    
    login_buttons = detector.find_action_buttons(test_html, 'login')
    print(f"\nLogin buttons: {len(login_buttons)}")
    for btn in login_buttons:
        print(f"  - {btn['text']} ({btn['tag']})")
    
    print("\n" + "=" * 70)
    print("✅ All Tests Completed!")
    print("=" * 70)
