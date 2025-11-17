"""
Inspect Lazada product page structure to find correct selectors
"""
import asyncio
from playwright.async_api import async_playwright

async def inspect_lazada():
    """Inspect Lazada search results page"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print("🔍 Loading Lazada search page...")
        await page.goto("https://www.lazada.vn/catalog/?q=tai+nghe+bluetooth", wait_until='domcontentloaded')
        
        print("⏳ Waiting for products...")
        await page.wait_for_selector('[data-qa-locator="product-item"]', timeout=15000)
        
        print("\n📦 Extracting first product structure...\n")
        
        # Get first product
        product = await page.query_selector('[data-qa-locator="product-item"]')
        
        if product:
            # Get all text content
            html = await product.inner_html()
            
            print("="*70)
            print("HTML STRUCTURE (first 2000 chars):")
            print("="*70)
            print(html[:2000])
            print("...")
            print("="*70)
            
            # Try different selectors
            print("\n🧪 Testing selectors:\n")
            
            # Title
            title_selectors = [
                '[class*="title"]',
                '.title',
                '[title]',
                'a[title]',
                '[data-qa-locator="product-item"] a'
            ]
            
            for selector in title_selectors:
                try:
                    elem = await product.query_selector(selector)
                    if elem:
                        text = await elem.inner_text()
                        if text.strip():
                            print(f"✅ Title selector: {selector}")
                            print(f"   Text: {text.strip()[:60]}...")
                            break
                except:
                    pass
            
            # Price
            price_selectors = [
                '.price',
                '[class*="price"]',
                '[class*="Price"]',
                'span[class*="currency"]',
                '.currency'
            ]
            
            for selector in price_selectors:
                try:
                    elem = await product.query_selector(selector)
                    if elem:
                        text = await elem.inner_text()
                        if '₫' in text or text.strip().isdigit():
                            print(f"✅ Price selector: {selector}")
                            print(f"   Text: {text.strip()}")
                            break
                except:
                    pass
            
            # Rating
            rating_selectors = [
                '.rating__score',
                '[class*="rating"]',
                '[class*="Rating"]',
                'span[class*="star"]'
            ]
            
            for selector in rating_selectors:
                try:
                    elem = await product.query_selector(selector)
                    if elem:
                        text = await elem.inner_text()
                        print(f"✅ Rating selector: {selector}")
                        print(f"   Text: {text.strip()}")
                        break
                except:
                    pass
            
            # Sold count
            sold_selectors = [
                '[class*="sold"]',
                '[class*="Sold"]',
                'span:has-text("đã bán")',
                'div:has-text("đã bán")'
            ]
            
            for selector in sold_selectors:
                try:
                    elem = await product.query_selector(selector)
                    if elem:
                        text = await elem.inner_text()
                        print(f"✅ Sold selector: {selector}")
                        print(f"   Text: {text.strip()}")
                        break
                except:
                    pass
            
            print("\n💡 Keeping browser open for 30s for manual inspection...")
            await asyncio.sleep(30)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(inspect_lazada())
