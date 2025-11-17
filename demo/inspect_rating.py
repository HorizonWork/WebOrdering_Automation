"""
Debug rating extraction - inspect actual Lazada HTML structure
"""
import asyncio
import re
from playwright.async_api import async_playwright

async def inspect_rating_structure():
    """Inspect how Lazada actually renders ratings"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print("🔍 Loading Lazada search page...")
        await page.goto("https://www.lazada.vn/catalog/?q=tai+nghe+bluetooth", wait_until='domcontentloaded')
        
        print("⏳ Waiting for products...")
        await page.wait_for_selector('[data-qa-locator="product-item"]', timeout=15000)
        
        # Get all products
        products = await page.query_selector_all('[data-qa-locator="product-item"]')
        
        print(f"\n📊 Found {len(products)} products\n")
        print("="*80)
        
        for i, product in enumerate(products[:5], 1):  # First 5 products
            print(f"\n🔍 PRODUCT {i}:")
            print("-"*80)
            
            # Get title
            try:
                title_elem = await product.query_selector('a[title]')
                if title_elem:
                    title = await title_elem.get_attribute('title')
                    print(f"Title: {title[:60]}...")
            except:
                pass
            
            # Test card-jfy-ratings (width-based)
            print("\n⭐ Testing '.card-jfy-ratings' (width-based):")
            try:
                rating_div = await product.query_selector('.card-jfy-ratings')
                if rating_div:
                    style = await rating_div.get_attribute('style')
                    print(f"   ✅ FOUND!")
                    print(f"   Style: {style}")
                    
                    if style:
                        match = re.search(r'width:\s*(\d+(?:\.\d+)?)%', style)
                        if match:
                            width = float(match.group(1))
                            rating = round(width / 20, 1)
                            print(f"   📊 Width: {width}% → Rating: {rating} stars")
                else:
                    print("   ❌ NOT FOUND")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Test alternative selectors
            print("\n🔎 Testing alternative rating selectors:")
            
            alt_selectors = [
                '.rating__score',
                '[class*="rating"]',
                '[class*="star"]',
                '.review-score',
                '[data-spm*="rating"]'
            ]
            
            for selector in alt_selectors:
                try:
                    elem = await product.query_selector(selector)
                    if elem:
                        html = await elem.inner_html()
                        text = await elem.inner_text()
                        classes = await elem.get_attribute('class')
                        print(f"   ✅ {selector}:")
                        print(f"      Classes: {classes}")
                        print(f"      Text: {text[:50] if text else '(empty)'}")
                        print(f"      HTML: {html[:100] if html else '(empty)'}")
                except Exception as e:
                    pass
            
            # Get full HTML of product for manual inspection
            print("\n📄 Full product HTML (rating section):")
            try:
                # Look for any div containing rating-related classes
                rating_containers = await product.query_selector_all('[class*="rating"], [class*="Rating"], [class*="star"], [class*="review"]')
                if rating_containers:
                    print(f"   Found {len(rating_containers)} rating-related elements")
                    for j, container in enumerate(rating_containers[:3], 1):
                        html = await container.evaluate('el => el.outerHTML')
                        print(f"\n   Element {j}:")
                        print(f"   {html[:200]}")
                else:
                    print("   ❌ No rating elements found")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print("\n" + "="*80)
        
        print("\n💡 Keeping browser open for 60s for manual inspection...")
        print("   → Open DevTools and inspect product cards manually")
        print("   → Look for rating elements and their structure")
        await asyncio.sleep(60)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(inspect_rating_structure())
