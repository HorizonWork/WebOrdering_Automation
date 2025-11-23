
import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1])) # scripts/ -> root
# Actually, debug_dom_extraction.py is in scripts/ so parents[1] is root
# Wait, it is in scripts/debug_dom_extraction.py? No, user put it in scripts/
# Let's check where I put it.
# I put it in d:\FA25\DSP\WebOrdering_Automation\scripts\debug_dom_extraction.py
# So parents[0] is scripts, parents[1] is WebOrdering_Automation (root)
# The error said No module named 'src'.
# If root is in path, 'import src' should work.
# Let's try adding absolute path explicitly to be safe.
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.perception.dom_distiller import DOMDistiller

async def debug_dom():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        url = "https://shopee.vn"
        print(f"Navigating to {url}...")
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        
        html = await page.content()
        print(f"Got HTML: {len(html)} chars")
        
        distiller = DOMDistiller()
        elements = distiller.extract_interactive_elements(html)
        
        print(f"Extracted {len(elements)} elements")
        
        if not elements:
            print("WARNING: No elements extracted!")
            # Debug noise removal
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            print(f"Soup title: {soup.title}")
            print(f"Soup body children: {len(list(soup.body.children)) if soup.body else 'No body'}")
        else:
            print("Top 5 elements:")
            for el in elements[:5]:
                print(f"- {el['tag']} {el['text'][:20]} ({el['selector']})")
                
        await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_dom())
