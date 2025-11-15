"""Test DOM Distiller - không cần ML models"""

import asyncio
from pathlib import Path
import sys

# Add project root
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

# Import trực tiếp, tránh __init__.py
from src.execution.browser_manager import BrowserManager
from src.perception.dom_distiller import DOMDistiller


async def test_dom_distiller():
    print("=" * 60)
    print("TEST 1: DOM Distiller (No ML needed)")
    print("=" * 60)

    manager = BrowserManager(headless=True)
    page = await manager.new_page()

    try:
        # Navigate to Lazada
        print("\n📍 Navigating to Lazada...")
        await page.goto("https://www.lazada.vn", timeout=30000)
        print(f"✅ Page loaded: {page.url}")

        # Get raw DOM
        raw_dom = await page.content()
        print(f"📄 Raw DOM size: {len(raw_dom):,} bytes")

        # Distill DOM
        print("\n🧹 Distilling DOM...")
        distiller = DOMDistiller()

        # Test different modes
        for mode in ["text_only", "interactive_only"]:
            distilled = distiller.distill(raw_dom, mode=mode)
            reduction = (1 - len(distilled) / len(raw_dom)) * 100
            print(
                f"  - Mode '{mode}': {len(distilled):,} bytes ({reduction:.1f}% reduction)"
            )

            # Verify cleanup
            assert "<script>" not in distilled, "Scripts should be removed"
            assert "<style>" not in distilled, "Styles should be removed"

        print("\n✅ DOM Distiller test PASSED!")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(test_dom_distiller())
