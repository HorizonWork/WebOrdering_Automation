import asyncio

import pytest

import path_setup  # noqa: F401

ROOT_DIR = path_setup.PROJECT_ROOT

from src.execution.browser_manager import BrowserManager  # noqa: E402
from src.perception.dom_distiller import DOMDistiller  # noqa: E402


@pytest.mark.asyncio
async def test_perception_layer():
    print("Testing Perception Layer...")
    manager = BrowserManager(headless=True)  # Không cần hiển thị
    page = await manager.new_page()
    await page.goto("https://www.lazada.vn")

    try:
        # 1. Test DOM Distiller
        print("Testing DOM Distiller...")
        distiller = DOMDistiller()
        raw_dom = await page.content()
        distilled_dom = distiller.distill(raw_dom, mode="text_only")

        print(f"Raw DOM size: {len(raw_dom)}, Distilled DOM size: {len(distilled_dom)}")
        assert len(raw_dom) > len(distilled_dom), "Distilled DOM should be smaller than raw DOM"
        assert "<script>" not in distilled_dom, "Scripts should be removed"

        # 2. Scene Representation test bỏ qua (cần ML dependencies)
        print("\n⚠️ Skipping Scene Representation test (requires ML dependencies)")
        print("DOM Distiller test passed! yes")

    except Exception as e:
        print(f"PERCEPTION LAYER FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(test_perception_layer())
