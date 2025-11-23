"""
Interaction Skills - User interaction operations
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.skills.base_skill import BaseSkill


class InteractionSkills(BaseSkill):
    """Interaction skill implementations"""
    
    def __init__(self):
        super().__init__("Interaction")
    
    async def click(self, page, selector: str, timeout: int = 5000, **kwargs):
        """Click element"""
        selectors = self._collect_selectors(selector, kwargs.get("fallback_selectors"))
        for idx, sel in enumerate(selectors):
            try:
                await page.click(sel, timeout=timeout)
                return self._success(f"Clicked {sel}")
            except Exception as e:
                if idx == len(selectors) - 1:
                    return self._error(f"Click failed ({sel}): {str(e)}")
                continue
    
    async def type(self, page, selector: str, text: str, clear: bool = False, timeout: int = 5000, **kwargs):
        """Type text"""
        try:
            if clear:
                await page.fill(selector, "")
            await page.type(selector, text, timeout=timeout)
            return self._success(f"Typed '{text}' into {selector}")
        except Exception as e:
            return self._error(f"Type failed: {str(e)}")
    
    async def fill(self, page, selector: str, text: str, timeout: int = 5000, **kwargs):
        """Fill input (faster than type)"""
        selectors = self._collect_selectors(selector, kwargs.get("fallback_selectors"))
        for idx, sel in enumerate(selectors):
            try:
                await page.fill(sel, text, timeout=timeout)
                return self._success(f"Filled {sel}")
            except Exception as e:
                if idx == len(selectors) - 1:
                    return self._error(f"Fill failed ({sel}): {str(e)}")
                continue
    
    async def select(self, page, selector: str, value: str, timeout: int = 5000, **kwargs):
        """Select dropdown option"""
        selectors = self._collect_selectors(selector, kwargs.get("fallback_selectors"))
        for idx, sel in enumerate(selectors):
            try:
                await page.select_option(sel, value, timeout=timeout)
                return self._success(f"Selected '{value}' in {sel}")
            except Exception as e:
                if idx == len(selectors) - 1:
                    return self._error(f"Select failed ({sel}): {str(e)}")
                continue
    
    async def hover(self, page, selector: str, timeout: int = 5000, **kwargs):
        """Hover over element"""
        try:
            await page.hover(selector, timeout=timeout)
            return self._success(f"Hovered over {selector}")
        except Exception as e:
            return self._error(f"Hover failed: {str(e)}")
    
    async def press(self, page, key: str = "Enter", **kwargs):
        """Press keyboard key"""
        try:
            if not key:
                key = "Enter"
            await page.keyboard.press(key)
            return self._success(f"Pressed key '{key}'")
        except Exception as e:
            return self._error(f"Press failed: {str(e)}")

    async def scroll(
        self,
        page,
        direction: str = "down",
        amount: Optional[int] = None,
        **kwargs,
    ):
        """Scroll page by roughly one viewport instead of jumping to full bottom/top."""
        try:
            direction = (direction or "down").lower()
            viewport_height = await page.evaluate("() => window.innerHeight") or 900
            step = int(amount) if amount else int(viewport_height * 0.75)
            step = max(100, step)

            if direction == "top":
                await page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
            elif direction == "bottom":
                await page.evaluate("() => window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
            else:
                delta = step if direction == "down" else -step
                await page.evaluate(
                    "(delta) => window.scrollBy({top: delta, behavior: 'smooth'})",
                    delta,
                )

            await page.wait_for_timeout(350)
            return self._success(f"Scrolled {direction} by {step}px")
        except Exception as e:
            return self._error(f"Scroll failed: {str(e)}")

    def _collect_selectors(self, primary: str, fallbacks):
        """Combine primary and fallback selectors into a list."""
        selectors = [primary] if primary else []
        if fallbacks and isinstance(fallbacks, (list, tuple)):
            selectors.extend([f for f in fallbacks if f])
        return selectors or [primary]
