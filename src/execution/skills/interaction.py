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
        try:
            await page.click(selector, timeout=timeout)
            return self._success(f"Clicked {selector}")
        except Exception as e:
            return self._error(f"Click failed: {str(e)}")
    
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
        try:
            await page.fill(selector, text, timeout=timeout)
            return self._success(f"Filled {selector}")
        except Exception as e:
            return self._error(f"Fill failed: {str(e)}")
    
    async def select(self, page, selector: str, value: str, timeout: int = 5000, **kwargs):
        """Select dropdown option"""
        try:
            await page.select_option(selector, value, timeout=timeout)
            return self._success(f"Selected '{value}' in {selector}")
        except Exception as e:
            return self._error(f"Select failed: {str(e)}")
    
    async def hover(self, page, selector: str, timeout: int = 5000, **kwargs):
        """Hover over element"""
        try:
            await page.hover(selector, timeout=timeout)
            return self._success(f"Hovered over {selector}")
        except Exception as e:
            return self._error(f"Hover failed: {str(e)}")
    
    async def press(self, page, key: str, **kwargs):
        """Press keyboard key"""
        try:
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
                try:
                    # Mouse wheel produces consistent incremental scroll instead
                    # of jumping straight to the footer.
                    await page.mouse.wheel(0, delta)
                except Exception:
                    await page.evaluate(
                        "(offset) => window.scrollBy({top: offset, behavior: 'smooth'})",
                        delta,
                    )

            await page.wait_for_timeout(350)
            return self._success(f"Scrolled {direction} by {step}px")
        except Exception as e:
            return self._error(f"Scroll failed: {str(e)}")
