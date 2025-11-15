"""
OmniPasser - heuristic login automation helper.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from playwright.async_api import Page

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class OmniPasser:
    """Heuristic login helper that fills email/password fields."""

    EMAIL_SELECTORS: List[str] = [
        "input[type='email']",
        "input[name='email']",
        "input[name='loginKey']",
        "#login-email",
        "input[data-login='email']",
    ]
    PASSWORD_SELECTORS: List[str] = [
        "input[type='password']",
        "input[name='password']",
        "#login-password",
        "input[data-login='password']",
    ]
    SUBMIT_SELECTORS: List[str] = [
        "button[type='submit']",
        "button[name='login']",
        "button[class*='login']",
        "#login-button",
    ]
    LOGGED_IN_MARKERS: List[str] = [
        "[class*='account']",
        "[data-testid='user-info']",
        ".navbar__username",
    ]

    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        self.email = email or settings.omnipasser_email
        self.password = password or settings.omnipasser_password

    async def login(self, page: Page, **_: Dict) -> Dict:
        if not self.email or not self.password:
            return {
                "status": "error",
                "message": "OmniPasser credentials missing (set OMNIPASSER_EMAIL/PASSWORD)",
            }
        try:
            if await self._is_logged_in(page):
                return {"status": "success", "message": "Already logged in"}

            email_selector = await self._find_visible(page, self.EMAIL_SELECTORS)
            password_selector = await self._find_visible(page, self.PASSWORD_SELECTORS)
            submit_selector = await self._find_visible(page, self.SUBMIT_SELECTORS)

            if not email_selector or not password_selector:
                return {"status": "error", "message": "Login inputs not found"}

            await page.fill(email_selector, self.email)
            await page.fill(password_selector, self.password)

            if submit_selector:
                await page.click(submit_selector)

            await page.wait_for_timeout(2000)
            if await self._is_logged_in(page):
                return {"status": "success", "message": "Login successful"}
            return {"status": "error", "message": "Login attempt did not complete"}
        except Exception as exc:  # pragma: no cover
            logger.error(f"OmniPasser login failed: {exc}", exc_info=True)
            return {"status": "error", "message": str(exc)}

    async def _find_visible(self, page: Page, selectors: List[str]) -> Optional[str]:
        for selector in selectors:
            try:
                locator = page.locator(selector)
                if await locator.count() > 0 and await locator.first.is_visible():
                    return selector
            except Exception:
                continue
        return None

    async def _is_logged_in(self, page: Page) -> bool:
        for selector in self.LOGGED_IN_MARKERS:
            try:
                locator = page.locator(selector)
                if await locator.count() > 0 and await locator.first.is_visible():
                    return True
            except Exception:
                continue
        return False
