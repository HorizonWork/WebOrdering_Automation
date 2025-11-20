"""
Lightweight CLI user interaction helper.

Used to confirm critical actions (login, payment, checkout, registration)
instead of silently skipping or failing.
"""

from __future__ import annotations

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserInteraction:
    """Simple synchronous CLI prompter."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def confirm(self, question: str, default: bool = False) -> bool:
        """Ask a yes/no question on CLI. Returns default if disabled."""
        if not self.enabled:
            return default
        suffix = " [Y/n]: " if default else " [y/N]: "
        try:
            ans = input(question + suffix).strip().lower()
        except EOFError:
            return default
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        return default

    def request_text(self, prompt: str) -> Optional[str]:
        """Ask user to provide a short text (e.g., email/OTP)."""
        if not self.enabled:
            return None
        try:
            ans = input(prompt + " (leave empty to skip): ").strip()
        except EOFError:
            return None
        return ans or None
