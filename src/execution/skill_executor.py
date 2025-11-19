"""
Skill Executor - Dispatches and executes skills on web pages.

Supports both legacy ReAct-style actions::

    {"skill": "goto", "params": {"url": "https://example.com"}}

and newer high-level DSL actions (Planner/Controller) with types like
NAVIGATE / CLICK / FILL / PRESS / SELECT_DROPDOWN / WAIT_FOR / CLICK_AT.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.execution.skills import (  # noqa: E402
    NavigationSkills,
    InteractionSkills,
    ObservationSkills,
    ValidationSkills,
    WaitSkills,
)
from src.execution.omni_passer import OmniPasser  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

Action = Dict[str, Any]
Result = Dict[str, Any]


class SkillExecutor:
    """
    Executes skills on browser pages.

    **Available low-level skills (legacy)**:
        - Navigation: goto, back, forward, reload, close
        - Interaction: click, type, fill, select, hover, press, scroll
        - Observation: screenshot, get_dom, get_text, get_url, get_title
        - Validation: check_exists, check_visible, check_enabled
        - Wait: wait_for, wait_for_selector, wait_for_navigation, wait (alias)
        - Special: complete, omni_login

    **High-level DSL (new)**:
        - NAVIGATE(url)          → goto(url)
        - CLICK(selector/id)     → click(selector)
        - FILL(selector, text)   → fill(selector, text)
        - PRESS(key)             → press(key)
        - SELECT_DROPDOWN(...)   → select(selector, value)
        - WAIT_FOR(selector|nav) → wait_for_selector / wait_for_navigation
        - CLICK_AT(bbox)         → mouse click at bbox center

    The DSL actions are translated into the low-level skills inside this class.
    """

    def __init__(self) -> None:
        """Initialize skill executor and underlying skill handlers."""
        # Initialize skill handlers
        self.navigation = NavigationSkills()
        self.interaction = InteractionSkills()
        self.observation = ObservationSkills()
        self.validation = ValidationSkills()
        self.wait = WaitSkills()
        self.omnipasser = OmniPasser()

        # Map low-level skill names to handlers
        self.skill_map = {
            # Navigation
            "goto": self.navigation.goto,
            "back": self.navigation.back,
            "forward": self.navigation.forward,
            "reload": self.navigation.reload,
            "close": self.navigation.close,

            # Interaction
            "click": self.interaction.click,
            "type": self.interaction.type,
            "fill": self.interaction.fill,
            "select": self.interaction.select,
            "hover": self.interaction.hover,
            "press": self.interaction.press,
            "scroll": self.interaction.scroll,

            # Observation
            "screenshot": self.observation.screenshot,
            "get_dom": self.observation.get_dom,
            "get_text": self.observation.get_text,
            "get_url": self.observation.get_url,
            "get_title": self.observation.get_title,

            # Validation
            "check_exists": self.validation.check_exists,
            "check_visible": self.validation.check_visible,
            "check_enabled": self.validation.check_enabled,

            # Wait
            "wait_for": self.wait.wait_for,
            "wait_for_selector": self.wait.wait_for_selector,
            "wait_for_navigation": self.wait.wait_for_navigation,
            # Alias used in some tests / logs (simple sleep-based wait)
            "wait": self.wait.wait_for,

            # Special
            "complete": self._complete,
            "omni_login": self.omnipasser.login,
        }

        logger.info("SkillExecutor initialized (%d skills)", len(self.skill_map))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, page, action: Action) -> Result:
        """
        Execute an action on the given Playwright page.

        Args:
            page: Playwright page instance.
            action:
                - Legacy form: ``{"skill": "click", "params": {"selector": "a"}}``
                - DSL form:    ``{"type": "CLICK", "dom_selector": "a", ...}``
                  or wrapped in ``{"chosen_action": {...}}`` from Controller.

        Returns:
            Result dict ``{"status": "...", "message": "...", "data": {...?}}``.
        """
        normalized = self._normalize_action(action)

        skill_name = normalized.get("skill")
        params: Dict[str, Any] = normalized.get("params", {}) or {}

        if not skill_name:
            return {
                "status": "error",
                "message": "No skill specified",
            }

        # Special high-level helper not backed by skills.*
        if skill_name == "click_at":
            return await self._click_at(page, **params)

        # Get skill handler
        skill_handler = self.skill_map.get(skill_name)

        if not skill_handler:
            logger.error("Unknown skill: %s", skill_name)
            return {
                "status": "error",
                "message": f"Unknown skill: {skill_name}",
            }

        # Execute skill
        try:
            logger.debug("Executing skill %s with params=%s", skill_name, params)

            result = await skill_handler(page, **params)

            logger.debug(
                "Skill %s finished with status=%s",
                skill_name,
                result.get("status"),
            )

            return result

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Skill execution failed: %s", exc, exc_info=True)
            return {
                "status": "error",
                "message": str(exc),
            }

    def get_available_skills(self) -> List[str]:
        """Return list of available low-level skill names."""
        return sorted(self.skill_map.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_action(self, action: Action) -> Action:
        """
        Normalize various action formats into the legacy
        ``{"skill": str, "params": dict}`` used internally.

        Supports:
            - Legacy ReAct actions with ``skill`` / ``params``.
            - Controller output with ``chosen_action``.
            - DSL actions with ``type`` field (NAVIGATE, CLICK, ...).
        """
        if not isinstance(action, dict):
            raise TypeError(f"Action must be a dict, got {type(action)!r}")

        # Legacy path: already in {skill, params} format
        if "skill" in action:
            skill = action["skill"]
            params = action.get("params") or {}
            return {"skill": skill, "params": params}

        # Unwrap controller output: {"chosen_action": {...}}
        chosen: Dict[str, Any] = action.get("chosen_action") or action

        # Try to detect DSL type
        action_type = (
            (chosen.get("type") or chosen.get("action_type") or "")
            .strip()
            .upper()
        )
        params_in: Dict[str, Any] = chosen.get("parameters") or {}

        # Common fields
        selector = (
            chosen.get("dom_selector")
            or chosen.get("selector")
            or params_in.get("dom_selector")
            or params_in.get("selector")
        )
        url = chosen.get("url") or params_in.get("url")

        # NAVIGATE → goto
        if action_type == "NAVIGATE":
            return {"skill": "goto", "params": {"url": url}}

        # CLICK → click
        if action_type == "CLICK":
            return {"skill": "click", "params": {"selector": selector}}

        # FILL → fill
        if action_type == "FILL":
            text = (
                chosen.get("text")
                or params_in.get("text")
                or params_in.get("value")
                or ""
            )
            return {
                "skill": "fill",
                "params": {"selector": selector, "text": text},
            }

        # PRESS → press
        if action_type == "PRESS":
            key = chosen.get("key") or params_in.get("key")
            return {"skill": "press", "params": {"key": key}}

        # SELECT_DROPDOWN → select
        if action_type == "SELECT_DROPDOWN":
            value = (
                chosen.get("value")
                or chosen.get("option_value")
                or params_in.get("value")
                or params_in.get("option_value")
            )
            return {
                "skill": "select",
                "params": {"selector": selector, "value": value},
            }

        # WAIT_FOR → wait_for_selector / wait_for_navigation
        if action_type == "WAIT_FOR":
            timeout = chosen.get("timeout") or params_in.get("timeout")
            state = (chosen.get("state") or params_in.get("state") or "visible").lower()
            wait_until = (
                chosen.get("wait_until")
                or params_in.get("wait_until")
                or "networkidle"
            )

            if selector:
                params: Dict[str, Any] = {
                    "selector": selector,
                    "state": state,
                }
                if timeout is not None:
                    params["timeout"] = timeout
                return {"skill": "wait_for_selector", "params": params}

            params = {"wait_until": wait_until}
            if timeout is not None:
                params["timeout"] = timeout
            return {"skill": "wait_for_navigation", "params": params}

        # CLICK_AT → special click_at helper
        if action_type == "CLICK_AT":
            bbox = chosen.get("bbox") or params_in.get("bbox")
            return {"skill": "click_at", "params": {"bbox": bbox}}

        # Fallback: if caller already provided a skill inside chosen, respect it
        if "skill" in chosen:
            skill = chosen["skill"]
            params = chosen.get("params") or params_in
            return {"skill": skill, "params": params or {}}

        # Last resort: return original dict; caller will see "Unknown skill"
        return action

    async def _complete(self, page, **kwargs: Any) -> Result:
        """Mark task as complete (special skill used by ReAct)."""
        message = kwargs.get("message", "Task completed")
        logger.info("Task completed: %s", message)
        return {
            "status": "success",
            "message": message,
            "data": {"completed": True},
        }

    async def _click_at(self, page, bbox: Optional[Dict[str, Any]] = None, **_: Any) -> Result:
        """
        Fallback CLICK_AT implementation using page.mouse.click on bbox center.

        Args:
            page: Playwright page.
            bbox: Dict with at least ``x``, ``y``, and optionally ``width``/``height``.
        """
        if not bbox:
            return {
                "status": "error",
                "message": "click_at requires bbox",
            }

        try:
            x = float(bbox.get("x", 0.0))
            y = float(bbox.get("y", 0.0))
            w = float(bbox.get("width", 0.0))
            h = float(bbox.get("height", 0.0))

            # Use center of bbox if width/height provided
            if w or h:
                x = x + w / 2.0
                y = y + h / 2.0

            await page.mouse.click(x, y)
            logger.info("CLICK_AT at (%.1f, %.1f)", x, y)

            return {
                "status": "success",
                "message": f"Clicked at ({x:.1f}, {y:.1f})",
                "data": {"x": x, "y": y},
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("click_at failed: %s", exc, exc_info=True)
            return {
                "status": "error",
                "message": str(exc),
            }


# ----------------------------------------------------------------------
# Manual test helper
# ----------------------------------------------------------------------

async def _test_skill_executor() -> None:
    """Basic manual test for SkillExecutor (for local debugging)."""
    from src.execution.browser_manager import BrowserManager  # noqa: WPS433
    import asyncio as _asyncio  # noqa: WPS433

    print("=" * 70)
    print("SkillExecutor - Manual Test")
    print("=" * 70 + "\n")

    manager = BrowserManager(headless=False, use_chrome_profile=False)
    executor = SkillExecutor()

    try:
        await manager.launch()
        page = await manager.new_page()

        # Test legacy action
        result = await executor.execute(
            page,
            {"skill": "goto", "params": {"url": "https://example.com"}},
        )
        print("goto:", result)

        # Test DSL-style NAVIGATE
        result = await executor.execute(
            page,
            {"type": "NAVIGATE", "url": "https://www.google.com"},
        )
        print("NAVIGATE:", result)

    finally:
        await manager.close()

    print("\nManual SkillExecutor test finished.")


if __name__ == "__main__":  # pragma: no cover - manual debug
    import asyncio as _asyncio

    _asyncio.run(_test_skill_executor())
