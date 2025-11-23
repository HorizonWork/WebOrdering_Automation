# -*- coding: utf-8 -*-
"""
Interactive Manual Data Collection

Goal: let a human perform actions in the browser, then capture
clean, structured steps (planner/controller/action) plus page state,
DOM, and screenshots for training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Page, async_playwright

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ManualStep:
    step: int
    timestamp: str
    planner: Dict[str, Any]
    controller: Dict[str, Any]
    action: Dict[str, Any]
    result: Dict[str, Any]
    page_state: Dict[str, Any]
    observation: Dict[str, Any]
    assets: Dict[str, str] = field(default_factory=dict)
    teacher_labels: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManualEpisode:
    episode_id: str
    goal: str
    start_url: str
    created_at: str
    steps: List[ManualStep]
    final_url: str
    success: bool
    total_steps: int
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class InteractiveCollector:
    """
    Interactive manual collector: user acts, script records structured steps.
    """

    def __init__(self) -> None:
        self.steps: List[ManualStep] = []
        self.episode_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.page: Optional[Page] = None

        self.manual_root = Path("data/manual")
        self.episode_dir = self.manual_root / "episodes"
        self.screenshot_dir = self.manual_root / "screenshots" / self.episode_id
        self.dom_dir = self.manual_root / "dom" / self.episode_id
        for d in (self.episode_dir, self.screenshot_dir, self.dom_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.max_dom_chars = 15000

        # Action types (aligned with DSL used elsewhere)
        self.action_types = [
            "CLICK (bam chuot)",
            "FILL (nhap lieu)",
            "PRESS_KEY (nhan phim)",
            "SCROLL (cuon trang)",
            "WAIT (cho doi)",
            "NAVIGATE (di toi URL)",
            "SELECT_DROPDOWN (chon gia tri)",
            "WAIT_FOR (doi selector/nav)",
            "CLICK_AT (toa do)",
            "COMPLETE (hoan thanh)",
            "OTHER (tuy chon)",
        ]

        # Plan types (rough high-level intents)
        self.plan_types = [
            "SEARCH_PRODUCT",
            "APPLY_FILTER",
            "SELECT_PRODUCT",
            "ADD_TO_CART",
            "CHECKOUT",
            "OTHER",
        ]

    async def collect(self, goal: str, start_url: str) -> None:
        """Main collection loop."""
        print("\n" + "=" * 70)
        print("THU THAP DU LIEU THU CONG - BAN CHUAN")
        print("=" * 70)
        print(f"\nMuc tieu: {goal}")
        print(f"Trang web: {start_url}")
        print("\n" + "=" * 70 + "\n")

        browser = None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False)
                context = await browser.new_context()
                self.page = await context.new_page()

                await self.page.goto(start_url, wait_until="domcontentloaded")
                print(f"Browser da mo: {start_url}\n")

                step_num = 1
                try:
                    while True:
                        input(
                            f"\n[Buoc {step_num}] Hoan thanh thao tac tren trinh duyet, "
                            "xong bam ENTER (Ctrl+C de dung)..."
                        )

                        observation, screenshot_bytes, full_dom = await self._capture_state()
                        assets = await self._save_assets(step_num, screenshot_bytes, full_dom)
                        page_state = self._build_page_state(observation, assets)

                        step_data = self._prompt_step_input(step_num, observation, page_state, assets)
                        self.steps.append(step_data)
                        print(f"Da luu buoc {step_num}")

                        if step_data.action.get("type") == "COMPLETE":
                            print("\nTask da danh dau hoan thanh.")
                            break

                        step_num += 1

                except KeyboardInterrupt:
                    print("\nDa dung thu thap theo yeu cau.")

                finally:
                    await self._save_episode(goal, start_url)
                    try:
                        if browser and browser.is_connected():
                            await browser.close()
                    except Exception:
                        pass

        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error("Loi he thong trong InteractiveCollector: %s", exc)
            import traceback

            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Capture utilities
    # ------------------------------------------------------------------ #

    async def _capture_state(self) -> Tuple[Dict[str, Any], bytes, str]:
        """Capture page observation + raw assets."""
        try:
            assert self.page is not None
            url = self.page.url
            title = await self.page.title()
            full_dom = await self.page.content()
            dom_excerpt = full_dom[: self.max_dom_chars]
            elements = await self._extract_elements()
            screenshot_bytes = await self.page.screenshot(full_page=True)

            observation: Dict[str, Any] = {
                "url": url,
                "title": title,
                "timestamp": datetime.now().isoformat(),
                "dom": dom_excerpt,
                "elements": elements,
                "interactive_elements": elements,
            }
            return observation, screenshot_bytes, full_dom
        except Exception as exc:
            logger.warning("Capture failed: %s", exc)
            fallback_obs = {"url": self.page.url if self.page else "", "error": "capture_failed"}
            return fallback_obs, b"", ""

    async def _extract_elements(self) -> List[Dict[str, Any]]:
        """Extract a lightweight list of interactive elements."""
        assert self.page is not None
        try:
            return await self.page.evaluate(
                """() => {
                    const elems = Array.from(
                        document.querySelectorAll('a, button, input, select, textarea, [role="button"]')
                    );
                    return elems.slice(0, 80).map((el, idx) => ({
                        id: idx,
                        tag: el.tagName.toLowerCase(),
                        text: (el.innerText || el.value || '').slice(0, 80).trim(),
                        type: el.type || '',
                        placeholder: el.placeholder || '',
                        href: el.href || '',
                        visible: el.offsetParent !== null
                    }));
                }"""
            )
        except Exception as exc:
            logger.warning("Element extraction failed: %s", exc)
            return []

    def _build_page_state(self, observation: Dict[str, Any], assets: Dict[str, str]) -> Dict[str, Any]:
        """
        Derive a page_state snapshot aligned with ObservationState schema used elsewhere.
        """
        elements = observation.get("interactive_elements") or observation.get("elements") or []
        dom_text = observation.get("dom") or ""
        url_lower = (observation.get("url", "") or "").lower()
        screenshot_id = assets.get("screenshot")

        if "cart" in url_lower:
            page_type = "cart"
        elif "checkout" in url_lower or "payment" in url_lower:
            page_type = "checkout"
        elif "login" in url_lower:
            page_type = "login"
        else:
            page_type = "unknown"

        return {
            "url": observation.get("url", ""),
            "page_type": page_type,
            "dom_state": {
                "title": observation.get("title", ""),
                "products": [],
                "filters": [],
            },
            "vision_state": {
                "screenshot_id": screenshot_id,
                "elements": elements,
            },
            "actions": [],
        }

    async def _save_assets(self, step_num: int, screenshot_bytes: bytes, dom_html: str) -> Dict[str, str]:
        """Persist screenshot and DOM to disk; return relative paths."""
        assets: Dict[str, str] = {}

        if screenshot_bytes:
            try:
                file_name = f"step_{step_num}.png"
                file_path = self.screenshot_dir / file_name
                file_path.write_bytes(screenshot_bytes)
                assets["screenshot"] = f"{self.episode_id}/{file_name}"
            except Exception as exc:
                logger.warning("Failed to save screenshot for step %s: %s", step_num, exc)

        if dom_html:
            try:
                file_name = f"step_{step_num}.html"
                file_path = self.dom_dir / file_name
                file_path.write_text(dom_html, encoding="utf-8")
                assets["dom"] = f"{self.episode_id}/{file_name}"
            except Exception as exc:
                logger.warning("Failed to save DOM for step %s: %s", step_num, exc)

        return assets

    # ------------------------------------------------------------------ #
    # Prompt logic
    # ------------------------------------------------------------------ #

    def _prompt_step_input(
        self,
        step_num: int,
        observation: Dict[str, Any],
        page_state: Dict[str, Any],
        assets: Dict[str, str],
    ) -> ManualStep:
        """Prompt user for plan/action details and build ManualStep."""
        print("\n" + "-" * 60)
        print(f"CHI TIET BUOC {step_num}")
        print(f"   URL: {observation.get('url', 'unknown')}")
        print("-" * 60)

        # Plan selection
        print("\nLoai hanh dong (Plan Step):")
        for i, ptype in enumerate(self.plan_types, 1):
            print(f"  {i}. {ptype}")
        plan_idx = self._get_number_input("Chon", 1, len(self.plan_types)) - 1
        plan_type = self.plan_types[plan_idx].strip().upper()
        if plan_type == "OTHER":
            other = input("   Nhap ten plan step (vd: LOGIN, CLOSE_POPUP): ").strip().upper()
            plan_type = other or "OTHER"
        plan_description = input("   Mo ta ngan cho plan step (enter de bo qua): ").strip()
        if not plan_description:
            plan_description = f"todo: describe plan step {plan_type.lower()}"

        # Action selection
        print("\nThao tac cu the (Action):")
        for i, atype in enumerate(self.action_types, 1):
            print(f"  {i}. {atype}")
        action_idx = self._get_number_input("Chon", 1, len(self.action_types)) - 1
        action_type = self.action_types[action_idx].split(" (")[0]
        if action_type == "OTHER":
            self._print_action_cheatsheet()
            custom = input("   Nhap action type (vd: CLICK, WAIT_FOR, SELECT_DROPDOWN, BACK, RELOAD): ").strip().upper()
            action_type = custom or "OTHER"

        action_params: Dict[str, Any] = {}
        description = ""
        action_id = f"{action_type.lower()}_step{step_num}"
        chosen_elem: Optional[Dict[str, Any]] = None

        # Offer quick element selection for element-based actions
        if action_type in {"CLICK", "FILL", "SELECT_DROPDOWN"}:
            chosen_elem = self._maybe_pick_element(observation)
            if chosen_elem is not None:
                elem_id = chosen_elem.get("id", "unknown")
                action_id = f"{action_type.lower()}_elem_{elem_id}"

        if action_type == "SCROLL":
            print("\n   Ban da cuon bao nhieu?")
            print("   1. Mot chut (~300px)")
            print("   2. Nua trang (~600px)")
            print("   3. Mot trang (~1000px)")
            print("   4. Den cuoi trang")
            scroll_opt = self._get_number_input("Chon muc do", 1, 4)
            mapping = {1: 300, 2: 600, 3: 1000, 4: 5000}
            desc_map = {1: "mot chut", 2: "nua trang", 3: "mot trang", 4: "xuong cuoi"}
            action_params["amount"] = mapping[scroll_opt]
            action_params["direction"] = "down"
            description = f"Cuon xuong {desc_map[scroll_opt]}"

        elif action_type == "FILL":
            text = input("   Ban da nhap noi dung gi: ").strip()
            action_params["text"] = text
            description = f"Nhap '{text}'" if text else ""
            if chosen_elem is not None:
                action_params["target_hint"] = chosen_elem.get("text") or chosen_elem.get("placeholder")

        elif action_type == "CLICK":
            target = input("   Bam vao dau (mo ta ngan): ").strip()
            action_params["target_hint"] = target
            description = f"Bam vao {target}" if target else ""
            if chosen_elem is not None:
                action_params["target_hint"] = chosen_elem.get("text") or target

        elif action_type == "PRESS_KEY":
            key = input("   Nhan phim nao (Enter, Esc...): ").strip() or "Enter"
            action_params["key"] = key
            description = f"Nhan phim {key}"

        elif action_type == "WAIT":
            sec_raw = input("   Doi khoang may giay (default 2): ").strip() or "2"
            duration = self._safe_float(sec_raw, default=2.0)
            action_params["duration"] = duration
            description = f"Doi {duration} giay"

        elif action_type == "NAVIGATE":
            url = input("   Di toi URL nao: ").strip()
            action_params["url"] = url
            description = f"Di toi {url}" if url else ""

        elif action_type == "SELECT_DROPDOWN":
            target = input("   Selector/vi tri dropdown: ").strip()
            value = input("   Gia tri (option_value) muon chon: ").strip()
            action_params["selector"] = target
            action_params["value"] = value
            description = f"Chon gia tri {value}" if value else "Chon tu dropdown"
            if chosen_elem is not None:
                action_params["selector"] = action_params["selector"] or chosen_elem.get("selector")
                action_params["target_hint"] = chosen_elem.get("text")

        elif action_type == "WAIT_FOR":
            selector = input("   Cho selector nao (bo trong neu doi navigation): ").strip()
            timeout_raw = input("   Timeout (ms, default 5000): ").strip() or "5000"
            state = input("   Trang thai (visible/attached/detached, default visible): ").strip().lower() or "visible"
            action_params["timeout"] = int(timeout_raw) if timeout_raw.isdigit() else 5000
            if selector:
                action_params["selector"] = selector
                action_params["state"] = state
                description = f"Cho {selector} ({state})"
            else:
                action_params["wait_until"] = "networkidle"
                description = "Cho navigation/idle"

        elif action_type == "CLICK_AT":
            x_raw = input("   Toa do x: ").strip() or "0"
            y_raw = input("   Toa do y: ").strip() or "0"
            w_raw = input("   Width (enter de bo qua): ").strip() or "0"
            h_raw = input("   Height (enter de bo qua): ").strip() or "0"
            action_params["bbox"] = {
                "x": self._safe_float(x_raw, 0.0),
                "y": self._safe_float(y_raw, 0.0),
                "width": self._safe_float(w_raw, 0.0),
                "height": self._safe_float(h_raw, 0.0),
            }
            description = f"Click tai ({action_params['bbox']['x']}, {action_params['bbox']['y']})"

        elif action_type == "COMPLETE":
            description = "Danh dau hoan thanh"

        # Optional free-form description override
        extra_desc = input("   Mo ta bo sung (enter de bo qua): ").strip()
        if extra_desc:
            description = extra_desc
        if not description:
            description = f"todo: describe {action_type.lower()}"

        reason = input("\nTai sao (thought/reasoning): ").strip()
        if not reason:
            reason = f"todo: add reasoning for {action_type.lower()}"

        planner_block = {
            "thought": reason,
            "next_plan_step": {
                "step_id": f"STEP_{step_num}",
                "type": plan_type,
                "description": plan_description,
            },
        }
        controller_block = {
            "thought": reason,
            "chosen_action": {
                "action_id": action_id,
                "type": action_type,
                "parameters": action_params,
                "description": description,
                "reason": reason,
            },
        }
        action_block = {
            "action_id": action_id,
            "type": action_type,
            "parameters": action_params,
            "description": description,
            "reason": reason,
        }
        result_block = {"status": "success", "message": "Recorded manually"}

        # Populate page_state.actions for downstream available_actions_flat
        action_entry = {
            "id": action_id,
            "type": action_type.lower(),
            "description": description,
            "dom_selector": action_params.get("selector") if isinstance(action_params, dict) else None,
            "vision_ref": str(chosen_elem.get("id")) if chosen_elem is not None else None,
        }
        try:
            page_state_actions = page_state.get("actions", [])
            page_state_actions.append(action_entry)
            page_state["actions"] = page_state_actions
        except Exception:
            page_state["actions"] = [action_entry]

        return ManualStep(
            step=step_num,
            timestamp=datetime.now().isoformat(),
            planner=planner_block,
            controller=controller_block,
            action=action_block,
            result=result_block,
            page_state=page_state,
            observation=observation,
            assets=assets,
            teacher_labels={
                "planner": planner_block,
                "controller": controller_block,
            },
        )

    def _get_number_input(self, prompt: str, min_val: int, max_val: int) -> int:
        while True:
            try:
                val_raw = input(f"{prompt} ({min_val}-{max_val}): ").strip()
                val = int(val_raw)
                if min_val <= val <= max_val:
                    return val
            except ValueError:
                pass
            print("Gia tri khong hop le, thu lai.")

    def _safe_float(self, raw: str, default: float) -> float:
        try:
            return float(raw)
        except ValueError:
            return default

    def _print_action_cheatsheet(self) -> None:
        """Display supported action types and common params to guide custom input."""
        print("\n--- Cheatsheet hanh dong chuan ---")
        print("  - NAVIGATE(url)")
        print("  - CLICK(selector)")
        print("  - FILL(selector, text)")
        print("  - PRESS_KEY(key)")
        print("  - SELECT_DROPDOWN(selector, value)")
        print("  - WAIT_FOR(selector?, state?, timeout?) | wait_until=networkidle")
        print("  - CLICK_AT(bbox: x,y,width,height)")
        print("  - SCROLL(direction=down, amount)")
        print("  - WAIT(duration)")
        print("  - BACK | RELOAD | FORWARD")
        print("------------------------------------\n")

    def _maybe_pick_element(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Offer a quick element pick list to attach action_id to an element.
        """
        elements = observation.get("interactive_elements") or observation.get("elements") or []
        if not elements:
            return None

        print("\nChon element (0 = bo qua):")
        show = elements[: min(10, len(elements))]
        for elem in show:
            text = (elem.get("text") or "").strip()
            print(f"  {elem.get('id', '-')}: {elem.get('tag', '')} | {text}")

        choice_raw = input("   ID element muon lien ket (0 de bo qua): ").strip() or "0"
        if not choice_raw.isdigit():
            return None
        choice = int(choice_raw)
        if choice == 0:
            return None

        for elem in elements:
            if elem.get("id") == choice:
                return elem
        return None

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    async def _save_episode(self, goal: str, start_url: str) -> None:
        """Persist the full episode to JSON."""
        final_url = self.page.url if self.page else start_url
        episode = ManualEpisode(
            episode_id=self.episode_id,
            goal=goal,
            start_url=start_url,
            created_at=datetime.now().isoformat(),
            steps=self.steps,
            final_url=final_url,
            success=bool(self.steps),
            total_steps=len(self.steps),
            metadata={"collector": "interactive_manual", "version": 2},
        )

        path = self.episode_dir / f"{self.episode_id}.json"
        path.write_text(
            json.dumps(asdict(episode), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\nDa luu episode: {path}")
        print(f"Tong so buoc: {len(self.steps)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive manual data collector")
    parser.add_argument("--goal", required=True, help="Nhiem vu muon thuc hien")
    parser.add_argument("--url", required=True, help="URL bat dau")
    args = parser.parse_args()

    collector = InteractiveCollector()
    await collector.collect(args.goal, args.url)


if __name__ == "__main__":
    asyncio.run(main())
