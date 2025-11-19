"""
Agent Orchestrator - Main Control Loop
Coordinates all 4 layers: Perception + Planning + Execution + (optional) Learning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings  # noqa: E402
from src.execution.browser_manager import BrowserManager  # noqa: E402
from src.execution.skill_executor import SkillExecutor  # noqa: E402
from src.models.phobert_encoder import PhoBERTEncoder  # noqa: E402
from src.models.vit5_planner import ViT5Planner  # noqa: E402
from src.orchestrator.safety_guardrails import SafetyGuardrails  # noqa: E402
from src.orchestrator.state_manager import StateManager  # noqa: E402
from src.perception.dom_distiller import DOMDistiller  # noqa: E402
from src.perception.vision_enhancer import VisionEnhancer  # noqa: E402
from src.perception.ui_detector import UIDetector  # noqa: E402
from src.planning.react_engine import ReActEngine  # noqa: E402
from src.planning.rule_policy import RulePolicy  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of task execution."""

    success: bool
    steps: int
    final_state: Dict
    history: List[Dict]
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict = field(default_factory=dict)
    final_url: Optional[str] = None
    summary: str = ""


class AgentOrchestrator:
    """
    Main orchestrator for WOA Agent.

    Responsibilities:
        1. Initialize all components (models, perception, planning, execution, safety).
        2. Run main control loop (ReAct-style or rule-based).
        3. Track state, history and metrics.
        4. Apply safety guardrails for URLs and actions.
    """

    def __init__(
        self,
        max_steps: Optional[int | None] = None,
        headless: Optional[bool | None] = None,
        enable_learning: bool = True,  # reserved for future use
        enable_guardrails: bool = True,
        phobert_checkpoint: str | None = None,
        vit5_checkpoint: str | None = None,
        user_data_dir: str | None = None,
    ) -> None:
        """
        Initialize orchestrator.

        Args:
            max_steps: Maximum execution steps (default from settings).
            headless: Run browser in headless mode.
            enable_learning: Enable learning layer (currently not wired).
            enable_guardrails: Enable safety guardrails.
            phobert_checkpoint: Optional path/name for PhoBERT model.
            vit5_checkpoint: Optional path/name for ViT5 planner checkpoint.
            user_data_dir: Path to Chrome user profile (for login persistence).
        """
        self.max_steps = max_steps or settings.max_steps
        self.headless = headless if headless is not None else settings.headless
        self.enable_learning = enable_learning
        self.enable_guardrails = enable_guardrails
        self.phobert_checkpoint = phobert_checkpoint
        self.vit5_checkpoint = vit5_checkpoint
        self.user_data_dir = user_data_dir

        # If using persistent profile (data collection), avoid headless to reduce anti-bot detection.
        if self.user_data_dir and self.headless:
            logger.warning(
                "Has user_data_dir but headless=True. "
                "Forcing headless=False to avoid anti-bot."
            )
            self.headless = False

        logger.info("Initializing Agent Orchestrator")
        logger.info("   Max steps: %s", self.max_steps)
        logger.info("   Headless: %s", self.headless)
        logger.info("   User Data Dir: %s", self.user_data_dir)
        logger.info("   Guardrails enabled: %s", self.enable_guardrails)

        self._init_components()

        logger.info("Agent Orchestrator ready.")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Initialize models, perception, planning, execution, safety."""
        logger.info("Initializing components...")

        # Models
        logger.info("  - Loading models...")
        phobert_model_name = self.phobert_checkpoint or "vinai/phobert-base-v2"
        self.phobert = PhoBERTEncoder(model_name=phobert_model_name)

        self.vit5 = (
            ViT5Planner(checkpoint_path=self.vit5_checkpoint)
            if self.vit5_checkpoint
            else ViT5Planner()
        )

        # Perception
        logger.info("  - Initializing perception...")
        self.dom_distiller = DOMDistiller()
        self.vision_enhancer = VisionEnhancer() if settings.enable_vision else None

        # Planning
        logger.info("  - Initializing planning (ReAct engine)...")
        self.react_engine = ReActEngine(planner=self.vit5, max_steps=self.max_steps)

        # Execution
        logger.info("  - Initializing execution...")
        try:
            self.browser_manager = BrowserManager(
                headless=self.headless,
                user_data_dir=self.user_data_dir,
            )
        except TypeError:
            # Fallback if BrowserManager does not yet support user_data_dir.
            logger.warning(
                "BrowserManager does not support user_data_dir. Falling back to headless only init."
            )
            self.browser_manager = BrowserManager(headless=self.headless)

        self.skill_executor = SkillExecutor()

        # State & Safety
        logger.info("  - Initializing state management...")
        self.state_manager = StateManager()

        if self.enable_guardrails:
            logger.info("  - Initializing safety guardrails...")
            self.guardrails = SafetyGuardrails()

        logger.info("All components initialized.")

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    async def execute_task(
        self,
        query: str,
        start_url: str,
        policy: str = "react",
    ) -> ExecutionResult:
        """
        Execute a task end-to-end.

        Args:
            query: User query in Vietnamese.
            start_url: Starting URL.
            policy: Execution policy ("react", "rules_shopee", "rules_lazada", "human_teleop").

        Returns:
            ExecutionResult with success status, history, etc.
        """
        start_time = datetime.now()
        page = None

        logger.info("=" * 70)
        logger.info("Starting Task Execution")
        logger.info("=" * 70)
        logger.info("Query: %s", query)
        logger.info("Start URL: %s", start_url)
        logger.info("Policy: %s", policy)
        logger.info("=" * 70)

        try:
            # Safety: check start URL
            if self.enable_guardrails:
                if not self.guardrails.check_url_allowed(start_url):
                    return ExecutionResult(
                        success=False,
                        steps=0,
                        final_state={},
                        history=[],
                        error=f"URL blocked by safety guardrails: {start_url}",
                        execution_time=0.0,
                        metadata={"reason": "guardrail_block", "policy": policy},
                        final_url=start_url,
                        summary=f"Blocked by safety guardrails for URL: {start_url}",
                    )

            # Initialize browser
            page = await self.browser_manager.new_page()

            # Navigate to start URL
            try:
                await page.goto(start_url, timeout=30000, wait_until="domcontentloaded")
            except Exception as nav_err:
                logger.warning(
                    "Navigation timeout or error: %s. Proceeding anyway.", nav_err
                )

            await self.browser_manager.wait_for_load(page)

            # Reset state
            self.react_engine.reset()
            self.state_manager.reset()

            # Dispatch to policy-specific loop
            if policy == "react":
                last_action = await self._run_react_loop(page, query)
            elif policy in ("rules_shopee", "rules_lazada"):
                last_action = await self._run_rule_loop(
                    page=page,
                    query=query,
                    start_url=start_url,
                    policy_name=policy,
                )
            elif policy == "human_teleop":
                last_action = await self._run_teleop_loop(
                    page=page, query=query, start_url=start_url
                )
            else:
                raise ValueError(f"Unknown policy: {policy}")

            # Final observation (for completeness)
            final_observation = await self._perceive(page)

            # Close browser
            await self.browser_manager.close()

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Build ExecutionResult
            success = bool(last_action and last_action.get("skill") == "complete")
            history = self.react_engine.get_history()
            progress = self.react_engine.analyze_progress()
            success_rate = progress.get("success_rate", 0.0)

            summary = (
                f"Task finished (policy={policy}) with skill='{last_action.get('skill')}' "
                if last_action
                else f"Task finished (policy={policy}) without last_action"
            )

            logger.info("=" * 70)
            logger.info("Execution finished. Success: %s", success)
            logger.info("Steps: %d", len(history))
            logger.info("Time: %.2fs", execution_time)
            logger.info("Success rate (ReAct steps): %.1f%%", success_rate * 100)
            logger.info("=" * 70)

            return ExecutionResult(
                success=success,
                steps=len(history),
                final_state={"observation": final_observation},
                history=history,
                error=None,
                execution_time=execution_time,
                metadata={"progress": progress, "policy": policy},
                final_url=page.url if page else start_url,
                summary=summary,
            )

        except Exception as exc:
            logger.error("Task execution failed: %s", exc, exc_info=True)

            # Try to close browser on error
            try:
                await self.browser_manager.close()
            except Exception:
                pass

            execution_time = (datetime.now() - start_time).total_seconds()

            history: List[Dict] = []
            if hasattr(self, "react_engine"):
                try:
                    history = self.react_engine.get_history()
                except Exception:
                    history = []

            return ExecutionResult(
                success=False,
                steps=len(history),
                final_state={},
                history=history,
                error=str(exc),
                execution_time=execution_time,
                metadata={"policy": policy},
                final_url=start_url,
                summary=f"Failed after {len(history)} steps (policy={policy}): {exc}",
            )

    # ------------------------------------------------------------------
    # Policy-specific loops
    # ------------------------------------------------------------------

    async def _run_react_loop(self, page, query: str) -> Optional[Dict]:
        """
        Standard ReAct + ViT5 control loop.

        Returns:
            Last executed action dict (or None).
        """
        step = 0
        last_action: Optional[Dict] = None

        while step < self.max_steps:
            step += 1
            logger.info("\n%s", "=" * 70)
            logger.info("Step %d/%d (policy=react)", step, self.max_steps)
            logger.info("%s", "=" * 70)

            # LAYER 1: PERCEPTION - Observe current state
            observation = await self._perceive(page)

            # Update state
            self.state_manager.update_state(observation)

            # LAYER 2: PLANNING - Decide next action
            thought, action = await self.react_engine.step(
                query=query,
                observation=observation,
                available_skills=self.skill_executor.get_available_skills(),
            )

            # Safety check on action
            if self.enable_guardrails:
                if not self.guardrails.check_action_allowed(action):
                    logger.warning(
                        "Action blocked by guardrails: %s", action.get("skill")
                    )
                    action = {
                        "skill": "complete",
                        "params": {"message": "Action blocked by guardrails"},
                    }

            # LAYER 3: EXECUTION - Execute action
            result = await self.skill_executor.execute(page, action)

            # Wait briefly for page to settle
            await asyncio.sleep(1)

            # Record step
            self.react_engine.add_step(
                step_num=step,
                thought=thought,
                action=action,
                observation=observation,
                result=result,
            )

            # Log step summary
            logger.info("Thought: %s...", thought[:80])
            logger.info("Action: %s(%s)", action.get("skill"), action.get("params"))
            logger.info("Result: %s", result.get("status"))

            last_action = action

            # Check completion
            if not self.react_engine.should_continue(last_action):
                logger.info("Task completed or max steps reached (ReAct decided).")
                break

        return last_action

    async def _run_rule_loop(
        self,
        page,
        query: str,
        start_url: str,
        policy_name: str,
    ) -> Optional[Dict]:
        """
        Heuristic rule-based policy loop.

        Uses DOMDistiller + UIDetector + RulePolicy to select actions,
        while still recording history via ReActEngine.add_step.

        Returns:
            Last executed action dict (or None).
        """
        logger.info(
            "Starting rule-based loop: policy=%s, start_url=%s", policy_name, start_url
        )

        ui_detector = UIDetector()
        rule_policy = RulePolicy()

        last_action: Optional[Dict] = None

        for step_idx in range(1, self.max_steps + 1):
            logger.info("\n%s", "=" * 70)
            logger.info(
                "[RULE] Step %d/%d (policy=%s)",
                step_idx,
                self.max_steps,
                policy_name,
            )
            logger.info("%s", "=" * 70)

            # Perception
            observation = await self._perceive(page)
            self.state_manager.update_state(observation)

            # Build page_state for heuristic policy
            page_state = self._build_page_state_for_rules(observation, ui_detector)

            # Select action via rules
            action = rule_policy.select_action(
                goal=query,
                page_state=page_state,
                policy_name=policy_name,
            )

            if action is None:
                logger.info("[RULE] No action returned by policy. Terminating.")
                break

            # Safety check on action
            if self.enable_guardrails:
                if not self.guardrails.check_action_allowed(action):
                    logger.warning("[RULE] Action blocked by guardrails: %s", action)
                    action = {
                        "skill": "complete",
                        "params": {
                            "message": "Action blocked by guardrails (rule_policy)"
                        },
                    }

            # Execute action
            result = await self.skill_executor.execute(page, action)
            await asyncio.sleep(1)

            # Record step in ReAct history format
            self.react_engine.add_step(
                step_num=step_idx,
                thought="(rule_policy)",
                action=action,
                observation=observation,
                result=result,
            )

            last_action = action

            # Simple termination conditions
            page_type = page_state.get("page_type", "generic")
            if page_type in ("checkout", "review_order"):
                logger.info(
                    "[RULE] Detected terminal page_type=%s. Stopping.", page_type
                )
                break

            # Also stop if rule explicitly completes
            if action.get("skill") == "complete":
                logger.info("[RULE] Received 'complete' action. Stopping.")
                break

        logger.info(
            "Rule-based loop finished after %d steps.", len(self.react_engine.get_history())
        )
        return last_action

    async def _run_teleop_loop(
        self,
        page,
        query: str,
        start_url: str,
    ) -> Optional[Dict]:
        """
        Placeholder for human teleoperation loop.

        Currently not implemented; records a single 'complete' step so that
        collectors can still save a trajectory without crashing.
        """
        logger.warning(
            "human_teleop policy not implemented yet. Ending immediately "
            "with a stub 'complete' action."
        )

        observation = await self._perceive(page)
        result: Dict[str, Any] = {
            "status": "success",
            "message": "human_teleop policy not implemented; auto-completed.",
        }
        action: Dict[str, Any] = {
            "skill": "complete",
            "params": {"message": "human_teleop stub"},
        }

        self.react_engine.add_step(
            step_num=1,
            thought="(human_teleop stub)",
            action=action,
            observation=observation,
            result=result,
        )

        return action

    def _build_page_state_for_rules(
        self,
        observation: Dict[str, Any],
        ui_detector: UIDetector,
    ) -> Dict[str, Any]:
        """
        Build a lightweight page_state snapshot for rule-based policies.

        Shape is intentionally similar to scripts.data_collection.collect_raw_trajectories.build_page_state.
        """
        url = observation.get("url", "")
        dom_html = observation.get("dom", "") or ""
        elements = (
            observation.get("interactive_elements")
            or observation.get("elements")
            or []
        )
        vision_state = observation.get("vision_state") or observation.get("vision") or {}

        ui_info: Dict[str, Any] = {"page_type": "generic"}
        if dom_html:
            try:
                ui_info = ui_detector.detect_all(dom_html) or {"page_type": "generic"}
            except Exception as e:
                logger.warning(f"UIDetector failed in rule loop: {e}")
                ui_info = {"page_type": "generic"}

        return {
            "url": url,
            "page_type": ui_info.get("page_type", "generic"),
            "dom_state": {
                "element_count": len(elements),
                "has_search": bool(ui_info.get("search_box", {}).get("found")),
            },
            "elements": elements,
            "ui_detection": ui_info,
            "vision_state": vision_state,
        }

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    async def _perceive(self, page) -> Dict:
        """
        Perception layer: Extract current state.

        Returns:
            Observation dict with {url, dom, screenshot, elements, vision?}
        """
        logger.debug("Perceiving current state...")

        # URL
        url = page.url

        # HTML & DOM distillation
        html = await self.browser_manager.get_html(page)
        dom_distilled = self.dom_distiller.distill(html, mode="text_only")

        # Interactive elements
        elements = self.dom_distiller.extract_interactive_elements(html)

        # Screenshot (raw bytes, primarily for logging / debug)
        screenshot = await self.browser_manager.screenshot(page)

        # Optional vision analysis
        vision_context = None
        if self.vision_enhancer:
            vision_context = await self.vision_enhancer.analyze_async(screenshot)

        observation: Dict[str, Any] = {
            "url": url,
            "dom": dom_distilled,
            "elements": elements,
            "interactive_elements": elements,
            "screenshot": screenshot,
            "timestamp": datetime.now().isoformat(),
            "vision": vision_context.__dict__ if vision_context else None,
        }

        logger.debug(
            "Observed DOM chars: %d, elements: %d", len(dom_distilled), len(elements)
        )

        return observation

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_execution_summary(self) -> str:
        """Get execution summary from ReAct engine."""
        return self.react_engine.get_summary()

    async def close(self) -> None:
        """Gracefully release browser resources."""
        if hasattr(self, "browser_manager") and self.browser_manager:
            await self.browser_manager.close()
        logger.info("AgentOrchestrator closed resources")


# Test & Usage Example (manual)
async def test_orchestrator() -> None:
    """Test orchestrator with a simple task on example.com."""
    print("=" * 70)
    print("Agent Orchestrator - Test")
    print("=" * 70 + "\n")

    orchestrator = AgentOrchestrator(
        max_steps=3,
        headless=True,
    )

    result = await orchestrator.execute_task(
        query="Mở trang ví dụ",
        start_url="https://example.com",
    )

    print("\n" + "=" * 70)
    print("Execution Result")
    print("=" * 70)
    print(f"Success: {result.success}")
    print(f"Steps: {result.steps}")
    print(f"Time: {result.execution_time:.2f}s")
    print(f"Error: {result.error or 'None'}")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())
