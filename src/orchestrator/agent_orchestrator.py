"""
Agent Orchestrator - Main Control Loop
Coordinates all 4 layers: Perception + Planning + Execution + (optional) Learning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
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
from src.planning.react_engine import ReActEngine  # noqa: E402
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
        2. Run main ReAct-style control loop.
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
        """
        self.max_steps = max_steps or settings.max_steps
        self.headless = headless if headless is not None else settings.headless
        self.enable_learning = enable_learning
        self.enable_guardrails = enable_guardrails
        self.phobert_checkpoint = phobert_checkpoint
        self.vit5_checkpoint = vit5_checkpoint

        logger.info("Initializing Agent Orchestrator")
        logger.info("   Max steps: %s", self.max_steps)
        logger.info("   Headless: %s", self.headless)
        logger.info("   Learning enabled: %s", self.enable_learning)
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
        context: Optional[Dict] = None,
    ) -> ExecutionResult:
        """
        Execute a task end-to-end.

        Args:
            query: User query in Vietnamese.
            start_url: Starting URL.
            context: Additional context (currently unused).

        Returns:
            ExecutionResult with success status, history, etc.
        """
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("Starting Task Execution")
        logger.info("=" * 70)
        logger.info("Query: %s", query)
        logger.info("Start URL: %s", start_url)
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
                        metadata={"reason": "guardrail_block"},
                        final_url=start_url,
                        summary=f"Blocked by safety guardrails for URL: {start_url}",
                    )

            # Initialize browser
            page = await self.browser_manager.new_page()

            # Navigate to start URL
            await page.goto(start_url, timeout=30000)
            await self.browser_manager.wait_for_load(page)

            # Reset state
            self.react_engine.reset()
            self.state_manager.reset()

            # Main execution loop
            step = 0
            last_action: Optional[Dict] = None

            while step < self.max_steps:
                step += 1
                logger.info("\n%s", "=" * 70)
                logger.info("Step %d/%d", step, self.max_steps)
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
                        logger.warning("Action blocked by guardrails: %s", action.get("skill"))
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

            # Final observation (for completeness)
            final_observation = await self._perceive(page)
            _ = final_observation  # currently unused, kept for potential future use

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
                f"Task finished with skill='{last_action.get('skill')}' "
                if last_action
                else "Task finished without last_action"
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
                metadata={"progress": progress},
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
                metadata={},
                final_url=start_url,
                summary=f"Failed after {len(history)} steps: {exc}",
            )

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

        logger.debug("Observed DOM chars: %d, elements: %d", len(dom_distilled), len(elements))

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

