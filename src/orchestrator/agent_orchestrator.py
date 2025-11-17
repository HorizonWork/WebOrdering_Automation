"""
Agent Orchestrator - Main Control Loop
Coordinates all 4 layers: Perception → Planning → Execution → Learning
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.phobert_encoder import PhoBERTEncoder, PhoBERTEncoderConfig # noqa: E402
from src.models.vit5_planner import ViT5Planner  # noqa: E402
from src.perception.dom_distiller import DOMDistiller  # noqa: E402
from src.planning.react_engine import ReActEngine  # noqa: E402
from src.execution.browser_manager import BrowserManager  # noqa: E402
from src.execution.skill_executor import SkillExecutor  # noqa: E402
from src.orchestrator.state_manager import StateManager  # noqa: E402
from src.orchestrator.safety_guardrails import SafetyGuardrails  # noqa: E402
from src.perception.vision_enhancer import VisionEnhancer  # noqa: E402
from config.settings import settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of task execution"""
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
    
    **Responsibilities**:
        1. Initialize all components
        2. Run main control loop (ReAct)
        3. Coordinate layers: Perception → Planning → Execution → Learning
        4. Handle errors and retries
        5. Track state and history
        6. Apply safety guardrails
    
    **Flow**:
        User Query → [Perception] → [Planning] → [Execution] → [Learning]
                          ↑                                          ↓
                          └──────────── Feedback Loop ──────────────┘
    """
    
    def __init__(
        self,
        max_steps: Optional[int | None] = None,
        headless: Optional[bool | None] = None,
        enable_learning: bool = True,
        enable_guardrails: bool = True,
        phobert_checkpoint: str | None = None,
        vit5_checkpoint: str | None = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            max_steps: Maximum execution steps (default from settings)
            headless: Run browser in headless mode
            enable_learning: Enable learning layer
            enable_guardrails: Enable safety guardrails
        """
        self.max_steps = max_steps or settings.max_steps
        self.headless = headless if headless is not None else settings.headless
        self.enable_learning = enable_learning
        self.enable_guardrails = enable_guardrails
        self.phobert_checkpoint = phobert_checkpoint
        self.vit5_checkpoint = vit5_checkpoint
        
        logger.info("🚀 Initializing Agent Orchestrator")
        logger.info(f"   Max steps: {self.max_steps}")
        logger.info(f"   Headless: {self.headless}")
        logger.info(f"   Learning: {self.enable_learning}")
        logger.info(f"   Guardrails: {self.enable_guardrails}")
        
        # Initialize components
        self._init_components()
        
        logger.info("✅ Agent Orchestrator ready!")
    
    def _init_components(self):
        """Initialize all components"""
        logger.info("Initializing components...")
        
        # Models
        logger.info("  → Loading models...")
        phobert_config = (
            PhoBERTEncoderConfig(model_name_or_path=self.phobert_checkpoint)
            if self.phobert_checkpoint
            else None
        )
        self.phobert = PhoBERTEncoder(override_config=phobert_config)
        self.vit5 = (
            ViT5Planner(checkpoint_path=self.vit5_checkpoint)
            if self.vit5_checkpoint
            else ViT5Planner()
        )

        # Perception
        logger.info("  → Initializing perception...")
        self.dom_distiller = DOMDistiller()
        self.vision_enhancer = VisionEnhancer() if settings.enable_vision else None
        
        # Planning
        logger.info("  → Initializing planning...")
        self.react_engine = ReActEngine(planner=self.vit5, max_steps=self.max_steps)
        
        # Execution
        logger.info("  → Initializing execution...")
        browser_config = dict(settings.browser_config)
        browser_config["headless"] = self.headless
        self.browser_manager = BrowserManager(**browser_config)
        self.skill_executor = SkillExecutor()
        
        # State & Safety
        logger.info("  → Initializing state management...")
        self.state_manager = StateManager()
        
        if self.enable_guardrails:
            logger.info("  → Initializing safety guardrails...")
            self.guardrails = SafetyGuardrails()
        
        logger.info("✓ All components initialized")
    
    async def execute_task(
        self,
        query: str,
        start_url: str,
        context: Optional[Dict] = None
    ) -> ExecutionResult:
        """
        Execute a task end-to-end.
        
        Args:
            query: User query in Vietnamese
            start_url: Starting URL
            context: Additional context
            
        Returns:
            ExecutionResult with success status, history, etc.
            
        Example:
            >>> orchestrator = AgentOrchestrator()
            >>> result = await orchestrator.execute_task(
            ...     query="Tìm áo khoác giá dưới 500k",
            ...     start_url="https://shopee.vn"
            ... )
            >>> print(f"Success: {result.success}")
            >>> print(f"Steps: {result.steps}")
        """
        start_time = datetime.now()
        
        logger.info("=" * 70)
        logger.info("🎯 Starting Task Execution")
        logger.info("=" * 70)
        logger.info(f"Query: {query}")
        logger.info(f"Start URL: {start_url}")
        logger.info("=" * 70)
        
        try:
            # Safety check
            if self.enable_guardrails:
                if not self.guardrails.check_url_allowed(start_url):
                    return ExecutionResult(
                        success=False,
                        steps=0,
                        final_state={},
                        history=[],
                        error=f"URL blocked by safety guardrails: {start_url}",
                        execution_time=0.0,
                        metadata={'reason': 'guardrail_block'},
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
            last_action = None
            
            while step < self.max_steps:
                step += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 Step {step}/{self.max_steps}")
                logger.info(f"{'='*70}")
                
                # LAYER 1: PERCEPTION - Observe current state
                observation = await self._perceive(page)
                
                # Update state
                self.state_manager.update_state(observation)
                
                # LAYER 2: PLANNING - Decide next action
                thought, action = await self.react_engine.step(
                    query=query,
                    observation=observation,
                    available_skills=self.skill_executor.get_available_skills()
                )
                
                # Safety check on action
                if self.enable_guardrails:
                    if not self.guardrails.check_action_allowed(action):
                        logger.warning(f"⚠️  Action blocked: {action['skill']}")
                        action = {
                            'skill': 'complete',
                            'params': {'message': 'Action blocked by guardrails'}
                        }
                
                # LAYER 3: EXECUTION - Execute action
                result = await self.skill_executor.execute(page, action)
                
                # Wait for changes
                await asyncio.sleep(1)
                
                # Record step
                self.react_engine.add_step(
                    step_num=step,
                    thought=thought,
                    action=action,
                    observation=observation,
                    result=result
                )
                
                # Log step summary
                logger.info(f"💭 Thought: {thought[:80]}...")
                logger.info(f"⚡ Action: {action['skill']}({action['params']})")
                logger.info(f"✅ Result: {result['status']}")
                
                last_action = action
                
                # Check completion
                if not self.react_engine.should_continue(last_action):
                    logger.info("🏁 Task completed or max steps reached")
                    break
            
            # Get final observation
            final_observation = await self._perceive(page)
            
            # Close browser
            await self.browser_manager.close()
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Build result
            success = last_action and last_action.get('skill') == 'complete'
            history = self.react_engine.get_history()
            
            final_url = page.url if page else start_url
            result = ExecutionResult(
                success=success,
                steps=step,
                final_state=final_observation,
                history=history,
                execution_time=execution_time,
                metadata={
                    'query': query,
                    'start_url': start_url,
                    'final_url': final_url
                },
                final_url=final_url,
                summary=f"Success={success}, steps={step}, final_url={final_url}",
            )
            
            # Log summary
            logger.info("\n" + "=" * 70)
            logger.info("📊 Execution Summary")
            logger.info("=" * 70)
            logger.info(f"✅ Success: {result.success}")
            logger.info(f"📈 Steps: {result.steps}")
            logger.info(f"⏱️  Time: {result.execution_time:.2f}s")
            logger.info(f"📊 Success rate: {self.react_engine.analyze_progress()['success_rate']:.1%}")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}", exc_info=True)
            
            # Close browser on error
            try:
                await self.browser_manager.close()
            except:  
                pass
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=False,
                steps=step if "step" in locals() else 0,  
                final_state={},
                history=self.react_engine.get_history()
                if hasattr(self, "react_engine")
                else [],
                error=str(e),
                execution_time=execution_time,
                final_url=start_url,
                summary=f"Failed after {step if 'step' in locals() else 0} steps: {e}",
            )
    
    async def _perceive(self, page) -> Dict:
        """
        Perception layer: Extract current state.
        
        Returns:
            Observation dict with {url, dom, screenshot, elements}
        """
        logger.debug("👁️  Perceiving current state...")
        
        # Get URL
        url = page.url
        
        # Get HTML
        html = await self.browser_manager.get_html(page)
        
        # Distill DOM
        dom_distilled = self.dom_distiller.distill(html, mode='text_only')
        
        # Extract interactive elements
        elements = self.dom_distiller.extract_interactive_elements(html)
        
        # Take screenshot (optional, for debugging)
        screenshot = await self.browser_manager.screenshot(page)
        
        vision_context = None
        if self.vision_enhancer:
            vision_context = await self.vision_enhancer.analyze_async(screenshot)

        observation = {
            'url': url,
            'dom': dom_distilled,
            'elements': elements,
            'interactive_elements': elements,  # maintain compatibility with planner expectations
            'screenshot': screenshot,
            'timestamp': datetime.now().isoformat(),
            'vision': vision_context.__dict__ if vision_context else None,
        }
        
        logger.debug(f"✓ Observed: {len(dom_distilled)} chars DOM, {len(elements)} elements")
        
        return observation
    
    def get_execution_summary(self) -> str:
        """Get execution summary"""
        return self.react_engine.get_summary()

    async def close(self) -> None:
        """Gracefully release resources."""
        if hasattr(self, "browser_manager") and self.browser_manager:
            await self.browser_manager.close()
        logger.info("AgentOrchestrator closed resources")
    
# Test & Usage Example
async def test_orchestrator():
    """Test orchestrator with simple task"""
    print("=" * 70)
    print("Agent Orchestrator - Test")
    print("=" * 70 + "\n")
    
    # Initialize
    orchestrator = AgentOrchestrator(
        max_steps=5,
        headless=False
    )
    
    # Execute task
    result = await orchestrator.execute_task(
        query="Tìm áo khoác giá rẻ",
        start_url="https://example.com"
    )
    
    # Print result
    print("\n" + "=" * 70)
    print("Execution Result")
    print("=" * 70)
    print(f"Success: {result.success}")
    print(f"Steps: {result.steps}")
    print(f"Time: {result.execution_time:.2f}s")
    print(f"Error: {result.error or 'None'}")
    
    # Print history
    print("\nExecution History:")
    for i, step in enumerate(result.history, 1):
        print(f"\n{i}. {step['action']['skill']}({step['action']['params']})")
        print(f"   Status: {step['result']['status']}")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())
