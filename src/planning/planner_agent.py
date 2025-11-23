"""
Planner Agent - High-Level Task Planning
Coordinates ReAct engine, Navigator, and ViT5 planner for task execution
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from src.models.vit5_planner import ViT5Planner
from src.planning.react_engine import ReActEngine
from src.planning.navigator_agent import NavigatorAgent
from src.planning.change_observer import ChangeObserver
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STUCK = "stuck"


@dataclass
class TaskPlan:
    """High-level task plan"""
    query: str
    steps: List[str]
    current_step: int
    status: TaskStatus
    confidence: float


class PlannerAgent:
    """
    High-level planner agent.
    
    **Responsibilities**:
        1. Generate high-level plan from query
        2. Coordinate ReAct execution
        3. Track progress against plan
        4. Detect stuck states
        5. Suggest plan adjustments
    
    **Architecture**:
        User Query
            ↓
        [Generate Plan] (ViT5)
            ↓
        [Execute Steps] (ReAct + Navigator)
            ↓
        [Monitor Progress]
            ↓
        [Complete / Adjust]
    
    **Features**:
        - Multi-step planning
        - Progress tracking
        - Stuck detection
        - Plan adjustment
        - Success estimation
    """
    
    def __init__(
        self,
        vit5_planner: Optional[ViT5Planner] = None,
        react_engine: Optional[ReActEngine] = None,
        navigator: Optional[NavigatorAgent] = None
    ):
        """
        Initialize planner agent.
        
        Args:
            vit5_planner: ViT5 planner for text generation
            react_engine: ReAct engine for execution
            navigator: Navigator agent for browser control
        """
        self.vit5 = vit5_planner or ViT5Planner()
        self.react = react_engine or ReActEngine(planner=self.vit5)
        self.navigator = navigator or NavigatorAgent()
        
        self.current_plan: Optional[TaskPlan] = None
        self.execution_history: List[Dict] = []
        
        logger.info("PlannerAgent initialized")
    
    def generate_plan(self, query: str, context: Optional[Dict] = None) -> TaskPlan:
        """
        Generate high-level plan from query.
        
        Args:
            query: User query in Vietnamese
            context: Additional context (url, constraints, etc.)
            
        Returns:
            TaskPlan with steps
            
        Example:
            >>> planner = PlannerAgent()
            >>> plan = planner.generate_plan("Mua áo khoác trên Shopee")
            >>> print(plan.steps)
            ['Vào trang Shopee', 'Tìm kiếm áo khoác', 'Lọc kết quả', 'Chọn sản phẩm', 'Thêm vào giỏ']
        """
        logger.info(f"🎯 Generating plan for: {query}")
        
        # Generate plan text using ViT5
        plan_text = self.vit5.generate_plan(query, context)
        
        # Parse plan into steps
        steps = self._parse_plan_steps(plan_text)
        
        plan = TaskPlan(
            query=query,
            steps=steps,
            current_step=0,
            status=TaskStatus.NOT_STARTED,
            confidence=0.8
        )
        
        self.current_plan = plan
        
        logger.info(f"yes Generated plan with {len(steps)} steps")
        for i, step in enumerate(steps, 1):
            logger.info(f"  {i}. {step}")
        
        return plan
    
    def _parse_plan_steps(self, plan_text: str) -> List[str]:
        """
        Parse plan text into individual steps.
        
        Args:
            plan_text: Generated plan text
            
        Returns:
            List of step descriptions
        """
        # Split by newlines and filter numbered items
        lines = plan_text.strip().split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove numbering (1., 2., -, •, etc.)
            import re
            cleaned = re.sub(r'^[\d\.\-\•\*]\s*', '', line)
            
            if cleaned:
                steps.append(cleaned)
        
        # If no steps parsed, use whole text as single step
        if not steps:
            steps = [plan_text.strip()]
        
        return steps
    
    def get_current_step(self) -> Optional[str]:
        """Get current step description"""
        if not self.current_plan:
            return None
        
        if self.current_plan.current_step >= len(self.current_plan.steps):
            return None
        
        return self.current_plan.steps[self.current_plan.current_step]
    
    def advance_step(self):
        """Move to next step in plan"""
        if not self.current_plan:
            return
        
        self.current_plan.current_step += 1
        
        # Update status
        if self.current_plan.current_step >= len(self.current_plan.steps):
            self.current_plan.status = TaskStatus.COMPLETED
            logger.info("🏁 All plan steps completed")
        else:
            logger.info(f"➡️  Advanced to step {self.current_plan.current_step + 1}/{len(self.current_plan.steps)}")
    
    def is_step_complete(self, observation: Dict) -> bool:
        """
        Check if current step is complete.
        
        Args:
            observation: Current state observation
            
        Returns:
            True if step is complete
        """
        if not self.current_plan:
            return False
        
        current_step = self.get_current_step()
        if not current_step:
            return True
        
        # Use ViT5 to check if observation matches step completion
        prompt = f"""Bước hiện tại: {current_step}

Trạng thái trang:
- URL: {observation.get('url', '')}
- DOM: {observation.get('dom', '')[:200]}...

Bước này đã hoàn thành chưa? Trả lời 'có' hoặc 'chưa':"""
        
        response = self.vit5.generate(prompt, max_length=10, temperature=0.3)
        
        completed = 'có' in response.lower() or 'hoàn thành' in response.lower()
        
        if completed:
            logger.info(f"yes Step completed: {current_step}")
        
        return completed
    
    def detect_stuck(self) -> bool:
        """
        Detect if execution is stuck.
        
        Returns:
            True if stuck
        """
        if not self.react:
            return False
        
        # Check if ReAct detected loop
        if self.react.detect_loop():
            logger.warning("⚠️  Stuck: Loop detected")
            return True
        
        # Check if too many failures
        progress = self.react.analyze_progress()
        if progress['steps'] > 10 and progress['success_rate'] < 0.3:
            logger.warning("⚠️  Stuck: Low success rate")
            return True
        
        # Check if same step repeated
        if self.current_plan:
            recent_steps = self.execution_history[-5:]
            if len(recent_steps) >= 5:
                same_step = all(
                    h.get('plan_step') == self.current_plan.current_step
                    for h in recent_steps
                )
                if same_step:
                    logger.warning("⚠️  Stuck: Same step repeated")
                    return True
        
        return False
    
    def suggest_plan_adjustment(self) -> Optional[TaskPlan]:
        """
        Suggest adjusted plan if stuck.
        
        Returns:
            Adjusted TaskPlan or None
        """
        if not self.current_plan or not self.detect_stuck():
            return None
        
        logger.info("🔄 Generating adjusted plan...")
        
        # Get current context
        last_observation = self.execution_history[-1].get('observation', {}) if self.execution_history else {}
        
        # Generate new plan
        context = {
            'url': last_observation.get('url', ''),
            'previous_plan': self.current_plan.steps,
            'stuck_at_step': self.current_plan.current_step,
            'reason': 'Execution stuck, need alternative approach'
        }
        
        new_plan = self.generate_plan(self.current_plan.query, context)
        
        logger.info("yes Generated adjusted plan")
        return new_plan
    
    def track_execution(
        self,
        action: Dict,
        result: Dict,
        observation: Dict
    ):
        """
        Track execution step.
        
        Args:
            action: Action executed
            result: Execution result
            observation: State observation
        """
        step_record = {
            'plan_step': self.current_plan.current_step if self.current_plan else -1,
            'action': action,
            'result': result,
            'observation': observation,
            'success': result.get('status') == 'success'
        }
        
        self.execution_history.append(step_record)
        
        # Check if step complete
        if self.current_plan and self.is_step_complete(observation):
            self.advance_step()
    
    def get_progress_summary(self) -> Dict:
        """
        Get progress summary.
        
        Returns:
            Summary dict
        """
        if not self.current_plan:
            return {
                'status': 'no_plan',
                'progress': 0.0,
                'current_step': None,
                'total_steps': 0
            }
        
        total = len(self.current_plan.steps)
        current = self.current_plan.current_step
        progress = current / total if total > 0 else 0.0
        
        return {
            'status': self.current_plan.status.value,
            'progress': progress,
            'current_step': self.get_current_step(),
            'total_steps': total,
            'steps_completed': current,
            'query': self.current_plan.query
        }
    
    def estimate_completion_time(self) -> Optional[float]:
        """
        Estimate remaining time to completion (seconds).
        
        Returns:
            Estimated seconds or None
        """
        if not self.current_plan or not self.execution_history:
            return None
        
        # Calculate average time per step
        total_time = sum(
            1.0 for _ in self.execution_history  # Simplified: assume 1s per action
        )
        
        steps_done = self.current_plan.current_step
        if steps_done == 0:
            return None
        
        avg_time_per_step = total_time / steps_done
        remaining_steps = len(self.current_plan.steps) - steps_done
        
        estimated = avg_time_per_step * remaining_steps
        
        return estimated
    
    def reset(self):
        """Reset planner state"""
        self.current_plan = None
        self.execution_history = []
        self.react.reset()
        self.navigator.reset()
        logger.info("PlannerAgent reset")


# Test & Example
if __name__ == "__main__":
    print("=" * 70)
    print("PlannerAgent - Test")
    print("=" * 70 + "\n")
    
    # Initialize
    planner = PlannerAgent()
    
    # Test 1: Generate plan
    print("🎯 Test 1: Generate Plan")
    print("=" * 70)
    query = "Mua áo khoác giá rẻ trên Shopee"
    plan = planner.generate_plan(query)
    
    print(f"\nQuery: {query}")
    print(f"Steps: {len(plan.steps)}")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step}")
    
    # Test 2: Progress tracking
    print("\n🎯 Test 2: Progress Tracking")
    print("=" * 70)
    
    # Simulate some steps
    for i in range(3):
        current = planner.get_current_step()
        print(f"\nStep {i+1}: {current}")
        
        # Mock execution
        action = {'skill': 'click', 'params': {'selector': '#btn'}}
        result = {'status': 'success', 'message': 'OK'}
        observation = {'url': 'https://shopee.vn', 'dom': '<div>test</div>'}
        
        planner.track_execution(action, result, observation)
        planner.advance_step()
    
    # Test 3: Progress summary
    print("\n🎯 Test 3: Progress Summary")
    print("=" * 70)
    summary = planner.get_progress_summary()
    print(f"\nStatus: {summary['status']}")
    print(f"Progress: {summary['progress']:.0%}")
    print(f"Steps completed: {summary['steps_completed']}/{summary['total_steps']}")
    print(f"Current step: {summary['current_step']}")
    
    # Test 4: Completion estimate
    print("\n🎯 Test 4: Completion Estimate")
    print("=" * 70)
    estimated = planner.estimate_completion_time()
    if estimated:
        print(f"\nEstimated time remaining: {estimated:.1f} seconds")
    else:
        print("\nNot enough data for estimate")
    
    print("\n" + "=" * 70)
    print("yes All Tests Completed!")
    print("=" * 70)
