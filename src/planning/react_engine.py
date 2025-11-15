"""
ReAct Engine - Reason + Act Loop
Implements ReAct pattern: Thought → Action → Observation → Repeat
Based on WebVoyager and Agent-E architectures
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.vit5_planner import ViT5Planner
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReActStep:
    """Single step in ReAct loop"""
    step_num: int
    thought: str
    action: Dict
    observation: Dict
    result: Dict
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ReActEngine:
    """
    ReAct (Reasoning + Acting) engine.
    
    **ReAct Pattern** (from Yao et al., 2022):
        1. Thought: "Cần tìm kiếm sản phẩm"
        2. Action: click(#search-box)
        3. Observation: Search box focused
        4. Thought: "Giờ nhập từ khóa"
        5. Action: type(#search-box, "áo khoác")
        6. ... (repeat until task complete)
    
    **Benefits**:
        - Interpretable reasoning
        - Error correction capability
        - Step-by-step progress tracking
        - Context maintenance
    
    **Components**:
        - ViT5 planner for thought/action generation
        - History tracking (last 10 steps)
        - Progress analysis
        - Completion detection
    """
    
    def __init__(
        self,
        planner: Optional[ViT5Planner] = None,
        max_steps: int = 30,
        history_window: int = 10
    ):
        """
        Initialize ReAct engine.
        
        Args:
            planner: ViT5 planner (creates new if None)
            max_steps: Maximum steps before termination
            history_window: Number of past steps to include in context
        """
        self.planner = planner or ViT5Planner()
        self.max_steps = max_steps
        self.history_window = history_window
        self.history: List[ReActStep] = []
        
        logger.info(f"ReAct Engine initialized (max_steps={max_steps}, history_window={history_window})")
    
    def reset(self):
        """Reset history"""
        self.history = []
        logger.info("ReAct Engine reset")
    
    async def step(
        self,
        query: str,
        observation: Dict,
        available_skills: Optional[List[str]] = None
    ) -> tuple[str, Dict]:
        """
        Execute one ReAct step: Thought → Action.
        
        Args:
            query: Original user query
            observation: Current state observation
            available_skills: Available skills for action generation
            
        Returns:
            (thought, action) tuple
            
        Example:
            >>> engine = ReActEngine()
            >>> thought, action = await engine.step(
            ...     query="Tìm áo khoác",
            ...     observation={'url': '...', 'dom': '...'},
            ...     available_skills=['goto', 'click', 'type']
            ... )
            >>> print(thought)
            "Cần click vào search box để tìm kiếm"
            >>> print(action)
            {'skill': 'click', 'params': {'selector': '#search'}}
        """
        step_num = len(self.history) + 1
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 ReAct Step {step_num}/{self.max_steps}")
        logger.info(f"{'='*70}")
        
        # Get recent history for context
        recent_history = self.get_history(last_n=self.history_window)
        
        # STEP 1: THINK - Generate reasoning thought
        logger.info("💭 Generating thought...")
        thought = self.planner.generate_thought(
            query=query,
            observation=observation,
            history=recent_history
        )
        logger.info(f"   Thought: {thought[:100]}...")
        
        # STEP 2: ACT - Generate action
        logger.info("⚡ Generating action...")
        action = self.planner.generate_action(
            query=query,
            observation=observation,
            history=recent_history,
            thought=thought,
            available_skills=available_skills
        )
        logger.info(f"   Action: {action['skill']}({action['params']})")
        logger.info(f"   Confidence: {action.get('confidence', 0):.2f}")
        
        return thought, action
    
    def add_step(
        self,
        step_num: int,
        thought: str,
        action: Dict,
        observation: Dict,
        result: Dict
    ):
        """
        Add step to history.
        
        Args:
            step_num: Step number
            thought: Reasoning thought
            action: Action taken
            observation: State observation
            result: Execution result
        """
        step = ReActStep(
            step_num=step_num,
            thought=thought,
            action=action,
            observation=observation,
            result=result
        )
        
        self.history.append(step)
        
        # Log summary
        status_emoji = "✅" if result.get('status') == 'success' else "❌"
        logger.info(f"{status_emoji} Step {step_num}: {action.get('skill')} - {result.get('status')}")
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get history as list of dictionaries.
        
        Args:
            last_n: Number of recent steps (None = all)
            
        Returns:
            List of step dictionaries
        """
        steps = self.history if last_n is None else self.history[-last_n:]
        
        return [
            {
                'step': s.step_num,
                'thought': s.thought,
                'action': s.action,
                'observation': s.observation,
                'result': s.result,
                'timestamp': s.timestamp
            }
            for s in steps
        ]
    
    def should_continue(self, last_action: Dict) -> bool:
        """
        Check if should continue ReAct loop.
        
        Args:
            last_action: Last action executed
            
        Returns:
            True if should continue, False if should stop
        """
        # Stop if task marked complete
        if last_action.get('skill') == 'complete':
            logger.info("🏁 Task marked as complete")
            return False
        
        # Stop if max steps reached
        if len(self.history) >= self.max_steps:
            logger.warning(f"⚠️  Max steps reached: {self.max_steps}")
            return False
        
        # Continue
        return True
    
    def analyze_progress(self) -> Dict:
        """
        Analyze execution progress.
        
        Returns:
            Progress metrics
        """
        if not self.history:
            return {
                'steps': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0,
                'last_action': None,
                'completion_status': 'not_started'
            }
        
        total = len(self.history)
        successful = sum(
            1 for s in self.history
            if s.result.get('status') == 'success'
        )
        failed = total - successful
        
        last_step = self.history[-1]
        completed = last_step.action.get('skill') == 'complete'
        
        return {
            'steps': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0.0,
            'last_action': last_step.action,
            'completion_status': 'completed' if completed else 'in_progress'
        }
    
    def get_summary(self) -> str:
        """
        Get execution summary as formatted text.
        
        Returns:
            Summary string
        """
        progress = self.analyze_progress()
        
        summary = f"""
{'='*70}
ReAct Execution Summary
{'='*70}
📊 Total steps: {progress['steps']}
✅ Successful: {progress['successful']}
❌ Failed: {progress['failed']}
📈 Success rate: {progress['success_rate']:.1%}
🏁 Status: {progress['completion_status']}
⚡ Last action: {progress.get('last_action', {}).get('skill', 'N/A')}
{'='*70}
"""
        return summary.strip()
    
    def get_trajectory(self) -> List[Dict]:
        """
        Get full trajectory for learning.
        
        Returns:
            List of (state, action, result) tuples
        """
        return [
            {
                'state': {
                    'url': s.observation.get('url', ''),
                    'dom': s.observation.get('dom', '')[:500],  # Truncate
                    'elements_count': len(s.observation.get('elements', []))
                },
                'thought': s.thought,
                'action': s.action,
                'result': s.result,
                'success': s.result.get('status') == 'success'
            }
            for s in self.history
        ]
    
    def detect_loop(self, window: int = 3) -> bool:
        """
        Detect if agent is stuck in a loop.
        
        Args:
            window: Number of recent steps to check
            
        Returns:
            True if loop detected
        """
        if len(self.history) < window * 2:
            return False
        
        recent = self.history[-window:]
        previous = self.history[-window*2:-window]
        
        # Check if same actions repeated
        recent_actions = [s.action['skill'] for s in recent]
        previous_actions = [s.action['skill'] for s in previous]
        
        if recent_actions == previous_actions:
            logger.warning(f"⚠️  Loop detected: {recent_actions}")
            return True
        
        return False
    
    def suggest_recovery(self) -> Dict:
        """
        Suggest recovery action if stuck.
        
        Returns:
            Recovery action
        """
        if self.detect_loop():
            logger.info("🔄 Suggesting recovery: reload page")
            return {
                'skill': 'reload',
                'params': {},
                'reason': 'Loop detected, reloading page'
            }
        
        # If too many failures, suggest screenshot
        progress = self.analyze_progress()
        if progress['success_rate'] < 0.5 and progress['steps'] > 5:
            logger.info("📸 Suggesting recovery: take screenshot")
            return {
                'skill': 'screenshot',
                'params': {'full_page': True},
                'reason': 'Low success rate, taking screenshot for debugging'
            }
        
        return None


# Test & Example
if __name__ == "__main__":
    import asyncio
    
    async def test_react_engine():
        print("=" * 70)
        print("ReAct Engine - Test")
        print("=" * 70 + "\n")
        
        # Initialize
        engine = ReActEngine(max_steps=10)
        
        # Mock observation
        observation = {
            'url': 'https://shopee.vn',
            'dom': '<div class="search"><input id="search-box"/><button>Tìm</button></div>',
            'elements': [
                {'selector': '#search-box', 'tag': 'input', 'text': ''},
                {'selector': 'button', 'tag': 'button', 'text': 'Tìm'}
            ]
        }
        
        # Simulate 3 steps
        query = "Tìm áo khoác giá rẻ"
        
        for i in range(3):
            print(f"\n{'='*70}")
            print(f"Step {i+1}")
            print(f"{'='*70}")
            
            # Generate thought & action
            thought, action = await engine.step(
                query=query,
                observation=observation,
                available_skills=['goto', 'click', 'type', 'complete']
            )
            
            print(f"\n💭 Thought: {thought}")
            print(f"⚡ Action: {action['skill']}({action['params']})")
            
            # Mock result
            result = {
                'status': 'success',
                'message': f'{action["skill"]} completed'
            }
            
            # Add to history
            engine.add_step(
                step_num=i+1,
                thought=thought,
                action=action,
                observation=observation,
                result=result
            )
            
            # Check continue
            if not engine.should_continue(action):
                break
        
        # Print summary
        print(engine.get_summary())
        
        # Analyze progress
        print("\n📊 Progress Analysis:")
        progress = engine.analyze_progress()
        for key, value in progress.items():
            print(f"  {key}: {value}")
    
    asyncio.run(test_react_engine())
