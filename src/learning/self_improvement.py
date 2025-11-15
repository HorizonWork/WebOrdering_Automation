"""
Self-Improvement - Learn from experience and improve over time
Analyzes trajectories and updates strategies
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.learning.memory.rail import RAILMemory
from src.learning.error_analyzer import ErrorAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SelfImprovement:
    """
    Self-improvement system for continuous learning.
    
    **Capabilities**:
        - Learn from successful trajectories
        - Analyze failure patterns
        - Update action strategies
        - Track performance metrics
        - Generate improvement recommendations
    
    **Learning Process**:
        1. Collect trajectories (success + failure)
        2. Analyze patterns
        3. Store successful strategies in RAIL
        4. Identify failure causes
        5. Suggest improvements
        6. Update success metrics
    
    **Metrics Tracked**:
        - Success rate over time
        - Common failure modes
        - Skill effectiveness
        - Task completion time
    """
    
    def __init__(
        self,
        rail_memory: Optional[RAILMemory] = None,
        error_analyzer: Optional[ErrorAnalyzer] = None
    ):
        """
        Initialize self-improvement system.
        
        Args:
            rail_memory: RAIL memory for storing experiences
            error_analyzer: Error analyzer for failure analysis
        """
        self.rail = rail_memory or RAILMemory()
        self.error_analyzer = error_analyzer or ErrorAnalyzer()
        
        # Metrics
        self.metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_steps': 0,
            'avg_steps_per_task': 0.0,
            'success_rate': 0.0,
            'skill_success_rates': {}
        }
        
        logger.info("SelfImprovement initialized")
    
    def learn_from_execution(
        self,
        query: str,
        trajectory: Dict,
        success: bool
    ):
        """
        Learn from execution trajectory.
        
        Args:
            query: Original query
            trajectory: Execution trajectory
            success: Whether execution succeeded
        """
        steps = trajectory.get('steps', [])
        
        logger.info(f"📚 Learning from execution: '{query[:50]}...' (success={success})")
        
        # Update metrics
        self._update_metrics(steps, success)
        
        if success:
            # Store successful experience in RAIL
            self.rail.add_experience(
                query=query,
                steps=steps,
                success=True,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'steps_count': len(steps)
                }
            )
            
            logger.info(f"✓ Stored successful strategy for: '{query[:50]}...'")
            
        else:
            # Analyze failure
            analysis = self.error_analyzer.analyze_trajectory({
                'query': query,
                'steps': steps
            })
            
            logger.warning(f"✗ Task failed: {analysis['total_errors']} errors")
            logger.info(f"   Most common issue: {analysis['most_common_category']}")
            logger.info(f"   Suggestions: {len(analysis['recovery_suggestions'])}")
            
            # Still store in RAIL for failure analysis
            self.rail.add_experience(
                query=query,
                steps=steps,
                success=False,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'error_analysis': analysis
                }
            )
    
    def _update_metrics(self, steps: List[Dict], success: bool):
        """Update performance metrics"""
        self.metrics['total_tasks'] += 1
        
        if success:
            self.metrics['successful_tasks'] += 1
        else:
            self.metrics['failed_tasks'] += 1
        
        # Update success rate
        self.metrics['success_rate'] = (
            self.metrics['successful_tasks'] / self.metrics['total_tasks']
        )
        
        # Update steps
        self.metrics['total_steps'] += len(steps)
        self.metrics['avg_steps_per_task'] = (
            self.metrics['total_steps'] / self.metrics['total_tasks']
        )
        
        # Update skill success rates
        for step in steps:
            action = step.get('action', {})
            skill = action.get('skill', 'unknown')
            result = step.get('result', {})
            
            if skill not in self.metrics['skill_success_rates']:
                self.metrics['skill_success_rates'][skill] = {
                    'total': 0,
                    'successful': 0,
                    'rate': 0.0
                }
            
            skill_metrics = self.metrics['skill_success_rates'][skill]
            skill_metrics['total'] += 1
            
            if result.get('status') == 'success':
                skill_metrics['successful'] += 1
            
            skill_metrics['rate'] = (
                skill_metrics['successful'] / skill_metrics['total']
            )
    
    def suggest_improvements(self) -> List[Dict]:
        """
        Generate improvement suggestions based on metrics.
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check overall success rate
        if self.metrics['success_rate'] < 0.5:
            suggestions.append({
                'priority': 'high',
                'category': 'success_rate',
                'issue': f"Low success rate: {self.metrics['success_rate']:.1%}",
                'suggestion': 'Cân nhắc fine-tune ViT5 với dữ liệu domain-specific'
            })
        
        # Check skill effectiveness
        for skill, metrics in self.metrics['skill_success_rates'].items():
            if metrics['total'] >= 5 and metrics['rate'] < 0.6:
                suggestions.append({
                    'priority': 'medium',
                    'category': 'skill_performance',
                    'issue': f"Skill '{skill}' has low success rate: {metrics['rate']:.1%}",
                    'suggestion': f"Cải thiện implementation của skill '{skill}' hoặc thêm retry logic"
                })
        
        # Check error patterns
        error_stats = self.error_analyzer.get_error_statistics()
        if error_stats['total_errors'] > 0:
            most_common_errors = sorted(
                error_stats['by_type'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for error_type, count in most_common_errors:
                if count >= 3:
                    suggestions.append({
                        'priority': 'medium',
                        'category': 'error_prevention',
                        'issue': f"Frequent error: {error_type} ({count} times)",
                        'suggestion': f"Thêm handling đặc biệt cho error type '{error_type}'"
                    })
        
        # Check efficiency
        if self.metrics['avg_steps_per_task'] > 15:
            suggestions.append({
                'priority': 'low',
                'category': 'efficiency',
                'issue': f"High average steps: {self.metrics['avg_steps_per_task']:.1f}",
                'suggestion': 'Tối ưu planning để giảm số bước thực hiện'
            })
        
        logger.info(f"💡 Generated {len(suggestions)} improvement suggestions")
        return suggestions
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        metrics = self.metrics
        
        report = f"""
{'='*70}
Self-Improvement Performance Report
{'='*70}

📊 Overall Performance:
  Total tasks: {metrics['total_tasks']}
  Successful: {metrics['successful_tasks']} ({metrics['success_rate']:.1%})
  Failed: {metrics['failed_tasks']}
  Avg steps per task: {metrics['avg_steps_per_task']:.1f}

🛠️  Skill Performance:
"""
        
        # Sort skills by success rate
        sorted_skills = sorted(
            metrics['skill_success_rates'].items(),
            key=lambda x: x[1]['rate'],
            reverse=True
        )
        
        for skill, skill_metrics in sorted_skills:
            report += f"  - {skill}: {skill_metrics['rate']:.1%} ({skill_metrics['successful']}/{skill_metrics['total']})\n"
        
        # Memory stats
        memory_stats = self.rail.get_statistics()
        report += f"\n📚 Memory Status:\n"
        report += f"  Trajectories stored: {memory_stats['trajectories']['total']}\n"
        report += f"  Successful examples: {memory_stats['trajectories']['successful']}\n"
        report += f"  Embeddings: {memory_stats['embeddings']}\n"
        
        # Error analysis
        error_stats = self.error_analyzer.get_error_statistics()
        report += f"\n⚠️  Error Analysis:\n"
        report += f"  Total errors: {error_stats['total_errors']}\n"
        
        if error_stats['by_category']:
            report += "  Top categories:\n"
            for cat, count in sorted(error_stats['by_category'].items(), key=lambda x: x[1], reverse=True)[:3]:
                report += f"    - {cat}: {count}\n"
        
        # Suggestions
        suggestions = self.suggest_improvements()
        if suggestions:
            report += f"\n💡 Improvement Suggestions:\n"
            for i, sug in enumerate(suggestions[:5], 1):
                report += f"  {i}. [{sug['priority'].upper()}] {sug['issue']}\n"
                report += f"     → {sug['suggestion']}\n"
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def save_state(self, path: str):
        """Save learning state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save RAIL memory
        self.rail.save(path / 'rail_memory')
        
        # Save metrics
        import json
        with open(path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"💾 Self-improvement state saved to {path}")
    
    def load_state(self, path: str):
        """Load learning state"""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Path not found: {path}")
            return
        
        # Load RAIL memory
        rail_path = path / 'rail_memory'
        if rail_path.exists():
            self.rail.load(rail_path)
        
        # Load metrics
        metrics_path = path / 'metrics.json'
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        
        logger.info(f"📂 Self-improvement state loaded from {path}")


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("SelfImprovement - Test")
    print("=" * 70 + "\n")
    
    si = SelfImprovement()
    
    # Simulate learning from executions
    print("Test 1: Learn from Executions")
    print("-" * 40)
    
    # Successful execution
    si.learn_from_execution(
        query="Tìm áo khoác",
        trajectory={
            'steps': [
                {'action': {'skill': 'goto'}, 'result': {'status': 'success'}},
                {'action': {'skill': 'type'}, 'result': {'status': 'success'}},
                {'action': {'skill': 'click'}, 'result': {'status': 'success'}}
            ]
        },
        success=True
    )
    
    # Failed execution
    si.learn_from_execution(
        query="Đăng nhập",
        trajectory={
            'steps': [
                {'action': {'skill': 'type'}, 'result': {'status': 'error', 'message': 'Selector not found'}},
                {'action': {'skill': 'click'}, 'result': {'status': 'error', 'message': 'Timeout'}}
            ]
        },
        success=False
    )
    
    print(f"✓ Learned from 2 executions")
    
    # Test 2: Get suggestions
    print("\n\nTest 2: Improvement Suggestions")
    print("-" * 40)
    
    suggestions = si.suggest_improvements()
    for sug in suggestions:
        print(f"[{sug['priority'].upper()}] {sug['issue']}")
        print(f"  → {sug['suggestion']}\n")
    
    # Test 3: Performance report
    print("\n\nTest 3: Performance Report")
    print("-" * 40)
    
    report = si.get_performance_report()
    print(report)
    
    print("\n✅ All Tests Completed!")
