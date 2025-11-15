"""
Metrics Tracker - Track performance metrics
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import json

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """
    Tracks performance metrics during execution.
    
    **Metrics**:
        - Execution time per skill
        - Success/failure rates
        - Step counts
        - Resource usage
    """
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_steps': 0,
            'total_time': 0.0,
            'skill_metrics': defaultdict(lambda: {
                'count': 0,
                'success': 0,
                'failure': 0,
                'total_time': 0.0
            }),
            'execution_history': []
        }
        
        logger.info("MetricsTracker initialized")
    
    def record_execution(
        self,
        success: bool,
        steps: int,
        execution_time: float,
        metadata: Optional[Dict] = None
    ):
        """Record execution metrics"""
        self.metrics['total_executions'] += 1
        
        if success:
            self.metrics['successful_executions'] += 1
        else:
            self.metrics['failed_executions'] += 1
        
        self.metrics['total_steps'] += steps
        self.metrics['total_time'] += execution_time
        
        # Add to history
        self.metrics['execution_history'].append({
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'steps': steps,
            'execution_time': execution_time,
            'metadata': metadata or {}
        })
        
        logger.debug(f"Recorded execution: success={success}, steps={steps}, time={execution_time:.2f}s")
    
    def record_skill(
        self,
        skill_name: str,
        success: bool,
        execution_time: float
    ):
        """Record skill execution"""
        skill_metrics = self.metrics['skill_metrics'][skill_name]
        
        skill_metrics['count'] += 1
        
        if success:
            skill_metrics['success'] += 1
        else:
            skill_metrics['failure'] += 1
        
        skill_metrics['total_time'] += execution_time
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        total = self.metrics['total_executions']
        
        if total == 0:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'avg_steps': 0.0,
                'avg_time': 0.0
            }
        
        return {
            'total_executions': total,
            'successful': self.metrics['successful_executions'],
            'failed': self.metrics['failed_executions'],
            'success_rate': self.metrics['successful_executions'] / total,
            'total_steps': self.metrics['total_steps'],
            'avg_steps': self.metrics['total_steps'] / total,
            'total_time': self.metrics['total_time'],
            'avg_time': self.metrics['total_time'] / total
        }
    
    def get_skill_stats(self) -> Dict:
        """Get per-skill statistics"""
        stats = {}
        
        for skill, metrics in self.metrics['skill_metrics'].items():
            count = metrics['count']
            if count > 0:
                stats[skill] = {
                    'count': count,
                    'success_rate': metrics['success'] / count,
                    'avg_time': metrics['total_time'] / count
                }
        
        return stats
    
    def get_report(self) -> str:
        """Generate metrics report"""
        summary = self.get_summary()
        skill_stats = self.get_skill_stats()
        
        report = f"""
{'='*70}
Performance Metrics Report
{'='*70}

📊 Overall Statistics:
  Total executions: {summary['total_executions']}
  Successful: {summary['successful']}
  Failed: {summary['failed']}
  Success rate: {summary['success_rate']:.1%}
  
  Total steps: {summary['total_steps']}
  Avg steps per execution: {summary['avg_steps']:.1f}
  
  Total time: {summary['total_time']:.2f}s
  Avg time per execution: {summary['avg_time']:.2f}s

🛠️  Skill Statistics:
"""
        
        # Sort skills by usage
        sorted_skills = sorted(
            skill_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for skill, stats in sorted_skills[:10]:  # Top 10
            report += f"  {skill}:\n"
            report += f"    Count: {stats['count']}\n"
            report += f"    Success rate: {stats['success_rate']:.1%}\n"
            report += f"    Avg time: {stats['avg_time']:.3f}s\n"
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def save(self, path: str):
        """Save metrics to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdict to dict
        metrics_copy = dict(self.metrics)
        metrics_copy['skill_metrics'] = dict(metrics_copy['skill_metrics'])
        
        with open(path, 'w') as f:
            json.dump(metrics_copy, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Metrics saved to {path}")
    
    def load(self, path: str):
        """Load metrics from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.metrics.update(data)
        self.metrics['skill_metrics'] = defaultdict(
            lambda: {'count': 0, 'success': 0, 'failure': 0, 'total_time': 0.0},
            data.get('skill_metrics', {})
        )
        
        logger.info(f"📂 Metrics loaded from {path}")
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()
        logger.info("Metrics reset")


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("MetricsTracker - Test")
    print("=" * 70 + "\n")
    
    tracker = MetricsTracker()
    
    # Record some executions
    tracker.record_execution(success=True, steps=5, execution_time=10.5)
    tracker.record_execution(success=True, steps=3, execution_time=5.2)
    tracker.record_execution(success=False, steps=2, execution_time=3.1)
    
    # Record skills
    tracker.record_skill('goto', success=True, execution_time=2.5)
    tracker.record_skill('click', success=True, execution_time=0.5)
    tracker.record_skill('type', success=True, execution_time=1.0)
    tracker.record_skill('click', success=False, execution_time=0.3)
    
    # Get summary
    print("Summary:")
    summary = tracker.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get report
    print("\n" + tracker.get_report())
    
    print("✅ MetricsTracker test completed!")
