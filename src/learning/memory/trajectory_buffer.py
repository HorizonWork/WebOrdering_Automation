"""
Trajectory Buffer - Store execution trajectories
Stores (state, action, result) sequences for learning
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import deque
import json
from datetime import datetime

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrajectoryBuffer:
    """
    Buffer for storing execution trajectories.
    
    **Trajectory**: Sequence of (state, action, result) tuples
    
    **Features**:
        - Store successful/failed trajectories
        - Query by task type
        - Filter by success rate
        - Circular buffer (max size)
        - Persistent storage
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize trajectory buffer.
        
        Args:
            max_size: Maximum trajectories to store
        """
        self.max_size = max_size
        self.trajectories = deque(maxlen=max_size)
        
        logger.info(f"TrajectoryBuffer initialized (max_size={max_size})")
    
    def add_trajectory(
        self,
        query: str,
        steps: List[Dict],
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """
        Add trajectory to buffer.
        
        Args:
            query: Original query
            steps: List of execution steps
            success: Whether trajectory succeeded
            metadata: Additional metadata
        """
        trajectory = {
            'query': query,
            'steps': steps,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.trajectories.append(trajectory)
        
        logger.info(f"Added trajectory (success={success}, steps={len(steps)})")
    
    def get_successful_trajectories(
        self,
        query_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Get successful trajectories.
        
        Args:
            query_filter: Filter by query keyword
            top_k: Return top K recent
            
        Returns:
            List of trajectories
        """
        results = [
            t for t in self.trajectories
            if t['success']
        ]
        
        # Filter by query
        if query_filter:
            query_lower = query_filter.lower()
            results = [
                t for t in results
                if query_lower in t['query'].lower()
            ]
        
        # Get top-k recent
        if top_k:
            results = list(reversed(results))[:top_k]
        
        logger.debug(f"Retrieved {len(results)} successful trajectories")
        return results
    
    def get_failed_trajectories(
        self,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Get failed trajectories for analysis"""
        results = [
            t for t in self.trajectories
            if not t['success']
        ]
        
        if top_k:
            results = list(reversed(results))[:top_k]
        
        return results
    
    def get_by_task_type(self, task_type: str) -> List[Dict]:
        """Get trajectories by task type"""
        return [
            t for t in self.trajectories
            if t.get('metadata', {}).get('task_type') == task_type
        ]
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics"""
        if not self.trajectories:
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0
            }
        
        total = len(self.trajectories)
        successful = sum(1 for t in self.trajectories if t['success'])
        failed = total - successful
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0.0
        }
    
    def clear(self):
        """Clear all trajectories"""
        self.trajectories.clear()
        logger.info("TrajectoryBuffer cleared")
    
    def save(self, path: str):
        """Save to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'max_size': self.max_size,
            'trajectories': list(self.trajectories)
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved {len(self.trajectories)} trajectories to {path}")
    
    def load(self, path: str):
        """Load from JSON file"""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.max_size = data.get('max_size', self.max_size)
        self.trajectories = deque(data.get('trajectories', []), maxlen=self.max_size)
        
        logger.info(f"📂 Loaded {len(self.trajectories)} trajectories from {path}")


if __name__ == "__main__":
    print("Testing TrajectoryBuffer...")
    
    buffer = TrajectoryBuffer(max_size=100)
    
    # Add test trajectories
    buffer.add_trajectory(
        query="Tìm áo khoác",
        steps=[
            {'action': 'goto', 'result': 'success'},
            {'action': 'type', 'result': 'success'}
        ],
        success=True,
        metadata={'task_type': 'search'}
    )
    
    buffer.add_trajectory(
        query="Đăng nhập",
        steps=[
            {'action': 'type', 'result': 'failed'}
        ],
        success=False,
        metadata={'task_type': 'login'}
    )
    
    # Get stats
    stats = buffer.get_statistics()
    print(f"yes Stats: {stats}")
    
    # Get successful
    successful = buffer.get_successful_trajectories()
    print(f"yes Successful: {len(successful)}")
    
    print("yes TrajectoryBuffer test passed!")
