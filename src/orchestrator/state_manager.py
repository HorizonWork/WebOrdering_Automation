"""
State Manager - Track and manage agent state
Maintains context across execution steps
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class State:
    """Agent state at a point in time"""
    url: str
    dom_hash: str
    elements_count: int
    timestamp: str
    metadata: Dict = field(default_factory=dict)


class StateManager:
    """
    Manages agent state across execution.
    
    **Responsibilities**:
        - Track current state
        - Maintain state history
        - Detect state changes
        - Provide context for planning
    
    **State includes**:
        - Current URL
        - DOM content (hashed)
        - Interactive elements
        - Execution metadata
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize state manager.
        
        Args:
            max_history: Maximum states to keep in history
        """
        self.max_history = max_history
        self.current_state: Optional[State] = None
        self.state_history: List[State] = []
        
        logger.info(f"StateManager initialized (max_history={max_history})")
    
    def update_state(self, observation: Dict):
        """
        Update current state from observation.
        
        Args:
            observation: Observation dict from perception layer
        """
        # Create state
        dom_hash = hash(observation.get('dom', ''))
        
        state = State(
            url=observation.get('url', ''),
            dom_hash=str(dom_hash),
            elements_count=len(observation.get('elements', [])),
            timestamp=observation.get('timestamp', datetime.now().isoformat()),
            metadata={
                'dom_size': len(observation.get('dom', '')),
                'has_screenshot': 'screenshot' in observation
            }
        )
        
        # Add to history
        if self.current_state:
            self.state_history.append(self.current_state)
            
            # Trim history
            if len(self.state_history) > self.max_history:
                self.state_history = self.state_history[-self.max_history:]
        
        self.current_state = state
        
        logger.debug(f"State updated: {state.url} ({state.elements_count} elements)")
    
    def get_current_state(self) -> Optional[State]:
        """Get current state"""
        return self.current_state
    
    def get_state_history(self, last_n: int = 10) -> List[State]:
        """Get last N states"""
        return self.state_history[-last_n:]
    
    def has_state_changed(self) -> bool:
        """Check if state changed from previous"""
        if len(self.state_history) < 1:
            return True
        
        prev_state = self.state_history[-1]
        curr_state = self.current_state
        
        if not curr_state:
            return False
        
        # Compare DOM hash
        return prev_state.dom_hash != curr_state.dom_hash
    
    def get_url_history(self) -> List[str]:
        """Get history of visited URLs"""
        urls = [s.url for s in self.state_history]
        if self.current_state:
            urls.append(self.current_state.url)
        return urls
    
    def is_url_visited(self, url: str) -> bool:
        """Check if URL was visited"""
        return url in self.get_url_history()
    
    def get_context_summary(self) -> Dict:
        """Get summary of current context"""
        return {
            'current_url': self.current_state.url if self.current_state else None,
            'elements_count': self.current_state.elements_count if self.current_state else 0,
            'history_length': len(self.state_history),
            'urls_visited': len(set(self.get_url_history())),
            'state_changed': self.has_state_changed()
        }
    
    def reset(self):
        """Reset state"""
        self.current_state = None
        self.state_history = []
        logger.info("State reset")
    
    def save_to_file(self, path: str):
        """Save state history to file"""
        data = {
            'current_state': self.current_state.__dict__ if self.current_state else None,
            'history': [s.__dict__ for s in self.state_history]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"State saved to {path}")
    
    def load_from_file(self, path: str):
        """Load state history from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Restore current state
        if data['current_state']:
            self.current_state = State(**data['current_state'])
        
        # Restore history
        self.state_history = [State(**s) for s in data['history']]
        
        logger.info(f"State loaded from {path}")


# Test
if __name__ == "__main__":
    print("Testing StateManager...\n")
    
    manager = StateManager()
    
    # Simulate state updates
    for i in range(5):
        observation = {
            'url': f'https://example.com/page{i}',
            'dom': f'<html>Page {i} content</html>',
            'elements': [{'id': j} for j in range(i * 2)],
            'timestamp': datetime.now().isoformat()
        }
        
        manager.update_state(observation)
        print(f"Step {i+1}: {manager.get_current_state().url}")
    
    # Get summary
    print("\nContext Summary:")
    summary = manager.get_context_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save/load
    manager.save_to_file("test_state.json")
    
    manager2 = StateManager()
    manager2.load_from_file("test_state.json")
    print(f"\nLoaded state: {manager2.get_current_state().url}")
    
    # Cleanup
    import os
    os.remove("test_state.json")
    print("\n✅ Test completed!")
