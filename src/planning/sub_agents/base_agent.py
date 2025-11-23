"""
Base Sub-Agent - Abstract base class for specialized agents
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseSubAgent(ABC):
    """
    Abstract base class for sub-agents.
    
    **Purpose**: 
        Specialized agents for specific tasks (login, payment, search, etc.)
    
    **Hierarchy**:
        AgentOrchestrator
            ├── PlannerAgent (general)
            └── Sub-Agents (specialized)
                ├── LoginAgent
                ├── PaymentAgent
                ├── SearchAgent
                └── ... (extensible)
    
    **Benefits**:
        - Domain expertise (e.g., login flows, payment forms)
        - Reusable across tasks
        - Easier to maintain & test
        - Modular architecture
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize sub-agent.
        
        Args:
            name: Agent name
            description: Agent description
        """
        self.name = name
        self.description = description
        self.success_count = 0
        self.failure_count = 0
        
        logger.info(f"🤖 {self.name} initialized: {self.description}")
    
    @abstractmethod
    async def can_handle(self, task: Dict, observation: Dict) -> bool:
        """
        Check if this agent can handle the task.
        
        Args:
            task: Task description
            observation: Current state
            
        Returns:
            True if can handle
        """
        pass
    
    @abstractmethod
    async def execute(
        self,
        task: Dict,
        page,
        observation: Dict
    ) -> Dict:
        """
        Execute the specialized task.
        
        Args:
            task: Task to execute
            page: Playwright page
            observation: Current state
            
        Returns:
            Result dict with {success, message, data}
        """
        pass
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        total = self.success_count + self.failure_count
        success_rate = self.success_count / total if total > 0 else 0.0
        
        return {
            'name': self.name,
            'success': self.success_count,
            'failure': self.failure_count,
            'total': total,
            'success_rate': success_rate
        }
    
    def _record_success(self):
        """Record successful execution"""
        self.success_count += 1
        logger.info(f"yes {self.name} succeeded (total: {self.success_count})")
    
    def _record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        logger.warning(f"no {self.name} failed (total: {self.failure_count})")
    
    def __repr__(self):
        return f"<{self.name}: {self.description}>"
