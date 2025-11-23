"""
RAIL Memory - Retrieval-Augmented In-context Learning
Combines vector store + trajectory buffer for few-shot learning
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.learning.memory.vector_store import VectorStore
from src.learning.memory.trajectory_buffer import TrajectoryBuffer
from src.models.phobert_encoder import PhoBERTEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAILMemory:
    """
    RAIL (Retrieval-Augmented In-context Learning) Memory.
    
    **Based on**: Agent-E and WebVoyager papers
    
    **How it works**:
        1. Store past trajectories with embeddings
        2. Given new query, retrieve similar past examples
        3. Use examples as few-shot context for planning
        4. Learn from successful executions
    
    **Components**:
        - VectorStore: Embedding-based retrieval
        - TrajectoryBuffer: Store execution history
        - PhoBERT: Encode queries for similarity
    
    **Benefits**:
        - Few-shot learning (no fine-tuning needed)
        - Learn from experience
        - Improve over time
        - Reuse successful strategies
    """
    
    def __init__(
        self,
        encoder: Optional[PhoBERTEncoder] = None,
        max_trajectories: int = 1000
    ):
        """
        Initialize RAIL memory.
        
        Args:
            encoder: PhoBERT encoder (creates new if None)
            max_trajectories: Max trajectories to store
        """
        self.encoder = encoder or PhoBERTEncoder()
        self.vector_store = VectorStore(dimension=768, metric='cosine')
        self.trajectory_buffer = TrajectoryBuffer(max_size=max_trajectories)
        
        logger.info(f"RAILMemory initialized (max_trajectories={max_trajectories})")
    
    def add_experience(
        self,
        query: str,
        steps: List[Dict],
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """
        Add execution experience to memory.
        
        Args:
            query: Original query
            steps: Execution steps
            success: Whether succeeded
            metadata: Additional info
        """
        # Encode query
        query_embedding = self.encoder.encode_text(query)
        
        # Add to trajectory buffer
        self.trajectory_buffer.add_trajectory(
            query=query,
            steps=steps,
            success=success,
            metadata=metadata
        )
        
        # Add to vector store (only successful ones)
        if success:
            self.vector_store.add(
                embedding=query_embedding,
                metadata={
                    'query': query,
                    'steps_count': len(steps),
                    'success': success,
                    'metadata': metadata or {}
                }
            )
            
            logger.info(f"yes Added successful experience: '{query[:50]}...'")
    
    def retrieve_similar_examples(
        self,
        query: str,
        top_k: int = 3,
        only_successful: bool = True
    ) -> List[Dict]:
        """
        Retrieve similar past examples for few-shot learning.
        
        Args:
            query: Current query
            top_k: Number of examples
            only_successful: Only return successful examples
            
        Returns:
            List of similar trajectories
        """
        # Encode query
        query_embedding = self.encoder.encode_text(query)
        
        # Search vector store
        search_results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 2  # Get more to filter
        )
        
        # Get full trajectories
        examples = []
        for idx, score, meta in search_results:
            # Find trajectory in buffer
            matching_trajs = [
                t for t in self.trajectory_buffer.trajectories
                if t['query'] == meta['query']
            ]
            
            if matching_trajs:
                traj = matching_trajs[-1]  # Most recent
                
                if only_successful and not traj['success']:
                    continue
                
                examples.append({
                    'query': traj['query'],
                    'steps': traj['steps'],
                    'similarity_score': score,
                    'success': traj['success']
                })
                
                if len(examples) >= top_k:
                    break
        
        logger.info(f"📚 Retrieved {len(examples)} similar examples for: '{query[:50]}...'")
        return examples
    
    def format_few_shot_context(
        self,
        query: str,
        examples: List[Dict]
    ) -> str:
        """
        Format examples as few-shot context for planner.
        
        Args:
            query: Current query
            examples: Retrieved examples
            
        Returns:
            Formatted context string
        """
        if not examples:
            return f"Nhiệm vụ: {query}\n"
        
        context = "Dưới đây là các ví dụ tương tự đã thực hiện thành công:\n\n"
        
        for i, example in enumerate(examples, 1):
            context += f"Ví dụ {i}:\n"
            context += f"  Truy vấn: {example['query']}\n"
            context += f"  Các bước:\n"
            
            for j, step in enumerate(example['steps'][:5], 1):  # Max 5 steps
                action = step.get('action', {})
                context += f"    {j}. {action.get('skill', 'unknown')}({action.get('params', {})})\n"
            
            context += "\n"
        
        context += f"Bây giờ hãy thực hiện nhiệm vụ: {query}\n"
        
        return context
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        traj_stats = self.trajectory_buffer.get_statistics()
        
        return {
            'trajectories': traj_stats,
            'embeddings': self.vector_store.size(),
            'retrieval_ready': self.vector_store.size() > 0
        }
    
    def save(self, path: str):
        """Save memory to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(path / 'vector_store')
        
        # Save trajectory buffer
        self.trajectory_buffer.save(path / 'trajectories.json')
        
        logger.info(f"💾 RAILMemory saved to {path}")
    
    def load(self, path: str):
        """Load memory from disk"""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Path not found: {path}")
            return
        
        # Load vector store
        vs_path = path / 'vector_store'
        if vs_path.exists():
            self.vector_store.load(vs_path)
        
        # Load trajectory buffer
        traj_path = path / 'trajectories.json'
        if traj_path.exists():
            self.trajectory_buffer.load(traj_path)
        
        logger.info(f"📂 RAILMemory loaded from {path}")


if __name__ == "__main__":
    print("Testing RAILMemory...")
    
    rail = RAILMemory()
    
    # Add experiences
    rail.add_experience(
        query="Tìm áo khoác giá rẻ",
        steps=[
            {'action': {'skill': 'goto', 'params': {'url': 'shopee.vn'}}},
            {'action': {'skill': 'type', 'params': {'text': 'áo khoác'}}}
        ],
        success=True
    )
    
    rail.add_experience(
        query="Tìm giày thể thao",
        steps=[
            {'action': {'skill': 'goto', 'params': {'url': 'shopee.vn'}}},
            {'action': {'skill': 'type', 'params': {'text': 'giày'}}}
        ],
        success=True
    )
    
    # Retrieve similar
    examples = rail.retrieve_similar_examples("Tìm áo sơ mi", top_k=2)
    print(f"yes Retrieved {len(examples)} examples")
    
    # Format context
    context = rail.format_few_shot_context("Tìm áo sơ mi", examples)
    print(f"yes Context length: {len(context)} chars")
    
    # Stats
    stats = rail.get_statistics()
    print(f"yes Stats: {stats}")
    
    print("yes RAILMemory test passed!")
