"""
Vector Store - Embedding Storage and Retrieval
Uses FAISS for efficient similarity search
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️  FAISS not available, using fallback numpy search")


class VectorStore:
    """
    Vector store for embedding-based retrieval.
    
    **Features**:
        - Add embeddings with metadata
        - Similarity search (cosine/L2)
        - Batch operations
        - Persistent storage
        - FAISS indexing
    """
    
    def __init__(
        self,
        dimension: int = 768,
        metric: str = 'cosine'
    ):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (768 for PhoBERT)
            metric: Distance metric ('cosine', 'l2')
        """
        self.dimension = dimension
        self.metric = metric
        
        # Storage
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.index = None
        
        # Initialize index
        self._init_index()
        
        logger.info(f"VectorStore initialized (dim={dimension}, metric={metric})")
    
    def _init_index(self):
        """Initialize FAISS index"""
        if not FAISS_AVAILABLE:
            return
        
        try:
            if self.metric == 'cosine':
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            
            logger.info("yes FAISS index created")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            self.index = None
    
    def add(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """Add single embedding"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Normalize for cosine
        if self.metric == 'cosine':
            embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
        
        # Add to storage
        self.embeddings.append(embedding.squeeze())
        self.metadata.append(metadata or {})
        
        # Add to index
        if self.index is not None:
            self.index.add(embedding.astype('float32'))
        
        logger.debug(f"Added embedding (total: {len(self.embeddings)})")
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict]
    ):
        """Add multiple embeddings"""
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have same length")
        
        # Normalize
        if self.metric == 'cosine':
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        # Add to storage
        for emb, meta in zip(embeddings, metadata_list):
            self.embeddings.append(emb)
            self.metadata.append(meta)
        
        # Add to index
        if self.index is not None:
            self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Added {len(embeddings)} embeddings (total: {len(self.embeddings)})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """Search for similar embeddings"""
        if len(self.embeddings) == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize
        if self.metric == 'cosine':
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Search
        if self.index is not None:
            distances, indices = self.index.search(
                query_embedding.astype('float32'),
                min(top_k, len(self.embeddings))
            )
            
            results = [
                (int(idx), float(dist), self.metadata[idx])
                for dist, idx in zip(distances[0], indices[0])
                if idx >= 0
            ]
        else:
            # Fallback numpy search
            embeddings_array = np.vstack(self.embeddings)
            
            if self.metric == 'cosine':
                scores = np.dot(embeddings_array, query_embedding.T).squeeze()
            else:
                scores = -np.linalg.norm(embeddings_array - query_embedding, axis=1)
            
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results = [
                (int(idx), float(scores[idx]), self.metadata[idx])
                for idx in top_indices
            ]
        
        return results
    
    def size(self) -> int:
        """Get number of stored embeddings"""
        return len(self.embeddings)
    
    def clear(self):
        """Clear all embeddings"""
        self.embeddings = []
        self.metadata = []
        self._init_index()
        logger.info("VectorStore cleared")
    
    def save(self, path: str):
        """Save to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self.embeddings:
            embeddings_array = np.vstack(self.embeddings)
            np.save(path / 'embeddings.npy', embeddings_array)
        
        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # Save index
        if self.index is not None and FAISS_AVAILABLE:
            faiss.write_index(self.index, str(path / 'index.faiss'))
        
        logger.info(f"💾 VectorStore saved to {path}")
    
    def load(self, path: str):
        """Load from disk"""
        path = Path(path)
        
        # Load embeddings
        emb_path = path / 'embeddings.npy'
        if emb_path.exists():
            embeddings_array = np.load(emb_path)
            self.embeddings = [emb for emb in embeddings_array]
        
        # Load metadata
        meta_path = path / 'metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load index
        index_path = path / 'index.faiss'
        if index_path.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(index_path))
        elif self.embeddings:
            self._init_index()
            if self.index is not None:
                emb_array = np.vstack(self.embeddings).astype('float32')
                self.index.add(emb_array)
        
        logger.info(f"📂 VectorStore loaded from {path} ({len(self.embeddings)} embeddings)")


if __name__ == "__main__":
    print("Testing VectorStore...")
    store = VectorStore(dimension=768)
    
    # Add test data
    for i in range(5):
        emb = np.random.randn(768)
        store.add(emb, {'id': i, 'text': f'Example {i}'})
    
    print(f"yes Added 5 embeddings")
    
    # Search
    query = np.random.randn(768)
    results = store.search(query, top_k=3)
    print(f"yes Search returned {len(results)} results")
    
    print("yes VectorStore test passed!")
