"""
Embedding Module - Generate embeddings for perception layer
Wraps PhoBERT for perception-specific use cases
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.phobert_encoder import PhoBERTEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerceptionEmbedding:
    """
    Embedding generator for perception layer.
    
    **Use Cases**:
        - Encode UI element text
        - Encode page content
        - Generate query embeddings
        - Semantic matching
    
    **Wrapper around PhoBERT**:
        - Provides perception-specific utilities
        - Caching for repeated elements
        - Batch processing optimization
    """
    
    def __init__(self, encoder: Optional[PhoBERTEncoder] = None):
        """
        Initialize perception embedding.
        
        Args:
            encoder: PhoBERT encoder (creates new if None)
        """
        self.encoder = encoder or PhoBERTEncoder()
        self.cache = {}  # Simple cache for repeated elements
        
        logger.info("PerceptionEmbedding initialized")
    
    def encode_ui_element(
        self,
        tag: str,
        text: str,
        attributes: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Encode UI element to embedding.
        
        Args:
            tag: HTML tag name
            text: Element text content
            attributes: Element attributes
            
        Returns:
            Embedding vector (768-dim)
        """
        # Create element description
        desc = f"{tag}"
        
        if text:
            desc += f" {text}"
        
        if attributes:
            # Add important attributes
            for key in ['placeholder', 'aria-label', 'title']:
                if key in attributes:
                    desc += f" {attributes[key]}"
        
        # Check cache
        cache_key = desc
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Encode
        embedding = self.encoder.encode_text(desc)
        
        # Cache result
        self.cache[cache_key] = embedding
        
        return embedding
    
    def encode_element_batch(
        self,
        elements: List[Dict]
    ) -> np.ndarray:
        """
        Encode multiple UI elements.
        
        Args:
            elements: List of element dicts with {tag, text, attributes}
            
        Returns:
            Embeddings array (n_elements, 768)
        """
        # Create descriptions
        descriptions = []
        for elem in elements:
            desc = f"{elem.get('tag', 'div')}"
            
            if elem.get('text'):
                desc += f" {elem['text']}"
            
            attrs = elem.get('attributes', {})
            for key in ['placeholder', 'aria-label', 'title']:
                if key in attrs:
                    desc += f" {attrs[key]}"
            
            descriptions.append(desc)
        
        # Batch encode
        embeddings = self.encoder.encode_text(descriptions)
        
        logger.info(f"yes Encoded {len(elements)} UI elements")
        
        return embeddings
    
    def encode_page_content(
        self,
        title: str,
        heading: str,
        description: str
    ) -> np.ndarray:
        """
        Encode page-level content.
        
        Args:
            title: Page title
            heading: Main heading
            description: Page description
            
        Returns:
            Page embedding
        """
        content = f"{title} {heading} {description}"
        return self.encoder.encode_text(content)
    
    def find_matching_elements(
        self,
        query: str,
        elements: List[Dict],
        top_k: int = 3
    ) -> List[tuple]:
        """
        Find elements matching query.
        
        Args:
            query: Search query
            elements: List of elements with embeddings
            top_k: Number of results
            
        Returns:
            List of (element, score) tuples
        """
        # Encode query
        query_embedding = self.encoder.encode_text(query)
        
        # Compute similarities
        results = []
        for elem in elements:
            if 'embedding' not in elem:
                continue
            
            elem_emb = elem['embedding']
            
            # Cosine similarity
            similarity = np.dot(query_embedding.flatten(), elem_emb.flatten()) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(elem_emb) + 1e-8
            )
            
            results.append((elem, float(similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache = {}
        logger.info("Embedding cache cleared")


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("PerceptionEmbedding - Test")
    print("=" * 70 + "\n")
    
    embedding = PerceptionEmbedding()
    
    # Test 1: Encode single element
    print("Test 1: Encode Single UI Element")
    print("-" * 40)
    
    emb = embedding.encode_ui_element(
        tag='input',
        text='',
        attributes={'placeholder': 'Tìm kiếm sản phẩm'}
    )
    
    print(f"yes Encoded input element")
    print(f"  Shape: {emb.shape}")
    print(f"  First 10 dims: {emb[0, :10]}\n")
    
    # Test 2: Batch encode
    print("\nTest 2: Batch Encode Elements")
    print("-" * 40)
    
    elements = [
        {'tag': 'button', 'text': 'Tìm kiếm', 'attributes': {}},
        {'tag': 'input', 'text': '', 'attributes': {'placeholder': 'Email'}},
        {'tag': 'a', 'text': 'Đăng nhập', 'attributes': {'href': '/login'}}
    ]
    
    embeddings = embedding.encode_element_batch(elements)
    print(f"yes Encoded {len(elements)} elements")
    print(f"  Shape: {embeddings.shape}\n")
    
    # Test 3: Find matching
    print("\nTest 3: Find Matching Elements")
    print("-" * 40)
    
    # Add embeddings to elements
    for i, elem in enumerate(elements):
        elem['embedding'] = embeddings[i]
    
    query = "nút tìm kiếm"
    matches = embedding.find_matching_elements(query, elements, top_k=2)
    
    print(f"Query: '{query}'\n")
    for i, (elem, score) in enumerate(matches, 1):
        print(f"{i}. {elem['tag']} - {elem['text']}")
        print(f"   Score: {score:.4f}\n")
    
    print("=" * 70)
    print("yes All Tests Completed!")
    print("=" * 70)
