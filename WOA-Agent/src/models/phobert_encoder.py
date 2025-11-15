"""
PhoBERT Encoder - Vietnamese Text Embedding
CRITICAL: This is for ENCODING ONLY, NOT text generation
Model: vinai/phobert-base-v2 (135M parameters, 768-dim embeddings)
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Optional
import numpy as np
from pathlib import Path
import yaml

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhoBERTEncoder:
    """
    PhoBERT wrapper for Vietnamese text embedding extraction.
    
    **CRITICAL DISTINCTION**:
        ✅ CORRECT: Use PhoBERT for ENCODING Vietnamese text → 768-dim vectors
        ❌ WRONG: Do NOT use PhoBERT for text GENERATION (use ViT5 instead)
    
    **Why?** PhoBERT is encoder-only (like BERT/RoBERTa), not encoder-decoder
    
    **Use Cases**:
        1. Encode user queries → semantic vectors
        2. Encode UI element text → vectors for matching
        3. Compute similarity between texts
        4. Vector database storage (RAIL memory)
        5. Few-shot retrieval
    
    **Architecture**:
        - Model: vinai/phobert-base-v2
        - Parameters: 135M
        - Output: 768-dimensional vectors
        - Input: Vietnamese text (max 256 tokens by default)
    
    **Performance**:
        - Encoding: ~50ms per text (GPU)
        - Batch encoding: ~200 texts/second (GPU, batch_size=32)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize PhoBERT encoder.
        
        Args:
            model_name: PhoBERT model identifier (default: vinai/phobert-base-v2)
            device: cuda/cpu/mps (default from settings)
            cache_dir: Model cache directory
        """
        # Load config from YAML if exists
        config_path = Path("config/models.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                phobert_config = config.get('phobert', {})
        else:
            phobert_config = {}
        
        # Settings
        self.model_name = model_name or phobert_config.get('model_name', 'vinai/phobert-base-v2')
        self.device = device or settings.device
        self.max_length = phobert_config.get('max_length', 256)
        self.batch_size = phobert_config.get('batch_size', 32)
        self.normalize = phobert_config.get('normalize', True)
        self.cache_dir = cache_dir or phobert_config.get('cache_dir', './checkpoints/phobert')
        
        logger.info(f"🚀 Loading PhoBERT from {self.model_name}")
        logger.info(f"📍 Device: {self.device}")
        logger.info(f"📏 Max length: {self.max_length}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        logger.info(f"🔄 Normalize: {self.normalize}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        logger.info("✓ Tokenizer loaded")
        
        # Load model
        logger.info("Loading model...")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        ).to(self.device)
        self.model.eval()
        logger.info("✓ Model loaded")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ PhoBERT ready!")
        logger.info(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        logger.info(f"   Embedding dim: 768")
    
    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: Optional[bool] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode Vietnamese text to embeddings (768-dim vectors).
        
        Args:
            texts: Single text or list of texts
            normalize: L2-normalize embeddings (default from config)
            show_progress: Show progress during encoding
            
        Returns:
            numpy array of shape (n_texts, 768)
            
        Example:
            >>> encoder = PhoBERTEncoder()
            >>> emb = encoder.encode_text("Tìm áo khoác giá rẻ")
            >>> emb.shape
            (1, 768)
            
            >>> embs = encoder.encode_text(["Text 1", "Text 2", "Text 3"])
            >>> embs.shape
            (3, 768)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Use config normalize if not specified
        if normalize is None:
            normalize = self.normalize
        
        embeddings = []
        total_texts = len(texts)
        
        # Process in batches
        for i in range(0, total_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_texts + self.batch_size - 1) // self.batch_size
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings (use [CLS] token, first token)
            outputs = self.model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
            
            if show_progress:
                progress = min(i + self.batch_size, total_texts)
                logger.info(f"📊 Batch {batch_num}/{total_batches}: {progress}/{total_texts} texts")
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        
        # L2 normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        logger.info(f"✓ Encoded {total_texts} texts → shape {embeddings.shape}")
        return embeddings
    
    def compute_similarity(
        self,
        query: str,
        candidates: List[str]
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            
        Returns:
            Array of similarity scores (0-1), shape (n_candidates,)
            
        Example:
            >>> encoder = PhoBERTEncoder()
            >>> similarities = encoder.compute_similarity(
            ...     query="Tìm áo khoác",
            ...     candidates=["Áo khoác nam", "Quần jean", "Giày"]
            ... )
            >>> similarities
            array([0.87, 0.32, 0.21])
        """
        # Encode with normalization for cosine similarity
        query_emb = self.encode_text(query, normalize=True)
        candidate_embs = self.encode_text(candidates, normalize=True)
        
        # Cosine similarity = dot product (when normalized)
        similarities = np.dot(candidate_embs, query_emb.T).squeeze()
        
        # Handle single candidate case
        if isinstance(similarities, (float, np.floating)):
            similarities = np.array([similarities])
        
        return similarities
    
    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find top-k most similar candidates to query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top matches to return
            
        Returns:
            List of (index, text, score) tuples, sorted by score (descending)
            
        Example:
            >>> encoder = PhoBERTEncoder()
            >>> matches = encoder.find_most_similar(
            ...     query="Tìm áo khoác màu đen",
            ...     candidates=["Áo khoác đen", "Áo khoác đỏ", "Quần jean"],
            ...     top_k=2
            ... )
            >>> for idx, text, score in matches:
            ...     print(f"{text}: {score:.4f}")
            Áo khoác đen: 0.9234
            Áo khoác đỏ: 0.7856
        """
        # Compute similarities
        similarities = self.compute_similarity(query, candidates)
        
        # Get top-k indices
        top_k = min(top_k, len(candidates))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = [
            (int(idx), candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def batch_similarity_matrix(
        self,
        texts_a: List[str],
        texts_b: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between two lists of texts.
        
        Args:
            texts_a: First list of texts
            texts_b: Second list of texts
            
        Returns:
            Similarity matrix of shape (len(texts_a), len(texts_b))
            
        Example:
            >>> encoder = PhoBERTEncoder()
            >>> queries = ["Tìm áo", "Tìm giày"]
            >>> items = ["Áo khoác", "Giày thể thao", "Quần"]
            >>> matrix = encoder.batch_similarity_matrix(queries, items)
            >>> matrix.shape
            (2, 3)
        """
        # Encode both lists
        embs_a = self.encode_text(texts_a, normalize=True)
        embs_b = self.encode_text(texts_b, normalize=True)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(embs_a, embs_b.T)
        
        logger.info(f"✓ Similarity matrix: {similarity_matrix.shape}")
        return similarity_matrix
    
    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Numpy array of embeddings
            path: File path (.npy)
        """
        np.save(path, embeddings)
        logger.info(f"💾 Saved embeddings to {path} ({embeddings.shape})")
    
    def load_embeddings(self, path: str) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            path: File path (.npy)
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = np.load(path)
        logger.info(f"📂 Loaded embeddings from {path} ({embeddings.shape})")
        return embeddings


# Test & Usage Examples
if __name__ == "__main__":
    print("=" * 70)
    print("PhoBERT Encoder - Test & Examples")
    print("=" * 70 + "\n")
    
    # Test data
    user_queries = [
        "Tìm áo khoác nam màu đen giá dưới 500k",
        "Mua giày thể thao chạy bộ Nike",
        "Tìm quán cà phê gần Quận 1"
    ]
    
    ui_elements = [
        "Áo khoác nam màu đen",
        "Áo khoác nữ màu đỏ",
        "Quần jean nam xanh",
        "Giày thể thao chạy bộ Nike",
        "Balo du lịch chống nước",
        "Áo sơ mi công sở trắng"
    ]
    
    # Initialize encoder
    print("Initializing PhoBERT Encoder...\n")
    encoder = PhoBERTEncoder()
    
    # Test 1: Encode single query
    print("=" * 70)
    print("Test 1: Encode Single Query")
    print("=" * 70)
    query = user_queries[0]
    query_emb = encoder.encode_text(query)
    print(f"\n📝 Query: {query}")
    print(f"📊 Embedding shape: {query_emb.shape}")
    print(f"🔢 First 10 dimensions: {query_emb[0, :10]}")
    print(f"📏 Vector norm: {np.linalg.norm(query_emb[0]):.4f}")
    print(f"✅ Normalized: {np.abs(np.linalg.norm(query_emb[0]) - 1.0) < 0.01}\n")
    
    # Test 2: Batch encode UI elements
    print("=" * 70)
    print("Test 2: Batch Encode UI Elements")
    print("=" * 70)
    ui_embs = encoder.encode_text(ui_elements, show_progress=True)
    print(f"\n📊 Encoded {len(ui_elements)} UI elements")
    print(f"📊 Embeddings shape: {ui_embs.shape}\n")
    
    # Test 3: Semantic similarity matching
    print("=" * 70)
    print("Test 3: Semantic Similarity Matching")
    print("=" * 70)
    similarities = encoder.compute_similarity(query, ui_elements)
    print(f"\n📝 Query: {query}\n")
    print("🎯 Similarity scores:")
    for idx, (element, score) in enumerate(zip(ui_elements, similarities)):
        emoji = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
        print(f"  {emoji} [{idx}] {element:<30} → {score:.4f}")
    
    # Test 4: Find top-k matches
    print("\n" + "=" * 70)
    print("Test 4: Find Top-3 Matches")
    print("=" * 70)
    matches = encoder.find_most_similar(query, ui_elements, top_k=3)
    print(f"\n📝 Query: {query}\n")
    print("🏆 Top 3 matches:")
    for rank, (idx, text, score) in enumerate(matches, 1):
        print(f"  {rank}. [{idx}] {text:<30} → {score:.4f}")
    
    # Test 5: Batch similarity matrix
    print("\n" + "=" * 70)
    print("Test 5: Batch Similarity Matrix")
    print("=" * 70)
    matrix = encoder.batch_similarity_matrix(user_queries[:2], ui_elements[:3])
    print(f"\n📊 Matrix shape: {matrix.shape}")
    print("📊 Similarity matrix:")
    print(matrix)
    
    # Test 6: Save/Load embeddings
    print("\n" + "=" * 70)
    print("Test 6: Save & Load Embeddings")
    print("=" * 70)
    cache_path = "test_embeddings.npy"
    encoder.save_embeddings(ui_embs, cache_path)
    loaded_embs = encoder.load_embeddings(cache_path)
    print(f"\n✅ Embeddings match: {np.allclose(ui_embs, loaded_embs)}")
    
    # Cleanup
    import os
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"🗑️  Cleaned up test file: {cache_path}")
    
    print("\n" + "=" * 70)
    print("✅ All Tests Completed Successfully!")
    print("=" * 70)
