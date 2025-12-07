# src/embeddings.py
"""Embedding management."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


class EmbeddingManager:
    """Manages document and query embeddings."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            print(f"Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded! Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()