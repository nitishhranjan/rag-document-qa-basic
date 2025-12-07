# src/retrievers/hybrid_retriever.py
"""Hybrid retrieval combining vector and BM25."""

from collections import defaultdict
from typing import List, Dict, Any

from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever
from ..config import HYBRID_ALPHA


class HybridRetriever:
    """Combines vector and BM25 retrieval."""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        chunks: List[Any],
        alpha: float = HYBRID_ALPHA
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_retriever: VectorRetriever instance
            bm25_retriever: BM25Retriever instance
            chunks: Original chunks for metadata
            alpha: Weight for vector search (0-1)
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.chunks = chunks
        self.alpha = alpha
    
    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: int = 20,
        bm25_top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final documents
            vector_top_k: Documents from vector search
            bm25_top_k: Documents from BM25 search
            
        Returns:
            List of document dictionaries with combined scores
        """
        # Get results from both methods
        vector_results = self.vector_retriever.retrieve(query, top_k=vector_top_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)
        
        # Merge by doc_index
        combined = defaultdict(lambda: {
            'doc_index': None,
            'content': None,
            'vector_score': 0.0,
            'bm25_score': 0.0,
            'combined_score': 0.0,
            'metadata': {}
        })
        
        # Add vector results
        for result in vector_results:
            doc_index = result.get("metadata", {}).get("doc_index")
            if doc_index is not None:
                combined[doc_index]['doc_index'] = doc_index
                combined[doc_index]['content'] = result['content']
                combined[doc_index]['vector_score'] = result['similarity_score']
                combined[doc_index]['metadata'] = result['metadata']
        
        # Add BM25 results
        for result in bm25_results:
            doc_index = result['doc_index']
            if doc_index in combined:
                combined[doc_index]['bm25_score'] = result['bm25_score']
            else:
                combined[doc_index]['doc_index'] = doc_index
                combined[doc_index]['content'] = result['content']
                combined[doc_index]['bm25_score'] = result['bm25_score']
                combined[doc_index]['metadata'] = result['metadata']
        
        # Normalize and combine scores
        combined_list = list(combined.items())
        
        vector_scores = [r['vector_score'] for _, r in combined_list]
        bm25_scores = [r['bm25_score'] for _, r in combined_list]
        
        norm_vector = self.normalize_scores(vector_scores)
        norm_bm25 = self.normalize_scores(bm25_scores)
        
        final_results = []
        for i, (doc_index, result) in enumerate(combined_list):
            result['combined_score'] = (
                self.alpha * norm_vector[i] +
                (1 - self.alpha) * norm_bm25[i]
            )
            result['vector_score_normalized'] = norm_vector[i]
            result['bm25_score_normalized'] = norm_bm25[i]
            final_results.append(result)
        
        # Sort and rank
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        for i, result in enumerate(final_results[:top_k]):
            result['rank'] = i + 1
        
        return final_results[:top_k]