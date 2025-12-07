# src/retrievers/bm25_retriever.py
"""BM25-based retrieval."""

import re
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Retrieves documents using BM25 keyword matching."""
    
    def __init__(self, documents: List[str], chunks: List[Any] = None):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of document text strings
            chunks: Original chunk objects for metadata
        """
        self.documents = documents
        self.chunks = chunks if chunks else []
        
        print("Building BM25 index...")
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"BM25 index built for {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents matching query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of document dictionaries
        """
        query_tokens = re.findall(r'\w+', query.lower())
        
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved_docs = []
        for i, idx in enumerate(top_indices):
            result = {
                "doc_index": int(idx),
                "content": self.documents[idx],
                "bm25_score": float(scores[idx]),
                "rank": i + 1
            }
            
            if self.chunks and idx < len(self.chunks):
                result["metadata"] = self.chunks[idx].metadata
                result["source_file"] = self.chunks[idx].metadata.get('source_file', 'unknown')
                result["page"] = self.chunks[idx].metadata.get('page_label', 'unknown')
            else:
                result["metadata"] = {}
                result["source_file"] = "unknown"
                result["page"] = "unknown"
            
            retrieved_docs.append(result)
        
        return retrieved_docs