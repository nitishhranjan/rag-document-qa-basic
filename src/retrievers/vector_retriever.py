# src/retrievers/vector_retriever.py
"""Vector-based retrieval."""

from typing import List, Dict, Any

from ..embeddings import EmbeddingManager
from ..vectorstore import VectorStore


class VectorRetriever:
    """Retrieves documents using vector similarity search."""
    
    def __init__(self, vectorstore: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever.
        
        Args:
            vectorstore: VectorStore instance
            embedding_manager: EmbeddingManager instance
        """
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents similar to query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            List of document dictionaries
        """
        query_embedding = self.embedding_manager.generate_embedding([query])[0]
        
        try:
            results = self.vectorstore.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc, metadata, distance, doc_id) in enumerate(
                    zip(documents, metadatas, distances, ids)
                ):
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": doc,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []