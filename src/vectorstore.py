# src/vectorstore.py
"""Vector store management."""

import os
import uuid
from typing import List, Any
import numpy as np
import chromadb

from .config import COLLECTION_NAME, VECTOR_STORE_DIR


class VectorStore:
    """Manages document embeddings in ChromaDB."""
    
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persistent_directory: str = VECTOR_STORE_DIR
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection
            persistent_directory: Path for persistent storage
        """
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.persistent_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persistent_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "Description": "RAG document collection",
                    "hnsw:space": "cosine"
                }
            )
            print(f"Vector store initialized: {self.collection_name}")
            print(f"Existing documents: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents
            embeddings: Numpy array of embeddings
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have same length")
        
        print(f"Adding {len(documents)} documents...")
        
        ids = []
        metadatas = []
        documents_text = []
        embedding_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            documents_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents_text,
                metadatas=metadatas,
                embeddings=embedding_list
            )
            print(f"Added {len(documents)} documents. Total: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
    
    def count(self) -> int:
        """Get document count."""
        return self.collection.count()