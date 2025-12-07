# src/config.py
"""Configuration settings for the RAG system."""

import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = "./data"
PDF_DIR = f"{DATA_DIR}/pdf"
VECTOR_STORE_DIR = f"{DATA_DIR}/vector_store"

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Vector store settings
COLLECTION_NAME = "pdf-documents"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
DEFAULT_TOP_K = 5
HYBRID_ALPHA = 0.7  # Weight for vector search (0.7 = 70% vector, 30% BM25)

# LLM settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 1024