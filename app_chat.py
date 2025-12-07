# app_chat.py
"""Streamlit chat interface for RAG Document QA System."""

# CRITICAL: Must be FIRST - before any other imports
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


# Lazy import function to delay loading torch-related modules
def get_rag_components():
    """Lazy import to avoid Streamlit/PyTorch conflict."""
    from src import (
        process_all_pdfs,
        split_documents,
        EmbeddingManager,
        VectorStore,
        VectorRetriever,
        BM25Retriever,
        HybridRetriever,
        enhanced_rag,
        get_llm
    )
    from src.config import PDF_DIR
    return {
        'process_all_pdfs': process_all_pdfs,
        'split_documents': split_documents,
        'EmbeddingManager': EmbeddingManager,
        'VectorStore': VectorStore,
        'VectorRetriever': VectorRetriever,
        'BM25Retriever': BM25Retriever,
        'HybridRetriever': HybridRetriever,
        'enhanced_rag': enhanced_rag,
        'get_llm': get_llm,
        'PDF_DIR': PDF_DIR
    }


@st.cache_resource
def initialize_system():
    """Initialize RAG system (cached for performance)."""
    # Import here to avoid early conflicts
    components = get_rag_components()
    
    with st.spinner("Initializing RAG system..."):
        # Load and chunk documents
        documents = components['process_all_pdfs'](components['PDF_DIR'])
        chunks = components['split_documents'](documents)
        
        # Initialize components
        embedding_manager = components['EmbeddingManager']()
        vectorstore = components['VectorStore']()
        
        # Add documents if empty
        if vectorstore.count() == 0:
            texts = [chunk.page_content for chunk in chunks]
            embeddings = embedding_manager.generate_embedding(texts)
            vectorstore.add_documents(chunks, embeddings)
        
        # Create retrievers
        vector_retriever = components['VectorRetriever'](vectorstore, embedding_manager)
        bm25_retriever = components['BM25Retriever'](
            documents=[chunk.page_content for chunk in chunks],
            chunks=chunks
        )
        hybrid_retriever = components['HybridRetriever'](
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            chunks=chunks
        )
        
        llm = components['get_llm']()
        
        return hybrid_retriever, llm


def main():
    """Main Streamlit chat app."""
    st.set_page_config(
        page_title="RAG Chat",
        page_icon="💬",
        layout="wide"
    )
    st.title("💬 RAG Document Chat")
    st.markdown("Chat with your documents using AI-powered retrieval")
    
    # Initialize system (cached)
    try:
        hybrid_retriever, llm = initialize_system()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                components = get_rag_components()
                result = components['enhanced_rag'](
                    query=prompt,
                    hybrid_retriever=hybrid_retriever,
                    llm=llm,
                    top_k=5,
                    include_sources=False  # Don't show sources in chat for cleaner UI
                )
                answer = result['raw_answer']  # Use raw_answer without citations
                st.markdown(answer)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Sidebar with clear chat option
    with st.sidebar:
        st.header("💬 Chat Options")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This chat interface uses:
        - **Hybrid Retrieval**: Vector + BM25 search
        - **LLM**: Groq (Llama 3.1)
        - **Vector Store**: ChromaDB
        """)


if __name__ == "__main__":
    main()