# app.py
"""Streamlit frontend for RAG Document QA System."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

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


@st.cache_resource
def initialize_system():
    """Initialize RAG system (cached for performance)."""
    with st.spinner("Initializing RAG system..."):
        # Load and chunk documents
        documents = process_all_pdfs(PDF_DIR)
        chunks = split_documents(documents)
        
        # Initialize components
        embedding_manager = EmbeddingManager()
        vectorstore = VectorStore()
        
        # Add documents if empty
        if vectorstore.count() == 0:
            texts = [chunk.page_content for chunk in chunks]
            embeddings = embedding_manager.generate_embedding(texts)
            vectorstore.add_documents(chunks, embeddings)
        
        # Create retrievers
        vector_retriever = VectorRetriever(vectorstore, embedding_manager)
        bm25_retriever = BM25Retriever(
            documents=[chunk.page_content for chunk in chunks],
            chunks=chunks
        )
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            chunks=chunks
        )
        
        llm = get_llm()
        
        return hybrid_retriever, llm, vectorstore.count()


def main():
    """Main Streamlit app."""
    # Page config
    st.set_page_config(
        page_title="RAG Document QA",
        page_icon="📚",
        layout="wide"
    )
    
    # Title
    st.title("📚 RAG Document QA System")
    st.markdown("Ask questions about your PDF documents using AI-powered retrieval")
    
    # Initialize system (cached)
    hybrid_retriever, llm, doc_count = initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Info")
        st.metric("Documents Indexed", doc_count)
        
        st.header("⚙️ Settings")
        top_k = st.slider("Number of documents to retrieve", 3, 10, 5)
        show_sources = st.checkbox("Show source citations", value=True)
        
        st.header("ℹ️ About")
        st.markdown("""
        This RAG system uses:
        - **Hybrid Retrieval**: Vector + BM25 search
        - **LLM**: Groq (Llama 3.1)
        - **Vector Store**: ChromaDB
        """)
    
    # Main content
    # Query input
    query = st.text_input(
        "💬 Ask a question about your documents:",
        placeholder="e.g., What is XGBoost?",
        key="query_input"
    )
    
    # Submit button
    if st.button("🔍 Search", type="primary") or query:
        if not query:
            st.warning("Please enter a question")
        else:
            # Show query
            st.markdown("### Question")
            st.info(query)
            
            # Get answer
            with st.spinner("Searching documents and generating answer..."):
                result = enhanced_rag(
                    query=query,
                    hybrid_retriever=hybrid_retriever,
                    llm=llm,
                    top_k=top_k,
                    include_sources=show_sources
                )
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(result['answer'])
            
            # Show metadata
            with st.expander("📊 Retrieval Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents Retrieved", result['metadata']['num_results'])
                with col2:
                    st.metric("Avg Relevance Score", f"{result['metadata']['avg_score']:.3f}")
            
            # Show sources
            if result['sources'] and show_sources:
                st.markdown("### 📑 Sources")
                for i, source in enumerate(result['sources'], 1):
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{i}. {source['source_file']}**")
                        with col2:
                            st.caption(f"Page {source['page']}")
                        with col3:
                            st.caption(f"Score: {source['combined_score']:.3f}")
    
    # Example queries
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    example_queries = [
        "What is XGBoost?",
        "How do embeddings work in machine learning?",
        "What is the role of deep learning in pedestrian detection?",
        "Explain attention mechanisms"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.query_input = example
                st.rerun()


if __name__ == "__main__":
    main()