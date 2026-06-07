# app_unified.py
"""Unified Streamlit app — PDF upload + chat with conversation history."""

# CRITICAL: Must be FIRST - before any other imports
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


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
    from src.document_loader import process_all_pdfs as _load_pdfs
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
    """Initialize RAG system from pre-built index (cached across sessions)."""
    components = get_rag_components()

    print("Running initialize_system()")

    documents = components['process_all_pdfs'](components['PDF_DIR'])
    chunks = components['split_documents'](documents)

    embedding_manager = components['EmbeddingManager']()
    vectorstore = components['VectorStore']()

    if vectorstore.count() == 0:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_manager.generate_embedding(texts)
        vectorstore.add_documents(chunks, embeddings)

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

    # Return mutable state that session can extend
    return {
        'embedding_manager': embedding_manager,
        'vectorstore': vectorstore,
        'all_chunks': list(chunks),
        'hybrid_retriever': hybrid_retriever,
        'llm': llm,
        'base_doc_names': sorted({
            Path(d.metadata.get('source_file', '')).stem
            for d in documents
            if d.metadata.get('source_file')
        }),
        'components': components
    }


def add_uploaded_pdf(system: dict, pdf_bytes: bytes, filename: str) -> int:
    """
    Process an uploaded PDF, embed it, and add it to the live retriever.
    Returns number of chunks added.
    """
    components = system['components']

    # Write bytes to a temp file so PyMuPDFLoader can read it
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # Load using existing loader
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(tmp_path)
        raw_docs = loader.load()

        # Tag with original filename
        for doc in raw_docs:
            doc.metadata['source_file'] = filename
            doc.metadata['file_type'] = 'pdf'

        # Chunk
        new_chunks = components['split_documents'](raw_docs)

        # Embed and add to vectorstore
        texts = [c.page_content for c in new_chunks]
        embeddings = system['embedding_manager'].generate_embedding(texts)
        system['vectorstore'].add_documents(new_chunks, embeddings)

        # Extend the full chunk list
        system['all_chunks'].extend(new_chunks)

        # Rebuild retrievers with updated chunks
        all_chunks = system['all_chunks']
        vector_retriever = components['VectorRetriever'](
            system['vectorstore'], system['embedding_manager']
        )
        bm25_retriever = components['BM25Retriever'](
            documents=[c.page_content for c in all_chunks],
            chunks=all_chunks
        )
        system['hybrid_retriever'] = components['HybridRetriever'](
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            chunks=all_chunks
        )

        return len(new_chunks)

    finally:
        os.unlink(tmp_path)


def main():
    st.set_page_config(
        page_title="RAG Document QA",
        page_icon="📚",
        layout="wide"
    )

    # ── Initialize system ────────────────────────────────────────────────────
    try:
        system = initialize_system()
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.stop()

    # ── Session state ────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []  # list of filenames uploaded this session

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📚 RAG Document QA")
        st.markdown("---")

        # Document list
        st.subheader("📄 Loaded Documents")
        for name in system['base_doc_names']:
            st.markdown(f"• {name}")

        if st.session_state.uploaded_docs:
            st.markdown("**Uploaded this session:**")
            for name in st.session_state.uploaded_docs:
                st.markdown(f"• 📎 {name}")

        st.markdown("---")

        # File uploader
        st.subheader("➕ Upload a PDF")
        uploaded_file = st.file_uploader(
            "Drop a PDF to add it to the knowledge base",
            type=["pdf"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_docs:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        n_chunks = add_uploaded_pdf(
                            system,
                            uploaded_file.read(),
                            uploaded_file.name
                        )
                        st.session_state.uploaded_docs.append(uploaded_file.name)
                        st.success(f"Added {n_chunks} chunks from {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Failed to process PDF: {e}")
            else:
                st.info(f"{uploaded_file.name} is already loaded.")

        st.markdown("---")

        # Chat controls
        st.subheader("💬 Chat")
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("Hybrid retrieval: Vector + BM25 · Groq Llama 3.1 · ChromaDB")

    # ── Main chat area ───────────────────────────────────────────────────────
    st.header("💬 Chat with your documents")

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources for assistant messages if present
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['source_file']}** — Page {s['page']} "
                            f"(score: {s['combined_score']:.3f})"
                        )

    # Chat input
    if prompt := st.chat_input("Ask anything about your documents…"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                components = get_rag_components()
                result = components['enhanced_rag'](
                    query=prompt,
                    hybrid_retriever=system['hybrid_retriever'],
                    llm=system['llm'],
                    top_k=5,
                    include_sources=False,
                    chat_history=st.session_state.messages[:-1]  # exclude current user msg
                )
                answer = result['raw_answer']
                sources = result['sources']

            st.markdown(answer)

            if sources:
                with st.expander("📎 Sources", expanded=False):
                    for s in sources:
                        st.markdown(
                            f"**{s['source_file']}** — Page {s['page']} "
                            f"(score: {s['combined_score']:.3f})"
                        )

        # Save assistant message with sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


if __name__ == "__main__":
    main()
