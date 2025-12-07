# main.py
"""Main entry point for the RAG system."""

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


def initialize_system():
    """Initialize all RAG components."""
    print("="*60)
    print("Initializing RAG System")
    print("="*60)
    
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
    
    print("\n✅ System initialized!")
    return hybrid_retriever, llm


def query(question: str, hybrid_retriever, llm):
    """Query the RAG system."""
    result = enhanced_rag(
        query=question,
        hybrid_retriever=hybrid_retriever,
        llm=llm,
        top_k=5
    )
    return result


if __name__ == "__main__":
    # Initialize
    hybrid_retriever, llm = initialize_system()
    
    # Example query
    question = "What is XGBoost?"
    result = query(question, hybrid_retriever, llm)
    
    print("\n" + "="*60)
    print(f"Question: {result['question']}")
    print("="*60)
    print(f"\nAnswer:\n{result['answer']}")