# src/rag_pipeline.py
"""RAG pipeline functions."""

from typing import Dict, Any
from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from .retrievers import HybridRetriever


# Prompt template
ENHANCED_PROMPT = """You are an expert assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "I don't have enough information."
3. Be concise and accurate
4. Mention sources when referencing information

Answer:"""


def get_llm():
    """Get LLM instance."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )


def enhanced_rag(
    query: str,
    hybrid_retriever: HybridRetriever,
    llm=None,
    top_k: int = 5,
    include_sources: bool = True
) -> Dict[str, Any]:
    """
    Enhanced RAG pipeline with hybrid retrieval.
    
    Args:
        query: User question
        hybrid_retriever: HybridRetriever instance
        llm: Language model (optional, will create if not provided)
        top_k: Number of documents to retrieve
        include_sources: Whether to include citations
        
    Returns:
        Dictionary with answer, sources, and metadata
    """
    if llm is None:
        llm = get_llm()
    
    # Retrieve
    results = hybrid_retriever.retrieve(query, top_k=top_k * 2)
    results = results[:top_k]
    
    if not results:
        return {
            'question': query,
            'answer': "I don't have enough information to answer.",
            'sources': [],
            'metadata': {'num_results': 0}
        }
    
    # Build context
    context_parts = []
    sources = []
    
    for i, result in enumerate(results, 1):
        source_file = result.get('metadata', {}).get('source_file', 'unknown')
        page = result.get('metadata', {}).get('page_label', 'unknown')
        
        context_parts.append(
            f"[Source {i}: {source_file}, Page {page}]\n{result['content']}"
        )
        
        sources.append({
            'source_file': source_file,
            'page': page,
            'combined_score': result.get('combined_score', 0.0),
            'rank': result.get('rank', i)
        })
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate answer
    prompt = ENHANCED_PROMPT.format(context=context, question=query)
    
    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer: {e}"
    
    # Add citations
    if include_sources and sources:
        citations = [
            f"[{i+1}] {s['source_file']} - Page {s['page']} (Score: {s['combined_score']:.3f})"
            for i, s in enumerate(sources)
        ]
        answer_with_citations = f"{answer}\n\n{'='*50}\nSources:\n" + "\n".join(citations)
    else:
        answer_with_citations = answer
    
    return {
        'question': query,
        'answer': answer_with_citations,
        'raw_answer': answer,
        'sources': sources,
        'metadata': {
            'num_results': len(results),
            'avg_score': sum(r['combined_score'] for r in results) / len(results)
        }
    }