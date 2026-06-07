# src/rag_pipeline.py
"""RAG pipeline functions."""

from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from .retrievers import HybridRetriever


# Prompt template — supports optional chat history
ENHANCED_PROMPT = """You are an expert assistant that answers questions based on the provided context.

{history_section}Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "I don't have enough information."
3. Be concise and accurate
4. If the question refers to something mentioned earlier in the conversation, use the chat history to understand what it refers to.

Answer:"""

# Condensation prompt — rewrites a follow-up question as a standalone question
CONDENSE_PROMPT = """Given the chat history below and a follow-up question, rewrite the follow-up as a \
standalone question that contains all necessary context for a search engine to retrieve relevant documents.
If the follow-up is already standalone, return it as-is.

Chat History:
{history}

Follow-up question: {question}

Standalone question:"""


def get_llm():
    """Get LLM instance."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )


def _build_history_text(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history as a readable string."""
    lines = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def condense_question(
    question: str,
    chat_history: List[Dict[str, str]],
    llm
) -> str:
    """
    Rewrite a follow-up question as a standalone question using chat history.
    If there's no history, returns the question unchanged.
    """
    if not chat_history:
        return question

    history_text = _build_history_text(chat_history)
    prompt = CONDENSE_PROMPT.format(history=history_text, question=question)

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        # Fall back to original question if condensation fails
        return question


def enhanced_rag(
    query: str,
    hybrid_retriever: HybridRetriever,
    llm=None,
    top_k: int = 5,
    include_sources: bool = True,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Enhanced RAG pipeline with hybrid retrieval and optional conversation history.

    Args:
        query: User question
        hybrid_retriever: HybridRetriever instance
        llm: Language model (optional, will create if not provided)
        top_k: Number of documents to retrieve
        include_sources: Whether to include citations
        chat_history: List of {"role": "user"|"assistant", "content": "..."} dicts

    Returns:
        Dictionary with answer, sources, and metadata
    """
    if llm is None:
        llm = get_llm()

    chat_history = chat_history or []

    # Condense follow-up questions into standalone queries for better retrieval
    retrieval_query = condense_question(query, chat_history, llm)

    # Retrieve using the condensed (standalone) question
    results = hybrid_retriever.retrieve(retrieval_query, top_k=top_k * 2)
    results = results[:top_k]

    if not results:
        return {
            'question': query,
            'answer': "I don't have enough information to answer.",
            'raw_answer': "I don't have enough information to answer.",
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

    # Build history section for the prompt (last 6 turns to keep context short)
    if chat_history:
        recent_history = chat_history[-6:]
        history_text = _build_history_text(recent_history)
        history_section = f"Chat History:\n{history_text}\n\n"
    else:
        history_section = ""

    # Generate answer
    prompt = ENHANCED_PROMPT.format(
        history_section=history_section,
        context=context,
        question=query
    )

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
            'avg_score': sum(r['combined_score'] for r in results) / len(results),
            'retrieval_query': retrieval_query  # useful for debugging
        }
    }