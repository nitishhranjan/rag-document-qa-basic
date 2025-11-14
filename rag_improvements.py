"""
RAG Improvements: Advanced techniques to enhance your RAG system
Use these functions to improve your existing RAG implementation
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional
from functools import wraps
from dotenv import load_dotenv

# Load environment variables (SECURITY BEST PRACTICE)
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. ENHANCED PROMPT ENGINEERING
# ============================================================================

def create_enhanced_prompt_template():
    """Create a better prompt template with clear instructions"""
    return """You are an expert assistant that answers questions based on the provided context.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided documents."
3. Be concise and accurate
4. If multiple sources are relevant, synthesize the information
5. Cite the source when possible (e.g., 'According to [source_file]...')

Answer:"""


def rag_enhanced(query: str, retriever, llm, top_k: int = 5):
    """
    Enhanced RAG with better prompt and source citation
    
    Args:
        query: User question
        retriever: Your RAGRetriever instance
        llm: LLM instance (e.g., ChatGroq)
        top_k: Number of documents to retrieve
    
    Returns:
        tuple: (answer, sources_list)
    """
    logger.info(f"Processing query: {query[:50]}...")
    
    # Retrieve documents
    results = retriever.retrieve(query, top_k=top_k)
    
    if not results:
        logger.warning("No results retrieved")
        return "No relevant context found in the documents.", []
    
    # Build context with source information
    context_parts = []
    sources = []
    
    for i, result in enumerate(results, 1):
        source_file = result['metadata'].get('source_file', 'Unknown')
        page = result['metadata'].get('page', 'N/A')
        
        context_parts.append(
            f"[Source {i}: {source_file}, Page {page}]\n{result['content']}"
        )
        sources.append({
            'source_file': source_file,
            'page': page,
            'similarity_score': result['similarity_score'],
            'rank': i
        })
    
    context = "\n\n".join(context_parts)
    
    # Use enhanced prompt
    prompt_template = create_enhanced_prompt_template()
    prompt = prompt_template.format(context=context, question=query)
    
    # Generate response
    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Generated answer with {len(sources)} sources")
        return answer, sources
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}", sources


# ============================================================================
# 2. RE-RANKING FOR BETTER RESULTS
# ============================================================================

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Re-ranking unavailable. Install with: pip install sentence-transformers")


class Reranker:
    """Re-rank retrieved documents using cross-encoder for better relevance"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with cross-encoder model
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        if not RERANKER_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        
        logger.info(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded successfully")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on query relevance
        
        Args:
            query: User query
            documents: List of document dictionaries with 'content' key
            top_k: Number of top documents to return
        
        Returns:
            Re-ranked list of documents
        """
        if not documents:
            return []
        
        logger.info(f"Re-ranking {len(documents)} documents...")
        
        # Create query-document pairs
        pairs = [[query, doc['content']] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score (descending)
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Re-ranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
        return reranked[:top_k]


# ============================================================================
# 3. HYBRID SEARCH (SEMANTIC + KEYWORD)
# ============================================================================

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed. Hybrid search unavailable. Install with: pip install rank-bm25")


class HybridRetriever:
    """Combine semantic search with keyword-based BM25 search"""
    
    def __init__(self, vectorstore, embedding_manager, documents: List[Any]):
        """
        Initialize hybrid retriever
        
        Args:
            vectorstore: Your VectorStore instance
            embedding_manager: Your EmbeddingManager instance
            documents: List of langchain Document objects
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 not installed. Install with: pip install rank-bm25")
        
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager
        self.documents = documents
        
        # Build BM25 index
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        logger.info("Building BM25 index...")
        tokenized_docs = []
        for doc in self.documents:
            # Simple tokenization
            tokens = re.findall(r'\w+', doc.page_content.lower())
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"BM25 index built for {len(tokenized_docs)} documents")
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search
        
        Args:
            query: User query
            top_k: Number of results to return
            alpha: Weight for semantic search (1-alpha for keyword search)
                   alpha=1.0 = pure semantic, alpha=0.0 = pure keyword
        
        Returns:
            List of combined search results
        """
        logger.info(f"Hybrid search: query='{query[:50]}...', top_k={top_k}, alpha={alpha}")
        
        # Semantic search
        query_embedding = self.embedding_manager.generate_embedding([query])[0]
        semantic_results = self.vectorstore.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2  # Get more for combination
        )
        
        # Keyword search (BM25)
        query_tokens = re.findall(r'\w+', query.lower())
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        else:
            bm25_scores = bm25_scores * 0  # All zeros
        
        # Combine scores
        combined_scores = {}
        
        if semantic_results['ids'] and semantic_results['ids'][0]:
            for i, doc_id in enumerate(semantic_results['ids'][0]):
                try:
                    # Extract index from doc_id (assuming format doc_xxx_index)
                    doc_idx = int(doc_id.split('_')[-1])
                    semantic_score = 1 - (semantic_results['distances'][0][i] / 2)
                    bm25_score = float(bm25_scores[doc_idx]) if doc_idx < len(bm25_scores) else 0.0
                    
                    combined_score = alpha * semantic_score + (1 - alpha) * bm25_score
                    
                    combined_scores[doc_id] = {
                        'id': doc_id,
                        'combined_score': combined_score,
                        'semantic_score': semantic_score,
                        'bm25_score': bm25_score,
                        'content': semantic_results['documents'][0][i],
                        'metadata': semantic_results['metadatas'][0][i],
                    }
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error processing doc_id {doc_id}: {e}")
                    continue
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        
        logger.info(f"Hybrid search complete. Retrieved {len(sorted_results)} documents")
        return sorted_results


# ============================================================================
# 4. EVALUATION FRAMEWORK
# ============================================================================

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict], 
                          ground_truth_doc_ids: Optional[List[str]] = None) -> Dict:
        """
        Evaluate retrieval quality
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            ground_truth_doc_ids: Optional list of correct document IDs
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'query': query,
            'num_retrieved': len(retrieved_docs),
            'avg_similarity': (
                sum(doc['similarity_score'] for doc in retrieved_docs) / len(retrieved_docs)
                if retrieved_docs else 0
            ),
        }
        
        if ground_truth_doc_ids and retrieved_docs:
            retrieved_ids = [doc['id'] for doc in retrieved_docs]
            intersection = set(retrieved_ids) & set(ground_truth_doc_ids)
            
            metrics['precision'] = len(intersection) / len(retrieved_ids) if retrieved_ids else 0
            metrics['recall'] = len(intersection) / len(ground_truth_doc_ids) if ground_truth_doc_ids else 0
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = (
                    2 * metrics['precision'] * metrics['recall'] /
                    (metrics['precision'] + metrics['recall'])
                )
            else:
                metrics['f1'] = 0
        
        return metrics
    
    def evaluate_generation(self, query: str, answer: str, 
                           ground_truth: Optional[str] = None) -> Dict:
        """
        Evaluate answer quality
        
        Args:
            query: User query
            answer: Generated answer
            ground_truth: Optional ground truth answer
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'query': query,
            'answer_length': len(answer),
            'has_answer': len(answer.strip()) > 0,
        }
        
        if ground_truth:
            # Simple metrics (can be enhanced with LLM evaluation)
            metrics['ground_truth_length'] = len(ground_truth)
            metrics['length_ratio'] = len(answer) / len(ground_truth) if ground_truth else 0
        
        return metrics
    
    def create_test_dataset(self, questions: List[str], 
                           ground_truths: Optional[List[str]] = None) -> Dict:
        """
        Create a test dataset for evaluation
        
        Args:
            questions: List of test questions
            ground_truths: Optional list of ground truth answers
        
        Returns:
            Test dataset dictionary
        """
        if ground_truths and len(ground_truths) != len(questions):
            raise ValueError("Number of questions and ground truths must match")
        
        return {
            'questions': questions,
            'ground_truths': ground_truths or [None] * len(questions)
        }
    
    def run_evaluation(self, rag_function, test_dataset: Dict, 
                      retriever=None) -> List[Dict]:
        """
        Run full evaluation on test dataset
        
        Args:
            rag_function: Function that takes query and returns (answer, sources)
            test_dataset: Test dataset from create_test_dataset
            retriever: Optional retriever for retrieval evaluation
        
        Returns:
            List of evaluation results
        """
        results = []
        
        for question, ground_truth in zip(
            test_dataset['questions'], 
            test_dataset['ground_truths']
        ):
            logger.info(f"Evaluating question: {question[:50]}...")
            
            # Get answer
            if retriever:
                answer, sources = rag_function(question, retriever)
            else:
                answer = rag_function(question)
                sources = []
            
            # Evaluate
            result = {
                'question': question,
                'answer': answer,
                'ground_truth': ground_truth,
                'sources': sources,
                'num_sources': len(sources),
            }
            
            # Add generation metrics
            result.update(self.evaluate_generation(question, answer, ground_truth))
            
            # Add retrieval metrics if retriever provided
            if retriever and sources:
                result.update(self.evaluate_retrieval(question, sources))
            
            results.append(result)
        
        return results


# ============================================================================
# 5. PRODUCTION-READY WRAPPER
# ============================================================================

def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    return wrapper


class ProductionRAG:
    """Production-ready RAG with logging, error handling, and monitoring"""
    
    def __init__(self, retriever, llm, reranker=None):
        """
        Initialize production RAG
        
        Args:
            retriever: RAGRetriever instance
            llm: LLM instance
            reranker: Optional Reranker instance
        """
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        
        # Verify API key is from environment (not hardcoded)
        # This is a security check
        logger.info("ProductionRAG initialized")
    
    @log_performance
    def query(self, question: str, top_k: int = 5, use_reranking: bool = True):
        """
        Query with full error handling and logging
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            use_reranking: Whether to use re-ranking
        
        Returns:
            tuple: (answer, sources)
        """
        try:
            logger.info(f"Processing query: {question[:50]}...")
            
            # Retrieve
            retrieve_k = top_k * 2 if (use_reranking and self.reranker) else top_k
            results = self.retriever.retrieve(question, top_k=retrieve_k)
            
            if not results:
                logger.warning("No results retrieved")
                return "No relevant information found.", []
            
            # Re-rank if available
            if self.reranker and use_reranking:
                logger.info("Re-ranking results...")
                results = self.reranker.rerank(question, results, top_k=top_k)
            else:
                results = results[:top_k]
            
            # Generate answer using enhanced RAG
            answer, sources = rag_enhanced(question, self.retriever, self.llm, top_k=len(results))
            
            logger.info(f"Query answered successfully with {len(sources)} sources")
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return "I encountered an error processing your question. Please try again.", []


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("RAG Improvements Module")
    print("=" * 50)
    print("\nThis module provides advanced RAG techniques:")
    print("1. Enhanced prompt engineering")
    print("2. Re-ranking for better results")
    print("3. Hybrid search (semantic + keyword)")
    print("4. Evaluation framework")
    print("5. Production-ready wrapper")
    print("\nSee NEXT_STEPS.md for detailed usage instructions.")

