"""
Advanced RAG Engine for easyResearch.

Implements production-grade retrieval with:
- Hybrid Search: Dense vectors + BM25 sparse retrieval
- Cross-Encoder Re-ranking optimized for RTX 3050 (4GB VRAM)
- Reciprocal Rank Fusion for score combination
- Full observability integration
"""

from __future__ import annotations

import gc
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from core.embedder import get_vector_store, embedding_model
from core.observability import RAGTracer, rag_logger, log_execution_time, log_gpu_memory
from config import (
    DEVICE,
    MAX_HISTORY_MESSAGES,
    RERANKER_MODEL,
    HYBRID_WEIGHT_RERANK,
    HYBRID_WEIGHT_BM25,
    MIN_SCORE_THRESHOLD,
    LLM_MODEL_GROQ,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    dense_k: int = 20              # Initial dense retrieval count
    sparse_k: int = 20             # BM25 retrieval count  
    rerank_top_k: int = 10         # Final docs after re-ranking
    
    # Score fusion weights
    dense_weight: float = 0.5
    sparse_weight: float = 0.2
    rerank_weight: float = 0.3
    
    # Thresholds
    min_score_threshold: float = MIN_SCORE_THRESHOLD
    
    # Re-ranker settings (optimized for 4GB VRAM)
    reranker_batch_size: int = 8   # Small batch for memory efficiency
    use_fp16: bool = True          # Half precision for memory savings


# Global re-ranker (lazy loaded)
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Lazy load cross-encoder with memory optimization."""
    global _reranker
    
    if _reranker is None:
        rag_logger.info(f"Loading cross-encoder: {RERANKER_MODEL} on {DEVICE}")
        
        # Configure for low VRAM usage
        _reranker = CrossEncoder(
            RERANKER_MODEL,
            device=DEVICE,
            max_length=512,  # Limit input length for memory
        )
        
        # Use FP16 if on CUDA for memory efficiency
        if DEVICE == "cuda" and torch.cuda.is_available():
            _reranker.model.half()
            rag_logger.info("Cross-encoder using FP16 for memory efficiency")
    
    return _reranker


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a question reformulation expert. Reformulate the user's latest "
        "question into a standalone question that can be understood WITHOUT the "
        "chat history.\n\n"
        "RULES:\n"
        "1. Replace pronouns (it, this, they, he, she…) with actual terms from history.\n"
        "2. Incorporate previous topic for follow-ups.\n"
        "3. If already self-contained, return AS-IS.\n"
        "4. NEVER answer the question.\n"
        "5. Keep the same language.\n"
        "6. Be concise but complete."
    ),
    ("placeholder", "{chat_history}"),
    ("human", "Reformulate this question: {input}"),
])

RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI research assistant with access to a document database.\n"
        "Answer the user's question based ONLY on the provided context below.\n\n"
        "GUIDELINES:\n"
        "1. If the answer is not in the context, say you don't know.\n"
        "2. Use the conversation summary for flow.\n"
        "3. Be concise. Use bullet points when appropriate.\n"
        "4. Cite source document names when possible.\n"
        "5. Answer in the SAME language as the question.\n\n"
        "CONVERSATION SUMMARY:\n{conversation_summary}",
    ),
    ("human", "RETRIEVED DOCUMENTS:\n{context}\n\nQUESTION:\n{question}"),
])

RAG_PROMPT_NO_HISTORY = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI research assistant with access to a document database.\n"
        "Answer the user's question based ONLY on the provided context below.\n\n"
        "GUIDELINES:\n"
        "1. If the answer is not in the context, say you don't know.\n"
        "2. Be concise. Use bullet points when appropriate.\n"
        "3. Cite source document names when possible.\n"
        "4. Answer in the SAME language as the question.",
    ),
    ("human", "RETRIEVED DOCUMENTS:\n{context}\n\nQUESTION:\n{question}"),
])


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Search Implementation
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


@log_execution_time
def bm25_search(
    documents: list[Document],
    query: str,
    top_k: int = 20,
) -> list[tuple[Document, float]]:
    """
    Perform BM25 sparse retrieval on documents.
    
    Returns documents with normalized BM25 scores.
    """
    if not documents:
        return []
    
    corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    
    # Normalize scores to [0, 1]
    max_score = float(max(scores)) if max(scores) > 0 else 1.0
    normalized_scores = [float(s) / max_score for s in scores]
    
    # Pair documents with scores and sort
    doc_scores = list(zip(documents, normalized_scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return doc_scores[:top_k]


@log_execution_time
def dense_search(
    collection_name: str,
    query: str,
    k: int = 20,
    filter_dict: dict | None = None,
) -> list[tuple[Document, float]]:
    """
    Perform dense vector similarity search.
    
    Returns documents with similarity scores.
    """
    db = get_vector_store(collection_name)
    
    if filter_dict:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
        conditions = []
        if "format" in filter_dict:
            conditions.append(
                FieldCondition(key="metadata.format", match=MatchValue(value=filter_dict["format"]))
            )
        if "source" in filter_dict:
            conditions.append(
                FieldCondition(key="metadata.source", match=MatchText(text=filter_dict["source"]))
            )
        qdrant_filter = Filter(must=conditions) if conditions else None
        
        results = db.similarity_search_with_score(
            query, k=k, filter=qdrant_filter
        )
    else:
        results = db.similarity_search_with_score(query, k=k)
    
    # Convert to (doc, score) tuples - note: Qdrant returns distance, lower is better
    # We invert to higher is better for consistency
    doc_scores = [(doc, 1.0 - min(score, 1.0)) for doc, score in results]
    
    return doc_scores


@log_gpu_memory
@log_execution_time
def rerank_documents(
    query: str,
    documents: list[Document],
    config: RetrievalConfig,
) -> list[tuple[Document, float]]:
    """
    Re-rank documents using cross-encoder.
    
    Optimized for RTX 3050 4GB VRAM with batch processing.
    """
    if not documents:
        return []
    
    reranker = get_reranker()
    
    # Create query-document pairs
    pairs = [[query, doc.page_content[:512]] for doc in documents]  # Truncate for memory
    
    # Batch prediction for memory efficiency
    all_scores = []
    batch_size = config.reranker_batch_size
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        
        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():  # Mixed precision
                    scores = reranker.predict(batch, show_progress_bar=False)
            else:
                scores = reranker.predict(batch, show_progress_bar=False)
        
        all_scores.extend(scores.tolist() if hasattr(scores, 'tolist') else list(scores))
        
        # Clear GPU cache after each batch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    gc.collect()
    
    # Normalize scores to [0, 1] using sigmoid-like transformation
    min_score, max_score = min(all_scores), max(all_scores)
    score_range = max_score - min_score if max_score != min_score else 1.0
    normalized_scores = [(s - min_score) / score_range for s in all_scores]
    
    # Pair and sort
    doc_scores = list(zip(documents, normalized_scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return doc_scores


def reciprocal_rank_fusion(
    rankings: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.
    
    RRF score = Σ 1 / (k + rank_i) for each ranking
    """
    doc_scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}
    
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            # Use content hash as doc identifier
            doc_id = hash(doc.page_content[:200])
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            
            doc_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]


@log_execution_time
def hybrid_search(
    collection_name: str,
    query: str,
    config: RetrievalConfig | None = None,
    filter_dict: dict | None = None,
) -> list[tuple[Document, float]]:
    """
    Perform hybrid search combining dense and sparse retrieval.
    
    Pipeline:
    1. Dense vector search (semantic similarity)
    2. BM25 sparse search (keyword matching)
    3. Reciprocal Rank Fusion
    4. Cross-encoder re-ranking
    """
    config = config or RetrievalConfig()
    
    # Step 1: Dense retrieval
    rag_logger.debug(f"Dense search: k={config.dense_k}")
    dense_results = dense_search(collection_name, query, k=config.dense_k, filter_dict=filter_dict)
    dense_docs = [doc for doc, _ in dense_results]
    
    if not dense_docs:
        rag_logger.warning("No documents found in dense search")
        return []
    
    # Step 2: BM25 on retrieved documents (not full corpus for efficiency)
    rag_logger.debug(f"BM25 search on {len(dense_docs)} docs")
    bm25_results = bm25_search(dense_docs, query, top_k=config.sparse_k)
    
    # Step 3: Reciprocal Rank Fusion
    rag_logger.debug("Combining with RRF")
    fused_results = reciprocal_rank_fusion([dense_results, bm25_results])
    
    # Take top candidates for re-ranking (limit for memory)
    candidates = [doc for doc, _ in fused_results[:min(len(fused_results), 30)]]
    
    # Step 4: Cross-encoder re-ranking
    rag_logger.debug(f"Re-ranking {len(candidates)} candidates")
    reranked = rerank_documents(query, candidates, config)
    
    # Apply score threshold and limit
    final_results = [
        (doc, score) for doc, score in reranked
        if score >= config.min_score_threshold
    ][:config.rerank_top_k]
    
    rag_logger.info(f"Hybrid search returned {len(final_results)} docs")
    return final_results


# ──────────────────────────────────────────────────────────────────────────────
# Query Processing
# ──────────────────────────────────────────────────────────────────────────────

def _needs_contextualization(question: str) -> bool:
    """Check if question needs context from chat history."""
    patterns = [
        r"\b(it|its|this|that|these|those|they|them|their|he|she|him|her)\b",
        r"\b(the same|above|previous|mentioned|said|such)\b",
        r"\b(what about|how about|and the|also the|another)\b",
        # Vietnamese
        r"\b(nó|này|đó|ở trên|như vậy|còn|thế thì|vậy thì)\b",
        r"\b(cái này|cái đó|điều đó|vấn đề này|chúng)\b",
    ]
    q_lower = question.lower()
    return any(re.search(p, q_lower) for p in patterns)


def _summarize_conversation(chat_history: list[dict], max_messages: int = 5) -> str:
    """Create a conversation summary for context."""
    if not chat_history or len(chat_history) <= 1:
        return "This is the beginning of the conversation."
    
    recent = chat_history[-max_messages:]
    parts = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "…" if len(msg["content"]) > 200 else msg["content"]
        parts.append(f"- {role}: {content}")
    
    return "\n".join(parts)


def _get_llm(
    api_key: str | None = None,
) -> ChatGroq:
    """Get configured LLM instance with error handling."""
    
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Groq API key required")
    
    from pydantic import SecretStr
    return ChatGroq(
        model=LLM_MODEL_GROQ,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        api_key=SecretStr(key),
    )


def contextualize_query(
    question: str,
    chat_history: list[dict],
    llm: ChatGroq,
) -> str:
    """Reformulate question to be standalone using chat history."""
    
    if not chat_history or not _needs_contextualization(question):
        return question
    
    try:
        # Convert to LangChain message format
        recent = chat_history[-MAX_HISTORY_MESSAGES:-1] if len(chat_history) > MAX_HISTORY_MESSAGES else chat_history[:-1]
        hist_lc = []
        for msg in recent:
            if msg["role"] == "user":
                hist_lc.append(HumanMessage(content=msg["content"]))
            else:
                hist_lc.append(AIMessage(content=msg["content"]))
        
        chain = CONTEXTUALIZE_PROMPT | llm
        result = chain.invoke({
            "chat_history": hist_lc,
            "input": question,
        })
        
        content = result.content
        standalone = content.strip() if isinstance(content, str) else str(content)
        rag_logger.debug(f"Contextualized: '{question[:50]}' -> '{standalone[:50]}'")
        return standalone
        
    except Exception as e:
        rag_logger.warning(f"Contextualization failed: {e}")
        return question


# ──────────────────────────────────────────────────────────────────────────────
# Main RAG Query Function
# ──────────────────────────────────────────────────────────────────────────────

def query_rag(
    question: str,
    collection_name: str,
    chat_history: list[dict] | None = None,
    k_target: int = 10,
    user_api_key: str | None = None,
    format_filter: str | None = None,
    source_filter: str | None = None,
    retrieval_config: RetrievalConfig | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Execute RAG query with hybrid search and full observability.
    
    Args:
        question: User's question
        collection_name: Qdrant collection/workspace name
        chat_history: Previous conversation messages
        k_target: Target number of documents to retrieve
        user_api_key: Optional user-provided API key
        format_filter: Filter by document format (e.g., "pdf")
        source_filter: Filter by source filename
        retrieval_config: Custom retrieval configuration
    
    Returns:
        Dictionary with answer, sources, debug info, and trace data
    """
    chat_history = chat_history or []
    config = retrieval_config or RetrievalConfig(rerank_top_k=k_target)
    
    # Build filter dict
    filter_dict = {}
    if format_filter:
        filter_dict["format"] = format_filter
    if source_filter:
        filter_dict["source"] = source_filter
    
    with RAGTracer(workspace=collection_name, query=question) as tracer:
        try:
            # Initialize LLM
            llm = _get_llm(user_api_key)
            model_name = LLM_MODEL_GROQ
            
            # Contextualize query if needed
            with tracer.stage("contextualization"):
                standalone_question = contextualize_query(
                    question, chat_history, llm
                )
                tracer.set_standalone_query(standalone_question)
            
            # Hybrid retrieval
            with tracer.stage("retrieval"):
                results = hybrid_search(
                    collection_name,
                    standalone_question,
                    config=config,
                    filter_dict=filter_dict if filter_dict else None,
                )
                
                tracer.log_retrieval([doc for doc, _ in results])
            
            if not results:
                return {
                    "answer": "I could not find relevant information in the documents.",
                    "sources": [],
                    "standalone_question": standalone_question,
                    "pipeline_info": {"status": "no_docs_found"},
                }
            
            # Log re-ranking scores
            tracer.log_reranking(
                [doc for doc, _ in results],
                [score for _, score in results],
            )
            
            # Build context
            final_docs = [doc for doc, _ in results]
            context_text = "\n\n---\n\n".join([
                f"[Source: {d.metadata.get('source', 'Unknown')}]\n"
                f"{d.metadata.get('parent_content', d.page_content)}"
                for d in final_docs
            ])
            
            # Generate response
            with tracer.stage("generation"):
                has_history = len(chat_history) > 1
                
                if has_history:
                    summary = _summarize_conversation(chat_history)
                    messages = RAG_PROMPT_WITH_HISTORY.format_messages(
                        context=context_text,
                        question=question,
                        conversation_summary=summary,
                    )
                else:
                    messages = RAG_PROMPT_NO_HISTORY.format_messages(
                        context=context_text,
                        question=question,
                    )
                
                response = llm.invoke(messages)
                content = response.content
                answer_text = content.strip() if isinstance(content, str) else str(content)
            
            # Collect sources
            source_names = list({d.metadata.get("source", "Unknown") for d in final_docs})
            
            # Log generation details
            tracer.log_generation(
                context_length=len(context_text),
                llm_provider="groq",
                llm_model=model_name,
                response_length=len(answer_text),
                sources=source_names,
            )
            
            return {
                "answer": answer_text,
                "sources": source_names,
                "standalone_question": standalone_question,
                "raw_docs": [
                    {
                        "source": d.metadata.get("source", "Unknown"),
                        "score": results[i][1] if i < len(results) else 0.0,
                        "preview": d.page_content[:200],
                    }
                    for i, d in enumerate(final_docs[:5])
                ],
                "pipeline_info": {
                    "trace_id": tracer.trace.trace_id,
                    "retrieval_count": len(results),
                    "model": model_name,
                    "contextualized": tracer.trace.contextualized,
                },
            }
            
        except ValueError as e:
            # API key or config errors
            return {
                "answer": f"❌ Configuration error: {e}",
                "sources": [],
                "pipeline_info": {"error": str(e)},
            }
        except Exception as e:
            rag_logger.exception("RAG query failed")
            return {
                "answer": f"❌ Query failed: {e}",
                "sources": [],
                "pipeline_info": {"error": str(e)},
            }


# Alias for backward compatibility
query_rag_system = query_rag
