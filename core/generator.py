"""
easyResearch for Big Data — Advanced RAG Generator
=====================================================
Hybrid Search pipeline adapted from easyResearch's generator.py:

    Question → Smart Contextualization → Vector Search (MMR)
                                              ↓
                                        BM25 Scoring
                                              ↓
                                  Cross-Encoder Reranking
                                              ↓
                   Hybrid Score (0.7×Rerank + 0.3×BM25)
                                              ↓
                     Parent Document Retrieval → LLM Answer

Supports: Groq (LLaMA 3.3 70B) and Google Gemini.
Metadata pre-filtering for format / source before vector search.
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from core.embedder import embedding_model
from config import (
    CHROMA_DIR,
    DEVICE,
    MAX_HISTORY_MESSAGES,
    RERANKER_MODEL,
    HYBRID_WEIGHT_RERANK,
    HYBRID_WEIGHT_BM25,
    MIN_SCORE_THRESHOLD,
)

load_dotenv()

# ── Reranker ────────────────────────────────────────────────────────────────
reranker_model = CrossEncoder(RERANKER_MODEL, device=DEVICE)


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

contextualize_q_system_prompt = (
    "You are a question reformulation expert. Reformulate the user's latest "
    "question into a standalone question that can be understood WITHOUT the "
    "chat history.\n\n"
    "RULES:\n"
    "1. Replace pronouns (it, this, they, he, she …) with actual terms from history.\n"
    "2. Incorporate previous topic for follow-ups.\n"
    "3. If already self-contained, return AS-IS.\n"
    "4. NEVER answer the question.\n"
    "5. Keep the same language.\n"
    "6. Be concise but complete."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "Reformulate this question: {input}"),
])

rag_prompt_with_history = ChatPromptTemplate.from_messages([
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

rag_prompt_no_history = ChatPromptTemplate.from_messages([
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


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _summarize_conversation(
    chat_history: list[dict],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> str:
    if not chat_history or len(chat_history) <= 1:
        return "This is the beginning of the conversation."
    recent = chat_history[-max_messages:]
    parts = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "…" if len(msg["content"]) > 200 else msg["content"]
        parts.append(f"- {role}: {content}")
    return "\n".join(parts)


def _needs_contextualization(question: str) -> bool:
    """Check if the question contains pronouns or follow-up markers."""
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


def _bm25_search(documents: list, query: str, top_k: int = 10) -> list:
    """BM25 keyword scoring over pre-retrieved documents."""
    if not documents:
        return []
    corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    for i, doc in enumerate(documents):
        doc.metadata["bm25_score"] = float(scores[i])
    return sorted(documents, key=lambda x: x.metadata["bm25_score"], reverse=True)[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN RAG FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def query_rag_system(
    question: str,
    collection_name: str,
    chat_history: list[dict] | None = None,
    k_target: int = 10,
    user_api_key: str | None = None,
    llm_provider: str = "groq",
    # Metadata filters (Big Data feature from EpsteinFiles-RAG)
    format_filter: str | None = None,
    source_filter: str | None = None,
) -> dict[str, Any]:
    """
    Hybrid RAG Pipeline:
      Vector Search + BM25 → Cross-Encoder Reranking → Parent Doc → LLM
    With optional metadata pre-filtering on format / source.
    """

    # ── 1. LLM initialisation ──────────────────────────────────────────
    if llm_provider == "gemini":
        sys_key = os.getenv("GOOGLE_API_KEY")
        key = user_api_key if user_api_key and user_api_key.strip() else sys_key
        if not key:
            return {"answer": "❌ Missing Google Gemini API Key.", "sources": []}
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                max_output_tokens=1024,
                google_api_key=key,
            )
        except Exception as e:
            return {"answer": f"❌ Gemini init error: {e}", "sources": []}
    else:
        sys_key = os.getenv("GROQ_API_KEY")
        key = user_api_key if user_api_key and user_api_key.strip() else sys_key
        if not key:
            return {"answer": "❌ Missing Groq API Key.", "sources": []}
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=1024,
                api_key=key,
            )
        except Exception as e:
            return {"answer": f"❌ LLM init error: {e}", "sources": []}

    # ── 2. Contextualisation ────────────────────────────────────────────
    standalone_question = question
    has_history = chat_history and len(chat_history) > 1
    need_context = has_history and _needs_contextualization(question)

    if need_context:
        try:
            chain = contextualize_q_prompt | llm
            recent = chat_history[-MAX_HISTORY_MESSAGES:-1] if len(chat_history) > MAX_HISTORY_MESSAGES else chat_history[:-1]
            hist_lc = []
            for msg in recent:
                if msg["role"] == "user":
                    hist_lc.append(HumanMessage(content=msg["content"]))
                else:
                    hist_lc.append(AIMessage(content=msg["content"]))
            standalone_question = chain.invoke({
                "chat_history": hist_lc,
                "input": question,
            }).content.strip()
            print(f"✨ Contextualized: {standalone_question}")
        except Exception as e:
            print(f"⚠️ Contextualization failed: {e}")

    # ── 3. Connect to ChromaDB ──────────────────────────────────────────
    db = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )

    # ── 4. Metadata pre-filter  (Big Data feature) ─────────────────────
    where_filter: dict | None = None
    conditions: list[dict] = []
    if format_filter:
        conditions.append({"format": {"$eq": format_filter}})
    if source_filter:
        conditions.append({"source": {"$contains": source_filter}})
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # ── 5. Vector search ────────────────────────────────────────────────
    if where_filter:
        all_docs = db.similarity_search(
            standalone_question,
            k=k_target * 2,
            filter=where_filter,
        )
    else:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_target * 2},
        )
        all_docs = retriever.invoke(standalone_question)

    if not all_docs:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
            "raw_docs": [],
            "pipeline_info": {"retrieval": "no_docs_found"},
        }

    # ── 6. BM25 scoring (hybrid component) ─────────────────────────────
    bm25_ranked = _bm25_search(all_docs.copy(), standalone_question, top_k=len(all_docs))
    bm25_scores = {
        hash(d.page_content[:100]): d.metadata.get("bm25_score", 0)
        for d in bm25_ranked
    }
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
    for doc in all_docs:
        h = hash(doc.page_content[:100])
        raw = bm25_scores.get(h, 0)
        doc.metadata["bm25_score"] = raw / max_bm25 if max_bm25 > 0 else 0

    # ── 7. Cross-Encoder reranking ──────────────────────────────────────
    pairs = [[standalone_question, doc.page_content] for doc in all_docs]
    rerank_scores = reranker_model.predict(pairs)

    for i, doc in enumerate(all_docs):
        doc.metadata["rerank_score"] = float(rerank_scores[i])
        doc.metadata["hybrid_score"] = (
            HYBRID_WEIGHT_RERANK * doc.metadata["rerank_score"]
            + HYBRID_WEIGHT_BM25 * doc.metadata["bm25_score"]
        )

    # Filter + sort
    filtered = [d for d in all_docs if d.metadata["hybrid_score"] >= MIN_SCORE_THRESHOLD]
    if not filtered:
        filtered = sorted(all_docs, key=lambda x: x.metadata["hybrid_score"], reverse=True)[:k_target]
    else:
        filtered = sorted(filtered, key=lambda x: x.metadata["hybrid_score"], reverse=True)[:k_target]

    final_docs = filtered
    top_scores = [round(d.metadata["hybrid_score"], 2) for d in final_docs[:3]]
    print(f"🎯 Selected {len(final_docs)} docs (scores: {top_scores}…)")

    if not final_docs:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
            "raw_docs": [],
            "pipeline_info": {"retrieval": "no_relevant_docs"},
        }

    # ── 8. Generate answer (using parent content for richer context) ────
    context_text = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source', 'Unknown')}]\n"
        f"{d.metadata.get('parent_content', d.page_content)}"
        for d in final_docs
    ])

    try:
        if has_history:
            summary = _summarize_conversation(chat_history)
            messages = rag_prompt_with_history.format_messages(
                context=context_text,
                question=question,
                conversation_summary=summary,
            )
        else:
            messages = rag_prompt_no_history.format_messages(
                context=context_text,
                question=question,
            )
        response = llm.invoke(messages)
        answer_text = response.content.strip()
    except Exception as e:
        answer_text = f"❌ API Error: {e}"

    source_names = list({d.metadata.get("source", "Unknown") for d in final_docs})

    return {
        "answer": answer_text,
        "sources": source_names,
        "raw_docs": [
            f"[Score: {d.metadata.get('hybrid_score', 0):.2f}] "
            f"{d.page_content[:200]}…"
            for d in final_docs
        ],
        "standalone_question": standalone_question if standalone_question != question else None,
        "pipeline_info": {
            "total_retrieved": len(all_docs),
            "final_docs": len(final_docs),
            "contextualized": need_context,
        },
    }
