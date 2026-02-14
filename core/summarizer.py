"""
easyResearch for Big Data — Auto-Summariser
=============================================
Generates a concise summary from the first few chunks of a notebook.
Supports Groq (LLaMA 3.3) and Google Gemini.
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research assistant. "
        "Summarise the following document snippets into a concise overview. "
        "Focus on: Main Topic, Key Objectives, and Target Audience. "
        "Answer in the language of the text (Vietnamese or English).",
    ),
    ("human", "Context to summarise:\n{context}"),
])


def generate_notebook_summary(
    chunks,
    api_key: str | None = None,
    llm_provider: str = "groq",
) -> str:
    """
    Build a summary from the first ~10 chunks (usually intro / abstract).
    """
    sample_context = "\n\n".join(
        chunk.page_content for chunk in chunks[:10]
    )

    if llm_provider == "gemini":
        final_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not final_key:
            return "⚠️ No Google Gemini API Key for summarisation."
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key=final_key,
        )
    else:
        final_key = api_key or os.getenv("GROQ_API_KEY")
        if not final_key:
            return "⚠️ No Groq API Key for summarisation."
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=final_key,
        )

    try:
        messages = _PROMPT.format_messages(context=sample_context)
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"❌ Summary error: {e}"
