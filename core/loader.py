"""
easyResearch for Big Data — Document Loader & Smart Splitter
==============================================================
Adapted from easyResearch's loader.py with Parent Document Retrieval.

* Small child chunks (400 chars) → precise vector search.
* Large parent chunks (2000 chars) → rich context sent to the LLM.
* Per-format splitting strategies (PDF/DOCX, Code, Data).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
    PARENT_OVERLAP,
    CHILD_OVERLAP,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-format splitting strategy
# ═══════════════════════════════════════════════════════════════════════════

def get_splitting_strategy(file_path: str):
    """
    Return (parent_size, child_size, parent_overlap, child_overlap, separators)
    tuned for the document type.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Strategy 1: Text documents (PDF, DOCX)
    if ext in (".pdf", ".docx", ".doc"):
        return 2500, 500, 300, 80, ["\n\n", "\n", ". ", " ", ""]

    # Strategy 2: Source code
    if ext in (".py", ".js", ".java", ".cpp", ".html"):
        return 1500, 400, 100, 30, [
            "\nclass ", "\ndef ", "\nfunction ",
            "\n\n", "\n", " ",
        ]

    # Strategy 3: Structured data
    if ext in (".json", ".csv", ".xml"):
        return 1000, 300, 50, 0, ["\n", "},", "],", " "]

    # Strategy 4: Default
    return PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, PARENT_OVERLAP, CHILD_OVERLAP, ["\n\n", "\n", " ", ""]


# ═══════════════════════════════════════════════════════════════════════════
#  Load + Split  (with Parent Document Retrieval)
# ═══════════════════════════════════════════════════════════════════════════

def load_and_split_document(
    file_path: str,
    use_parent_retrieval: bool = True,
) -> list[Document]:
    """
    Load a file and split it into chunks.

    When *use_parent_retrieval* is True the function creates small child chunks
    for vector search with the full parent chunk content stored in metadata /
    ``parent_content`` so the LLM receives wider context.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ── File loader dispatch ────────────────────────────────────────────
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    filename = os.path.basename(file_path)

    parent_size, child_size, p_overlap, c_overlap, separators = get_splitting_strategy(file_path)

    # ── Simple single-level splitting ───────────────────────────────────
    if not use_parent_retrieval:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=p_overlap,
            separators=separators,
        )
        splits = splitter.split_documents(docs)
        for i, split in enumerate(splits):
            split.metadata["source"] = filename
            split.metadata["chunk_index"] = i
            split.metadata["parent_content"] = split.page_content
            uid = f"{filename}_{i}"
            split.id = hashlib.sha256(uid.encode()).hexdigest()
        return splits

    # ── Parent Document Retrieval ───────────────────────────────────────
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=p_overlap,
        separators=separators,
    )
    parent_docs = parent_splitter.split_documents(docs)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=c_overlap,
        separators=separators,
    )

    all_child_chunks: list[Document] = []

    for parent_idx, parent_doc in enumerate(parent_docs):
        parent_content = parent_doc.page_content
        temp_doc = Document(
            page_content=parent_content,
            metadata=parent_doc.metadata.copy(),
        )
        child_chunks = child_splitter.split_documents([temp_doc])

        for child_idx, child_chunk in enumerate(child_chunks):
            child_chunk.metadata["source"] = filename
            child_chunk.metadata["parent_index"] = parent_idx
            child_chunk.metadata["child_index"] = child_idx
            child_chunk.metadata["chunk_index"] = len(all_child_chunks)
            child_chunk.metadata["parent_content"] = parent_content
            child_chunk.metadata["parent_page"] = parent_doc.metadata.get("page", 0)

            uid = f"{filename}_p{parent_idx}_c{child_idx}"
            child_chunk.id = hashlib.sha256(uid.encode()).hexdigest()
            all_child_chunks.append(child_chunk)

    return all_child_chunks


def load_document_simple(file_path: str) -> list[Document]:
    """Load a document without splitting — useful for summarisation."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()
