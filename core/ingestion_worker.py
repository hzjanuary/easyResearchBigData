"""
easyResearch for Big Data — Ingestion Worker  (Big Data Engine)
================================================================
Full pipeline:  Clean → Chunk → Embed → Store

Design goals (merged from EpsteinFiles-RAG + easyResearch):
──────────────────────────────────────────────────────────────
1. **Flexible** – accepts any folder of PDF / TXT / DOCX / code files.
2. **RTX 3050 safe** – batch embedding (size 32) with explicit
   ``torch.cuda.empty_cache()`` between batches so 4 GB VRAM is never exceeded.
3. **Multiprocessing** – CPU-bound Clean & Chunk phases use all B760 cores
   via ``ProcessPoolExecutor``.
4. **Metadata-rich** – file path, date, format, page count survive into
   ChromaDB for pre-vector metadata filters.
5. **Async-friendly** – ``run_pipeline_async()`` runs in a daemon thread so
   the Streamlit UI stays responsive during large-scale processing.
6. **UI-tuneable** – chunk_size / overlap can be overridden per run.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    CHROMA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
    MAX_WORKERS,
    LOG_FILE,
    UPLOAD_DIR,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Progress state  (read by Streamlit in the main thread)
# ═══════════════════════════════════════════════════════════════════════════

class PipelineStatus:
    """Thread-safe mutable status consumed by the Streamlit UI."""

    def __init__(self) -> None:
        self.stage: str = "idle"          # idle|cleaning|chunking|embedding|done|error
        self.progress: float = 0.0        # 0.0 – 1.0
        self.message: str = ""
        self.error: str | None = None
        self.docs_cleaned: int = 0
        self.chunks_created: int = 0
        self.chunks_embedded: int = 0
        self.elapsed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "stage", "progress", "message", "error",
            "docs_cleaned", "chunks_created", "chunks_embedded", "elapsed",
        )}


# Module-level singleton
pipeline_status = PipelineStatus()


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 — CLEAN  (multiprocessing via cleaner_pro.py)
# ═══════════════════════════════════════════════════════════════════════════

def _stage_clean(source_dir: Path) -> list[dict[str, Any]]:
    from core.cleaner_pro import clean_documents

    def _cb(done: int, total: int) -> None:
        pipeline_status.progress = done / max(total, 1)
        pipeline_status.message = f"Cleaning file {done}/{total}"

    pipeline_status.stage = "cleaning"
    pipeline_status.progress = 0.0
    log.info("▶ Stage 1/3 — Cleaning documents from %s", source_dir)

    docs = clean_documents(source_dir, progress_callback=_cb)
    pipeline_status.docs_cleaned = len(docs)
    log.info("  ✔ %d documents cleaned", len(docs))
    return docs


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 — CHUNK  (CPU-bound, multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_single_doc(args: tuple) -> list[dict[str, Any]]:
    """Chunk one document dict. Runs inside a worker process."""
    doc, chunk_size, chunk_overlap = args
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    parts = splitter.split_text(doc["text"])
    chunks: list[dict[str, Any]] = []
    for idx, text in enumerate(parts):
        chunks.append({
            "text": text,
            "metadata": {
                "source": doc["file"],
                "file_path": doc.get("file_path", doc["file"]),
                "format": doc.get("format", "unknown"),
                "modified_date": doc.get("modified_date", ""),
                "chunk_index": idx,
            },
        })
    return chunks


def _stage_chunk(
    docs: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    pipeline_status.stage = "chunking"
    pipeline_status.progress = 0.0
    log.info("▶ Stage 2/3 — Chunking %d documents (size=%d, overlap=%d)",
             len(docs), chunk_size, chunk_overlap)

    all_chunks: list[dict[str, Any]] = []
    total = len(docs)

    args = [(doc, chunk_size, chunk_overlap) for doc in docs]

    if total > 50:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for i, result in enumerate(pool.map(_chunk_single_doc, args, chunksize=32)):
                all_chunks.extend(result)
                pipeline_status.progress = (i + 1) / total
                pipeline_status.message = f"Chunking doc {i + 1}/{total}"
    else:
        for i, a in enumerate(args):
            all_chunks.extend(_chunk_single_doc(a))
            pipeline_status.progress = (i + 1) / total
            pipeline_status.message = f"Chunking doc {i + 1}/{total}"

    # Dedup via SHA-256 (same approach as EpsteinFiles-RAG)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in all_chunks:
        h = hashlib.sha256(c["text"].lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)

    pipeline_status.chunks_created = len(unique)
    log.info("  ✔ %d unique chunks (from %d raw)", len(unique), len(all_chunks))
    return unique


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 — EMBED  (GPU-bound, careful with 4 GB VRAM)
# ═══════════════════════════════════════════════════════════════════════════

def _stage_embed(
    chunks: list[dict[str, Any]],
    collection_name: str,
    reset_db: bool = True,
) -> None:
    import shutil
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import EMBEDDING_MODEL, EMBED_BATCH_SIZE as BS, DEVICE as DEV

    pipeline_status.stage = "embedding"
    pipeline_status.progress = 0.0
    log.info("▶ Stage 3/3 — Embedding %d chunks (device=%s, batch=%d)",
             len(chunks), DEV, BS)

    if reset_db and Path(CHROMA_DIR).exists():
        # Only reset the specific collection if possible, else full reset
        log.info("  Resetting Chroma DB …")
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEV},
        encode_kwargs={"batch_size": BS, "normalize_embeddings": True},
    )

    db = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    total = len(chunks)
    for i in range(0, total, BS):
        end = min(i + BS, total)
        batch = chunks[i:end]

        texts = [c["text"] for c in batch]
        metas = [c["metadata"] for c in batch]
        ids = [hashlib.sha256(c["text"].encode()).hexdigest() for c in batch]

        db.add_texts(texts=texts, metadatas=metas, ids=ids)

        # ── VRAM housekeeping (critical for RTX 3050) ───────────
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        pipeline_status.progress = end / total
        pipeline_status.chunks_embedded = end
        pipeline_status.message = f"Embedded {end}/{total} chunks"
        log.info("  Embedded %d / %d", end, total)

    log.info("  ✔ Chroma DB stored → %s", CHROMA_DIR)


# ═══════════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    source_dir: Path | None = None,
    collection_name: str = "default_notebook",
    reset_db: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """
    Synchronous Clean → Chunk → Embed pipeline.

    Parameters
    ----------
    source_dir      : Folder with raw documents.  Defaults to UPLOAD_DIR.
    collection_name : ChromaDB collection to write to.
    reset_db        : Wipe existing Chroma DB before embedding.
    chunk_size      : Characters per chunk (UI-tuneable).
    chunk_overlap   : Overlap between chunks (UI-tuneable).
    """
    global pipeline_status
    pipeline_status = PipelineStatus()
    src = source_dir or UPLOAD_DIR
    t0 = time.perf_counter()

    try:
        docs = _stage_clean(src)
        if not docs:
            pipeline_status.stage = "error"
            pipeline_status.error = "No valid documents found after cleaning."
            return pipeline_status.to_dict()

        chunks = _stage_chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            pipeline_status.stage = "error"
            pipeline_status.error = "No chunks produced."
            return pipeline_status.to_dict()

        _stage_embed(chunks, collection_name=collection_name, reset_db=reset_db)

        pipeline_status.stage = "done"
        pipeline_status.progress = 1.0
        pipeline_status.elapsed = time.perf_counter() - t0
        pipeline_status.message = (
            f"Pipeline complete — {pipeline_status.docs_cleaned} docs, "
            f"{pipeline_status.chunks_created} chunks, "
            f"{pipeline_status.chunks_embedded} embedded "
            f"in {pipeline_status.elapsed:.1f}s"
        )
        log.info(pipeline_status.message)

    except Exception as exc:
        pipeline_status.stage = "error"
        pipeline_status.error = str(exc)
        log.exception("Pipeline failed")

    return pipeline_status.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
#  Async wrapper  (Streamlit background ingestion)
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline_async(
    source_dir: Path | None = None,
    collection_name: str = "default_notebook",
    reset_db: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    """
    Fire-and-forget from the Streamlit main thread.
    Runs the full pipeline in a background daemon thread.
    The UI reads ``pipeline_status`` to display progress.
    """
    import threading

    thread = threading.Thread(
        target=run_pipeline,
        kwargs={
            "source_dir": source_dir,
            "collection_name": collection_name,
            "reset_db": reset_db,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        daemon=True,
    )
    thread.start()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    result = run_pipeline()
    print(json.dumps(result, indent=2))
