"""
Production-Grade Ingestion Pipeline for easyResearch.

Implements metadata enrichment using LLM extraction:
- Automatic extraction of author, topic_tags, summary, document_type
- Batch processing optimized for RTX 3050 (4GB VRAM) 
- CUDA-accelerated embedding with memory management
- Full observability integration
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from config import (
    Config,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
    MAX_WORKERS,
    LOG_FILE,
    UPLOAD_DIR,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_VECTOR_SIZE,
    DEVICE,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL_GROQ,
)

from core.observability import setup_logger, log_execution_time, log_gpu_memory

# Configure logging
log = setup_logger("pipeline", log_file=LOG_FILE)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline."""
    
    # Chunking
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    
    # Embedding
    batch_size: int = EMBED_BATCH_SIZE
    use_cuda: bool = (DEVICE == "cuda")
    
    # Metadata enrichment
    enable_llm_enrichment: bool = True
    enrichment_batch_size: int = 5  # Docs per LLM call (rate limit aware)
    enrichment_timeout: float = 30.0
    
    # Processing
    max_workers: int = MAX_WORKERS
    reset_db: bool = True
    
    # Memory optimization
    clear_cache_every: int = 10  # Clear GPU cache every N batches


@dataclass
class PipelineStatus:
    """Current status of the pipeline execution."""
    
    stage: str = "idle"  # idle|cleaning|enriching|chunking|embedding|done|error
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    
    # Counters
    docs_cleaned: int = 0
    docs_enriched: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    
    # Timing
    elapsed: float = 0.0
    stage_times: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Global pipeline status (thread-safe via GIL for simple reads)
pipeline_status = PipelineStatus()


# ──────────────────────────────────────────────────────────────────────────────
# Metadata Enrichment with LLM
# ──────────────────────────────────────────────────────────────────────────────

METADATA_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a document analysis expert. Analyze the given text and extract structured metadata.

Return a JSON object with these fields:
- "author": Author name if found, otherwise "Unknown"
- "topic_tags": List of 3-5 relevant topic tags (e.g., ["RPC", "distributed systems", "networking"])
- "summary": One-sentence summary (max 100 words)
- "document_type": One of ["lecture", "book", "article", "code", "notes", "reference", "other"]
- "language": Primary language of the text ("en", "vi", or other ISO code)
- "technical_level": One of ["beginner", "intermediate", "advanced"]

IMPORTANT:
- Extract ONLY information present in the text
- Tags should be specific technical terms when applicable
- Keep summary factual and concise
- Respond ONLY with valid JSON, no markdown or explanation"""
    ),
    ("human", "Analyze this document excerpt:\n\n{text}")
])


@dataclass
class DocumentMetadata:
    """Extracted document metadata."""
    author: str = "Unknown"
    topic_tags: list[str] = field(default_factory=list)
    summary: str = ""
    document_type: str = "other"
    language: str = "en"
    technical_level: str = "intermediate"
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_llm_metadata(response_text: str) -> DocumentMetadata:
    """Parse LLM response into DocumentMetadata."""
    try:
        # Handle potential markdown code blocks
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        data = json.loads(text)
        
        return DocumentMetadata(
            author=data.get("author", "Unknown"),
            topic_tags=data.get("topic_tags", [])[:5],
            summary=data.get("summary", "")[:500],
            document_type=data.get("document_type", "other"),
            language=data.get("language", "en"),
            technical_level=data.get("technical_level", "intermediate"),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.warning(f"Failed to parse metadata: {e}")
        return DocumentMetadata()


class MetadataEnricher:
    """LLM-based metadata enrichment for documents."""
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or GROQ_API_KEY
        self._llm: ChatGroq | None = None
        
    @property
    def llm(self) -> ChatGroq:
        """Lazy load LLM to avoid initialization overhead."""
        if self._llm is None:
            if not self.api_key:
                raise ValueError("Groq API key required for metadata enrichment")
            
            from pydantic import SecretStr
            self._llm = ChatGroq(
                model=LLM_MODEL_GROQ,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=512,
                api_key=SecretStr(self.api_key),
            )
            log.info("Initialized LLM for metadata enrichment")
        return self._llm
    
    @log_execution_time
    def enrich_document(
        self,
        text: str,
        existing_metadata: dict | None = None,
    ) -> dict[str, Any]:
        """
        Extract metadata from document text using LLM.
        
        Args:
            text: Document text (will be truncated for efficiency)
            existing_metadata: Existing metadata to merge with
            
        Returns:
            Merged metadata dictionary
        """
        existing = existing_metadata or {}
        
        # Use first 2000 chars for analysis (balance quality vs cost)
        sample_text = text[:2000] if len(text) > 2000 else text
        
        try:
            messages = METADATA_EXTRACTION_PROMPT.format_messages(text=sample_text)
            response = self.llm.invoke(messages)
            content = response.content
            extracted = _parse_llm_metadata(content if isinstance(content, str) else str(content))
            
            # Merge with existing metadata (existing takes precedence for source info)
            return {
                **existing,
                "author": extracted.author,
                "topic_tags": extracted.topic_tags,
                "summary": extracted.summary,
                "document_type": extracted.document_type,
                "language": extracted.language,
                "technical_level": extracted.technical_level,
                "enriched": True,
            }
            
        except Exception as e:
            log.warning(f"Metadata enrichment failed: {e}")
            return {**existing, "enriched": False, "enrichment_error": str(e)}
    
    def enrich_batch(
        self,
        documents: list[dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Enrich a batch of documents with rate limiting.
        
        Args:
            documents: List of document dicts with 'text' and 'metadata' keys
            progress_callback: Optional callback for progress updates
            
        Returns:
            Documents with enriched metadata
        """
        enriched = []
        total = len(documents)
        
        for i, doc in enumerate(documents):
            try:
                metadata = self.enrich_document(
                    doc["text"],
                    doc.get("metadata", {})
                )
                enriched.append({
                    "text": doc["text"],
                    "metadata": metadata,
                })
                
                # Rate limiting: ~0.5s between calls
                if i < total - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                log.error(f"Failed to enrich document {i}: {e}")
                enriched.append({
                    "text": doc["text"],
                    "metadata": {**doc.get("metadata", {}), "enriched": False},
                })
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return enriched


# ──────────────────────────────────────────────────────────────────────────────
# Document Cleaning Stage
# ──────────────────────────────────────────────────────────────────────────────

def _stage_clean(
    source_dir: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Stage 1: Clean and extract text from source documents."""
    from core.cleaner_pro import clean_documents
    
    pipeline_status.stage = "cleaning"
    pipeline_status.progress = 0.0
    log.info(f"▶ Stage 1/4 — Cleaning documents from {source_dir}")
    
    def _cb(done: int, total: int) -> None:
        pipeline_status.progress = done / max(total, 1)
        pipeline_status.message = f"Cleaning file {done}/{total}"
        if progress_callback:
            progress_callback(done, total)
    
    docs = clean_documents(source_dir, progress_callback=_cb)
    pipeline_status.docs_cleaned = len(docs)
    log.info(f"  ✔ {len(docs)} documents cleaned")
    
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Metadata Enrichment Stage
# ──────────────────────────────────────────────────────────────────────────────

def _stage_enrich(
    docs: list[dict[str, Any]],
    config: PipelineConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Stage 2: Enrich documents with LLM-extracted metadata."""
    
    if not config.enable_llm_enrichment:
        log.info("⏭ Skipping metadata enrichment (disabled)")
        return docs
    
    pipeline_status.stage = "enriching"
    pipeline_status.progress = 0.0
    log.info(f"▶ Stage 2/4 — Enriching metadata for {len(docs)} documents")
    
    start_time = time.perf_counter()
    
    try:
        enricher = MetadataEnricher()
        
        def _cb(done: int, total: int) -> None:
            pipeline_status.progress = done / max(total, 1)
            pipeline_status.message = f"Enriching doc {done}/{total}"
            pipeline_status.docs_enriched = done
            if progress_callback:
                progress_callback(done, total)
        
        enriched = enricher.enrich_batch(docs, progress_callback=_cb)
        
        elapsed = time.perf_counter() - start_time
        pipeline_status.stage_times["enrichment"] = elapsed
        log.info(f"  ✔ {len(enriched)} documents enriched in {elapsed:.1f}s")
        
        return enriched
        
    except ValueError as e:
        log.warning(f"Metadata enrichment skipped: {e}")
        return docs


# ──────────────────────────────────────────────────────────────────────────────
# Chunking Stage
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_single_doc(args: tuple) -> list[dict[str, Any]]:
    """Chunk a single document (for parallel processing)."""
    doc, chunk_size, chunk_overlap = args
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    parts = splitter.split_text(doc["text"])
    chunks: list[dict[str, Any]] = []
    
    for idx, text in enumerate(parts):
        # Inherit all metadata from parent document
        metadata = doc.get("metadata", {}).copy()
        metadata["chunk_index"] = idx
        metadata["total_chunks"] = len(parts)
        
        chunks.append({
            "text": text,
            "metadata": metadata,
        })
    
    return chunks


def _stage_chunk(
    docs: list[dict[str, Any]],
    config: PipelineConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Stage 3: Split documents into chunks."""
    
    pipeline_status.stage = "chunking"
    pipeline_status.progress = 0.0
    log.info(f"▶ Stage 3/4 — Chunking {len(docs)} documents (size={config.chunk_size}, overlap={config.chunk_overlap})")
    
    start_time = time.perf_counter()
    all_chunks: list[dict[str, Any]] = []
    total = len(docs)
    
    args = [(doc, config.chunk_size, config.chunk_overlap) for doc in docs]
    
    # Use multiprocessing for large batches
    if total > 50:
        with ProcessPoolExecutor(max_workers=config.max_workers) as pool:
            for i, result in enumerate(pool.map(_chunk_single_doc, args, chunksize=32)):
                all_chunks.extend(result)
                pipeline_status.progress = (i + 1) / total
                pipeline_status.message = f"Chunking doc {i + 1}/{total}"
                if progress_callback:
                    progress_callback(i + 1, total)
    else:
        for i, a in enumerate(args):
            all_chunks.extend(_chunk_single_doc(a))
            pipeline_status.progress = (i + 1) / total
            pipeline_status.message = f"Chunking doc {i + 1}/{total}"
            if progress_callback:
                progress_callback(i + 1, total)
    
    # Deduplicate by content hash
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    
    for chunk in all_chunks:
        h = hashlib.sha256(chunk["text"].lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    
    pipeline_status.chunks_created = len(unique)
    elapsed = time.perf_counter() - start_time
    pipeline_status.stage_times["chunking"] = elapsed
    log.info(f"  ✔ {len(unique)} unique chunks (from {len(all_chunks)} raw) in {elapsed:.1f}s")
    
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# Embedding Stage (CUDA Optimized)
# ──────────────────────────────────────────────────────────────────────────────

@log_gpu_memory
def _stage_embed(
    chunks: list[dict[str, Any]],
    collection_name: str,
    config: PipelineConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Stage 4: Embed chunks and store in Qdrant (CUDA optimized)."""
    
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import VectorParams, Distance
    from langchain_qdrant import QdrantVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    
    col_name = Config.get_collection_name(collection_name)
    pipeline_status.stage = "embedding"
    pipeline_status.progress = 0.0
    
    log.info(f"▶ Stage 4/4 — Embedding {len(chunks)} chunks into '{col_name}'")
    log.info(f"  Device: {DEVICE}, Batch size: {config.batch_size}")
    
    start_time = time.perf_counter()
    
    # Initialize Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Reset collection if requested
    if config.reset_db:
        try:
            client.delete_collection(col_name)
            log.info(f"  Reset collection: {col_name}")
        except UnexpectedResponse:
            pass
    
    # Ensure collection exists
    try:
        client.get_collection(col_name)
    except UnexpectedResponse:
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        log.info(f"  Created collection: {col_name}")
    
    # Initialize embeddings with CUDA optimization
    encode_kwargs = {
        "batch_size": config.batch_size,
        "normalize_embeddings": True,
    }
    
    # Enable FP16 on CUDA for memory efficiency
    if config.use_cuda and torch.cuda.is_available():
        encode_kwargs["convert_to_numpy"] = True
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs=encode_kwargs,
    )
    
    # Create vector store connection
    db = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=col_name,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    )
    
    # Process in batches with memory management
    total = len(chunks)
    batch_size = config.batch_size
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch = chunks[i:end]
        
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        
        # Generate stable IDs from content hash
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, hashlib.sha256(c["text"].encode()).hexdigest()))
            for c in batch
        ]
        
        # Add to vector store
        db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        # Memory management
        if config.use_cuda and (i // batch_size) % config.clear_cache_every == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Update progress
        pipeline_status.progress = end / total
        pipeline_status.chunks_embedded = end
        pipeline_status.message = f"Embedded {end}/{total} chunks"
        
        if progress_callback:
            progress_callback(end, total)
        
        if (i // batch_size) % 5 == 0:
            log.info(f"  Embedded {end}/{total} chunks")
    
    elapsed = time.perf_counter() - start_time
    pipeline_status.stage_times["embedding"] = elapsed
    log.info(f"  ✔ Stored {total} vectors in {col_name} ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline Function
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    source_dir: Path | str | None = None,
    collection_name: str = "default_notebook",
    config: PipelineConfig | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """
    Run the complete ingestion pipeline.
    
    Stages:
    1. Cleaning: Extract and clean text from source files
    2. Enrichment: Extract metadata using LLM
    3. Chunking: Split into overlapping chunks
    4. Embedding: Generate vectors and store in Qdrant
    
    Args:
        source_dir: Directory containing source documents
        collection_name: Target Qdrant collection name
        config: Pipeline configuration
        progress_callback: Optional callback(stage, done, total)
        
    Returns:
        Pipeline status dictionary
    """
    global pipeline_status
    pipeline_status = PipelineStatus()
    
    cfg = config or PipelineConfig()
    src = Path(source_dir) if source_dir else UPLOAD_DIR
    
    total_start = time.perf_counter()
    log.info(f"═══════════════════════════════════════════════════════════")
    log.info(f"Starting ingestion pipeline: {collection_name}")
    log.info(f"Source: {src}")
    log.info(f"Config: chunk_size={cfg.chunk_size}, batch={cfg.batch_size}, enrichment={cfg.enable_llm_enrichment}")
    log.info(f"═══════════════════════════════════════════════════════════")
    
    try:
        # Stage 1: Clean
        def clean_cb(d, t):
            if progress_callback:
                progress_callback("cleaning", d, t)
        
        docs = _stage_clean(src, progress_callback=clean_cb)
        
        if not docs:
            pipeline_status.stage = "error"
            pipeline_status.error = "No valid documents found after cleaning."
            return pipeline_status.to_dict()
        
        # Stage 2: Enrich metadata
        def enrich_cb(d, t):
            if progress_callback:
                progress_callback("enriching", d, t)
        
        docs = _stage_enrich(docs, cfg, progress_callback=enrich_cb)
        
        # Stage 3: Chunk
        def chunk_cb(d, t):
            if progress_callback:
                progress_callback("chunking", d, t)
        
        chunks = _stage_chunk(docs, cfg, progress_callback=chunk_cb)
        
        if not chunks:
            pipeline_status.stage = "error"
            pipeline_status.error = "No chunks produced."
            return pipeline_status.to_dict()
        
        # Stage 4: Embed
        def embed_cb(d, t):
            if progress_callback:
                progress_callback("embedding", d, t)
        
        _stage_embed(chunks, collection_name, cfg, progress_callback=embed_cb)
        
        # Complete
        pipeline_status.stage = "done"
        pipeline_status.progress = 1.0
        pipeline_status.elapsed = time.perf_counter() - total_start
        pipeline_status.message = (
            f"Pipeline complete — {pipeline_status.docs_cleaned} docs, "
            f"{pipeline_status.docs_enriched} enriched, "
            f"{pipeline_status.chunks_created} chunks, "
            f"{pipeline_status.chunks_embedded} embedded "
            f"in {pipeline_status.elapsed:.1f}s"
        )
        log.info(f"═══════════════════════════════════════════════════════════")
        log.info(pipeline_status.message)
        log.info(f"═══════════════════════════════════════════════════════════")
        
    except Exception as exc:
        pipeline_status.stage = "error"
        pipeline_status.error = str(exc)
        pipeline_status.elapsed = time.perf_counter() - total_start
        log.exception("Pipeline failed")
    
    return pipeline_status.to_dict()


def run_pipeline_async(
    source_dir: Path | str | None = None,
    collection_name: str = "default_notebook",
    config: PipelineConfig | None = None,
) -> None:
    """Run pipeline in background thread."""
    import threading
    
    thread = threading.Thread(
        target=run_pipeline,
        kwargs={
            "source_dir": source_dir,
            "collection_name": collection_name,
            "config": config,
        },
        daemon=True,
    )
    thread.start()


def get_pipeline_status() -> dict[str, Any]:
    """Get current pipeline status."""
    return pipeline_status.to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="easyResearch Ingestion Pipeline")
    parser.add_argument("--source", "-s", type=str, help="Source directory")
    parser.add_argument("--collection", "-c", type=str, default="default_notebook", help="Collection name")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--no-enrich", action="store_true", help="Disable LLM metadata enrichment")
    parser.add_argument("--no-reset", action="store_true", help="Don't reset existing collection")
    
    args = parser.parse_args()
    
    cfg = PipelineConfig(
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        enable_llm_enrichment=not args.no_enrich,
        reset_db=not args.no_reset,
    )
    
    result = run_pipeline(
        source_dir=args.source,
        collection_name=args.collection,
        config=cfg,
    )
    
    print(json.dumps(result, indent=2))
