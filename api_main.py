"""
Production-Grade FastAPI Backend for easyResearch RAG System.

Features:
- Hybrid search with re-ranking
- Qdrant connection retry with exponential backoff
- Groq API rate limiting with retry
- Batch processing endpoints
- Full observability integration
- OpenAPI documentation
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    Config,
    QDRANT_HOST,
    QDRANT_PORT,
    API_HOST,
    API_PORT,
    DEVICE,
)

# Import core modules
from core.rag_engine import query_rag, RetrievalConfig
from core.pipeline import run_pipeline, run_pipeline_async, get_pipeline_status, PipelineConfig
from core.embedder import (
    add_to_vector_db,
    get_all_notebooks,
    get_notebook_stats,
    delete_notebook,
    delete_file_from_notebook,
    get_total_db_size,
    check_qdrant_health,
)
from core.loader import load_and_split_document
from core.observability import (
    rag_logger,
    get_current_metrics,
    get_recent_traces,
    clear_logs,
    MetricsCalculator,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Constants
# ──────────────────────────────────────────────────────────────────────────────

API_VERSION = "0.1.0"
API_TITLE = "easyResearch RAG API"
API_DESCRIPTION = """
Production-grade RAG API with hybrid search, re-ranking, and Big Data pipelines.

## Features
- **Hybrid Search**: Dense vectors + BM25 sparse retrieval
- **Cross-Encoder Re-ranking**: ms-marco-MiniLM-L-6-v2
- **Metadata Enrichment**: LLM-extracted tags and summaries
- **Full Observability**: Tracing, metrics, and logging

## Workspaces
Each workspace has isolated documents and chat history.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Retry Decorators for Resilience
# ──────────────────────────────────────────────────────────────────────────────

class QdrantConnectionError(Exception):
    """Custom exception for Qdrant connection issues."""
    pass


class GroqRateLimitError(Exception):
    """Custom exception for Groq rate limiting."""
    pass


def with_qdrant_retry(func):
    """Decorator for Qdrant operations with exponential backoff."""
    @functools.wraps(func)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, QdrantConnectionError)),
        reraise=True,
    )
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                rag_logger.warning(f"Qdrant connection error, retrying: {e}")
                raise QdrantConnectionError(str(e)) from e
            raise
    return wrapper


def with_groq_rate_limit(func):
    """Decorator for Groq API calls with rate limit handling."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit indicators
                if "rate" in error_msg or "429" in error_msg or "limit" in error_msg:
                    delay = base_delay * (2 ** attempt)
                    rag_logger.warning(f"Rate limited, waiting {delay}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(delay)
                    
                    if attempt == max_retries - 1:
                        raise GroqRateLimitError(f"Rate limit exceeded after {max_retries} retries") from e
                else:
                    raise
        
        raise GroqRateLimitError("Max retries exceeded")
    return wrapper


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., min_length=1, max_length=4000, description="User's question")
    collection_name: str = Field(default="Default_Project", description="Workspace/collection name")
    chat_history: list[dict] = Field(default_factory=list, description="Previous conversation messages")
    k_target: int = Field(default=10, ge=1, le=50, description="Number of documents to retrieve")
    format_filter: str | None = Field(default=None, description="Filter by document format")
    source_filter: str | None = Field(default=None, description="Filter by source filename")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RPC in distributed systems?",
                "collection_name": "network_programming",
                "k_target": 10,
            }
        }


class AskResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: list[str]
    standalone_question: str | None = None
    pipeline_info: dict = Field(default_factory=dict)
    raw_docs: list[dict] | None = None


class UploadRequest(BaseModel):
    """Request model for file upload configuration."""
    collection_name: str = Field(default="Default_Project")
    use_parent_retrieval: bool = Field(default=True, description="Enable parent-child chunking")


class UploadResponse(BaseModel):
    """Response model for file upload."""
    filename: str
    chunks: int
    collection: str
    metadata: dict | None = None


class PipelineRequest(BaseModel):
    """Request model for ingestion pipeline."""
    collection_name: str = Field(default="Default_Project")
    source_dir: str | None = Field(default=None, description="Source directory path")
    chunk_size: int = Field(default=400, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=500)
    batch_size: int = Field(default=32, ge=1, le=128)
    enable_enrichment: bool = Field(default=True, description="Enable LLM metadata enrichment")
    reset_db: bool = Field(default=True, description="Reset existing collection")


class PipelineResponse(BaseModel):
    """Response model for pipeline status."""
    stage: str
    progress: float
    message: str
    error: str | None = None
    docs_cleaned: int = 0
    docs_enriched: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    elapsed: float = 0.0


class WorkspaceStats(BaseModel):
    """Workspace statistics model."""
    name: str
    chunks: int
    files: list[str]
    size_mb: float
    metadata: dict | None = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: Literal["ok", "degraded", "error"]
    version: str
    device: str
    gpu_name: str | None
    gpu_memory_mb: float | None
    qdrant: dict
    db_size_mb: float


class MetricsResponse(BaseModel):
    """RAG metrics response model."""
    hit_rate: float
    mrr: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_retrieval_time_ms: float
    avg_rerank_time_ms: float
    avg_generation_time_ms: float
    avg_total_time_ms: float
    p95_total_time_ms: float
    avg_docs_retrieved: float
    avg_context_length: float


# ──────────────────────────────────────────────────────────────────────────────
# Application Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    rag_logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    rag_logger.info(f"Device: {DEVICE}")
    
    # Check Qdrant connection
    status = check_qdrant_health()
    if status.get("status") == "ok":
        rag_logger.info(f"✅ Qdrant connected: {QDRANT_HOST}:{QDRANT_PORT}")
    else:
        rag_logger.warning(f"⚠️ Qdrant health check failed: {status.get('error')}")
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        rag_logger.info(f"🖥️ GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    
    yield
    
    # Shutdown
    rag_logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(QdrantConnectionError)
async def qdrant_error_handler(request, exc: QdrantConnectionError):
    rag_logger.error(f"Qdrant connection error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Database connection error",
            "detail": str(exc),
            "retry_after": 5,
        },
    )


@app.exception_handler(GroqRateLimitError)
async def rate_limit_handler(request, exc: GroqRateLimitError):
    rag_logger.warning(f"Rate limit error: {exc}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc),
            "retry_after": 60,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Query Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["Query"])
@with_groq_rate_limit
async def ask_question(req: AskRequest):
    """
    Execute a RAG query with hybrid search and re-ranking.
    
    The query goes through:
    1. Question contextualization (if chat history exists)
    2. Dense vector search
    3. BM25 sparse ranking
    4. Reciprocal Rank Fusion
    5. Cross-encoder re-ranking
    6. LLM response generation
    """
    try:
        config = RetrievalConfig(rerank_top_k=req.k_target)
        
        result = query_rag(
            question=req.question,
            collection_name=req.collection_name,
            chat_history=req.chat_history,
            k_target=req.k_target,
            format_filter=req.format_filter,
            source_filter=req.source_filter,
            retrieval_config=config,
        )
        
        return AskResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            standalone_question=result.get("standalone_question"),
            pipeline_info=result.get("pipeline_info", {}),
            raw_docs=result.get("raw_docs"),
        )
        
    except Exception as e:
        rag_logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/{collection_name}", tags=["Query"])
async def search_documents(
    collection_name: str,
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(default=10, ge=1, le=50, description="Number of results"),
    format_filter: str | None = Query(default=None, description="Filter by format"),
):
    """
    Perform semantic search without LLM generation.
    
    Returns raw document matches with scores.
    """
    from core.rag_engine import hybrid_search
    
    try:
        filter_dict = {"format": format_filter} if format_filter else None
        config = RetrievalConfig(rerank_top_k=k)
        
        results = hybrid_search(collection_name, q, config=config, filter_dict=filter_dict)
        
        return {
            "query": q,
            "count": len(results),
            "results": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "score": round(score, 4),
                    "content": doc.page_content[:500],
                    "metadata": {
                        k: v for k, v in doc.metadata.items()
                        if k not in ["parent_content"]
                    },
                }
                for doc, score in results
            ],
        }
        
    except Exception as e:
        rag_logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
@with_qdrant_retry
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="Default_Project"),
    use_parent_retrieval: bool = Form(default=True),
):
    """
    Upload and index a single document.
    
    Supports: PDF, DOCX, TXT, PY, JS, JSON, CSV
    """
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    allowed_extensions = {".pdf", ".docx", ".txt", ".py", ".js", ".json", ".csv"}
    
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(allowed_extensions)}",
        )
    
    upload_dir = Config.get_workspace_dir(collection_name)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(upload_dir)) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        chunks = load_and_split_document(tmp_path, use_parent_retrieval=use_parent_retrieval)
        add_to_vector_db(chunks, collection_name=collection_name)
        
        return UploadResponse(
            filename=filename,
            chunks=len(chunks),
            collection=collection_name,
            metadata={"parent_retrieval": use_parent_retrieval},
        )
        
    except Exception as e:
        rag_logger.exception(f"Upload failed: {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/pipeline/start", response_model=PipelineResponse, tags=["Ingestion"])
async def start_pipeline(
    req: PipelineRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start the ingestion pipeline in the background.
    
    Stages:
    1. Document cleaning and text extraction
    2. LLM metadata enrichment (optional)
    3. Chunking with deduplication
    4. CUDA-accelerated embedding
    """
    config = PipelineConfig(
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        batch_size=req.batch_size,
        enable_llm_enrichment=req.enable_enrichment,
        reset_db=req.reset_db,
    )
    
    source_dir = Path(req.source_dir) if req.source_dir else Config.get_workspace_dir(req.collection_name)
    
    # Start in background
    background_tasks.add_task(
        run_pipeline,
        source_dir=source_dir,
        collection_name=req.collection_name,
        config=config,
    )
    
    return PipelineResponse(
        stage="starting",
        progress=0.0,
        message=f"Pipeline started for {req.collection_name}",
    )


@app.get("/pipeline/status", response_model=PipelineResponse, tags=["Ingestion"])
async def pipeline_status():
    """Get current pipeline execution status."""
    status = get_pipeline_status()
    return PipelineResponse(**status)


# ──────────────────────────────────────────────────────────────────────────────
# Workspace Management Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/workspaces", tags=["Workspace"])
async def list_workspaces():
    """List all available workspaces."""
    workspaces = get_all_notebooks()
    return {
        "workspaces": workspaces,
        "count": len(workspaces),
    }


@app.get("/workspaces/{workspace_name}", response_model=WorkspaceStats, tags=["Workspace"])
async def get_workspace(workspace_name: str):
    """Get detailed statistics for a workspace."""
    stats = get_notebook_stats(workspace_name)
    return WorkspaceStats(
        name=workspace_name,
        chunks=stats.get("chunks", 0),
        files=stats.get("files", []),
        size_mb=stats.get("size_mb", 0.0),
    )


@app.delete("/workspaces/{workspace_name}", tags=["Workspace"])
async def remove_workspace(workspace_name: str):
    """Delete a workspace and all its data."""
    success = delete_notebook(workspace_name)
    
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Also clean up upload directory
    upload_dir = Config.get_workspace_dir(workspace_name)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    
    return {"deleted": workspace_name, "status": "success"}


@app.delete("/workspaces/{workspace_name}/files/{filename}", tags=["Workspace"])
async def remove_file_from_workspace(workspace_name: str, filename: str):
    """Remove a specific file from a workspace."""
    deleted_count = delete_file_from_notebook(workspace_name, filename)
    
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="File not found in workspace")
    
    return {
        "deleted": filename,
        "chunks_removed": deleted_count,
        "workspace": workspace_name,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Observability Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check():
    """
    Comprehensive health check.
    
    Checks Qdrant connection, GPU availability, and system resources.
    """
    qdrant_status = check_qdrant_health()
    
    gpu_name = None
    gpu_memory = None
    device = DEVICE
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    
    status = "ok"
    if qdrant_status.get("status") != "ok":
        status = "degraded"
    
    return HealthResponse(
        status=status,
        version=API_VERSION,
        device=device,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory,
        qdrant=qdrant_status,
        db_size_mb=get_total_db_size(),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Observability"])
async def get_metrics():
    """
    Get RAG performance metrics.
    
    Includes hit rate, MRR, latency percentiles, and volume metrics.
    """
    metrics = get_current_metrics()
    return MetricsResponse(**metrics)


@app.get("/traces", tags=["Observability"])
async def get_traces(
    limit: int = Query(default=50, ge=1, le=500, description="Number of traces to return"),
):
    """Get recent RAG pipeline traces."""
    traces = get_recent_traces(limit=limit)
    return {
        "count": len(traces),
        "traces": traces,
    }


@app.delete("/logs", tags=["Observability"])
async def clear_all_logs():
    """Clear all log files (traces, metrics)."""
    clear_logs()
    return {"status": "cleared"}


# ──────────────────────────────────────────────────────────────────────────────
# Batch Processing Endpoints
# ──────────────────────────────────────────────────────────────────────────────

class BatchQueryRequest(BaseModel):
    """Request for batch query processing."""
    questions: list[str] = Field(..., min_length=1, max_length=20)
    collection_name: str = Field(default="Default_Project")
    k_target: int = Field(default=5)


@app.post("/batch/query", tags=["Batch"])
async def batch_query(req: BatchQueryRequest):
    """
    Process multiple queries in batch.
    
    Useful for evaluation and testing.
    Limited to 20 questions per batch.
    """
    results = []
    
    for i, question in enumerate(req.questions):
        try:
            result = query_rag(
                question=question,
                collection_name=req.collection_name,
                k_target=req.k_target,
            )
            results.append({
                "index": i,
                "question": question,
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "success": True,
            })
            
            # Rate limit between queries
            await asyncio.sleep(0.5)
            
        except Exception as e:
            results.append({
                "index": i,
                "question": question,
                "error": str(e),
                "success": False,
            })
    
    return {
        "total": len(req.questions),
        "successful": sum(1 for r in results if r["success"]),
        "results": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
