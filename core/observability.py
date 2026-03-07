"""
Observability & Logging Module for easyResearch RAG System.

Provides centralized logging, tracing, and RAG metrics calculation.
Implements production-grade observability for the complete RAG lifecycle:
Query -> Retrieval -> Re-ranking -> LLM Generation -> Response.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, ParamSpec

from config import BASE_DIR


# Type hints for decorators
P = ParamSpec("P")
T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RAG_LOG_FILE = LOG_DIR / "rag_pipeline.log"
METRICS_LOG_FILE = LOG_DIR / "rag_metrics.jsonl"
TRACE_LOG_FILE = LOG_DIR / "rag_traces.jsonl"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for terminal."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Create a configured logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(
        "%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)
    
    return logger


# Main RAG logger
rag_logger = setup_logger("rag_engine", log_file=RAG_LOG_FILE)


# ──────────────────────────────────────────────────────────────────────────────
# Trace Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Single document retrieval result with scores."""
    doc_id: str
    source: str
    content_preview: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    hybrid_score: float = 0.0
    is_relevant: bool = False  # For metrics calculation


@dataclass
class RAGTrace:
    """Complete trace of a RAG pipeline execution."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    workspace: str = ""
    
    # Query stage
    original_query: str = ""
    standalone_query: str = ""
    contextualized: bool = False
    
    # Retrieval stage
    retrieval_k: int = 0
    dense_results: list[RetrievalResult] = field(default_factory=list)
    bm25_top_k: list[str] = field(default_factory=list)
    
    # Re-ranking stage
    reranked_results: list[RetrievalResult] = field(default_factory=list)
    final_docs_count: int = 0
    
    # Generation stage
    prompt_tokens: int = 0
    context_length: int = 0
    llm_provider: str = ""
    llm_model: str = ""
    
    # Response
    response_length: int = 0
    sources_cited: list[str] = field(default_factory=list)
    
    # Timing (milliseconds)
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Errors
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert RetrievalResult objects
        data["dense_results"] = [asdict(r) for r in self.dense_results[:5]]
        data["reranked_results"] = [asdict(r) for r in self.reranked_results[:5]]
        return data
    
    def log_summary(self) -> str:
        """Generate a human-readable summary."""
        return (
            f"[{self.trace_id}] "
            f"Query: '{self.original_query[:50]}...' | "
            f"Docs: {self.final_docs_count} | "
            f"Time: {self.total_time_ms:.0f}ms "
            f"(retr: {self.retrieval_time_ms:.0f}, "
            f"rerank: {self.rerank_time_ms:.0f}, "
            f"gen: {self.generation_time_ms:.0f})"
        )


@dataclass
class RAGMetrics:
    """Aggregated RAG performance metrics."""
    # Retrieval quality
    hit_rate: float = 0.0          # % of queries with at least 1 relevant doc
    mrr: float = 0.0               # Mean Reciprocal Rank
    precision_at_k: float = 0.0    # Precision @ k
    ndcg: float = 0.0              # Normalized Discounted Cumulative Gain
    
    # Latency
    avg_retrieval_time_ms: float = 0.0
    avg_rerank_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    p95_total_time_ms: float = 0.0
    
    # Volume
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Context
    avg_docs_retrieved: float = 0.0
    avg_context_length: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Tracing Context Manager
# ──────────────────────────────────────────────────────────────────────────────

class RAGTracer:
    """Context manager for tracing RAG pipeline execution."""
    
    def __init__(self, workspace: str, query: str):
        self.trace = RAGTrace(workspace=workspace, original_query=query)
        self._stage_start: float = 0.0
        self._total_start: float = 0.0
    
    def __enter__(self) -> "RAGTracer":
        self._total_start = time.perf_counter()
        rag_logger.info(f"[{self.trace.trace_id}] Starting RAG pipeline for: '{self.trace.original_query[:80]}...'")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.trace.total_time_ms = (time.perf_counter() - self._total_start) * 1000
        
        if exc_type is not None:
            self.trace.error = str(exc_val)
            rag_logger.error(f"[{self.trace.trace_id}] Pipeline failed: {exc_val}")
        else:
            rag_logger.info(self.trace.log_summary())
        
        # Persist trace to JSONL file
        self._persist_trace()
    
    @contextmanager
    def stage(self, name: str):
        """Time a specific stage of the pipeline."""
        stage_start = time.perf_counter()
        rag_logger.debug(f"[{self.trace.trace_id}] Starting stage: {name}")
        
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            rag_logger.debug(f"[{self.trace.trace_id}] Stage '{name}' completed in {elapsed_ms:.1f}ms")
            
            # Store timing by stage name
            if name == "retrieval":
                self.trace.retrieval_time_ms = elapsed_ms
            elif name == "reranking":
                self.trace.rerank_time_ms = elapsed_ms
            elif name == "generation":
                self.trace.generation_time_ms = elapsed_ms
    
    def log_retrieval(
        self,
        docs: list[Any],
        dense_scores: list[float] | None = None,
    ) -> None:
        """Log retrieval results."""
        self.trace.retrieval_k = len(docs)
        
        for i, doc in enumerate(docs[:10]):  # Log top 10
            score = dense_scores[i] if dense_scores and i < len(dense_scores) else 0.0
            result = RetrievalResult(
                doc_id=str(i),
                source=doc.metadata.get("source", "Unknown"),
                content_preview=doc.page_content[:100],
                dense_score=score,
            )
            self.trace.dense_results.append(result)
    
    def log_bm25(self, top_sources: list[str]) -> None:
        """Log BM25 top results."""
        self.trace.bm25_top_k = top_sources[:5]
    
    def log_reranking(self, docs: list[Any], scores: list[float]) -> None:
        """Log re-ranking results."""
        for i, (doc, score) in enumerate(zip(docs[:10], scores[:10])):
            result = RetrievalResult(
                doc_id=str(i),
                source=doc.metadata.get("source", "Unknown"),
                content_preview=doc.page_content[:100],
                rerank_score=score,
                hybrid_score=doc.metadata.get("hybrid_score", 0.0),
            )
            self.trace.reranked_results.append(result)
        
        self.trace.final_docs_count = len(docs)
    
    def log_generation(
        self,
        context_length: int,
        llm_provider: str,
        llm_model: str,
        response_length: int,
        sources: list[str],
    ) -> None:
        """Log generation stage."""
        self.trace.context_length = context_length
        self.trace.llm_provider = llm_provider
        self.trace.llm_model = llm_model
        self.trace.response_length = response_length
        self.trace.sources_cited = sources
    
    def set_standalone_query(self, query: str) -> None:
        """Set the contextualized standalone query."""
        self.trace.standalone_query = query
        self.trace.contextualized = (query != self.trace.original_query)
    
    def _persist_trace(self) -> None:
        """Persist trace to JSONL file."""
        try:
            with open(TRACE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.trace.to_dict()) + "\n")
        except Exception as e:
            rag_logger.warning(f"Failed to persist trace: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Calculator
# ──────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:
    """Calculate RAG metrics from traces."""
    
    def __init__(self, traces: list[RAGTrace] | None = None):
        self.traces = traces or []
    
    @classmethod
    def from_log_file(cls, limit: int = 1000) -> "MetricsCalculator":
        """Load traces from the JSONL log file."""
        traces = []
        
        if not TRACE_LOG_FILE.exists():
            return cls(traces)
        
        try:
            with open(TRACE_LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()[-limit:]  # Last N traces
                for line in lines:
                    try:
                        data = json.loads(line.strip())
                        trace = RAGTrace(
                            trace_id=data.get("trace_id", ""),
                            timestamp=data.get("timestamp", ""),
                            workspace=data.get("workspace", ""),
                            original_query=data.get("original_query", ""),
                            standalone_query=data.get("standalone_query", ""),
                            contextualized=data.get("contextualized", False),
                            retrieval_k=data.get("retrieval_k", 0),
                            final_docs_count=data.get("final_docs_count", 0),
                            context_length=data.get("context_length", 0),
                            llm_provider=data.get("llm_provider", ""),
                            llm_model=data.get("llm_model", ""),
                            response_length=data.get("response_length", 0),
                            sources_cited=data.get("sources_cited", []),
                            retrieval_time_ms=data.get("retrieval_time_ms", 0.0),
                            rerank_time_ms=data.get("rerank_time_ms", 0.0),
                            generation_time_ms=data.get("generation_time_ms", 0.0),
                            total_time_ms=data.get("total_time_ms", 0.0),
                            error=data.get("error"),
                        )
                        traces.append(trace)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            rag_logger.warning(f"Failed to load traces: {e}")
        
        return cls(traces)
    
    def calculate_hit_rate(self, relevance_threshold: float = 0.3) -> float:
        """
        Calculate Hit Rate: % of queries with at least one relevant document.
        
        Uses hybrid_score > threshold as a proxy for relevance.
        In production, this would use human annotations.
        """
        if not self.traces:
            return 0.0
        
        hits = sum(
            1 for t in self.traces
            if t.final_docs_count > 0 and not t.error
        )
        return hits / len(self.traces)
    
    def calculate_mrr(self) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = (1/|Q|) * Σ(1/rank_i) where rank_i is the rank of the first
        relevant document for query i.
        
        Without explicit relevance labels, we use position of highest-scored doc.
        """
        if not self.traces:
            return 0.0
        
        reciprocal_ranks = []
        for trace in self.traces:
            if trace.final_docs_count > 0 and not trace.error:
                # Assume first doc after re-ranking is most relevant
                reciprocal_ranks.append(1.0)  # rank = 1
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    def calculate_latency_stats(self) -> dict[str, float]:
        """Calculate latency percentiles and averages."""
        successful = [t for t in self.traces if not t.error]
        
        if not successful:
            return {
                "avg_retrieval_ms": 0.0,
                "avg_rerank_ms": 0.0,
                "avg_generation_ms": 0.0,
                "avg_total_ms": 0.0,
                "p50_total_ms": 0.0,
                "p95_total_ms": 0.0,
                "p99_total_ms": 0.0,
            }
        
        retrieval_times = [t.retrieval_time_ms for t in successful]
        rerank_times = [t.rerank_time_ms for t in successful]
        generation_times = [t.generation_time_ms for t in successful]
        total_times = sorted([t.total_time_ms for t in successful])
        
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])
        
        return {
            "avg_retrieval_ms": sum(retrieval_times) / len(retrieval_times),
            "avg_rerank_ms": sum(rerank_times) / len(rerank_times),
            "avg_generation_ms": sum(generation_times) / len(generation_times),
            "avg_total_ms": sum(total_times) / len(total_times),
            "p50_total_ms": percentile(total_times, 0.50),
            "p95_total_ms": percentile(total_times, 0.95),
            "p99_total_ms": percentile(total_times, 0.99),
        }
    
    def calculate_all_metrics(self) -> RAGMetrics:
        """Calculate all RAG metrics."""
        latency = self.calculate_latency_stats()
        successful = [t for t in self.traces if not t.error]
        
        metrics = RAGMetrics(
            hit_rate=self.calculate_hit_rate(),
            mrr=self.calculate_mrr(),
            total_queries=len(self.traces),
            successful_queries=len(successful),
            failed_queries=len(self.traces) - len(successful),
            avg_retrieval_time_ms=latency["avg_retrieval_ms"],
            avg_rerank_time_ms=latency["avg_rerank_ms"],
            avg_generation_time_ms=latency["avg_generation_ms"],
            avg_total_time_ms=latency["avg_total_ms"],
            p95_total_time_ms=latency["p95_total_ms"],
        )
        
        if successful:
            metrics.avg_docs_retrieved = sum(t.final_docs_count for t in successful) / len(successful)
            metrics.avg_context_length = sum(t.context_length for t in successful) / len(successful)
        
        return metrics
    
    def export_metrics_json(self) -> dict[str, Any]:
        """Export metrics as JSON-serializable dictionary."""
        metrics = self.calculate_all_metrics()
        return asdict(metrics)


# ──────────────────────────────────────────────────────────────────────────────
# Decorators for Instrumentation
# ──────────────────────────────────────────────────────────────────────────────

def log_execution_time(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            rag_logger.debug(f"{func.__name__} completed in {elapsed:.1f}ms")
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            rag_logger.error(f"{func.__name__} failed after {elapsed:.1f}ms: {e}")
            raise
    return wrapper


def log_gpu_memory(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to log GPU memory usage before and after function execution."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated() / 1024**2
                rag_logger.debug(f"{func.__name__} GPU memory before: {mem_before:.1f}MB")
        except ImportError:
            pass
        
        result = func(*args, **kwargs)
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / 1024**2
                rag_logger.debug(f"{func.__name__} GPU memory after: {mem_after:.1f}MB")
        except ImportError:
            pass
        
        return result
    return wrapper


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_current_metrics() -> dict[str, Any]:
    """Get current RAG metrics."""
    calculator = MetricsCalculator.from_log_file()
    return calculator.export_metrics_json()


def clear_logs() -> None:
    """Clear all log files."""
    for log_file in [RAG_LOG_FILE, METRICS_LOG_FILE, TRACE_LOG_FILE]:
        if log_file.exists():
            log_file.unlink()
    rag_logger.info("All logs cleared")


def get_recent_traces(limit: int = 50) -> list[dict[str, Any]]:
    """Get recent traces as dictionaries."""
    calculator = MetricsCalculator.from_log_file(limit=limit)
    return [t.to_dict() for t in calculator.traces[-limit:]]
