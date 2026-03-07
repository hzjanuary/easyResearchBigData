"""
easyResearch Core Module.

Production-grade RAG components:
- rag_engine: Hybrid search with re-ranking
- pipeline: Metadata enrichment pipeline  
- observability: Logging and metrics
- embedder: Vector embedding utilities
- loader: Document loading and splitting
- generator: Legacy RAG functions (use rag_engine for new code)
"""

# Re-export main components for convenience
from core.rag_engine import (
    query_rag,
    query_rag_system,  # Backward compatibility
    RetrievalConfig,
    hybrid_search,
    rerank_documents,
)

from core.pipeline import (
    run_pipeline,
    run_pipeline_async,
    get_pipeline_status,
    PipelineConfig,
    PipelineStatus,
    MetadataEnricher,
)

from core.observability import (
    RAGTracer,
    RAGMetrics,
    MetricsCalculator,
    rag_logger,
    get_current_metrics,
    get_recent_traces,
)

from core.embedder import (
    get_vector_store,
    add_to_vector_db,
    get_all_notebooks,
    get_notebook_stats,
    delete_notebook,
    check_qdrant_health,
)

from core.loader import (
    load_and_split_document,
    load_document_simple,
)

__all__ = [
    # RAG Engine
    "query_rag",
    "query_rag_system",
    "RetrievalConfig",
    "hybrid_search",
    "rerank_documents",
    # Pipeline
    "run_pipeline",
    "run_pipeline_async",
    "get_pipeline_status",
    "PipelineConfig",
    "PipelineStatus",
    "MetadataEnricher",
    # Observability
    "RAGTracer",
    "RAGMetrics",
    "MetricsCalculator",
    "rag_logger",
    "get_current_metrics",
    "get_recent_traces",
    # Embedder
    "get_vector_store",
    "add_to_vector_db",
    "get_all_notebooks",
    "get_notebook_stats",
    "delete_notebook",
    "check_qdrant_health",
    # Loader
    "load_and_split_document",
    "load_document_simple",
]
