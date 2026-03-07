# 🧠 easyResearch for Big Data

> **v2.1.0** — Production-Grade AI RAG System

High-performance **RAG (Retrieval-Augmented Generation)** system combining **easyResearch**'s workspace UI with **EpsteinFiles-RAG**'s big data pipeline — featuring **multi-workspace isolation**, **hybrid search**, **cross-encoder re-ranking**, and **full observability**.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.9+-red.svg)](https://qdrant.tech)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRODUCTION RAG SYSTEM v2.1                         │
└─────────────────────────────────────────────────────────────────────────────┘

                     ┌──────────────────────────────────────┐
                     │         Multi-Workspace Manager      │
                     │  Config.get_collection_name()        │
                     │  Config.get_workspace_dir()          │
                     └────────────────┬─────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Workspace A    │        │  Workspace B    │        │  Workspace C    │
│  ws_project_a   │        │  ws_project_b   │        │  ws_project_c   │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Ingestion Pipeline (pipeline.py)                           NEW v2.1   │
│                                                                         │
│  Stage 1 — Clean  (multiprocessing)                                     │
│  Stage 2 — Enrich (LLM metadata: author, tags, summary, doc_type)  ✨   │
│  Stage 3 — Chunk  (SHA-256 dedup)                                       │
│  Stage 4 — Embed  (CUDA FP16, batch=32) → ws_{workspace} collection     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Qdrant (Docker)    │
                         │  localhost:6333     │
                         │  Enriched Payloads: │
                         │  - topic_tags       │
                         │  - document_type    │
                         │  - summary          │
                         └──────────┬──────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Hybrid Search Pipeline (rag_engine.py)                     NEW v2.1   │
│                                                                         │
│  Question → Contextualisation                                           │
│           → Dense Vector Search (Qdrant)                                │
│           → BM25 Sparse Search (keyword: RPC, RMI, OSI)            ✨   │
│           → Reciprocal Rank Fusion (RRF)                           ✨   │
│           → Cross-Encoder Reranking (FP16, batch=8)                ✨   │
│           → Parent Document Retrieval                                   │
│           → LLM Answer (Groq/Gemini)                                    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Observability Layer (observability.py)                     NEW v2.1   │
│                                                                         │
│  • Full pipeline tracing (trace_id, timing, scores)                     │
│  • RAG Metrics: Hit Rate, MRR, P50/P95/P99 latency                      │
│  • JSONL trace logs: logs/rag_traces.jsonl                              │
│  • GPU memory monitoring                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

### Core RAG
- **Multi-Workspace Isolation** — each project has its own Qdrant collection, uploads, summaries, and chat history
- **Hybrid Search** — Dense vectors + BM25 sparse + Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Re-ranking** — `ms-marco-MiniLM-L-6-v2` optimized for RTX 3050 (FP16, batch=8)
- **Parent Document Retrieval** — small child chunks (400 chars) for search, large parent chunks (2000 chars) for LLM context

### Production Pipeline (v2.1) ✨
- **Metadata Enrichment** — LLM extracts `author`, `topic_tags`, `summary`, `document_type`, `language`, `technical_level`
- **Filtered Retrieval** — query by format, source, or topic tags stored in Qdrant payload
- **CUDA Optimization** — FP16 embeddings, batch processing with memory management for 4GB VRAM
- **Big Data Pipeline** — parallel cleaning & chunking via `ProcessPoolExecutor`

### Observability (v2.1) ✨
- **Full Tracing** — Query → Retrieval → Re-ranking → Generation with timing
- **RAG Metrics** — Hit Rate, MRR, P50/P95/P99 latency
- **JSONL Logs** — `logs/rag_traces.jsonl` for analysis

### API & Resilience (v2.1) ✨
- **FastAPI Backend** — OpenAPI docs at `/docs`
- **Qdrant Retry** — Exponential backoff for connection errors
- **Groq Rate Limiting** — Automatic retry with backoff
- **Batch Endpoints** — Process multiple queries for evaluation

### Other Features
- **Qdrant Vector Database** — production-ready vector store running on Docker
- **Multilingual** — `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, Vietnamese + English)
- **Dual LLM** — Groq (LLaMA 3.3 70B) / Google Gemini 2.5 Flash
- **Dataset Download** — import from HuggingFace Hub or Kaggle
- **Smart Contextualisation** — detects pronouns/follow-up patterns and reformulates questions
- **Dark Theme UI** — AnythingLLM-inspired zinc dark theme in Streamlit

## Tech Stack

| Component     | Technology                                               |
| ------------- | -------------------------------------------------------- |
| UI            | Streamlit                                                |
| API           | FastAPI + Uvicorn                                        |
| Vector DB     | **Qdrant** (Docker, localhost:6333)                      |
| Embedding     | sentence-transformers (MiniLM-L12-v2, 384 dim)           |
| Reranker      | cross-encoder (ms-marco-MiniLM-L-6-v2)                   |
| BM25          | rank-bm25                                                |
| LLM           | Groq (LLaMA 3.3 70B Versatile) / Google Gemini 2.5 Flash |
| Framework     | LangChain + langchain-qdrant                             |
| GPU           | PyTorch CUDA 12.6                                        |
| Observability | Custom tracing + JSONL logs ✨                            |
| Resilience    | tenacity (retry with backoff) ✨                          |

## Usage Examples (v2.1)

### Using the New RAG Engine

```python
from core.rag_engine import query_rag, RetrievalConfig

# Configure retrieval
config = RetrievalConfig(
    dense_k=20,           # Initial dense retrieval
    sparse_k=20,          # BM25 retrieval
    rerank_top_k=10,      # Final docs after re-ranking
    reranker_batch_size=8 # Optimized for RTX 3050
)

# Execute query with hybrid search
result = query_rag(
    question="What is RPC in distributed systems?",
    collection_name="network_programming",
    chat_history=[],
    k_target=10,
    llm_provider="groq",
    retrieval_config=config
)

print(result["answer"])
print(result["sources"])
print(result["pipeline_info"])  # trace_id, timing info
```

### Running the Enrichment Pipeline

```python
from core.pipeline import run_pipeline, PipelineConfig

config = PipelineConfig(
    chunk_size=400,
    chunk_overlap=80,
    batch_size=32,
    enable_llm_enrichment=True,  # Extract metadata with LLM
    reset_db=True
)

result = run_pipeline(
    source_dir="uploads/my_workspace",
    collection_name="my_workspace",
    config=config
)

print(f"Processed: {result['docs_cleaned']} docs, {result['chunks_embedded']} chunks")
```

### Accessing Metrics

```python
from core.observability import get_current_metrics, get_recent_traces

# Get RAG performance metrics
metrics = get_current_metrics()
print(f"Hit Rate: {metrics['hit_rate']:.2%}")
print(f"MRR: {metrics['mrr']:.3f}")
print(f"P95 Latency: {metrics['p95_total_time_ms']:.0f}ms")

# Get recent traces for debugging
traces = get_recent_traces(limit=10)
for trace in traces:
    print(f"[{trace['trace_id']}] {trace['original_query'][:50]}... ({trace['total_time_ms']:.0f}ms)")
```

## Project Structure

```
easyResearchforBigData/
├── app.py                    # Streamlit UI
├── main.py                   # Legacy FastAPI (backward compat)
├── api_main.py               # ✨ Production FastAPI v2.1 (use this)
├── config.py                 # Central configuration + Config class
├── requirements.txt
├── .env                      # API keys (GROQ_API_KEY, GOOGLE_API_KEY)
├── .gitignore
├── core/
│   ├── __init__.py           # ✨ Module exports
│   ├── rag_engine.py         # ✨ Hybrid search + re-ranking (NEW)
│   ├── pipeline.py           # ✨ Metadata enrichment pipeline (NEW)
│   ├── observability.py      # ✨ Logging, tracing, metrics (NEW)
│   ├── cleaner_pro.py        # Multi-format text extraction & cleaning
│   ├── loader.py             # Document loading & smart splitting
│   ├── embedder.py           # Embedding & Qdrant management
│   ├── generator.py          # Legacy RAG (use rag_engine.py)
│   ├── summarizer.py         # Auto-summarisation
│   └── ingestion_worker.py   # Legacy pipeline (use pipeline.py)
├── logs/                     # ✨ Observability logs (NEW)
│   ├── rag_pipeline.log      # General RAG logs
│   ├── rag_traces.jsonl      # JSONL trace logs
│   └── rag_metrics.jsonl     # Metrics data
├── uploads/
│   └── {workspace_name}/     # Per-workspace uploads
└── database/
    ├── summaries/
    │   └── {workspace_name}/ # Per-workspace summaries
    └── chat_history/
        └── {workspace_name}/ # Per-workspace chat JSON
```

## Multi-Workspace Architecture

Each workspace is completely isolated with its own:

| Resource          | Path / Name                               |
| ----------------- | ----------------------------------------- |
| Qdrant Collection | `ws_{workspace_name}`                     |
| Upload Directory  | `uploads/{workspace_name}/`               |
| Summary Directory | `database/summaries/{workspace_name}/`    |
| Chat History      | `database/chat_history/{workspace_name}/` |

The `Config` class in `config.py` provides centralized workspace management:

```python
from config import Config

# Get workspace-specific paths
upload_dir = Config.get_workspace_dir("My Project")     # uploads/my_project/
summary_dir = Config.get_summary_dir("My Project")      # database/summaries/my_project/
chat_dir = Config.get_chat_dir("My Project")            # database/chat_history/my_project/
collection = Config.get_collection_name("My Project")   # ws_my_project
```

## Quick Start

### 1. Start Qdrant (Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or with persistent storage:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 2. Clone & install

```bash
git clone <repo-url>
cd easyResearchforBigData
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file:

```env
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...

# Optional Qdrant settings (defaults shown)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 4. Run

**Streamlit UI:**

```bash
streamlit run app.py
```

**FastAPI v2.1 (Production):**

```bash
uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload
```

**Pipeline CLI:**

```bash
python -m core.pipeline --source uploads/my_workspace --collection my_workspace
```

**Legacy FastAPI (backward compat):**

```bash
uvicorn main:app --reload
```

## API Endpoints

### Query Endpoints
| Method | Endpoint                | Description                                |
| ------ | ----------------------- | ------------------------------------------ |
| POST   | `/ask`                  | RAG query with hybrid search + re-ranking  |
| GET    | `/search/{collection}`  | Semantic search without LLM generation     |
| POST   | `/batch/query`          | Process multiple queries (max 20)          |

### Ingestion Endpoints
| Method | Endpoint            | Description                                    |
| ------ | ------------------- | ---------------------------------------------- |
| POST   | `/upload`           | Upload & embed a single document               |
| POST   | `/pipeline/start`   | Start background ingestion with enrichment ✨  |
| GET    | `/pipeline/status`  | Get current pipeline status                    |

### Workspace Endpoints
| Method | Endpoint                           | Description                  |
| ------ | ---------------------------------- | ---------------------------- |
| GET    | `/workspaces`                      | List all workspaces          |
| GET    | `/workspaces/{name}`               | Workspace statistics         |
| DELETE | `/workspaces/{name}`               | Delete workspace             |
| DELETE | `/workspaces/{name}/files/{file}`  | Remove file from workspace   |

### Observability Endpoints ✨
| Method | Endpoint   | Description                              |
| ------ | ---------- | ---------------------------------------- |
| GET    | `/health`  | Health check (GPU, Qdrant, version)      |
| GET    | `/metrics` | RAG metrics (Hit Rate, MRR, latency)     |
| GET    | `/traces`  | Recent pipeline traces                   |
| DELETE | `/logs`    | Clear all log files                      |

## Configuration

### Core Settings (`config.py`)

| Parameter               | Default                               | Description                       |
| ----------------------- | ------------------------------------- | --------------------------------- |
| `QDRANT_HOST`           | localhost                             | Qdrant server host                |
| `QDRANT_PORT`           | 6333                                  | Qdrant server port                |
| `QDRANT_VECTOR_SIZE`    | 384                                   | Vector dimensions (MiniLM-L12-v2) |
| `EMBEDDING_MODEL`       | paraphrase-multilingual-MiniLM-L12-v2 | Multilingual embedding            |
| `EMBED_BATCH_SIZE`      | 32                                    | Batch size for 4 GB VRAM          |
| `RERANKER_MODEL`        | cross-encoder/ms-marco-MiniLM-L-6-v2  | Cross-encoder for re-ranking      |
| `DEFAULT_CHUNK_SIZE`    | 400                                   | Characters per chunk              |
| `DEFAULT_CHUNK_OVERLAP` | 80                                    | Overlap between chunks            |
| `SEARCH_K`              | 12                                    | Final docs returned               |
| `HYBRID_WEIGHT_RERANK`  | 0.7                                   | Cross-encoder weight              |
| `HYBRID_WEIGHT_BM25`    | 0.3                                   | BM25 keyword weight               |
| `MAX_WORKERS`           | cpu_count - 2                         | Parallel cleaning/chunking        |
| `LLM_MODEL_GROQ`        | llama-3.3-70b-versatile               | Groq model ID                     |
| `LLM_MODEL_GEMINI`      | gemini-2.5-flash                      | Gemini model ID                   |

### RetrievalConfig (v2.1) ✨

```python
from core.rag_engine import RetrievalConfig

config = RetrievalConfig(
    dense_k=20,              # Initial dense retrieval count
    sparse_k=20,             # BM25 retrieval count
    rerank_top_k=10,         # Final docs after re-ranking
    dense_weight=0.5,        # Dense score weight
    sparse_weight=0.2,       # BM25 score weight
    rerank_weight=0.3,       # Re-ranker score weight
    min_score_threshold=0.1, # Minimum hybrid score
    reranker_batch_size=8,   # Batch size for RTX 3050
    use_fp16=True            # Half precision for memory
)
```

### PipelineConfig (v2.1) ✨

```python
from core.pipeline import PipelineConfig

config = PipelineConfig(
    chunk_size=400,              # Chunk size
    chunk_overlap=80,            # Overlap
    batch_size=32,               # Embedding batch size
    use_cuda=True,               # Use GPU
    enable_llm_enrichment=True,  # LLM metadata extraction
    enrichment_batch_size=5,     # Docs per LLM call
    max_workers=4,               # Parallel workers
    reset_db=True,               # Reset collection
    clear_cache_every=10         # GPU cache clearing frequency
)
```

### Config Class Methods

| Method                           | Returns                               |
| -------------------------------- | ------------------------------------- |
| `Config.get_collection_name(ws)` | `ws_{workspace_name}`                 |
| `Config.get_workspace_dir(ws)`   | `Path("uploads/{workspace}/")`        |
| `Config.get_summary_dir(ws)`     | `Path("database/summaries/{ws}/")`    |
| `Config.get_chat_dir(ws)`        | `Path("database/chat_history/{ws}/")` |

## Hardware Requirements

- **GPU:** NVIDIA RTX 3050 (4 GB VRAM) or better — optimized for low VRAM
- **CPU:** Any multi-core processor
- **RAM:** 16 GB recommended
- **Storage:** Depends on dataset size
- **Docker:** Required for Qdrant

### GPU Memory Optimization (v2.1)

The system is specifically optimized for RTX 3050 with 4GB VRAM:

| Component       | Optimization                              |
| --------------- | ----------------------------------------- |
| Cross-Encoder   | FP16 precision, batch size 8              |
| Embeddings      | Batch size 32, cache cleared every 10 batches |
| Re-ranker       | Lazy loading, truncated inputs (512 tokens) |
| GPU Cache       | `torch.cuda.empty_cache()` after batches  |

## License

MIT License - See [LICENSE](LICENSE) file for more details.

---

<p align="center">
  Made with ❤️ by easyResearch<br>
  <sub>v2.1.0 — Production-Grade RAG System</sub>
</p>
