# 🧠 easyResearch for Big Data

High-performance **RAG (Retrieval-Augmented Generation)** system combining **easyResearch**'s workspace UI with **EpsteinFiles-RAG**'s big data pipeline — featuring **multi-workspace isolation** for managing separate research projects.

## Architecture

```
                     ┌──────────────────────────────────────┐
                     │         Multi-Workspace Manager      │
                     │  Config.get_collection_name()        │
                     │  Config.get_workspace_dir()          │
                     └────────────────┬─────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Workspace A    │        │  Workspace B    │        │  Workspace C    │
│  ws_project_a   │        │  ws_project_b   │        │  ws_project_c   │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Ingestion Pipeline (ingestion_worker.py)                               │
│                                                                         │
│  Stage 1 — Clean  (multiprocessing)                                     │
│  Stage 2 — Chunk  (SHA-256 dedup)                                       │
│  Stage 3 — Embed  (GPU batched, 32/batch) → ws_{workspace} collection   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Qdrant (Docker)    │
                         │  localhost:6333     │
                         │  Collections:       │
                         │  - ws_project_a     │
                         │  - ws_project_b     │
                         │  - ws_project_c     │
                         └──────────┬──────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Hybrid Search Pipeline (generator.py)                                  │
│                                                                         │
│  Question → Contextualisation → Vector MMR                              │
│           → BM25 Scoring                                                │
│           → Cross-Encoder Reranking                                     │
│           → Hybrid Score (0.7×RE + 0.3×BM)                              │
│           → Parent Document Retrieval                                   │
│           → LLM Answer                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Workspace Isolation** — each project has its own Qdrant collection, uploads, summaries, and chat history
- **Hybrid Search** — Vector (MMR) + BM25 + Cross-Encoder reranking for high relevance
- **Qdrant Vector Database** — production-ready vector store running on Docker
- **Parent Document Retrieval** — small child chunks (400 chars) for search, large parent chunks (2000 chars) for LLM context
- **Big Data Pipeline** — parallel cleaning & chunking via `ProcessPoolExecutor`, VRAM-safe batch embedding with `torch.cuda.empty_cache()`
- **Multilingual** — `paraphrase-multilingual-MiniLM-L12-v2` embedding model (384 dimensions, Vietnamese + English)
- **Dual LLM** — Groq (LLaMA 3.3 70B Versatile) and Google Gemini 2.5 Flash
- **Dataset Download** — import datasets directly from HuggingFace Hub or Kaggle
- **Smart Contextualisation** — detects pronouns/follow-up patterns and reformulates questions
- **Dark Theme UI** — AnythingLLM-inspired zinc dark theme in Streamlit

## Tech Stack

| Component | Technology                                               |
| --------- | -------------------------------------------------------- |
| UI        | Streamlit                                                |
| API       | FastAPI + Uvicorn                                        |
| Vector DB | **Qdrant** (Docker, localhost:6333)                      |
| Embedding | sentence-transformers (MiniLM-L12-v2, 384 dim)           |
| Reranker  | cross-encoder (ms-marco-MiniLM-L-6-v2)                   |
| BM25      | rank-bm25                                                |
| LLM       | Groq (LLaMA 3.3 70B Versatile) / Google Gemini 2.5 Flash |
| Framework | LangChain + langchain-qdrant                             |
| GPU       | PyTorch CUDA 12.6                                        |

## Project Structure

```
easyResearchforBigData/
├── app.py                    # Streamlit UI
├── main.py                   # FastAPI REST API
├── config.py                 # Central configuration + Config class
├── requirements.txt
├── .env                      # API keys (GROQ_API_KEY, GOOGLE_API_KEY)
├── .gitignore
├── core/
│   ├── cleaner_pro.py        # Multi-format text extraction & cleaning
│   ├── loader.py             # Document loading & smart splitting
│   ├── embedder.py           # Embedding & Qdrant management
│   ├── generator.py          # Hybrid RAG search & LLM generation
│   ├── summarizer.py         # Auto-summarisation
│   └── ingestion_worker.py   # Full pipeline: Clean → Chunk → Embed
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

**FastAPI (optional):**

```bash
uvicorn main:app --reload
```

## API Endpoints

| Method | Endpoint            | Description                       |
| ------ | ------------------- | --------------------------------- |
| POST   | `/ask`              | Query the hybrid RAG pipeline     |
| POST   | `/upload`           | Upload & embed a document         |
| GET    | `/notebooks`        | List all workspaces               |
| GET    | `/stats/{name}`     | Workspace statistics              |
| DELETE | `/notebooks/{name}` | Delete a workspace                |
| GET    | `/health`           | Health check (GPU, Qdrant status) |

## Configuration

Key settings in `config.py`:

| Parameter               | Default                               | Description                       |
| ----------------------- | ------------------------------------- | --------------------------------- |
| `QDRANT_HOST`           | localhost                             | Qdrant server host                |
| `QDRANT_PORT`           | 6333                                  | Qdrant server port                |
| `QDRANT_VECTOR_SIZE`    | 384                                   | Vector dimensions (MiniLM-L12-v2) |
| `EMBEDDING_MODEL`       | paraphrase-multilingual-MiniLM-L12-v2 | Multilingual embedding            |
| `EMBED_BATCH_SIZE`      | 32                                    | Batch size for 4 GB VRAM          |
| `DEFAULT_CHUNK_SIZE`    | 400                                   | Characters per chunk              |
| `DEFAULT_CHUNK_OVERLAP` | 80                                    | Overlap between chunks            |
| `SEARCH_K`              | 12                                    | Final docs returned               |
| `HYBRID_WEIGHT_RERANK`  | 0.7                                   | Cross-encoder weight              |
| `HYBRID_WEIGHT_BM25`    | 0.3                                   | BM25 keyword weight               |
| `MAX_WORKERS`           | cpu_count - 2                         | Parallel cleaning/chunking        |
| `LLM_MODEL_GROQ`        | llama-3.3-70b-versatile               | Groq model ID                     |
| `LLM_MODEL_GEMINI`      | gemini-2.5-flash                      | Gemini model ID                   |

### Config Class Methods

| Method                           | Returns                               |
| -------------------------------- | ------------------------------------- |
| `Config.get_collection_name(ws)` | `ws_{workspace_name}`                 |
| `Config.get_workspace_dir(ws)`   | `Path("uploads/{workspace}/")`        |
| `Config.get_summary_dir(ws)`     | `Path("database/summaries/{ws}/")`    |
| `Config.get_chat_dir(ws)`        | `Path("database/chat_history/{ws}/")` |

## Hardware Requirements

- **GPU:** NVIDIA RTX 3050 (4 GB VRAM) or better
- **CPU:** Any multi-core processor
- **RAM:** 16 GB recommended
- **Storage:** Depends on dataset size
- **Docker:** Required for Qdrant

## License

MIT License - See [LICENSE](LICENSE) file for more details.

---

<p align="center">
  Made with ❤️ by easyResearch
</p>
