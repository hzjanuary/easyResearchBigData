# 🧠 easyResearch for Big Data

High-performance **RAG (Retrieval-Augmented Generation)** system combining **easyResearch**'s workspace UI with **EpsteinFiles-RAG**'s big data pipeline. Designed for consumer hardware — **RTX 3050 (4 GB VRAM)** + **Huananzhi B760** CPU.

## Architecture

```
Upload / Download Dataset
         │
         ▼
┌─────────────────────────────────────────────┐
│  Ingestion Pipeline (ingestion_worker.py)   │
│                                             │
│  Stage 1 — Clean  (multiprocessing)         │
│  Stage 2 — Chunk  (SHA-256 dedup)           │
│  Stage 3 — Embed  (GPU batched, 32/batch)   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
            ┌──────────┐
            │ ChromaDB │
            └────┬─────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Hybrid Search Pipeline (generator.py)      │
│                                             │
│  Question → Contextualisation → Vector MMR  │
│           → BM25 Scoring                    │
│           → Cross-Encoder Reranking         │
│           → Hybrid Score (0.7×RE + 0.3×BM)  │
│           → Parent Document Retrieval       │
│           → LLM Answer                      │
└─────────────────────────────────────────────┘
```

## Features

- **Hybrid Search** — Vector (MMR) + BM25 + Cross-Encoder reranking for high relevance
- **Parent Document Retrieval** — small child chunks (400 chars) for search, large parent chunks (2000 chars) for LLM context
- **Big Data Pipeline** — parallel cleaning & chunking via `ProcessPoolExecutor`, VRAM-safe batch embedding with `torch.cuda.empty_cache()`
- **Multilingual** — `paraphrase-multilingual-MiniLM-L12-v2` embedding model (Vietnamese + English)
- **Dual LLM** — Groq (LLaMA 3.3 70B) and Google Gemini 2.5 Flash
- **Dataset Download** — import datasets directly from HuggingFace Hub or Kaggle
- **Workspace Management** — create, switch, delete workspaces with per-workspace chat history
- **Smart Contextualisation** — detects pronouns/follow-up patterns and reformulates questions
- **Dark Theme UI** — AnythingLLM-inspired zinc dark theme in Streamlit

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Vector DB | ChromaDB |
| Embedding | sentence-transformers (MiniLM-L12-v2) |
| Reranker | cross-encoder (ms-marco-MiniLM-L-6-v2) |
| BM25 | rank-bm25 |
| LLM | Groq (LLaMA 3.3 70B) / Google Gemini 2.5 Flash |
| Framework | LangChain |
| GPU | PyTorch CUDA 12.6 |

## Project Structure

```
easyResearchforBigData/
├── app.py                    # Streamlit UI
├── main.py                   # FastAPI REST API
├── config.py                 # Central configuration
├── requirements.txt
├── .env                      # API keys (GROQ_API_KEY, GOOGLE_API_KEY)
├── .gitignore
├── core/
│   ├── cleaner_pro.py        # Multi-format text extraction & cleaning
│   ├── loader.py             # Document loading & smart splitting
│   ├── embedder.py           # Embedding & ChromaDB management
│   ├── generator.py          # Hybrid RAG search & LLM generation
│   ├── summarizer.py         # Auto-summarisation
│   └── ingestion_worker.py   # Full pipeline: Clean → Chunk → Embed
├── uploads/                  # Raw documents
└── database/
    ├── chroma_db/            # Vector database
    └── chat_history/         # Per-workspace chat JSON
```

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd easyResearchforBigData
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file:

```env
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
```

### 3. Run

**Streamlit UI:**
```bash
streamlit run app.py
```

**FastAPI (optional):**
```bash
uvicorn main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ask` | Query the hybrid RAG pipeline |
| POST | `/upload` | Upload & embed a document |
| GET | `/notebooks` | List all workspaces |
| GET | `/stats/{name}` | Workspace statistics |
| DELETE | `/notebooks/{name}` | Delete a workspace |
| GET | `/health` | Health check (GPU info, DB size) |

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | Multilingual embedding |
| `EMBED_BATCH_SIZE` | 32 | Batch size for 4 GB VRAM |
| `DEFAULT_CHUNK_SIZE` | 400 | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | 80 | Overlap between chunks |
| `SEARCH_K` | 12 | Final docs returned |
| `HYBRID_WEIGHT_RERANK` | 0.7 | Cross-encoder weight |
| `HYBRID_WEIGHT_BM25` | 0.3 | BM25 keyword weight |
| `MAX_WORKERS` | cpu_count - 2 | Parallel cleaning/chunking |

## Hardware Requirements

- **GPU:** NVIDIA RTX 3050 (4 GB VRAM) or better
- **CPU:** Any multi-core processor (optimised for Huananzhi B760)
- **RAM:** 16 GB recommended
- **Storage:** Depends on dataset size

## License

MIT
