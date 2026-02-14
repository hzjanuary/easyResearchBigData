import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
UPLOAD_DIR      = BASE_DIR / "uploads"
CHROMA_DIR      = str(BASE_DIR / "database" / "chroma_db")
CHAT_DIR        = str(BASE_DIR / "database" / "chat_history")
LOG_FILE        = BASE_DIR / "database" / "ingestion.log"

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Cleaning / Chunking ────────────────────────────────────────────────────
MIN_DOC_LENGTH      = 100          # ignore docs shorter than this (chars)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".py", ".js", ".json", ".csv"}

# Parent Document Retrieval — small child chunks for search, large parents for context
PARENT_CHUNK_SIZE   = 2000
CHILD_CHUNK_SIZE    = 400
PARENT_OVERLAP      = 200
CHILD_OVERLAP       = 50

# Big‑Data ingestion overrides (UI‑tuneable)
DEFAULT_CHUNK_SIZE  = 400
DEFAULT_CHUNK_OVERLAP = 80
MAX_WORKERS         = max(1, (os.cpu_count() or 4) - 2)  # leave 2 cores free

# ── Embedding (RTX 3050 4 GB VRAM) ─────────────────────────────────────────
EMBEDDING_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE    = 32           # STRICT small batches → fits in 4 GB VRAM
EMBED_DEVICE        = DEVICE

# ── Reranker ────────────────────────────────────────────────────────────────
RERANKER_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Retrieval ───────────────────────────────────────────────────────────────
SEARCH_TYPE         = "mmr"        # Maximal Marginal Relevance (diversity)
SEARCH_K            = 12           # final docs returned
SEARCH_FETCH_K      = 60           # candidates considered before MMR re-rank
HYBRID_WEIGHT_RERANK = 0.7
HYBRID_WEIGHT_BM25   = 0.3
MIN_SCORE_THRESHOLD  = 0.1

# ── LLM ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL_GROQ      = "llama-3.3-70b-versatile"
LLM_MODEL_GEMINI    = "gemini-2.5-flash"
LLM_TEMPERATURE     = 0.2
LLM_MAX_TOKENS      = 1024

# ── History ─────────────────────────────────────────────────────────────────
MAX_HISTORY_MESSAGES = 10

# ── Streamlit / API ────────────────────────────────────────────────────────
API_HOST            = "127.0.0.1"
API_PORT            = 8000

# ── Ensure directories exist ───────────────────────────────────────────────
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
Path(CHAT_DIR).mkdir(parents=True, exist_ok=True)
