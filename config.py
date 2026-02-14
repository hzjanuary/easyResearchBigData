import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

BASE_DIR        = Path(__file__).resolve().parent
UPLOAD_DIR      = BASE_DIR / "uploads"
CHROMA_DIR      = str(BASE_DIR / "database" / "chroma_db")
CHAT_DIR        = str(BASE_DIR / "database" / "chat_history")
LOG_FILE        = BASE_DIR / "database" / "ingestion.log"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_DOC_LENGTH       = 100
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".py", ".js", ".json", ".csv"}

PARENT_CHUNK_SIZE    = 2000
CHILD_CHUNK_SIZE     = 400
PARENT_OVERLAP       = 200
CHILD_OVERLAP        = 50

DEFAULT_CHUNK_SIZE   = 400
DEFAULT_CHUNK_OVERLAP = 80
MAX_WORKERS          = max(1, (os.cpu_count() or 4) - 2)

EMBEDDING_MODEL      = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE     = 32
EMBED_DEVICE         = DEVICE

RERANKER_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SEARCH_TYPE          = "mmr"
SEARCH_K             = 12
SEARCH_FETCH_K       = 60
HYBRID_WEIGHT_RERANK = 0.7
HYBRID_WEIGHT_BM25   = 0.3
MIN_SCORE_THRESHOLD  = 0.1

GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY       = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL_GROQ       = "llama-3.3-70b-versatile"
LLM_MODEL_GEMINI     = "gemini-2.5-flash"
LLM_TEMPERATURE      = 0.2
LLM_MAX_TOKENS       = 1024

MAX_HISTORY_MESSAGES = 10

API_HOST             = "127.0.0.1"
API_PORT             = 8000

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
Path(CHAT_DIR).mkdir(parents=True, exist_ok=True)
