import os, shutil, tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.loader import load_and_split_document
from core.embedder import (
    add_to_vector_db,
    get_all_notebooks,
    get_notebook_stats,
    delete_notebook,
    get_total_db_size,
    check_qdrant_health,
)
from core.generator import query_rag_system
from core.summarizer import generate_notebook_summary
from config import Config, QDRANT_HOST, QDRANT_PORT


app = FastAPI(
    title="easyResearch for Big Data",
    version="2.0.0",
    description="High-performance RAG API with hybrid search, reranking & Big Data pipelines",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    collection_name: str = "Default_Project"
    chat_history: list = Field(default_factory=list)
    k_target: int = 10
    format_filter: str | None = None
    source_filter: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    standalone_question: str | None = None
    pipeline_info: dict = Field(default_factory=dict)


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    collection: str



@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        result = query_rag_system(
            req.question,
            collection_name=req.collection_name,
            chat_history=req.chat_history,
            k_target=req.k_target,
            format_filter=req.format_filter,
            source_filter=req.source_filter,
        )
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            standalone_question=result.get("standalone_question"),
            pipeline_info=result.get("pipeline_info", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    collection_name: str = Form("Default_Project"),
):
    suffix = Path(file.filename).suffix
    upload_dir = Config.get_workspace_dir(collection_name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(upload_dir)) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = load_and_split_document(tmp_path)
        add_to_vector_db(chunks, collection_name=collection_name)
        return UploadResponse(
            filename=file.filename,
            chunks=len(chunks),
            collection=collection_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/notebooks")
async def list_notebooks():
    notebooks = get_all_notebooks()
    return {"notebooks": notebooks, "count": len(notebooks)}


@app.get("/stats/{collection_name}")
async def stats(collection_name: str):
    return get_notebook_stats(collection_name)


@app.delete("/notebooks/{collection_name}")
async def remove_notebook(collection_name: str):
    success = delete_notebook(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"deleted": collection_name}


@app.get("/health")
async def health():
    import torch
    qdrant_status = check_qdrant_health()
    return {
        "status": "ok" if qdrant_status.get("status") == "ok" else "degraded",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "db_size_mb": get_total_db_size(),
        "qdrant": qdrant_status,
    }


@app.on_event("startup")
async def startup_event():
    status = check_qdrant_health()
    if status.get("status") != "ok":
        print(f"⚠️ Qdrant health check failed: {status.get('error')}")
        print(f"   Make sure Qdrant is running at {QDRANT_HOST}:{QDRANT_PORT}")
    else:
        print(f"✅ Qdrant connected: {QDRANT_HOST}:{QDRANT_PORT}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
