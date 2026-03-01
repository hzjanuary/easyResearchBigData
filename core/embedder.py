
from __future__ import annotations

import gc
from typing import Callable

import torch
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    Config,
    DEVICE,
    EMBEDDING_MODEL,
    EMBED_BATCH_SIZE,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_VECTOR_SIZE,
)

print(f"🚀 easyResearch running on device: {DEVICE.upper()}")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def check_qdrant_health() -> dict:
    try:
        client = get_qdrant_client()
        info = client.get_collections()
        return {
            "status": "ok",
            "host": f"{QDRANT_HOST}:{QDRANT_PORT}",
            "collections": [c.name for c in info.collections],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def ensure_collection_exists(collection_name: str) -> None:
    client = get_qdrant_client()
    try:
        client.get_collection(collection_name)
    except UnexpectedResponse:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"✨ Created collection: {collection_name}")


def get_vector_store(collection_name: str) -> QdrantVectorStore:
    col_name = Config.get_collection_name(collection_name)
    ensure_collection_exists(col_name)
    return QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=col_name,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    )


def add_to_vector_db(
    chunks,
    collection_name: str,
    batch_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
):
    col_name = Config.get_collection_name(collection_name)
    ensure_collection_exists(col_name)
    
    db = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=col_name,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.id for chunk in chunks]

    bs = batch_size or EMBED_BATCH_SIZE
    total = len(chunks)

    print(f"📥 Embedding {total} chunks into '{col_name}' (batch={bs}) …")

    for i in range(0, total, bs):
        end = min(i + bs, total)
        db.add_texts(
            texts=texts[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"   ✅ Batch {i} → {end}")
        if progress_callback:
            progress_callback(end, total)

    return db


def get_retriever(collection_name: str, k: int = 5, fetch_k: int = 20):
    db = get_vector_store(collection_name)
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )


def get_notebook_stats(notebook_name: str) -> dict:
    col_name = Config.get_collection_name(notebook_name)
    stats = {"chunks": 0, "files": [], "size_mb": 0.0}
    try:
        client = get_qdrant_client()
        info = client.get_collection(col_name)
        stats["chunks"] = info.points_count or 0

        if stats["chunks"] > 0:
            points, _ = client.scroll(
                collection_name=col_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            sources = {
                p.payload.get("metadata", {}).get("source", "")
                for p in points
                if p.payload
            }
            stats["files"] = sorted([s for s in sources if s])

        if hasattr(info, "disk_data_size"):
            stats["size_mb"] = round((info.disk_data_size or 0) / (1024 * 1024), 2)

    except UnexpectedResponse:
        pass
    except Exception as e:
        print(f"⚠️ Stats error for {col_name}: {e}")
    return stats


def get_total_db_size() -> float:
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        total = 0
        for col in collections.collections:
            try:
                info = client.get_collection(col.name)
                total += info.disk_data_size or 0
            except Exception:
                pass
        return round(total / (1024 * 1024), 2)
    except Exception:
        return 0.0


def get_all_notebooks() -> list[str]:
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        workspaces = []
        for c in collections.collections:
            if c.name.startswith("ws_"):
                ws_name = c.name[3:]
                workspaces.append(ws_name)
        return workspaces
    except Exception as e:
        print(f"⚠️ List collections error: {e}")
        return []


def delete_file_from_notebook(notebook_name: str, source_name: str) -> int:
    col_name = Config.get_collection_name(notebook_name)
    try:
        client = get_qdrant_client()
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        points, _ = client.scroll(
            collection_name=col_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source_name),
                    )
                ]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )

        if points:
            ids = [p.id for p in points]
            client.delete(
                collection_name=col_name,
                points_selector=ids,
            )
            print(f"🗑️ Deleted {len(ids)} chunks of '{source_name}' from {col_name}")
            return len(ids)
        return 0
    except Exception as e:
        print(f"❌ Delete file error: {e}")
        return 0


def delete_notebook(notebook_name: str) -> bool:
    col_name = Config.get_collection_name(notebook_name)
    try:
        client = get_qdrant_client()
        client.delete_collection(col_name)
        print(f"🗑️ Deleted collection: {col_name}")
        return True
    except Exception as e:
        print(f"❌ Delete collection error: {e}")
        return False
