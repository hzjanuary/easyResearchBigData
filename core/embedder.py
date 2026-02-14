
from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path

import chromadb
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, DEVICE, EMBEDDING_MODEL, EMBED_BATCH_SIZE

print(f"🚀 easyResearch running on device: {DEVICE.upper()}")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)


def add_to_vector_db(
    chunks,
    collection_name: str = "default_notebook",
    batch_size: int | None = None,
    progress_callback=None,
):
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.id for chunk in chunks]

    bs = batch_size or EMBED_BATCH_SIZE
    total = len(chunks)

    print(f"📥 Embedding {total} chunks into '{collection_name}' (batch={bs}) …")

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


def get_retriever(collection_name: str = "default_notebook", k: int = 5, fetch_k: int = 20):
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )


def get_notebook_stats(notebook_name: str) -> dict:
    stats = {"chunks": 0, "files": [], "size_mb": 0.0}
    try:
        if not os.path.exists(CHROMA_DIR):
            return stats

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        target = None
        for col in client.list_collections():
            if col.name == notebook_name:
                target = col
                break
        if not target:
            return stats

        collection = client.get_collection(notebook_name)
        stats["chunks"] = collection.count()

        if stats["chunks"] > 0:
            result = collection.get(include=["metadatas"])
            if result and result["metadatas"]:
                sources = {
                    meta["source"]
                    for meta in result["metadatas"]
                    if meta and "source" in meta
                }
                stats["files"] = sorted(sources)

        col_uuid = str(target.id)
        dir_path = os.path.join(CHROMA_DIR, col_uuid)
        if os.path.exists(dir_path):
            total_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(dir_path)
                for f in fns
            )
            stats["size_mb"] = round(total_size / (1024 * 1024), 2)

    except Exception as e:
        print(f"⚠️ Stats error for {notebook_name}: {e}")
    return stats


def get_total_db_size() -> float:
    try:
        if not os.path.exists(CHROMA_DIR):
            return 0.0
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(CHROMA_DIR)
            for f in fns
        )
        return round(total / (1024 * 1024), 2)
    except Exception:
        return 0.0


def get_all_notebooks() -> list[str]:
    try:
        if not os.path.exists(CHROMA_DIR):
            return []
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return [c.name for c in client.list_collections()]
    except Exception as e:
        print(f"⚠️ List notebooks error: {e}")
        return []


def delete_file_from_notebook(notebook_name: str, source_name: str) -> int:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(notebook_name)
        result = collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(result["ids"], result["metadatas"])
            if meta and meta.get("source") == source_name
        ]
        if ids_to_delete:
            BATCH = 500
            for i in range(0, len(ids_to_delete), BATCH):
                collection.delete(ids=ids_to_delete[i : i + BATCH])
            print(f"🗑️ Deleted {len(ids_to_delete)} chunks of '{source_name}'")
        return len(ids_to_delete)
    except Exception as e:
        print(f"❌ Delete file error: {e}")
        return 0


def delete_notebook(notebook_name: str) -> bool:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        target = None
        for col in client.list_collections():
            if col.name == notebook_name:
                target = col
                break

        col_uuid = str(target.id) if target else None
        client.delete_collection(notebook_name)
        print(f"🗑️ Deleted collection: {notebook_name}")

        if col_uuid:
            dir_path = os.path.join(CHROMA_DIR, col_uuid)
            if os.path.exists(dir_path):
                time.sleep(0.5)
                shutil.rmtree(dir_path, ignore_errors=True)
                print(f"📂 Cleaned up folder: {dir_path}")

        return True
    except Exception as e:
        print(f"❌ Delete notebook error: {e}")
        return False
