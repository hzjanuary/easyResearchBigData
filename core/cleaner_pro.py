
from __future__ import annotations

import datetime
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from config import (
    MIN_DOC_LENGTH,
    MAX_WORKERS,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)

log = logging.getLogger(__name__)


def _extract_pdf(path: Path) -> tuple[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    meta = {"page_count": len(reader.pages)}
    return "\n\n".join(pages), meta


def _extract_txt(path: Path) -> tuple[str, dict]:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc), {}
        except (UnicodeDecodeError, ValueError):
            continue
    return path.read_text(encoding="utf-8", errors="replace"), {}


def _extract_docx(path: Path) -> tuple[str, dict]:
    import docx2txt

    text = docx2txt.process(str(path))
    return text or "", {}


def _extract_code(path: Path) -> tuple[str, dict]:
    return _extract_txt(path)[0], {"language": path.suffix.lstrip(".")}


_EXTRACTORS: dict[str, Callable] = {
    ".pdf":  _extract_pdf,
    ".txt":  _extract_txt,
    ".docx": _extract_docx,
    ".py":   _extract_code,
    ".js":   _extract_code,
    ".json": _extract_code,
    ".csv":  _extract_code,
}


_HEADER_FOOTER_RE = re.compile(
    r"(?m)^(Page \d+ of \d+|Confidential|DRAFT|©.*|All rights reserved\.?)$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = _HEADER_FOOTER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\S\n]+\n", "\n", text)
    text = re.sub(r"\f", "\n\n", text)
    return text.strip()


def _process_single_file(
    file_path_str: str,
    base_dir_str: str,
) -> dict[str, Any] | None:
    path = Path(file_path_str)
    base = Path(base_dir_str)
    ext = path.suffix.lower()

    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        return None

    try:
        raw_text, extra_meta = extractor(path)
    except Exception as exc:
        log.warning("Failed to read %s: %s", path.name, exc)
        return None

    text = clean_text(raw_text)
    if len(text) < MIN_DOC_LENGTH:
        return None

    stat = path.stat()
    modified_date = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()

    try:
        rel_path = str(path.relative_to(base))
    except ValueError:
        rel_path = path.name

    doc = {
        "file": path.name,
        "file_path": rel_path,
        "format": ext.lstrip("."),
        "modified_date": modified_date,
        "size_bytes": stat.st_size,
        "text": text,
    }
    doc.update(extra_meta)
    return doc


def discover_files(source_dir: Path) -> list[Path]:
    files = [
        p for p in source_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    ]
    log.info("Discovered %d documents in %s", len(files), source_dir)
    return sorted(files)


def clean_documents(
    source_dir: Path,
    *,
    max_workers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    files = discover_files(source_dir)
    if not files:
        log.warning("No documents found — nothing to clean.")
        return []

    workers = max_workers or MAX_WORKERS
    total = len(files)
    docs: list[dict[str, Any]] = []
    done = 0

    log.info("Cleaning %d files with %d workers …", total, workers)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_single_file, str(f), str(source_dir)): f
            for f in files
        }
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                docs.append(result)
            if progress_callback:
                progress_callback(done, total)

    docs.sort(key=lambda d: d["file"])
    log.info("Cleaning complete → %d documents", len(docs))
    return docs
