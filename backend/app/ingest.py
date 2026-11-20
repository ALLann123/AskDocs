from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import db
from app.config import settings
from app.loaders import load_file
from app.vectorstore import delete_document_chunks, get_vector_store

BATCH_SIZE = 16


def stored_path(doc: dict[str, Any]) -> Path:
    """Files are saved as <doc_id>.<ext>. The user's filename is never used on disk,
    which rules out path-traversal tricks like '../../etc/passwd'."""
    return settings.upload_dir / f"{doc['id']}.{doc['file_type']}"


def ingest_document(doc_id: str) -> None:
    """Runs in a background thread. Updates the SQLite row as it goes."""
    doc = db.get_document(doc_id)
    if doc is None:
        return

    try:
        db.update_document(doc_id, status="processing", progress=2, error=None)

        # 1. Load
        pages = load_file(stored_path(doc))
        page_count = len(pages) if doc["file_type"] == "pdf" else None

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError(
                "No extractable text found. If this is a scanned PDF, it needs OCR first."
            )

        # 3. Attach metadata
        for index, chunk in enumerate(chunks):
            chunk.metadata.update(
                {"doc_id": doc_id, "filename": doc["filename"], "chunk_index": index}
            )
        ids = [f"{doc_id}:{index}" for index in range(len(chunks))]

        # 4. Embed + store in batches so we can report progress
        store = get_vector_store()
        for start in range(0, len(chunks), BATCH_SIZE):
            end = start + BATCH_SIZE
            store.add_documents(documents=chunks[start:end], ids=ids[start:end])
            progress = 5 + int(95 * min(end, len(chunks)) / len(chunks))
            db.update_document(doc_id, progress=min(progress, 99))

        db.update_document(
            doc_id,
            status="ready",
            progress=100,
            page_count=page_count,
            chunk_count=len(chunks),
        )

    except Exception as exc:  # noqa: BLE001: we really do want to catch everything here
        delete_document_chunks(doc_id)  # don't leave half-indexed documents behind
        db.update_document(doc_id, status="failed", error=str(exc)[:500])


def delete_document_data(doc_id: str) -> None:
    """Remove a document everywhere: vectors, file on disk, database row."""
    doc = db.get_document(doc_id)
    if doc is None:
        return
    delete_document_chunks(doc_id)
    stored_path(doc).unlink(missing_ok=True)
    db.delete_document(doc_id)