import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    size_bytes  INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',   -- queued|processing|ready|failed
    progress    INTEGER NOT NULL DEFAULT 0,       -- 0..100
    error       TEXT,
    page_count  INTEGER,
    chunk_count INTEGER,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,                    -- user|assistant
    content     TEXT NOT NULL,
    meta_json   TEXT,                             -- sources + which were cited
    created_at  TEXT NOT NULL
);
"""

DOC_UPDATE_FIELDS = {"status", "progress", "error", "page_count", "chunk_count"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------
def create_document(doc_id: str, filename: str, file_type: str,
                    sha256: str, size_bytes: int) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, file_type, sha256, size_bytes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, filename, file_type, sha256, size_bytes, _now()),
        )


def get_document(doc_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def find_by_hash(sha256: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def update_document(doc_id: str, **fields: Any) -> None:
    assert fields.keys() <= DOC_UPDATE_FIELDS, "unexpected field"
    sets = ", ".join(f"{name} = ?" for name in fields)
    with connect() as conn:
        conn.execute(f"UPDATE documents SET {sets} WHERE id = ?", (*fields.values(), doc_id))


def delete_document(doc_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def fail_stuck_documents() -> None:
    """If the server died mid-ingestion, don't leave documents 'processing' forever."""
    with connect() as conn:
        conn.execute(
            "UPDATE documents SET status = 'failed', error = 'Interrupted by server restart' "
            "WHERE status IN ('queued', 'processing')"
        )


# ----------------------------------------------------------------------
# Threads and messages
# ----------------------------------------------------------------------
def create_thread(title: str) -> str:
    thread_id = uuid.uuid4().hex
    now = _now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now),
        )
    return thread_id


def get_thread(thread_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
    return dict(row) if row else None


def list_threads() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM threads ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_thread(thread_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))


def add_message(thread_id: str, role: str, content: str,
                meta: dict[str, Any] | None = None) -> int:
    now = _now()
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO messages (thread_id, role, content, meta_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, json.dumps(meta) if meta else None, now),
        )
        conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))
        return cursor.lastrowid


def get_messages(thread_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY id", (thread_id,)
        ).fetchall()
    messages = []
    for row in rows:
        item = dict(row)
        meta = json.loads(item.pop("meta_json") or "{}")
        item["sources"] = meta.get("sources", [])
        item["used_sources"] = meta.get("used_sources", [])
        messages.append(item)
    return messages


def recent_messages(thread_id: str, limit: int = 6) -> list[dict[str, Any]]:
    return get_messages(thread_id)[-limit:]