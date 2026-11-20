from typing import Any

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str
    progress: int
    error: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    created_at: str
    duplicate: bool = False  # true if an upload matched an existing file


class Source(BaseModel):
    id: int                   # the citation number: [1], [2], ...
    doc_id: str
    filename: str
    page: int | None = None   # PDFs only
    chunk_index: int
    text: str                 # the exact passage
    score: float              # 0..1, higher = more relevant (1 - cosine distance)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    document_ids: list[str] | None = None  # None = search everything


class ThreadOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[dict[str, Any]] = []
    used_sources: list[int] = []
    created_at: str