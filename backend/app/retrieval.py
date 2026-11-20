from app.config import settings
from app.schemas import Source
from app.vectorstore import get_vector_store


def retrieve(query: str, document_ids: list[str] | None = None) -> list[Source]:
    """Semantic search → filter by relevance → number the survivors 1..n."""
    store = get_vector_store()

    # Chroma metadata filter: only search inside the chosen documents
    where = {"doc_id": {"$in": document_ids}} if document_ids else None

    # Over-fetch, then cut by the distance threshold
    results = store.similarity_search_with_score(
        query, k=settings.top_k * 2, filter=where
    )

    sources: list[Source] = []
    for document, distance in results:  # already sorted: smallest distance first
        if distance > settings.max_distance:
            continue
        meta = document.metadata
        sources.append(
            Source(
                id=len(sources) + 1,
                doc_id=meta["doc_id"],
                filename=meta["filename"],
                page=meta.get("page"),
                chunk_index=meta["chunk_index"],
                text=document.page_content,
                score=round(max(0.0, 1.0 - distance), 3),
            )
        )
        if len(sources) == settings.top_k:
            break
    return sources


def format_context(sources: list[Source]) -> str:
    """The text block the LLM reads. Each chunk is labeled with its citation number."""
    blocks = []
    for source in sources:
        location = f", page {source.page}" if source.page else ""
        blocks.append(f"[{source.id}] (file: {source.filename}{location})\n{source.text}")
    return "\n\n---\n\n".join(blocks)