import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from app import db
from app.llm import get_llm
from app.prompts import ANSWER_PROMPT, FOLLOWUP_PROMPT, NOT_FOUND, REWRITE_PROMPT
from app.retrieval import format_context, retrieve
from app.schemas import ChatRequest

# Matches [1], [2][3] and also [1, 2]
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

Event = tuple[str, Any]  # (event_name, json_serializable_data)


def extract_citations(answer: str, source_count: int) -> list[int]:
    """Which source numbers did the model actually cite? Ignores out-of-range numbers."""
    used: set[int] = set()
    for match in CITATION_RE.finditer(answer):
        for number in match.group(1).split(","):
            n = int(number)
            if 1 <= n <= source_count:
                used.add(n)
    return sorted(used)


async def rewrite_question(question: str, history: list[dict[str, Any]]) -> str:
    """Turn a follow-up into a standalone question. Skipped for the first message."""
    if not history:
        return question

    transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history)
    chain = REWRITE_PROMPT | get_llm() | StrOutputParser()
    rewritten = (await chain.ainvoke({"history": transcript, "question": question})).strip()
    return rewritten or question


async def suggest_followups(question: str, answer: str) -> list[str]:
    try:
        chain = FOLLOWUP_PROMPT | get_llm() | StrOutputParser()
        raw = await chain.ainvoke({"question": question, "answer": answer[:1500]})
    except Exception:  # noqa: BLE001: follow-ups are a nice-to-have, never fatal
        return []

    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip() for line in raw.splitlines()]
    return [line for line in lines if line.endswith("?")][:3]


async def answer_stream(req: ChatRequest) -> AsyncIterator[Event]:
    # ---- Thread bookkeeping -------------------------------------------------
    thread_id = req.thread_id
    if thread_id is None or db.get_thread(thread_id) is None:
        thread_id = db.create_thread(title=req.question[:60])
    yield "thread", {"thread_id": thread_id}

    history = db.recent_messages(thread_id, limit=6)  # read BEFORE saving this turn
    db.add_message(thread_id, "user", req.question)

    # ---- 1. Rewrite follow-ups into standalone search queries -----------------
    query = await rewrite_question(req.question, history)
    yield "status", {"stage": "searching", "query": query}

    # ---- 2. Retrieve (blocking I/O → run in a thread so we don't freeze the server)
    sources = await asyncio.to_thread(retrieve, query, req.document_ids)
    yield "sources", [source.model_dump() for source in sources]

    # ---- 3. Nothing relevant? Say so without calling the LLM -------------------
    if not sources:
        db.add_message(thread_id, "assistant", NOT_FOUND, {"sources": [], "used_sources": []})
        yield "token", {"text": NOT_FOUND}
        yield "done", {"thread_id": thread_id, "used_sources": []}
        return

    # ---- 4. Stream the answer ----------------------------------------------------
    yield "status", {"stage": "answering"}
    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()

    parts: list[str] = []
    async for piece in chain.astream({"context": format_context(sources), "question": query}):
        if piece:
            parts.append(piece)
            yield "token", {"text": piece}
    answer = "".join(parts).strip()

    # ---- 5. Validate citations and save ------------------------------------------
    used = extract_citations(answer, len(sources))
    message_id = db.add_message(
        thread_id,
        "assistant",
        answer,
        {"sources": [s.model_dump() for s in sources], "used_sources": used},
    )
    yield "done", {"thread_id": thread_id, "message_id": message_id, "used_sources": used}

    # ---- 6. Related questions (after `done`, so the UI can finalize immediately) -
    questions = await suggest_followups(req.question, answer)
    if questions:
        yield "followups", {"questions": questions}