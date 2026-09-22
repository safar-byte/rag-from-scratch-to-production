"""The HTTP service.

Two endpoints matter:

    POST /query      retrieve + generate, returning the full per-stage trace
    POST /retrieve   retrieval only, no model call

`/retrieve` exists because it is what you actually reach for when debugging. Nearly
every bad RAG answer is a bad retrieval, and being able to see the candidates without
waiting on generation turns a two-minute cycle into a two-second one.

The response deliberately carries far more than the answer text: per-stage scores for
every chunk, the assembled prompt, token counts, cost, and timings. That payload is what
the inspector UI renders, and it is the difference between a system you can debug and
one you can only re-run and hope about.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ragkit.config import get_settings
from ragkit.generate import build_prompt
from ragkit.pipeline import RagPipeline, RetrievalStrategy
from ragkit.types import Answer, Scored

app = FastAPI(title="ragkit", version="0.1.0")

# The inspector runs on a different port in development. Tightened in any real
# deployment — this is a dev convenience, not a pattern to copy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipelines: dict[tuple[str, str, float | None, bool], RagPipeline] = {}


def get_pipeline(
    strategy: str = "rerank",
    transform: str = "identity",
    score_floor: float | None = None,
    strict_prompt: bool = False,
) -> RagPipeline:
    """Pipelines are cached per configuration.

    Model loading dominates cost — BGE plus a cross-encoder is ~40s — so constructing a
    pipeline per request would make the service unusable. The cache key is the full
    configuration, so switching strategy in the UI reuses the already-loaded models.
    """
    key = (strategy, transform, score_floor, strict_prompt)
    if key not in _pipelines:
        _pipelines[key] = RagPipeline(
            get_settings(),
            strategy=strategy,
            transform=transform,
            score_floor=score_floor,
            strict_prompt=strict_prompt,
        )
    return _pipelines[key]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=25)
    strategy: str = "rerank"
    transform: str = "identity"
    score_floor: float | None = None
    strict_prompt: bool = False


def _serialise(scored: Scored) -> dict[str, Any]:
    return {
        "chunk_id": scored.chunk.chunk_id,
        "doc_id": scored.chunk.doc_id,
        "title": scored.chunk.title,
        "text": scored.chunk.text,
        "context": scored.chunk.context,
        "start_char": scored.chunk.start_char,
        "end_char": scored.chunk.end_char,
        "rank": scored.rank,
        "score": round(scored.score, 6),
        # The per-stage breakdown is the whole point: "which stage put this here" is
        # the question you ask when a result looks wrong.
        "scores": {k: round(v, 6) for k, v in scored.scores.items()},
    }


def _serialise_answer(answer: Answer, question: str) -> dict[str, Any]:
    return {
        "answer": answer.text,
        "abstained": bool(answer.trace.get("abstained", False)),
        "citations": [
            {"doc_id": c.doc_id, "title": c.title, "cited_text": c.cited_text}
            for c in answer.citations
        ],
        "contexts": [_serialise(c) for c in answer.contexts],
        "prompt": build_prompt(question, answer.contexts),
        "usage": {
            "input_tokens": answer.usage.input_tokens,
            "output_tokens": answer.usage.output_tokens,
            "cache_read_tokens": answer.usage.cache_read_tokens,
            "cost_usd": answer.usage.cost_usd,
        },
        "trace": answer.trace,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "profile": settings.profile.value,
        "chunks_indexed": RagPipeline(settings).count(),
        "strategies": [s.value for s in RetrievalStrategy],
    }


@app.post("/retrieve")
def retrieve(request: QueryRequest) -> dict[str, Any]:
    """Retrieval only. No model call, so this stays fast enough to iterate with."""
    pipeline = get_pipeline(request.strategy, request.transform)
    started = time.perf_counter()
    try:
        results = pipeline.retrieve(request.question, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001 - surface the reason rather than a 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "question": request.question,
        "retriever": pipeline.retriever.name,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "contexts": [_serialise(r) for r in results],
    }


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    pipeline = get_pipeline(
        request.strategy, request.transform, request.score_floor, request.strict_prompt
    )
    try:
        answer = pipeline.ask(request.question, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"question": request.question, **_serialise_answer(answer, request.question)}
