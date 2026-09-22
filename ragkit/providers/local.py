"""The offline backend: sentence-transformers, a cross-encoder, and Ollama.

Everything here runs on your machine with no API key. It is slower and the answers
are weaker than the cloud path, but it means the repo is genuinely runnable by anyone
who clones it — and it means you can iterate on retrieval logic for free, which is
most of what you actually spend time on.

Heavy imports are deferred into the constructors so that merely importing this module
(which the provider factory does) never pulls in torch.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ragkit.config import Settings
from ragkit.types import Chunk, Usage


class SentenceTransformerEmbedder:
    """Local dense embeddings via sentence-transformers.

    Defaults to BGE-small: 384 dimensions, ~130MB, and genuinely competitive for its
    size. BGE wants an instruction prefix on queries but not on documents, which is
    exactly the asymmetry the Embedder protocol exists to express.
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        self._settings = settings
        self._model = SentenceTransformer(settings.local_embedding_model)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._settings.local_embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 256
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(self.QUERY_INSTRUCTION + text, normalize_embeddings=True)
        return vector.tolist()


class CrossEncoderReranker:
    """Local cross-encoder reranking."""

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import CrossEncoder

        self._settings = settings
        self._model = CrossEncoder(settings.local_rerank_model)

    @property
    def name(self) -> str:
        return self._settings.local_rerank_model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        pairs = [(query, chunk.embed_text) for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]


class OllamaGenerator:
    """Generation through a locally running Ollama server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.Client(base_url=settings.ollama_host, timeout=300.0)

    @property
    def name(self) -> str:
        return self._settings.local_generation_model

    # A reasoning model spends tokens thinking before it writes anything, and that
    # spend comes out of the same budget as the answer. With too small a budget the
    # thinking is truncated and `response` comes back EMPTY — a blank answer with no
    # error, which reads downstream as a model that refused. Measured on qwen3:4b:
    # a one-line answer needed ~270 tokens, of which ~250 were thinking.
    THINKING_HEADROOM = 1024

    def generate(self, *, system: str, prompt: str, max_tokens: int = 1024) -> tuple[str, Usage]:
        payload: dict[str, Any] = {
            "model": self._settings.local_generation_model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            # Temperature 0: RAG answers should be reproducible, and the eval harness
            # in lesson 03 is meaningless if the same input gives different output.
            "options": {
                "temperature": 0.0,
                "num_predict": max_tokens + self.THINKING_HEADROOM,
            },
        }
        try:
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._settings.ollama_host}. "
                "Start it with `ollama serve`, then "
                f"`ollama pull {self._settings.local_generation_model}`."
            ) from exc

        body = response.json()
        usage = Usage(
            input_tokens=int(body.get("prompt_eval_count", 0)),
            output_tokens=int(body.get("eval_count", 0)),
            cost_usd=0.0,  # local inference is free; the cost column stays honest at 0
        )

        text = (body.get("response") or "").strip()
        thinking = (body.get("thinking") or "").strip()

        # Ollama returns a reasoning model's chain of thought in a separate `thinking`
        # field. An empty `response` alongside a non-empty `thinking` means generation
        # stopped mid-reasoning, so raise instead of handing back "" — a silent empty
        # answer would be scored as a refusal and quietly wreck the eval.
        if not text and thinking:
            raise RuntimeError(
                f"{self._settings.local_generation_model} used its whole token budget "
                f"thinking ({len(thinking)} chars) and produced no answer. Raise "
                "max_tokens, or use a non-reasoning model."
            )
        return text, usage

    def generate_json(self, *, system: str, prompt: str, max_tokens: int = 1024) -> Any:
        """Ask for JSON back.

        Used by the LLM-judge and the agentic graders. Small local models drift out of
        JSON often enough that the caller must handle a parse failure rather than
        assume success.
        """
        payload = {
            "model": self._settings.local_generation_model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }
        response = self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        return json.loads(response.json()["response"])
