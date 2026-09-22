"""The cloud backend: Claude for generation, Voyage for embeddings and reranking.

Anthropic does not serve an embeddings endpoint, so the cloud profile is genuinely
two vendors. That is the honest shape of a production RAG stack and the reason the
provider abstraction has three separate protocols instead of one "LLM" object.

Claude API details that are easy to get wrong from memory, all applied below:
  * Model IDs are bare — `claude-opus-5`, never a date-suffixed variant.
  * `thinking={"type": "adaptive"}`; `budget_tokens` is rejected with a 400 on Opus 5.
    Depth is tuned with `output_config={"effort": ...}` instead.
  * Prompt caching is a prefix match: `tools` -> `system` -> `messages`. Stable content
    first, the user's question last, or the cache never hits.
"""

from __future__ import annotations

from typing import Any

from ragkit.config import Settings
from ragkit.types import Chunk, Usage

# Per-million-token prices, used only to populate the cost column in the eval report.
# Approximate and worth re-checking; the point is relative cost between pipelines.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
}
# A cache read is billed at a small fraction of a fresh input token — which is the
# entire argument for lesson 12.
CACHE_READ_DISCOUNT = 0.1


class VoyageEmbedder:
    """Dense embeddings via Voyage AI."""

    def __init__(self, settings: Settings) -> None:
        import voyageai

        settings.require_cloud_keys()
        self._settings = settings
        self._client = voyageai.Client(api_key=settings.voyage_api_key)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        # Probed once rather than hardcoded, so swapping the model in .env cannot leave
        # a stale dimension behind and corrupt the index.
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    @property
    def name(self) -> str:
        return self._settings.cloud_embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        # Voyage caps batch size; chunk the request rather than let a large corpus 400.
        for i in range(0, len(texts), 128):
            result = self._client.embed(
                texts[i : i + 128],
                model=self._settings.cloud_embedding_model,
                input_type="document",
            )
            out.extend(result.embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed(
            [text], model=self._settings.cloud_embedding_model, input_type="query"
        )
        return result.embeddings[0]


class VoyageReranker:
    """Hosted cross-encoder reranking via Voyage."""

    def __init__(self, settings: Settings) -> None:
        import voyageai

        settings.require_cloud_keys()
        self._settings = settings
        self._client = voyageai.Client(api_key=settings.voyage_api_key)

    @property
    def name(self) -> str:
        return self._settings.cloud_rerank_model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        result = self._client.rerank(
            query=query,
            documents=[c.embed_text for c in chunks],
            model=self._settings.cloud_rerank_model,
            top_k=min(top_k, len(chunks)),
        )
        return [(chunks[r.index], float(r.relevance_score)) for r in result.results]


class ClaudeGenerator:
    """Generation via the Claude API.

    `cheap=True` selects the small model for bulk work. `effort` defaults to "low" for
    answer generation: a RAG answer is a reading-comprehension task over context you
    already retrieved, so paying for deep reasoning on every query is usually waste.
    Raise it for the agentic pipelines in lesson 10, where the model actually has to
    decide things.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        cheap: bool = False,
        effort: str = "low",
    ) -> None:
        import anthropic

        settings.require_cloud_keys()
        self._settings = settings
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.cloud_cheap_model if cheap else settings.cloud_generation_model
        self._effort = effort

    @property
    def name(self) -> str:
        return self._model

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 4096,
        cache_system: bool = False,
    ) -> tuple[str, Usage]:
        """One answer from one prompt.

        `cache_system=True` marks the system prompt as a cache breakpoint. Worth it
        only when that prefix is byte-identical across requests and long enough to
        clear the model's minimum cacheable prefix — see lesson 12.
        """
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if cache_system:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}

        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
        )

        # Always check stop_reason before reading content: a safety refusal comes back
        # as HTTP 200 with stop_reason="refusal" and no answer text.
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            category = getattr(detail, "category", None) if detail else None
            raise RuntimeError(f"Claude declined this request (category: {category}).")

        # With adaptive thinking the response carries thinking blocks alongside text;
        # take only the text blocks.
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return text.strip(), self._usage(response)

    def _usage(self, response: Any) -> Usage:
        raw = response.usage
        cache_read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
        fresh_in = int(raw.input_tokens)
        out = int(raw.output_tokens)

        in_price, out_price = PRICES_USD_PER_MTOK.get(self._model, (0.0, 0.0))
        cost = (
            fresh_in * in_price
            + cache_write * in_price * 1.25  # a cache write costs a little extra
            + cache_read * in_price * CACHE_READ_DISCOUNT
            + out * out_price
        ) / 1_000_000

        return Usage(
            input_tokens=fresh_in,
            output_tokens=out,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=cost,
        )
