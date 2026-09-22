"""Assembling the prompt from retrieved context.

The part of a RAG system most likely to be written once and never looked at again, and
one of the two places hallucination is actually decided (the other is retrieval).

Three choices below are load-bearing:

* **The context is numbered and labelled with its source.** Without stable labels the
  model cannot cite, and you cannot check whether it cited correctly.
* **The model is told explicitly to say when the context is insufficient.** Left to
  itself it will answer anyway from parametric knowledge, which is the failure mode
  RAG exists to prevent. This instruction is necessary but not sufficient — it reduces
  ungrounded answers, it does not eliminate them, which is why lesson 03 measures
  groundedness rather than trusting the prompt.
* **Context comes before the question.** It is the stable part of the prompt, so it
  belongs in the cacheable prefix (lesson 12), and models attend well to instructions
  that come last.
"""

from __future__ import annotations

from ragkit.types import Scored

SYSTEM_PROMPT = """You answer questions using only the context provided.

Rules:
- Use only information from the numbered context passages. Do not use prior knowledge.
- Cite the passages you used with their numbers in square brackets, like [1] or [2][4].
- If the context does not contain enough information to answer, say exactly what is
  missing. Do not guess, and do not fill gaps from memory.
- Be concise and direct. Do not restate the question or describe what you are about to do.
"""


def format_context(results: list[Scored], max_chars: int | None = None) -> str:
    """Render retrieved chunks as numbered passages.

    `max_chars` is a crude budget guard. It truncates whole passages from the end
    rather than cutting one mid-sentence, because a half passage is worse than no
    passage: it can strand a claim away from its qualifier.
    """
    blocks: list[str] = []
    used = 0
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        header = f"[{index}] {chunk.title or chunk.doc_id} ({chunk.doc_id})"
        body = f"{chunk.context}\n{chunk.text}".strip() if chunk.context else chunk.text
        block = f"{header}\n{body}"
        if max_chars is not None and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[Scored], max_context_chars: int | None = None) -> str:
    if not results:
        return (
            "No context passages were retrieved.\n\n"
            f"Question: {question}\n\n"
            "Say that the corpus contains nothing relevant to this question."
        )
    context = format_context(results, max_chars=max_context_chars)
    return f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
