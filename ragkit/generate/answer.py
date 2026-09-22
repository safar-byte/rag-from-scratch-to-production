"""Turning retrieved context into an answer with citations."""

from __future__ import annotations

import re
import time

from ragkit.generate.prompt import STRICT_SYSTEM_PROMPT, SYSTEM_PROMPT, build_prompt
from ragkit.providers.base import Generator
from ragkit.types import Answer, Citation, Scored

# Matches [1], [2][4], [1, 3] — the bracket form the system prompt asks for.
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(text: str, results: list[Scored]) -> list[Citation]:
    """Resolve [n] markers in the answer back to the chunks they point at.

    This is the *fallback* citation mechanism: parsing markers the model was asked to
    emit. It is unreliable by nature — the model can cite a passage it did not use, or
    use one it did not cite, and nothing here can detect either. Lesson 12 replaces it
    on the cloud path with the Citations API, which returns spans the model actually
    grounded in rather than numbers it typed.
    """
    seen: set[int] = set()
    citations: list[Citation] = []
    for match in CITATION_PATTERN.finditer(text):
        for raw in match.group(1).split(","):
            index = int(raw.strip())
            if index in seen or not 1 <= index <= len(results):
                continue
            seen.add(index)
            chunk = results[index - 1].chunk
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    cited_text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                )
            )
    return citations


def generate_answer(
    question: str,
    results: list[Scored],
    generator: Generator,
    *,
    max_tokens: int = 1024,
    max_context_chars: int | None = None,
    strict: bool = False,
) -> Answer:
    """Generate an answer over the retrieved context.

    `strict=True` swaps in the more forceful refusal prompt. It is a separate flag
    rather than the default so that lesson 03 can measure what the wording is actually
    worth, instead of the repo quietly assuming it helps.
    """
    prompt = build_prompt(question, results, max_context_chars=max_context_chars)
    system = STRICT_SYSTEM_PROMPT if strict else SYSTEM_PROMPT

    started = time.perf_counter()
    text, usage = generator.generate(system=system, prompt=prompt, max_tokens=max_tokens)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return Answer(
        text=text,
        citations=extract_citations(text, results),
        contexts=results,
        usage=usage,
        trace={
            "generator": generator.name,
            "strict_prompt": strict,
            "generation_ms": round(elapsed_ms, 1),
            "prompt_chars": len(prompt),
            "n_contexts": len(results),
        },
    )
