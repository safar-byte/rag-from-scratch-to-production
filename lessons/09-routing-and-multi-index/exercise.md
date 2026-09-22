# Exercise — Lesson 09

## 1. Audit the routing decisions

```bash
python lessons/09-routing-and-multi-index/run.py
```

Read all 44. **Do you agree with every one?** Find a question you would have routed
differently and work out which pattern did or did not fire.

Two lookups go to lexical and eight do not. Look at those eight — should any of them be
lexical? What would it cost if they were?

## 2. Measure it

```bash
python -m ragkit.eval.run --retrieval-only --strategy rerank
python -m ragkit.eval.run --retrieval-only --strategy router
```

The effect will be small — only 4 questions are routed differently. **Check the right
thing:** did `vocab_mismatch` hold up, and did the lexical-routed lookups improve?

An aggregate that barely moves while the intended subset improves is a success. Learning
to read that, rather than the headline number, is most of what this lesson teaches.

## 3. Break it deliberately

Invert the router — send prose to BM25 and identifiers to dense:

```python
HeuristicRouter(lexical=semantic_retriever, semantic=lexical_retriever)
```

Measure. The damage should be large and concentrated in `vocab_mismatch`. That asymmetry
is why the router defaults to semantic when unsure, and seeing the size of it is worth
more than reading the argument.

## 4. Heuristic versus model

```bash
python - <<'EOF'
from ragkit.config import get_settings
from ragkit.eval.golden import load_golden
from ragkit.pipeline import RagPipeline
from ragkit.retrieve.router import LlmRouter, looks_lexical

p = RagPipeline(get_settings())
router = LlmRouter(p.generator, p._bm25 and object(), object())
for q in load_golden()[:12]:
    heuristic = "lexical" if looks_lexical(q.question) else "semantic"
    print(f"{heuristic:9s} | {q.question[:60]}")
EOF
```

Extend that to call the LLM router and compare the two classifications. Where do they
disagree? Is the model right, and is it right often enough to justify a generation per
query?

## 5. Route something else

The pattern generalises. Pick one and implement it:

- **Skip reranking** when the top dense score is already above a threshold. Saves the
  cross-encoder pass on easy queries. What threshold, and what does it cost in quality?
- **Skip query transformation** on keyword-shaped queries (lesson 06, exercise 5).
- **Route by cost**: cheap model for lookups, expensive for multi-hop.

Measure whichever you choose. The interesting number is not the quality delta but the
*cost* delta at equal quality.

## 6. When routing goes wrong

A router is a single point of failure that fails silently — a misrouted query returns
plausible-looking worse results, with no error.

What would you log to detect it in production? What would the alert threshold be? (Hint:
the router's decisions are recorded in `HeuristicRouter.decisions`, and lesson 11's
observability notes cover the rest.)

## Record

In `PROGRESS.md`: whether the routed subset improved, and which extra routing rule from
step 5 you would ship.
