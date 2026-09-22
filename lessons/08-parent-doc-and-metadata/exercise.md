# Exercise — Lesson 08

## 1. See the expansion

```bash
python lessons/08-parent-doc-and-metadata/run.py "how does chunk overlap work"
```

For each pair, ask: **could the model have answered from the child alone?** If yes, the
expansion bought nothing but tokens. Find a case where it genuinely helped, and one
where it did not.

## 2. Sweep the window

```bash
for w in 0 300 600 1200; do
  echo "=== window $w"
  python -m ragkit.eval.run --retrieval-only --strategy parent --top-k 5
done
```

(Edit `parent_window` in the pipeline, or call `RagPipeline(strategy="parent",
parent_window=w)` directly.)

**Predict what recall will do.** The answer should be "almost nothing" — expansion
changes what the generator sees, not what retrieval finds. If recall moves a lot,
something is wrong; work out what before continuing.

Then measure what it costs: at window 1200 with top-k 5, how many characters of prompt?
Compare against the model's context budget.

## 3. Force a merge

Two nearby hits should collapse into one window rather than being returned twice:

```bash
python lessons/08-parent-doc-and-metadata/run.py "chunking" --window 1500
```

Look for `merged N windows` in the panel title. Then set the window to 50 and confirm
the merging stops.

Why does returning both matter? Count the wasted characters, and think about which
passage got pushed out of the top k to make room for a near-duplicate.

## 4. Pre-filter versus post-filter

```bash
python lessons/08-parent-doc-and-metadata/run.py "locking" --filter 07-operations.md
```

Now implement post-filtering yourself: retrieve the global top 5 and discard anything
not from `07-operations.md`.

```bash
python - <<'EOF'
from ragkit.config import get_settings
from ragkit.pipeline import RagPipeline
p = RagPipeline(get_settings(), strategy="dense")
hits = p.retrieve("locking", top_k=5)
kept = [h for h in hits if h.chunk.doc_id == "07-operations.md"]
print(f"post-filter kept {len(kept)} of {len(hits)}")
EOF
```

How many survive? Now imagine a corpus 100× larger where that document is 1% of it. How
many would survive then — and would anything have told you?

## 5. The silent-drop trap

```bash
python - <<'EOF'
from ragkit.config import get_settings
from ragkit.pipeline import RagPipeline
p = RagPipeline(get_settings(), strategy="bm25")
p.retriever.retrieve("locking", top_k=5, where={"doc_id": "07-operations.md"})
EOF
```

It raises. Consider the alternative: a `where` clause accepted and silently ignored. In
a multi-tenant system that is not a bug, it is a data leak, and every test would still
pass. Where else in this codebase could an ignored argument do the same?

## 6. When is a separate index the right call?

Work out the selectivity at which pre-filtering stops being enough and a per-partition
index is warranted. Reason about what an HNSW walk does when 99% of the nodes it
traverses are ineligible.

## Record

In `PROGRESS.md`: the window size you would ship and why, and your answer to step 4
about what happens at scale.
