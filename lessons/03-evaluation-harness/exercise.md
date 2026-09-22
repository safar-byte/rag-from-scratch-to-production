# Exercise — Lesson 03

## 1. Confirm reproducibility

```bash
python -m ragkit.eval.run --retrieval-only
python -m ragkit.eval.run --retrieval-only
```

The numbers must be identical. If they are not, something in the pipeline is
non-deterministic, and until you find it you cannot attribute any delta to your change
rather than to noise. Everything after this depends on this holding.

## 2. Find the one question with headroom

`multi_hop recall@3 = 0.900` means one of the five multi-hop questions misses one of its
two documents even at depth 3. Find it:

```bash
python -m ragkit.eval.run --retrieval-only --write --label "scratch"
python - <<'EOF'
import json, glob
run = json.load(open(sorted(glob.glob("benchmarks/runs/*scratch*.json"))[-1]))
for r in run["results"]:
    if r["kind"] == "multi_hop":
        print(r["id"], r["retrieval"]["recall@3"], r["retrieved"][:3])
        print("   ", r["question"][:90])
EOF
```

Which document is missing, and why do you think dense retrieval does not surface it?
Write your hypothesis down. You will test it in lesson 04.

Then delete the `scratch` row from `benchmarks/results.md` — the table is for real runs.

## 3. Break a metric and watch the tests catch it

In `ragkit/eval/metrics.py`, change `recall_at_k` to use `len(retrieved[:k])` as the
denominator instead of `len(relevant)`.

```bash
python -m pytest tests/test_metrics.py -q
```

Read which assertions fail. That change turns recall into something closer to precision —
a plausible-looking edit that silently redefines the number every later decision in this
repo is based on. Put it back.

## 4. Validate the judge

This is the exercise that matters most, and the one everyone skips.

```bash
python -m ragkit.eval.run --write --label "judge check"        # LLM judge
python -m ragkit.eval.run --no-llm-judge --write --label "det" # deterministic
```

Open both JSON files in `benchmarks/runs/`. Pick ten questions, read the answer, and
grade the groundedness yourself from 0 to 1 before looking at either machine score.

Now compare all three. Where does the LLM judge disagree with you? Where does the
deterministic grader? A judge that agrees with you on eight of ten is usable; one that
agrees on four is generating numbers, not measurements.

Record what you find. Until you have done this once, every groundedness number in this
repo is an unvalidated claim.

## 5. Fix the corpus problem

The harness says the corpus is saturated. Do something about it and watch the numbers
respond:

```bash
python -m ragkit.eval.run --retrieval-only --top-k 1
```

Then add three or four new documents to `data/` on topics adjacent to the existing ones —
close enough to act as genuine distractors, not obviously unrelated. Add two golden
questions for them. Re-ingest, re-run.

Did the saturation warning go away? Did recall drop? **A drop here is good news** — it
means the benchmark can now tell methods apart.

## 6. Write a question that breaks it

Try to write a golden question that is fair — genuinely answerable from exactly one
document in `data/` — but that dense retrieval gets wrong at k=3.

This is harder than it sounds, and the difficulty is itself the lesson. Vocabulary
mismatch is your best tool: ask using words that do not appear in the document.

If you find one, add it to `golden.yaml`. It will be the most valuable question in the
set, because it is the only one with anything to prove.

## Record

In `PROGRESS.md`:

- your hypothesis from step 2, for checking in lesson 04
- your judge-agreement rate from step 4 — the number that decides how much to trust
  every groundedness score from here on
- whether step 5 cleared the saturation warning
