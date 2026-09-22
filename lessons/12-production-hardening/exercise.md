# Exercise — Lesson 12

## 1. Watch the gate decide

```bash
python lessons/12-production-hardening/run.py
```

Four of the five probes are classified correctly. The fifth —
`SHARD_REPLICATION_FACTOR` — is unanswerable and scores *above* the floor.

**Explain why in one sentence,** without using the word "threshold". Then say what would
catch it.

## 2. Use the inspector properly

```bash
uvicorn ragkit.api.main:app --port 8000
cd ui && npm run dev
```

Ask "Why do I get charged twice?" with strategy `rerank`, then switch to `dense`, then
`bm25`. Watch the score chips change shape.

Find a query where the reranker badge shows a passage moving **down**. Read it. Do you
agree it was overranked? That is lesson 05's R@1 dip (0.824 → 0.811) made concrete.

## 3. Prove caching works, or find out it does not

Cloud profile only:

```bash
RAG_PROFILE=cloud python lessons/12-production-hardening/run.py
```

Call 2 must show `cache_read > 0`.

Now break it deliberately: add `f"Current time: {datetime.now()}"` to the top of
`SYSTEM_PROMPT` and run again. The cache reads drop to zero and the cost roughly
doubles, with nothing reporting a problem.

That is the failure mode. A timestamp, a request id, an unsorted dict — any of them.
Which other things in this codebase could end up in a prompt prefix and vary per request?

## 4. Cost per completed task

Pick a question the local profile gets wrong. Work out:

- cost of one wrong answer, plus the retry, plus the user's clarifying question
- cost of one correct answer from a better pipeline

Which is cheaper *per completed task*? This is the calculation that decides model choice,
and it is almost never the same answer as cost per request.

## 5. Diagnose three failures from the trace alone

Use `benchmarks/runs/*.json` and `first_relevant_rank`. For three failing questions,
classify each:

- correct chunk **absent at any depth** → fix retrieval
- correct chunk at **rank 40, shortlist 25** → deepen the shortlist
- correct chunk **reached the prompt**, answer still wrong → prompt or model

Then fix one and measure. Being able to tell these apart *before* changing anything is
the whole skill this course was building toward.

## 6. Ship-readiness review

Go through the repo and list what you would not deploy as-is. Some starters:

- CORS allows `localhost:5173` unconditionally
- no authentication on the API
- no rate limiting
- `ingest` rebuilds the whole index with no versioning (lesson 12's own README says what
  the fix looks like)
- the corpus is 16 documents and most golden questions are saturated

For each: how would you fix it, and does it block a first deployment or not?

## 7. Read the table

Open [`benchmarks/results.md`](../../benchmarks/results.md) and read every row in order.

Which technique gave the largest gain? Which gave none? Which made things worse? Would
you have predicted any of it before starting?

That table, not the thirteen techniques, is what the course was for.
