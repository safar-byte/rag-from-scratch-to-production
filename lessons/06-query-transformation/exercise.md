# Exercise — Lesson 06

## 1. Look at what each technique actually produces

```bash
python lessons/06-query-transformation/run.py "why do I get charged twice"
python lessons/06-query-transformation/run.py "What is FUSION_CONSTANT set to?"
```

Compare the two. On the conversational question, does the rewrite keep the useful terms?
On the identifier question, does any technique improve on the original — or do they all
paraphrase away the one token that mattered?

Note the `transform ms` column against the ~1s retrieval takes. That ratio is the whole
argument.

## 2. Measure, do not assume

```bash
for t in identity rewrite hyde multiquery stepback; do
  echo "=== $t"
  python -m ragkit.eval.run --retrieval-only --strategy rerank --transform $t
done
```

**Predict the ranking before you run it.** Which will win, and by how much?

This takes a while — one model call per question per technique. That wait is data too:
it is exactly the cost you would pay per query in production.

## 3. Explain the result

The likely outcome is that none of them beats `identity` by much, and some lose. Before
concluding that query transformation is useless, work out *why* on this corpus:

- 17 of 44 questions are saturated at 1.000. What can a transformation improve there?
- `vocab R@5` already reached 1.000 in lesson 05. What is left for a rewrite to fix?

Then answer the real question: **on what kind of corpus would these techniques win?**
Describe it concretely — corpus shape, question shape, what the baseline pipeline looks
like. If you can do that, you understand the technique better than a benchmark win would
have taught you.

## 4. Catch HyDE failing

HyDE writes a fake answer and retrieves with it. Ask something the model has no
knowledge of:

```bash
python lessons/06-query-transformation/run.py "What is the default value of SHARD_REPLICATION_FACTOR?"
```

Read the hypothetical passage. It will be fluent, specific, and about a setting that
does not exist. That passage is then used as the query vector.

This is the failure mode worth internalising: HyDE does not degrade gracefully. It fails
by being precisely wrong, and it looks exactly like working.

## 5. Route instead of applying

Step-back helps why-questions and adds noise to lookups. Rather than applying it always,
route it: use `looks_lexical` from `ragkit/retrieve/router.py` to send identifier queries
straight through untransformed, and apply step-back only to the rest.

Measure it. Does selective application beat both "always" and "never"?

## Record

In `PROGRESS.md`: the measured ranking from step 2, and your answer to step 3 — the
description of a corpus where these would win. That answer is the transferable part.
