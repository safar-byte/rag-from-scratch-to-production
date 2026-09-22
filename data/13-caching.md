# Caching in a Retrieval Pipeline

There are four distinct caches in a mature RAG system, and they fail in different ways.
Treating them as one thing is how a cache ends up serving stale answers nobody can
explain.

## Embedding cache

Query embeddings are deterministic for a given model and input, so an exact-match cache
on the query string is free correctness. Hit rates are high in production because real
query distributions have a long head: a small number of questions account for a large
share of traffic.

The invalidation key must include the model identifier. A cache keyed on query text alone
will serve vectors from the previous model after an upgrade, producing a silent and total
failure in which every result is meaningless and nothing raises.

## Retrieval cache

Caching the ranked result list for a query is riskier, because the correct answer changes
whenever the corpus changes. The key must include the index generation, and entries must
be dropped when that generation is retired. A retrieval cache with no generation in its
key is the most common source of "the document is indexed but search cannot find it".

## Prompt prefix cache

This is the one with the largest effect on cost. A language model charges far less for
input tokens it has already processed, provided the prefix of the request is byte
identical. In a RAG system the stable prefix is the system prompt, the tool definitions,
and in some designs the source document itself.

The rule that matters: the prefix must be genuinely unchanged. A timestamp, a request
identifier, a randomly ordered JSON object, or a tool list assembled from an unordered
set will all break the match, and nothing reports the failure. The only reliable check is
to read the cache-read token count off the response and assert that it is greater than
zero on a repeated request.

Ordering follows from this. Stable content goes first and volatile content last, which
means the user's question belongs at the end of the prompt, after the context.

## Answer cache

Caching the final answer is the cheapest and the most dangerous. It is safe for identical
questions against an unchanged index, and unsafe the moment either varies. Personalised
or permission-scoped answers must never share a cache entry across users, and the way
that goes wrong is a permission leak rather than a stale result.

## What not to cache

Do not cache the reranker output separately from the retrieval it reordered. The two are
a unit: a reranked list keyed on the query alone becomes wrong as soon as the underlying
candidates change, and the resulting inconsistency is nearly impossible to reproduce.
