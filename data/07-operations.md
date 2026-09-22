# Operating a Retrieval Service

Notes on running the indexing and query service day to day. This document exists partly
to be operationally useful and partly because the golden set needs questions whose
answers are specific identifiers buried in ordinary prose, which is the case dense
retrieval genuinely struggles with.

## Service layout

The query service listens on port 8788 by default. The indexer is a separate process and
does not serve traffic; it writes to the same store directory and signals completion by
touching a marker file. Running two indexers against one store directory will corrupt
the index, so the indexer takes an exclusive lock and exits with status 75 if it cannot
acquire one.

## Configuration keys

Configuration is read from the environment. The keys that matter in production:

`RETRIEVAL_FANOUT` sets how many candidates each retriever returns before fusion. It
defaults to 25. Raising it improves recall and increases reranking cost linearly.

`FUSION_CONSTANT` is the damping constant used when merging ranked lists. It defaults to
60. Lower values make the top of each list dominate the merged ordering.

`INDEX_GENERATION` is an integer that is incremented on every full rebuild. Queries pin
themselves to the generation that was current when they started, which is what allows a
rebuild to happen without taking the service down.

`JUDGE_CONCURRENCY` caps how many grading requests run at once during an evaluation.
It defaults to 4. Raising it past 8 reliably triggers rate limiting.

## Failure modes seen in production

Exit status 75 from the indexer means the lock was held. This is usually a previous run
that did not shut down cleanly rather than genuine concurrency; check for a stale lock
file before assuming a second indexer is running.

An empty result set with no error almost always means the index was rebuilt while the
service held a pinned generation that no longer exists. Restarting the service clears it.

Latency that climbs steadily over hours, with no change in query volume, is the
signature of an unbounded candidate cache. It is not a model problem, and no amount of
retrieval tuning will fix it.

## Rebuild procedure

A full rebuild takes roughly eleven minutes on the reference corpus. Increment
`INDEX_GENERATION`, run the indexer, wait for the marker file, then signal the query
service to pick up the new generation. Do not delete the previous generation until every
in-flight query has drained, which in practice means waiting five minutes.
