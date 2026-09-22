# Observability for Retrieval Systems

A RAG pipeline that logs only its inputs and outputs is not debuggable. When an answer
is wrong, the question is always which stage went wrong, and that is not recoverable
from the final text.

## Log the stages, not the result

Every query should emit a record containing the original question, the transformed query
if any, the candidate identifiers and scores from each retriever, the fused ordering, the
reranked ordering, which chunks reached the prompt, the token counts, and the stage
timings. That record is what lets someone answer "why did it say that" three days later.

The identifiers matter more than the text. Logging chunk contents is expensive and often
impossible where the corpus is sensitive, but logging chunk identifiers and scores costs
almost nothing and preserves the ability to replay.

## The diagnostic that matters most

Compare the rank of the correct chunk before and after each stage. A pipeline with
retrieval, fusion, and reranking has three points where the right answer can be lost, and
only a per-stage record distinguishes them.

Three characteristic patterns:

The correct chunk never appears at any depth. Retrieval failed, and reranking and
prompting are irrelevant. Look at chunking, the embedding model, or the query.

The correct chunk appears at rank 40 and the shortlist is 25. Retrieval found it and the
shortlist was too shallow. This is the most common and most easily fixed failure.

The correct chunk reaches the prompt and the answer is still wrong. Retrieval is fine.
The problem is the prompt, the model, or the context being diluted by irrelevant passages
ranked above it.

## Score distributions

Track the distribution of top-1 scores over time, not just the mean. A drifting corpus, a
changed embedding model, or a silently truncated index all show up as a shift in that
distribution well before anyone reports a bad answer.

A sudden collapse in top-1 scores across all queries almost always means the index was
rebuilt with a different embedding model than the one serving queries. This is a common
and total failure: nothing errors, and every result is meaningless.

## Sampling and cost

Full per-stage logging is affordable at low volume and expensive at high volume. Sample
it: log everything for a small percentage of traffic, log identifiers and timings for all
of it, and log everything unconditionally for queries whose top score falls below a
threshold, since those are the ones most likely to have failed.
