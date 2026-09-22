# Reranking

A reranker takes the shortlist that retrieval produced and reorders it by relevance to
the query. It is usually the single largest quality improvement available to a RAG
system, and it is also the easiest to deploy incorrectly.

## Bi-encoders versus cross-encoders

A bi-encoder, which is what an ordinary embedding model is, encodes the query and each
document separately, so document vectors can be computed once at ingest time and
searched in milliseconds. The query and the document never meet until the similarity
calculation, which limits how well the model can judge relevance.

A cross-encoder feeds the query and the document through the model together and outputs
a single relevance score. Attention runs across both texts, so it can weigh how the
query relates to each part of the document. This is far more accurate and far slower,
because nothing can be precomputed. A cross-encoder scoring one million chunks per query
is not a system; it is a stalled request.

## The over-fetch pattern

The standard arrangement is therefore two stages. Cheap retrieval fetches a shortlist of
25 to 100 candidates, and the cross-encoder reranks that shortlist down to the 3 to 5
chunks that reach the generator.

The critical consequence is that recall at the shortlist depth is a hard ceiling on
final quality. A reranker can only reorder what retrieval already found. If the answer
is not in the top 50, no reranker will put it in the top 5. When a reranker
disappoints, the problem is nearly always retrieval depth or recall, not the reranker.

## Cost

Reranking adds one model call per query over the whole shortlist, typically 50 to 300
milliseconds locally. Hosted rerankers bill per document scored, so shortlist depth is a
direct cost lever: doubling the shortlist doubles the reranking bill and usually buys
much less than double the improvement.
