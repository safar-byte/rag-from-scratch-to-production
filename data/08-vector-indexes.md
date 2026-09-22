# Vector Index Structures

Exact nearest-neighbour search compares the query against every vector in the corpus.
That is fine up to tens of thousands of vectors and hopeless beyond it, so production
systems use approximate nearest neighbour indexes that trade a little recall for a large
speedup.

## Flat

A flat index stores vectors contiguously and scans all of them. Recall is exactly 1.0 by
construction, because nothing is skipped. Query time grows linearly with corpus size.
Below roughly 50,000 vectors a flat index is usually the right answer, and reaching for
anything cleverer is premature.

## HNSW

Hierarchical Navigable Small World builds a layered graph. Upper layers are sparse and
used for coarse navigation; the bottom layer contains every vector. A search descends
from the top, greedily walking toward the query at each level.

Three parameters matter. `M` is the number of neighbours each node keeps, typically 16
to 64; higher values improve recall and increase memory. `ef_construction` controls how
hard the builder works to find good neighbours, typically 100 to 500. `ef_search` is the
size of the candidate list at query time and is the main recall/latency dial — it can be
changed per query without rebuilding, which makes it the parameter to tune first.

HNSW is memory-hungry: the graph itself often costs as much as the vectors. It also does
not support deletion cleanly. Deleted nodes are marked as tombstones and continue to
occupy memory and participate in traversal until the index is rebuilt.

## IVF

Inverted File indexes cluster the vectors, usually with k-means, and search only the
clusters nearest the query. `nlist` is the number of clusters and `nprobe` is how many
are searched. IVF needs a training pass over a representative sample before it can index
anything, which makes it awkward for corpora that grow continuously.

## Quantisation

Product quantisation compresses vectors by splitting them into subvectors and replacing
each with a codebook entry. It can cut memory by a factor of 8 to 32 at a measurable
cost in recall. The usual arrangement is to search the compressed index for a generous
candidate list and then rescore those candidates against the uncompressed vectors, which
recovers most of the lost recall for a small amount of extra work.

## Choosing

Below 50,000 vectors use a flat index. Between 50,000 and roughly 10 million, HNSW with
`ef_search` tuned against a recall target. Beyond that, or when memory is the binding
constraint, IVF with product quantisation and a rescoring pass. Measure recall against a
flat index on a sample before trusting any approximate configuration.
