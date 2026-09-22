# Hybrid Search and Reciprocal Rank Fusion

Hybrid search runs a dense (embedding) retriever and a sparse (keyword) retriever over
the same corpus and merges their results. It exists because the two methods fail in
different, complementary ways.

## BM25

BM25 is the standard lexical ranking function. It scores a document by the frequency of
the query terms in it, discounted by how common those terms are across the whole corpus,
and normalised by document length. It has two tuning parameters: k1, which controls
term-frequency saturation and usually sits between 1.2 and 2.0, and b, which controls
length normalisation and defaults to 0.75.

BM25 excels at exactly what dense retrieval is worst at: exact identifiers, rare terms,
error codes, and proper nouns. It has no notion of meaning, so it cannot match the word
car to the word automobile.

## Reciprocal Rank Fusion

The problem with merging two rankings is that their scores are not comparable. A cosine
similarity of 0.82 and a BM25 score of 14.3 live on different scales, and normalising
them is fragile because BM25 has no fixed upper bound.

Reciprocal Rank Fusion sidesteps this by discarding the scores and using only the ranks.
The fused score for a document d is the sum, over every retriever r, of one divided by
the quantity k plus the rank of d in that retriever's list. Ranks are 1-based, and
documents absent from a list contribute nothing from it. The constant k is conventionally
set to 60.

The constant k damps the influence of the very top ranks. With k equal to 60, the gap
between rank 1 and rank 2 is small, so a document ranked moderately well by both
retrievers can outrank a document ranked first by only one. That behaviour is the entire
point: agreement between two independent methods is strong evidence of relevance.

RRF needs no training, no score normalisation, and no per-corpus tuning, which is why it
is a strong default despite its simplicity.
