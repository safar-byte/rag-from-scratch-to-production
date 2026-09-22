# Chunking

Chunking splits documents into retrievable units. It is the least glamorous part of a
RAG system and frequently the highest-leverage one, because no downstream technique can
recover information that chunking destroyed.

## The core tension

Small chunks give precise retrieval, because the matching passage is not diluted by
surrounding irrelevant text, but they fragment context, so a fact that needs two
sentences to state may be split across two chunks and neither one answers the question.
Large chunks preserve context but dilute the embedding: a 2000-token chunk covering
twelve topics has an embedding that is a blurry average of all twelve and is strongly
retrieved by nothing.

A common starting point is 512 tokens with 64 tokens of overlap. These are starting
points, not recommendations. The right size depends on the corpus and should be chosen
by measurement, not by reputation.

## Strategies

Fixed-size chunking splits every N characters or tokens. It is trivial to implement and
it cuts sentences in half.

Recursive chunking splits on a priority list of separators: paragraph breaks, then line
breaks, then sentence boundaries, then words, falling back to the next separator only
when a chunk is still too large. This respects document structure and is the sensible
default.

Semantic chunking embeds each sentence and starts a new chunk where the similarity
between consecutive sentences drops below a threshold, so boundaries fall where the
topic actually shifts. It costs one embedding call per sentence at ingest time.

Structural chunking uses the document's own markup: Markdown headings, HTML sections,
code function boundaries. Where the structure exists, it is usually better than anything
inferred.

## Overlap

Overlap repeats the tail of one chunk at the head of the next so a fact straddling a
boundary survives in at least one chunk. Typical overlap is 10 to 20 percent of chunk
size. Overlap costs storage and introduces near-duplicate results, which is a reason to
deduplicate before passing chunks to the generator.
