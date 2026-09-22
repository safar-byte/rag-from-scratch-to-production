# Embeddings and Vector Search

An embedding model maps text to a fixed-length vector of floating point numbers such
that texts with similar meanings land near each other. Retrieval then becomes a nearest
neighbour search in that space.

## Dimensionality

The number of components in the vector is the model's dimension. BGE-small produces
384-dimensional vectors. Larger models typically produce 768, 1024, or 1536 dimensions.
Higher dimension usually means better quality and always means more storage and slower
search: an index of one million chunks at 1024 dimensions in float32 occupies roughly
4 gigabytes before any index overhead.

## Distance metrics

Cosine similarity measures the angle between two vectors and ignores their magnitude.
It is the default for text retrieval because document length should not affect
relevance. Dot product is equivalent to cosine similarity when all vectors are
normalised to unit length, and normalising at write time lets you use the cheaper dot
product at query time. Euclidean distance is rarely the right choice for text.

## Asymmetric models

Some embedding models are asymmetric: they expect a short instruction prefix on queries
but not on documents. BGE models are trained this way. The prefix
"Represent this sentence for searching relevant passages:" is prepended to the query
only. Forgetting this prefix does not raise an error. It silently costs several points
of recall, which is the worst kind of bug because the system still appears to work.

## The fundamental limitation

Embeddings capture semantic similarity, not lexical precision. A query for error code
TX-4491 will not reliably retrieve the chunk containing TX-4491, because the code is
close to meaningless to the embedding model and looks much like TX-4492. This is the
single strongest argument for hybrid retrieval, and it is why identifiers, product
codes, error numbers, and proper nouns are the classic failure mode of pure dense search.
