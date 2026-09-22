# Contextual Retrieval

Contextual retrieval addresses a specific and very common failure: a chunk that is
meaningless once separated from its document.

## The problem

Consider a chunk that reads, in full, "Revenue grew 3% over the previous quarter."

Which company? Which quarter? The chunk does not say, because the surrounding document
said it. Embedded as-is, this chunk will not be retrieved by a question such as "How did
Acme perform in Q2 2024?", because the words Acme and Q2 2024 appear nowhere in it.
Pronouns, relative dates, and unqualified references such as "the system", "this
approach", or "the above" all produce chunks that are individually unretrievable.

## The technique

Before embedding, ask a cheap language model to write one or two sentences situating
each chunk within its source document, and prepend that to the chunk. In the example
above the prefix might read: "This chunk is from Acme Corp's Q2 2024 earnings report, in
the section on the cloud infrastructure segment."

The prefix is embedded and indexed with the chunk. The original text is preserved
separately so citations still point at the real span rather than at generated text. Both
the dense index and the BM25 index benefit, because the prefix adds the exact keywords a
user is likely to search for.

## Making it affordable

The naive implementation sends the entire source document to the model once per chunk,
which for a 200-chunk document means sending that document 200 times. Two mechanisms fix
this.

Prompt caching keeps the document as a stable prompt prefix across all the chunks
derived from it, so it is processed once and read from cache thereafter, at a small
fraction of the input cost.

Batch processing runs the whole pass asynchronously at roughly half price, which is
appropriate because this is a one-time ingest-time cost with no latency requirement.

Used together, contextualising a corpus becomes a modest one-off expense rather than a
prohibitive one. The cost is paid once at ingest; every query afterwards benefits.
