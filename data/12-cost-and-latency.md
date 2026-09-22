# Cost and Latency

A retrieval technique that improves quality is not automatically worth deploying. The
question is always what it cost, and the answer has two currencies.

## Where the latency goes

For a typical cloud-backed pipeline, an approximate per-query breakdown:

Query embedding is 10 to 50 milliseconds. Vector search over a well-tuned index is 5 to
50 milliseconds. BM25 over the same corpus is 1 to 10 milliseconds and is essentially
free by comparison. Reranking a shortlist of 25 is 50 to 300 milliseconds and is usually
the largest non-generation cost. Generation is 500 to 5000 milliseconds and dominates
everything else.

The consequence is that retrieval optimisation rarely improves perceived latency. Halving
a 40 millisecond search inside a 3 second response is invisible. Retrieval work should be
justified by quality, and latency work should start with generation: streaming, a smaller
model, shorter outputs, or fewer sequential model calls.

Query transformation is the exception worth watching, because it inserts a full
generation before retrieval even starts. A rewrite step can easily double end-to-end
latency while improving recall by a few points.

## Where the money goes

Generation dominates cost as it dominates latency, and output tokens typically cost
several times what input tokens cost. Embedding is a one-time ingest cost plus a tiny
per-query cost. Hosted reranking bills per document scored, so shortlist depth is a
direct and linear cost lever.

Three levers reduce spend without reducing quality, and should be exhausted before any
lever that trades quality away.

Caching a stable prompt prefix makes repeated input tokens far cheaper, and the prefix in
a RAG system, meaning the system prompt, the tool definitions, and in some designs the
document itself, is exactly the part that repeats.

Batching moves work with no latency requirement onto an asynchronous path at roughly half
price. Ingest-time passes over a corpus belong there.

Right-sizing the model matters more than it looks. Grading, classification, and
contextualisation are easier tasks than answering, and a small model does them at a
fraction of the cost with no measurable quality loss.

## Cost per completed task

The number to optimise is cost per completed task, not cost per request. A cheaper model
that needs two retries and a clarifying turn is not cheaper. An agentic pipeline making
four model calls but answering correctly the first time may well beat a single call that
fails a third of the time and sends the user back to ask again.
