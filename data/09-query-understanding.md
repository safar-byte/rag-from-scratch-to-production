# Query Understanding

The user's question is an input to a retrieval system, not a search query. The gap
between the two is where a surprising amount of retrieval quality is lost.

## Why questions make poor queries

Questions carry words that are useless for matching. "Can you tell me roughly what the
default timeout is?" contains one term that matters. Conversational framing, politeness,
and hedging all dilute the embedding and add noise to lexical matching.

Questions also frequently omit the terms that would find the answer. Someone asking
"why is it so slow after a while?" will not match a document about unbounded cache
growth, because they share no vocabulary at all. This is the vocabulary mismatch problem
and it is the oldest problem in information retrieval.

## Rewriting

Query rewriting asks a model to turn the question into a better search query: strip the
framing, keep the content terms, expand obvious abbreviations. It is cheap, it is easy to
inspect, and it helps most on conversational input. It helps very little on input that
was already a keyword query, and it can hurt by discarding a term that mattered.

## HyDE

Hypothetical Document Embeddings inverts the problem. Instead of improving the query,
the model writes a fabricated passage that *would* answer the question, and that passage
is embedded and used as the query vector. The reasoning is that a hypothetical answer
sits closer in embedding space to the real answer than the question does, because
answers resemble answers.

HyDE helps most when questions and documents are written in very different registers. It
is actively harmful when the model lacks the domain knowledge to write a plausible
answer, because it then hallucinates a passage about the wrong subject and retrieves
confidently against it. It also costs a full generation before retrieval even starts.

## Multi-query

Multi-query generates several paraphrases of the question, retrieves for each, and fuses
the results. It improves recall because different phrasings surface different chunks,
and it is a natural fit for rank fusion. It multiplies retrieval cost by the number of
variants, and the gains flatten quickly — beyond about four variants the additional
queries mostly return what the earlier ones already found.

## Step-back prompting

Step-back prompting asks a more general question first. For "why did the indexer exit
with status 75", the step-back question is "how does the indexer handle locking". The
general question retrieves the background material that makes the specific answer
interpretable. It works well for why-questions and poorly for lookups, where the general
version just adds noise.

## When to use none of these

Every technique here costs at least one extra model call before retrieval begins, and
each one can make retrieval worse. Route them: apply transformation only to queries that
look conversational or that return low-scoring results on a first pass. Applying all of
them to every query is the most common way to make a RAG system slower and no better.
