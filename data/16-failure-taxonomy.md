# A Taxonomy of Retrieval Failures

When an answer is wrong, the useful question is which stage produced the wrongness.
These are the recurring categories, roughly in the order you should check them.

## The passage was never indexed

The most embarrassing and most common. The source file failed extraction, was filtered
out by a suffix check, was skipped as a duplicate, or sat in a directory the loader never
walked. No amount of retrieval tuning helps.

Check first, and check by searching the index for a distinctive phrase rather than by
reasoning about the pipeline.

## The passage was destroyed by chunking

It was indexed, but the fact was split across a boundary so that no single chunk states
it. A heading was separated from its section. A table row lost its header. The passage
exists in the corpus and does not exist in any retrievable unit.

The signature is that a phrase search finds fragments but no chunk contains the whole
fact.

## Vocabulary mismatch

The chunk is intact and the query shares no terms with it. The user asks about being
"charged twice" and the document says "duplicate transaction". Dense retrieval is meant
to bridge this and does so imperfectly; lexical retrieval cannot bridge it at all.

The signature is that the chunk is retrieved immediately once you rephrase the query in
the document's language.

## Lexical specificity

The mirror image. The query hinges on an identifier, a version number, or a rare proper
noun that carries little meaning to an embedding model, so dense retrieval returns
passages that are topically adjacent but do not contain the term.

The signature is that the correct chunk is absent from dense results and first in lexical
ones.

## Ranked too low

The correct chunk was retrieved, at rank 40, and the shortlist was 25. This is the most
tractable failure in the list and the one most often misdiagnosed as something harder.
Always look at the rank before concluding anything else.

## Crowded out by near-duplicates

The top five results are five versions of the same passage, and the complementary
information that would have completed the answer sits at rank six. Deduplication at
ingest prevents this; deduplication at query time mitigates it.

## Context dilution

Everything needed was in the prompt, alongside eight passages that were not. The model
attended to the wrong one. More context is not monotonically better, and this failure
gets worse as top-k grows, which is why sweeping top-k upward eventually degrades
quality.

## Ungrounded generation

Retrieval worked, the context contained the answer, and the model answered from its
parametric knowledge instead. Or the context did not contain the answer and the model
answered anyway. Instructing the model to refuse reduces this and does not eliminate it,
which is why it has to be measured rather than assumed.

## Stale index

The corpus changed and the index did not. The answer is correct with respect to what was
indexed and wrong with respect to the world. Nothing in the retrieval path can detect
this; only ingest-side freshness tracking can.
