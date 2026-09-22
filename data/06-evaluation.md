# Evaluating RAG Systems

A RAG system you cannot measure is a RAG system you cannot improve. Evaluation splits
cleanly into two halves that fail independently and must be measured separately.

## Retrieval metrics

These ask whether the right chunks were found, and need a golden set mapping each
question to the chunks that genuinely answer it.

Recall@k is the fraction of relevant chunks that appear in the top k results. It is the
most important single retrieval number, because a chunk that was never retrieved cannot
be reranked, cited, or reasoned over.

Precision@k is the fraction of the top k results that are relevant.

Mean Reciprocal Rank, or MRR, is the average over all questions of one divided by the
rank of the first relevant result. It rewards putting a correct answer at the very top
and ignores everything after it.

Normalised discounted cumulative gain, or nDCG@k, rewards relevant results
proportionally to how highly they rank, discounting by the logarithm of position. It is
the right metric when relevance is graded rather than binary.

## Generation metrics

These ask whether the answer was any good given what was retrieved.

Groundedness, also called faithfulness, measures whether every claim in the answer is
supported by the retrieved context. Low groundedness means hallucination.

Answer relevance measures whether the answer actually addresses the question. An answer
can be perfectly grounded and completely unresponsive.

Context precision measures how much of the supplied context was actually needed. Low
context precision means you are paying for tokens that contribute nothing.

## Separating the two

The diagnostic value of measuring both halves is that it tells you where to work. High
retrieval scores with low groundedness is a prompting or model problem. Low retrieval
scores with high groundedness means the generator is doing well with bad material and
the fix belongs in retrieval. A single end-to-end score cannot distinguish these, which
is why it is the wrong thing to optimise.

## LLM-as-judge

Groundedness and relevance are graded by a language model against a rubric, since no
string-matching metric captures them. Judges should run at temperature zero for
reproducibility, and a cheap fast model is usually sufficient, because judging is an
easier task than answering. Judges are biased and imperfect: validate them against human
labels on a sample before trusting them, and never compare scores produced by different
judges.
