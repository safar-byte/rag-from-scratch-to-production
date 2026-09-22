# Corpus

Six short original documents about retrieval systems, written for this repo. They are
original work (MIT, same as the repo) rather than a scraped public-domain dump, for
three reasons: the licensing is unambiguous, the corpus is small enough to commit and
re-index in seconds, and the facts in it are specific enough that the golden Q&A set in
`ragkit/eval/golden.yaml` has verifiable answers.

It is also deliberately awkward in places. Numbers are scattered across documents, some
questions need two documents to answer, and a few terms are defined in one file and used
in another. A corpus that is too clean makes every retrieval strategy look equally good,
which defeats the purpose of the eval harness.
