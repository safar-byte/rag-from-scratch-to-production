# Getting Documents In

Retrieval quality is bounded by what the ingest pipeline produced, and most corpora
arrive in formats that lose information on the way in.

## PDFs

PDF is a layout format, not a document format. It records where glyphs sit on a page, not
what the reading order is, so extraction is inference rather than parsing. Multi-column
layouts interleave if the extractor reads across the page. Tables collapse into
whitespace-separated runs that are meaningless once chunked. Headers and footers repeat
on every page and, left in, become the most frequent strings in the corpus, which skews
term statistics enough to degrade lexical scoring measurably.

Scanned PDFs contain no text at all and need optical character recognition, whose error
rate is highly sensitive to scan quality. A corpus that is silently half image-only is a
common and hard-to-spot problem; count extracted characters per page and flag pages below
a threshold.

## HTML

The main content is usually a small fraction of the bytes. Navigation, cookie notices,
sidebars, and footers repeat across every page and will otherwise dominate the corpus.
Strip them by extracting the main content region rather than the whole body.

Preserve heading structure while stripping presentation. Headings are the strongest
retrieval signal a structured document has.

## Tables

Tables are the hardest common case. A table chunked as text loses the association between
a cell and its column header, so a row reading "3.2 | 47 | passed" is unretrievable and
uninterpretable. Two workable approaches: render each row as a sentence that repeats the
column names, or keep the table intact as one chunk and accept the size. Splitting a
table across chunks without repeating the header is always wrong.

## Code

Code should be chunked on function and class boundaries, not on line counts. A function
split in half retrieves poorly and cannot be reasoned about. Keep the enclosing file path
and the signature with each chunk, since both carry strong retrieval signal that the body
alone does not.

## Deduplication

Real corpora contain near-duplicates: the same policy in three places, a document and its
revision, boilerplate repeated across hundreds of files. Near-duplicates crowd out
diversity in results, so the top five become five copies of one passage and the answer
loses everything the other candidates would have supplied.

Deduplicate at ingest by content hash for exact matches, and by a similarity threshold for
near matches. Deduplicating at query time instead is possible but wasteful, since the
duplicates have already displaced better candidates by then.

## Metadata to keep

Capture at ingest: the source path or URL, a document title, the modification time, the
content hash, and any field you might later filter on. Adding a filter field afterwards
means a full re-ingest, and re-ingesting a large corpus to add a field you could have
captured the first time is a distinctly avoidable afternoon.
