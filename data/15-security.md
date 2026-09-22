# Security and Privacy

A retrieval system is a machine for surfacing information to whoever asks. That is the
feature and it is also the risk.

## Permission enforcement

Permissions must be enforced at retrieval, not at generation. Filtering the answer after
the model has already seen restricted context is not a control: the model can paraphrase,
summarise, or be induced to reveal what it was shown. The restricted document must never
enter the prompt.

This means permission state has to be available at query time, denormalised onto the
chunk, and refreshed when it changes. A document whose access was revoked but whose index
entry still carries the old permission remains retrievable, and the gap between the two
is measured in whatever your re-ingest interval is.

## Prompt injection through documents

A retrieved chunk is untrusted input. If the corpus contains user-supplied content, a
document can contain text addressed to the model rather than to the reader, instructing
it to ignore its rules or to reveal earlier context. The model has no reliable way to
distinguish an instruction from the operator and an instruction embedded in a passage it
was asked to summarise.

Mitigations are partial and should be layered. Delimit retrieved content clearly and
state in the system prompt that it is data, not instructions. Never let retrieved content
decide whether a tool runs. Treat any corpus with user-writable content as hostile, and
assume that a sufficiently determined injection will eventually work.

## Data leakage through the index

Embeddings are not a privacy boundary. Approximate inversion of text from embedding
vectors is well demonstrated, so an embedding index of sensitive documents should be
protected to the same standard as the documents themselves. Storing the index in a less
protected system than the source corpus is a common and significant mistake.

Chunk text stored alongside the vectors is, obviously, the plaintext itself.

## Logging

Query logs are sensitive. People ask retrieval systems things they would not put in a
ticket, and a full query log is a record of what everyone in an organisation wanted to
know and when. Retain it briefly, restrict access, and prefer logging chunk identifiers
over chunk contents.

## Personally identifiable information

Detect and redact before indexing, not after retrieval. Once a name or an account number
is in the index it is retrievable, it is in the backups, and removing it means a
re-ingest. Redaction at ingest is cheap and redaction afterwards is a project.
