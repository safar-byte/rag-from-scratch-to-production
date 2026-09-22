# Lesson 11 — GraphRAG

```bash
python lessons/11-graphrag/run.py
python lessons/11-graphrag/run.py "How does BM25 relate to HNSW?"
```

## The idea

Vector search retrieves passages that *resemble* the query. Some questions are not about
resemblance at all:

- "How do X and Y relate?"
- "What else is affected by Z?"
- "What depends on this component?"

These need the **connections between** passages, and an embedding of a passage does not
encode what it connects to. GraphRAG builds that structure explicitly: entities as nodes,
relationships as edges, retrieval as a walk.

## What this implementation does

The simplest thing that answers the question:

```
nodes  = entities, and the chunks they appear in
edges  = two entities co-occurring in the same chunk, weighted by frequency
```

Co-occurrence is a weak notion of "related" — it records that two things were discussed
together, not *how* they relate. A full GraphRAG extracts typed relations ("X depends on
Y"), which is strictly better and costs a model call per chunk to build, and again on
every corpus change.

## The result on this corpus, which is the lesson

```
entities: 34    edges: 4    mean degree: 1.6
```

**That is not a graph. It is a scatter.** Four edges across sixteen documents means
traversal has essentially nowhere to go — a one-hop walk from most entities reaches
nothing at all.

The reason is structural, not a tuning problem. This corpus is **sixteen independent
explanatory essays**. Each document explains one topic thoroughly and barely mentions the
others. There is almost no entity co-occurrence to exploit, because the documents were
not written as an interconnected knowledge base.

GraphRAG shines on corpora with the opposite shape: incident reports that reference the
same services, papers citing each other, org documents naming the same people and teams,
codebases where components genuinely depend on components. Entities must **recur across
documents in varying combinations** for co-occurrence to mean anything.

No amount of parameter tuning fixes a corpus that is the wrong shape. Building the graph
and measuring it is what tells you that, in about thirty seconds — and that is the
correct outcome of this lesson, not a diagram.

## Measured against the golden set

The structural argument above is confirmed by the numbers, and they are worse than
"unhelpful":

| kind | n | R@5 (graph) | R@5 (rerank) |
|---|---|---|---|
| lookup | 10 | 0.400 | 1.000 |
| conceptual | 9 | 0.333 | 1.000 |
| multi_hop | 7 | 0.143 | 0.929 |
| **vocab_mismatch** | 8 | **0.000** | **1.000** |
| all | 44 | **0.409** | 0.987 |

**The graph returned nothing at all for 34 of 44 questions.** No entity from the query
appears in the graph, so there is nowhere to start a walk, and `GraphRetriever` correctly
returns an empty list rather than inventing a seed.

`vocab_mismatch` scoring 0.000 is the clearest signal. Those questions are phrased in a
user's words rather than the document's — "Why do I get charged twice?" — so they contain
no entity names by construction. **A graph keyed on entities cannot answer a question
that names no entities**, which is precisely the case dense retrieval exists for.

(The `unanswerable` row scores 1.000, which is an artifact: with no relevant document,
retrieving nothing is correct by definition. Ignore it.)

This is not a tuning result. It is what a graph looks like over a corpus with no
structure to traverse, and it is why the honest deployment of GraphRAG is *alongside*
vector retrieval for the minority of questions that name entities and need their
connections — never instead of it.

## Two bugs found while building it

Both are the silent kind, and both are instructive about entity extraction generally.

**The Title Case pattern used `\s`,** which matches newlines, so it ran across line
breaks and swallowed the start of the next sentence — producing entities like
`"Vector Search An"` and `"Dimensionality The"`. Fixed with a literal space class, plus a
plausibility check that rejects a trailing stopword. A related case: `"Chunking Chunking"`,
from a heading immediately followed by its own body text.

**Singleton entities were dropped entirely.** They appear in one chunk, so they cannot
form edges — but dropping them from the node set too made `FUSION_CONSTANT` unfindable
even though the corpus documents it. They are now excluded from edge building and kept
for seeding a walk.

The general point: entity extraction is where GraphRAG quality is actually decided, it is
lossy in ways that do not raise errors, and the junk it produces is plausible enough to
survive a casual look at the output.

## Deciding whether you need this

Do not start from "should we add a knowledge graph". Start from your failing questions:

1. Take the questions your current pipeline gets wrong.
2. How many require **connecting information across documents** rather than finding the
   right passage?
3. If that number is small, GraphRAG will not help, however good the demos look.

Then check the corpus: build the graph and look at mean degree. Below ~3, there is not
enough structure to traverse. This repo scores 1.6.

## Cost

A second index that must be kept current. Entity extraction is lossy. Co-occurrence
produces hub nodes connecting everything to everything, which makes traversal
meaningless — `max_degree` exists to prune them.

**Reach for this last.** On most corpora, hybrid retrieval plus reranking answers the
questions a graph would, at a fraction of the build and maintenance cost.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 12 — Production hardening](../12-production-hardening/).
