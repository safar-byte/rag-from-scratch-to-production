"""Building and querying an entity graph over the corpus.

The structure is deliberately the simplest thing that answers the question GraphRAG
exists for:

    nodes  = entities, and the chunks they appear in
    edges  = two entities co-occurring in the same chunk, weighted by how often

Co-occurrence is a weak notion of "related" — it records that two things were discussed
together, not *how* they relate. A full GraphRAG extracts typed relations ("X depends on
Y", "X is a kind of Y"), which is strictly better and costs a model call per chunk to
build and re-build on every corpus change.

## What this buys, and what it costs

The win is questions vector search structurally cannot answer: "how do X and Y relate",
"what else is affected by Z". Those need the *connections* between passages, and an
embedding of a passage does not encode what it connects to.

The cost is real and usually understated. The graph is a second index that must be kept
current, entity extraction is lossy, and co-occurrence produces hub nodes that connect
to everything and make traversal meaningless — hence `max_degree` below.

**Reach for this last.** On most corpora hybrid retrieval plus reranking answers the
questions a graph would, at a fraction of the build and maintenance cost. The honest
test is whether your failing questions are relational; if they are not, a graph will not
help, however impressive the visualisation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from ragkit.graph.extract import extract_heuristic
from ragkit.types import Chunk, Scored


@dataclass(slots=True)
class EntityGraph:
    # entity -> chunk ids it appears in
    entity_chunks: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # entity -> {neighbour: co-occurrence count}
    edges: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    chunks: dict[str, Chunk] = field(default_factory=dict)
    display: dict[str, str] = field(default_factory=dict)  # lowercase key -> original

    @property
    def n_entities(self) -> int:
        return len(self.entity_chunks)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.edges.values()) // 2

    def neighbours(self, entity: str, limit: int = 10) -> list[tuple[str, int]]:
        items = sorted(self.edges.get(entity, {}).items(), key=lambda kv: kv[1], reverse=True)
        return items[:limit]

    def stats(self) -> dict[str, Any]:
        degrees = [len(v) for v in self.edges.values()]
        return {
            "entities": self.n_entities,
            "edges": self.n_edges,
            "mean_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
            "max_degree": max(degrees, default=0),
        }


def build_graph(
    chunks: list[Chunk],
    *,
    min_frequency: int = 2,
    max_degree: int = 40,
    extractor: Any = None,
) -> EntityGraph:
    """Build the co-occurrence graph.

    `min_frequency` drops entities seen in only one chunk: they cannot connect anything,
    so they add nodes and no edges.

    `max_degree` prunes hubs. An entity appearing in most chunks connects everything to
    everything, and traversal through it returns the whole corpus — the graph equivalent
    of a stopword. Pruning hubs is what keeps traversal selective.
    """
    extract = extractor or extract_heuristic
    graph = EntityGraph()
    per_chunk: dict[str, list[str]] = {}

    for chunk in chunks:
        graph.chunks[chunk.chunk_id] = chunk
        entities = extract(chunk)
        keys = []
        for entity in entities:
            key = entity.lower()
            graph.display.setdefault(key, entity)
            keys.append(key)
        per_chunk[chunk.chunk_id] = keys
        for key in keys:
            graph.entity_chunks[key].add(chunk.chunk_id)

    # Singletons are excluded from EDGE building - appearing in one chunk, they cannot
    # connect anything - but they are KEPT in entity_chunks so a query naming one can
    # still seed a walk. Dropping them entirely made `FUSION_CONSTANT` unfindable even
    # though the corpus documents it, which is a silly way to lose a lookup.
    keep = {e for e, ids in graph.entity_chunks.items() if len(ids) >= min_frequency}

    for chunk_id, keys in per_chunk.items():
        present = sorted({k for k in keys if k in keep})
        for a, b in combinations(present, 2):
            graph.edges[a][b] = graph.edges[a].get(b, 0) + 1
            graph.edges[b][a] = graph.edges[b].get(a, 0) + 1
        _ = chunk_id

    for entity, neighbours in list(graph.edges.items()):
        if len(neighbours) > max_degree:
            strongest = sorted(neighbours.items(), key=lambda kv: kv[1], reverse=True)[:max_degree]
            graph.edges[entity] = dict(strongest)

    return graph


class GraphRetriever:
    """Retrieve by walking entity connections rather than by similarity.

    Finds entities mentioned in the query, walks `hops` steps out, and returns chunks
    containing the entities reached. Scoring is by how many query-connected entities a
    chunk contains — a crude relevance signal, and the point is that it is a *different*
    signal from cosine similarity, so it surfaces different chunks.

    Best used alongside vector retrieval rather than instead of it, which is why the
    lesson fuses the two with RRF rather than replacing one with the other.
    """

    def __init__(self, graph: EntityGraph, hops: int = 1) -> None:
        self._graph = graph
        self._hops = hops

    @property
    def name(self) -> str:
        return f"graph(h={self._hops})"

    def _seed_entities(self, query: str) -> list[str]:
        lowered = query.lower()
        return [e for e in self._graph.entity_chunks if e in lowered]

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        seeds = self._seed_entities(query)
        if not seeds:
            # No entity from the query is in the graph, so there is nothing to walk.
            # Returning nothing is correct and important: a graph retriever that
            # invents a starting point returns confident noise.
            return []

        reached: dict[str, int] = {e: 0 for e in seeds}
        frontier = list(seeds)
        for hop in range(1, self._hops + 1):
            nxt: list[str] = []
            for entity in frontier:
                for neighbour, _weight in self._graph.neighbours(entity):
                    if neighbour not in reached:
                        reached[neighbour] = hop
                        nxt.append(neighbour)
            frontier = nxt
            if not frontier:
                break

        # Score a chunk by how many reached entities it holds, discounted by hop
        # distance so directly-mentioned entities dominate.
        scores: dict[str, float] = defaultdict(float)
        for entity, distance in reached.items():
            weight = 1.0 / (1 + distance)
            for chunk_id in self._graph.entity_chunks.get(entity, ()):
                scores[chunk_id] += weight

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[Scored] = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = self._graph.chunks.get(chunk_id)
            if chunk is None:
                continue
            out.append(
                Scored(
                    chunk=chunk,
                    score=score,
                    scores={"graph": round(score, 4), "seed_entities": float(len(seeds))},
                    rank=rank,
                )
            )
        return out
