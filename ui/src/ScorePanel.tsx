import type { Context } from "./types";

/**
 * The per-stage score breakdown for one retrieved chunk.
 *
 * This component is the reason the UI exists. An answer tells you what the system
 * concluded; this tells you *why* a particular passage was in front of it — which
 * retriever surfaced it, where each one ranked it, and whether the reranker moved it.
 * Almost every bad RAG answer is a bad retrieval, and that is invisible without this.
 */

/** Which stage produced each score key, so they can be grouped rather than dumped. */
const STAGE_LABELS: Record<string, string> = {
  dense: "dense cosine",
  bm25: "bm25",
  dense_rank: "dense rank",
  bm25_rank: "bm25 rank",
  rrf: "rrf fused",
  rrf_dense: "rrf from dense",
  rrf_bm25: "rrf from bm25",
  rerank: "cross-encoder",
  rank_before_rerank: "rank before rerank",
  graph: "graph",
  route: "routed lexical",
  n_variants: "query variants",
  n_hops: "hops",
  child_chars: "child size",
  parent_chars: "parent size",
  merged_windows: "windows merged",
  seed_entities: "seed entities",
};

function formatScore(key: string, value: number): string {
  if (key.endsWith("_rank") || key === "route" || key.startsWith("n_")) {
    return String(Math.round(value));
  }
  if (key.endsWith("_chars") || key === "merged_windows" || key === "seed_entities") {
    return String(Math.round(value));
  }
  return value.toFixed(4);
}

export function ScorePanel({ context }: { context: Context }) {
  const entries = Object.entries(context.scores);
  if (entries.length === 0) return null;

  const movement = context.scores.rank_before_rerank;
  const moved = movement !== undefined ? Math.round(movement) - context.rank : 0;

  return (
    <div className="scores">
      {moved !== 0 && (
        <span className={moved > 0 ? "badge up" : "badge down"}>
          {moved > 0 ? `↑ ${moved}` : `↓ ${Math.abs(moved)}`} by reranker
        </span>
      )}
      {context.scores.route === 1 && <span className="badge route">routed to lexical</span>}
      {entries.map(([key, value]) => (
        <span className="score" key={key}>
          <span className="score-key">{STAGE_LABELS[key] ?? key}</span>
          <span className="score-value">{formatScore(key, value)}</span>
        </span>
      ))}
    </div>
  );
}
