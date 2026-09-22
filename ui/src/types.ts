export interface Context {
  chunk_id: string;
  doc_id: string;
  title: string;
  text: string;
  context: string;
  start_char: number;
  end_char: number;
  rank: number;
  score: number;
  /**
   * Per-stage scores. The keys vary by pipeline, which is the point — a hybrid result
   * carries `dense_rank` and `bm25_rank`, a reranked one carries `rank_before_rerank`.
   * The inspector renders whatever is present rather than assuming a fixed shape.
   */
  scores: Record<string, number>;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cost_usd: number;
}

export interface QueryResponse {
  question: string;
  answer: string;
  abstained: boolean;
  citations: { doc_id: string; title: string; cited_text: string }[];
  contexts: Context[];
  prompt: string;
  usage: Usage;
  trace: Record<string, unknown>;
}

export interface RetrieveResponse {
  question: string;
  retriever: string;
  elapsed_ms: number;
  contexts: Context[];
}

export interface Health {
  status: string;
  profile: string;
  chunks_indexed: number;
  strategies: string[];
}

export const API = "http://localhost:8000";
