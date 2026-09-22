import { useEffect, useState } from "react";
import { ScorePanel } from "./ScorePanel";
import { API, type Context, type Health, type QueryResponse, type RetrieveResponse } from "./types";

const STRATEGIES = ["dense", "bm25", "hybrid", "rerank", "parent", "router"];
const TRANSFORMS = ["identity", "rewrite", "hyde", "multiquery", "stepback"];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [question, setQuestion] = useState("Why do I get charged twice?");
  const [strategy, setStrategy] = useState("rerank");
  const [transform, setTransform] = useState("identity");
  const [topK, setTopK] = useState(5);
  const [scoreFloor, setScoreFloor] = useState<number | null>(null);
  const [strictPrompt, setStrictPrompt] = useState(false);

  const [contexts, setContexts] = useState<Context[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [retriever, setRetriever] = useState("");
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setError(`Cannot reach the API at ${API}. Is uvicorn running?`));
  }, []);

  const body = () => ({
    question,
    top_k: topK,
    strategy,
    transform,
    score_floor: scoreFloor,
    strict_prompt: strictPrompt,
  });

  async function call<T>(path: string): Promise<T | null> {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body()),
      });
      if (!response.ok) throw new Error((await response.json()).detail ?? response.statusText);
      return (await response.json()) as T;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setBusy(false);
    }
  }

  /** Retrieval only. The fast path, and the one to reach for when debugging. */
  async function runRetrieve() {
    setResult(null);
    const data = await call<RetrieveResponse>("/retrieve");
    if (data) {
      setContexts(data.contexts);
      setRetriever(data.retriever);
      setElapsed(data.elapsed_ms);
    }
  }

  async function runQuery() {
    const data = await call<QueryResponse>("/query");
    if (data) {
      setResult(data);
      setContexts(data.contexts);
      setRetriever(String(data.trace.retriever ?? ""));
      setElapsed(Number(data.trace.total_ms ?? 0));
    }
  }

  return (
    <div className="app">
      <header>
        <h1>ragkit inspector</h1>
        {health && (
          <span className="meta">
            {health.profile} · {health.chunks_indexed} chunks indexed
          </span>
        )}
      </header>

      <section className="controls">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void runQuery();
          }}
          placeholder="Ask something. Cmd/Ctrl+Enter to answer."
          rows={2}
        />

        <div className="row">
          <label>
            strategy
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              {STRATEGIES.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </label>
          <label>
            transform
            <select value={transform} onChange={(e) => setTransform(e.target.value)}>
              {TRANSFORMS.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
          <label>
            top-k
            <input
              type="number"
              min={1}
              max={25}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={scoreFloor !== null}
              onChange={(e) => setScoreFloor(e.target.checked ? 0.6 : null)}
            />
            abstain below 0.60
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={strictPrompt}
              onChange={(e) => setStrictPrompt(e.target.checked)}
            />
            strict refusal prompt
          </label>
        </div>

        <div className="row">
          <button onClick={() => void runRetrieve()} disabled={busy}>
            Retrieve only
          </button>
          <button className="primary" onClick={() => void runQuery()} disabled={busy}>
            {busy ? "Working…" : "Answer"}
          </button>
          {retriever && (
            <span className="meta">
              {retriever}
              {elapsed !== null && ` · ${Math.round(elapsed)}ms`}
            </span>
          )}
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      {result && (
        <section className={result.abstained ? "answer abstained" : "answer"}>
          <h2>{result.abstained ? "Refused" : "Answer"}</h2>
          <p>{result.answer}</p>
          {result.abstained && (
            <p className="meta">
              {String(result.trace.abstain_reason ?? result.trace.crag_reason ?? "")}
            </p>
          )}
          {result.citations.length > 0 && (
            <p className="meta">Cited: {result.citations.map((c) => c.doc_id).join(", ")}</p>
          )}
          <p className="meta">
            {result.usage.input_tokens}+{result.usage.output_tokens} tokens
            {result.usage.cache_read_tokens > 0 &&
              ` · ${result.usage.cache_read_tokens} from cache`}
            {result.usage.cost_usd > 0 && ` · $${result.usage.cost_usd.toFixed(5)}`}
          </p>
          <button className="link" onClick={() => setShowPrompt(!showPrompt)}>
            {showPrompt ? "hide" : "show"} assembled prompt
          </button>
          {showPrompt && <pre className="prompt">{result.prompt}</pre>}
        </section>
      )}

      <section className="contexts">
        <h2>
          Retrieved context
          {contexts.length > 0 && <span className="meta"> — {contexts.length} passages</span>}
        </h2>
        {contexts.length === 0 && !busy && (
          <p className="meta">
            Nothing retrieved yet. Try &ldquo;Retrieve only&rdquo; — it skips the model, so it
            returns in milliseconds once the embedder is warm.
          </p>
        )}
        {contexts.map((c) => (
          <article key={c.chunk_id}>
            <header>
              <span className="rank">{c.rank}</span>
              <span className="title">{c.title || c.doc_id}</span>
              <span className="meta">
                {c.doc_id} · chars {c.start_char}–{c.end_char}
              </span>
            </header>
            {c.context && <p className="ctx-prefix">{c.context}</p>}
            <p className="chunk-text">{c.text}</p>
            <ScorePanel context={c} />
          </article>
        ))}
      </section>
    </div>
  );
}
