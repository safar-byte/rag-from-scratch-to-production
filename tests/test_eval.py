"""The evaluation harness itself.

A bug here is worse than a bug in retrieval: retrieval bugs show up as bad answers,
harness bugs show up as good numbers. These tests run entirely offline with fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.config import Settings
from ragkit.eval.golden import GoldenQuestion, load_golden, validate_golden
from ragkit.eval.judge import DeterministicJudge, looks_like_refusal
from ragkit.eval.report import build_row
from ragkit.eval.run import dedupe_to_documents, evaluate_retrieval, run_eval
from ragkit.pipeline import RagPipeline
from ragkit.types import Chunk, Scored

# ---- golden set ------------------------------------------------------------


def test_the_shipped_golden_set_is_valid() -> None:
    # Guards the failure where a golden question names a document that no longer
    # exists: it becomes permanently unanswerable and reads as a mysterious dent in
    # recall that nobody can explain.
    problems = validate_golden(load_golden())
    assert problems == [], problems


def test_golden_set_covers_every_question_kind() -> None:
    kinds = {q.kind for q in load_golden()}
    assert {"lookup", "conceptual", "multi_hop", "unanswerable"} <= kinds


def test_validation_catches_a_missing_document(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("content", encoding="utf-8")
    questions = [GoldenQuestion(id="q1", question="?", relevant=["ghost.md"], kind="lookup")]
    problems = validate_golden(questions, corpus_dir=tmp_path)
    assert any("ghost.md" in p for p in problems)


def test_validation_catches_a_mislabelled_unanswerable(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("content", encoding="utf-8")
    questions = [
        GoldenQuestion(id="q1", question="?", relevant=["real.md"], kind="unanswerable"),
        GoldenQuestion(id="q2", question="?", relevant=[], kind="lookup"),
    ]
    problems = validate_golden(questions, corpus_dir=tmp_path)
    assert len(problems) == 2


def test_validation_catches_duplicate_ids(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("content", encoding="utf-8")
    questions = [
        GoldenQuestion(id="q1", question="a", relevant=["real.md"], kind="lookup"),
        GoldenQuestion(id="q1", question="b", relevant=["real.md"], kind="lookup"),
    ]
    assert any("duplicate" in p for p in validate_golden(questions, corpus_dir=tmp_path))


# ---- document-level collapsing --------------------------------------------


def test_chunk_ranking_collapses_to_document_ranking_keeping_best_rank() -> None:
    # Three chunks of doc a occupy the top; doc a must count once, at rank 1.
    assert dedupe_to_documents(["a", "a", "b", "a", "c"]) == ["a", "b", "c"]


def test_document_ranking_preserves_order() -> None:
    assert dedupe_to_documents(["c", "b", "a"]) == ["c", "b", "a"]


# ---- scoring ---------------------------------------------------------------


def test_retrieval_scoring_covers_every_k() -> None:
    question = GoldenQuestion(id="q", question="?", relevant=["a.md"], kind="lookup")
    scores = evaluate_retrieval(question, ["a.md", "b.md", "c.md"])
    for k in (1, 3, 5):
        assert scores[f"recall@{k}"] == 1.0
        assert f"ndcg@{k}" in scores
    assert scores["mrr"] == 1.0


# ---- the deterministic judge ----------------------------------------------


def _contexts(text: str) -> list[Scored]:
    return [Scored(chunk=Chunk(chunk_id="c", doc_id="d.md", text=text), score=1.0)]


def test_refusal_detection() -> None:
    assert looks_like_refusal("The context does not contain that information.")
    assert looks_like_refusal("I cannot answer from the provided passages.")
    assert not looks_like_refusal("The default port is 8788.")


def test_unanswerable_question_scores_on_whether_it_refused() -> None:
    judge = DeterministicJudge()
    question = GoldenQuestion(
        id="q", question="Capital of France?", relevant=[], kind="unanswerable"
    )

    refused = judge.grade(question, "The context does not contain this.", _contexts("irrelevant"))
    assert refused.correct_refusal == 1.0
    assert refused.groundedness == 1.0

    confabulated = judge.grade(question, "It is Paris.", _contexts("irrelevant"))
    assert confabulated.correct_refusal == 0.0
    assert confabulated.groundedness == 0.0


def test_relevance_tracks_expected_strings() -> None:
    judge = DeterministicJudge()
    question = GoldenQuestion(
        id="q", question="What port?", relevant=["d.md"], answer_contains=["8788"], kind="lookup"
    )
    context = _contexts("The query service listens on port 8788 by default.")

    assert judge.grade(question, "It listens on port 8788.", context).relevance == 1.0
    assert judge.grade(question, "It listens on port 9000.", context).relevance == 0.0


def test_groundedness_proxy_rewards_reusing_context_vocabulary() -> None:
    judge = DeterministicJudge()
    question = GoldenQuestion(id="q", question="?", relevant=["d.md"], kind="conceptual")
    context = _contexts("The indexer takes an exclusive lock and exits with status 75.")

    grounded = judge.grade(question, "The indexer takes an exclusive lock.", context)
    invented = judge.grade(
        question, "Kubernetes rebalances shards across availability zones.", context
    )
    assert grounded.groundedness > invented.groundedness


# ---- the report writer -----------------------------------------------------


def test_ceiling_flag_is_carried_onto_the_row() -> None:
    summary = {
        "config": {"profile": "local", "retriever": "dense", "chunks_indexed": 80},
        "retrieval": {"recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "mrr": 1.0, "ndcg@5": 1.0},
        "generation": {},
        "ceiling_reached": True,
        "mean_latency_ms": 120.0,
        "total_cost_usd": 0.0,
    }
    # A number that cannot be trusted must never appear in the table without its caveat.
    assert "ceiling" in build_row(summary, "test run")


def test_missing_metrics_render_as_a_dash_not_a_crash() -> None:
    summary = {
        "config": {"profile": "local", "retriever": "dense", "chunks_indexed": 80},
        "retrieval": {"recall@1": 0.5},
        "generation": {},
        "ceiling_reached": False,
        "mean_latency_ms": 10.0,
        "total_cost_usd": 0.0,
    }
    row = build_row(summary, "partial")
    assert "| - |" in row


# ---- end to end ------------------------------------------------------------


def test_run_eval_end_to_end_with_fakes(pipeline: RagPipeline, temp_settings: Settings) -> None:
    pipeline.ingest()
    summary = run_eval(temp_settings, pipeline=pipeline, retrieval_only=True, limit=6, quiet=True)

    assert len(summary["results"]) == 6
    assert summary["config"]["profile"] == "local"
    for k in (1, 3, 5):
        assert 0.0 <= summary["retrieval"][f"recall@{k}"] <= 1.0
    assert "max_recall@1" in next(iter(summary["by_kind"].values()))


def test_run_eval_refuses_an_invalid_golden_set(
    pipeline: RagPipeline, temp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ragkit.eval.run.validate_golden", lambda _q: ["q99: relevant document 'gone.md' missing"]
    )
    pipeline.ingest()
    with pytest.raises(ValueError, match="Golden set is invalid"):
        run_eval(temp_settings, pipeline=pipeline, retrieval_only=True, quiet=True)
