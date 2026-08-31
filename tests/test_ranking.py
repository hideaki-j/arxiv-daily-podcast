from __future__ import annotations

from datetime import datetime, timezone

from ir_arxiv_ranker.models import Paper
from ir_arxiv_ranker.paper_state import needs_scoring_score
from ir_arxiv_ranker.ranking import (
    ScoringAspect,
    aggregate_score,
    applicable_scoring_aspects,
    load_scoring_aspects,
    rank_from_scores,
    score_papers,
)


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        arxiv_id=f"2406.{paper_id}",
        title=title,
        authors=["Test Author"],
        published="2026-06-01T00:00:00Z",
        updated="2026-06-01T00:00:00Z",
        summary="A test abstract.",
        pdf_url=f"https://arxiv.org/pdf/2406.{paper_id}.pdf",
    )


def test_score_papers_scores_each_paper_and_orders_by_aggregate(monkeypatch):
    captured = {}

    def fake_batch_call_llm_json(**kwargs):
        captured.update(kwargs)
        return [
            {
                "id": "P1",
                "scores": {"survey": 2, "legal_domain": 0, "prompting": 1},
            },
            {
                "id": "P2",
                "scores": {"survey": 0, "legal_domain": 2, "prompting": 1},
            },
        ]

    monkeypatch.setattr("ir_arxiv_ranker.ranking.batch_call_llm_json", fake_batch_call_llm_json)
    aspects = [
        ScoringAspect("survey", "survey papers", 1.0, "negative"),
        ScoringAspect("legal_domain", "legal domain", 1.0, "positive"),
        ScoringAspect("prompting", "prompting techniques", 2.0, "positive"),
    ]

    rankings = score_papers(
        client=object(),
        model="test-model",
        scoring_prompt_template=(
            "{{ paper.id }} {% for aspect in aspects %}{{ aspect.key }} {% endfor %}"
        ),
        papers=[_paper("P1", "Survey"), _paper("P2", "Legal prompting")],
        top_n=2,
        author_influence_by_id={"P1": 6, "P2": 0},
        aspects=aspects,
        max_workers=150,
    )

    assert len(captured["prompts"]) == 2
    assert captured["max_workers"] == 150
    assert rankings.total_score_by_id == {"P1": 6.0, "P2": 4.0}
    assert rankings.final_ranking == ["P1", "P2"]
    assert rankings.scores_by_id["P1"] == {
        "survey": 2,
        "legal_domain": 0,
        "prompting": 1,
        "author_influence_score": 6,
    }
    assert rankings.tldr_by_id == {}


def test_rank_from_scores_orders_without_llm_call():
    rankings = rank_from_scores(
        papers=[_paper("P1", "Lower"), _paper("P2", "Higher")],
        top_n=1,
        scores_by_id={"P1": {"legal_domain": 1}, "P2": {"legal_domain": 2}},
        total_score_by_id={"P1": 1.0, "P2": 2.0},
        tldr_by_id={"P1": "One.", "P2": "Two."},
    )

    assert rankings.final_ranking == ["P2"]
    assert rankings.scores_by_id["P1"] == {"legal_domain": 1}


def test_forward_only_aspect_is_not_applied_to_legacy_record(tmp_path):
    path = tmp_path / "scoring_aspects.yaml"
    path.write_text(
        """
positive:
  legal_domain:
    label: legal domain
    weight: 1
  agentic_search_or_deep_research:
    label: agentic search / deep research
    guidance: Score retrieval-only work as 0.
    weight: 1
    effective_from: "2026-08-31T20:31:40Z"
"""
    )

    aspects = load_scoring_aspects(path)
    legacy_aspects = applicable_scoring_aspects(aspects, "2026-08-31T20:00:00Z")
    future_aspects = applicable_scoring_aspects(aspects, "2026-08-31T21:00:00Z")
    legacy_record = {
        "scoring_scores": {"legal_domain": 2},
        "scoring_total_score": 2.0,
        "scoring_input_hash": "same",
        "scoring_scored_input_hash": "same",
    }

    assert [aspect.key for aspect in legacy_aspects] == ["legal_domain"]
    assert [aspect.key for aspect in future_aspects] == [
        "legal_domain",
        "agentic_search_or_deep_research",
    ]
    assert not needs_scoring_score(
        legacy_record, [aspect.key for aspect in legacy_aspects]
    )
    assert aggregate_score(legacy_record["scoring_scores"], aspects) == 2.0


def test_forward_only_aspect_is_still_required_for_new_llm_scores(monkeypatch):
    captured = {}
    aspects = [
        ScoringAspect("legal_domain", "legal domain", 1.0, "positive"),
        ScoringAspect(
            "agentic_search_or_deep_research",
            "agentic search / deep research",
            1.0,
            "positive",
            guidance="Score retrieval-only work as 0.",
            effective_from=datetime(2026, 8, 31, 20, 31, 40, tzinfo=timezone.utc),
        ),
    ]

    def fake_batch_call_llm_json(**kwargs):
        captured.update(kwargs)
        return [
            {
                "id": "P1",
                "scores": {
                    "legal_domain": 0,
                    "agentic_search_or_deep_research": 2,
                },
            }
        ]

    monkeypatch.setattr("ir_arxiv_ranker.ranking.batch_call_llm_json", fake_batch_call_llm_json)

    rankings = score_papers(
        client=object(),
        model="test-model",
        scoring_prompt_template=(
            "{% for aspect in aspects %}{{ aspect.key }}: {{ aspect.guidance }}{% endfor %}"
        ),
        papers=[_paper("P1", "Agentic search")],
        top_n=1,
        aspects=aspects,
    )

    score_schema = captured["response_formats"][0]["schema"]["properties"]["scores"]
    assert "agentic_search_or_deep_research" in score_schema["required"]
    assert "Score retrieval-only work as 0." in captured["prompts"][0]
    assert rankings.scores_by_id["P1"]["agentic_search_or_deep_research"] == 2
    assert rankings.total_score_by_id["P1"] == 2.0
