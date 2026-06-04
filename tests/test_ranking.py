from __future__ import annotations

from ir_arxiv_ranker.models import Paper
from ir_arxiv_ranker.ranking import RankingAspect, rank_from_scores, rank_papers


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


def test_rank_papers_scores_each_paper_and_orders_by_aggregate(monkeypatch):
    captured = {}

    def fake_batch_call_llm_json(**kwargs):
        captured.update(kwargs)
        return [
            {
                "id": "P1",
                "scores": {"survey": 2, "legal_domain": 0, "prompting": 1},
                "tldr": "Paper one summary.",
            },
            {
                "id": "P2",
                "scores": {"survey": 0, "legal_domain": 2, "prompting": 1},
                "tldr": "Paper two summary.",
            },
        ]

    monkeypatch.setattr("ir_arxiv_ranker.ranking.batch_call_llm_json", fake_batch_call_llm_json)
    aspects = [
        RankingAspect("survey", "survey papers", 1.0, "negative"),
        RankingAspect("legal_domain", "legal domain", 1.0, "positive"),
        RankingAspect("prompting", "prompting techniques", 2.0, "positive"),
    ]

    rankings = rank_papers(
        client=object(),
        model="test-model",
        scoring_prompt_template=(
            "{{ paper.id }} {% for aspect in aspects %}{{ aspect.key }} {% endfor %}"
        ),
        papers=[_paper("P1", "Survey"), _paper("P2", "Legal prompting")],
        top_n=2,
        author_influence_by_id={"P1": 5, "P2": 0},
        aspects=aspects,
        max_workers=150,
    )

    assert len(captured["prompts"]) == 2
    assert captured["max_workers"] == 150
    assert rankings.total_score_by_id == {"P1": 5.0, "P2": 4.0}
    assert rankings.final_ranking == ["P1", "P2"]
    assert rankings.scores_by_id["P1"] == {
        "survey": 2,
        "legal_domain": 0,
        "prompting": 1,
        "author_influence_score": 5,
    }
    assert rankings.tldr_by_id["P1"] == "Paper one summary."


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
