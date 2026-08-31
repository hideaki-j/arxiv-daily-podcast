from __future__ import annotations

import csv
import json

from ir_arxiv_ranker.models import Paper
from ir_arxiv_ranker.output import write_csv, write_results_json
from ir_arxiv_ranker.ranking import Rankings


def _paper(paper_id: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        arxiv_id=f"2406.{paper_id}",
        title=f"Paper {paper_id}",
        authors=["Test Author"],
        published="2026-06-01T00:00:00Z",
        updated="2026-06-01T00:00:00Z",
        summary="A test abstract.",
        pdf_url=f"https://arxiv.org/pdf/2406.{paper_id}.pdf",
    )


def test_outputs_support_mixed_legacy_and_new_rubric_scores(tmp_path):
    papers = [_paper("P1"), _paper("P2")]
    rankings = Rankings(
        automatic_eval_ranking=["P2", "P1"],
        user_simulator_ranking=["P2", "P1"],
        final_ranking=["P2", "P1"],
        tldr_by_id={},
        scores_by_id={
            "P1": {"legal_domain": 1},
            "P2": {"legal_domain": 1, "agentic_search_or_deep_research": 2},
        },
        total_score_by_id={"P1": 1.0, "P2": 3.0},
    )

    csv_path = write_csv(tmp_path, {paper.paper_id: paper for paper in papers}, rankings)
    json_path = write_results_json(tmp_path, papers, rankings)

    with csv_path.open(newline="") as handle:
        csv_scores = {
            row["id"]: json.loads(row["scoring_scores"])
            for row in csv.DictReader(handle)
        }
    payload = json.loads(json_path.read_text())

    assert "agentic_search_or_deep_research" not in csv_scores["P1"]
    assert csv_scores["P2"]["agentic_search_or_deep_research"] == 2
    assert "agentic_search_or_deep_research" not in payload["rankings"]["scoring_scores"]["P1"]
    assert payload["rankings"]["scoring_scores"]["P2"][
        "agentic_search_or_deep_research"
    ] == 2
