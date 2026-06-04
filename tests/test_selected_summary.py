from __future__ import annotations

from ir_arxiv_ranker.models import Paper
from ir_arxiv_ranker.selected_summary import generate_selected_summaries_batch


def _paper(paper_id: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        arxiv_id=f"2406.{paper_id}",
        title="Selected Paper",
        authors=["Test Author"],
        published="2026-06-01T00:00:00Z",
        updated="2026-06-01T00:00:00Z",
        summary="A test abstract.",
        pdf_url=f"https://arxiv.org/pdf/2406.{paper_id}.pdf",
    )


def test_generate_selected_summaries_batch_validates_sections(monkeypatch):
    def fake_batch_call_llm_json(**kwargs):
        return [
            {
                "id": "P1",
                "tldr": "A concise summary.",
                "background": "Background text.",
                "existing_problem": "Problem text.",
                "proposed_method": "Method text.",
                "results": "Results text.",
            }
        ]

    monkeypatch.setattr(
        "ir_arxiv_ranker.selected_summary.batch_call_llm_json",
        fake_batch_call_llm_json,
    )
    monkeypatch.setattr(
        "ir_arxiv_ranker.selected_summary._extract_pdf_text",
        lambda path: "Main paper text.",
    )

    summaries = generate_selected_summaries_batch(
        client=object(),
        model="test-model",
        prompt_template="{{ paper.id }} {{ paper_text }}",
        papers=[_paper("P1")],
        pdf_paths=[object()],
    )

    assert summaries["P1"]["tldr"] == "A concise summary."
    assert summaries["P1"]["existing_problem"] == "Problem text."
