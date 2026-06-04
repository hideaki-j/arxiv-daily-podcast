from __future__ import annotations

from ir_arxiv_ranker.models import Paper
from ir_arxiv_ranker.paper_state import (
    base_arxiv_id,
    load_paper_state,
    mark_sent,
    merge_discovered_papers,
    records_to_papers,
    save_paper_state,
    set_affiliations,
    unsent_records,
)


def _paper(arxiv_id: str, paper_id: str = "IR001", title: str = "Paper") -> Paper:
    return Paper(
        paper_id=paper_id,
        arxiv_id=arxiv_id,
        title=title,
        authors=["Jimmy Lin"],
        published="2026-06-01T00:00:00Z",
        updated="2026-06-01T00:00:00Z",
        summary="A test summary.",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


def test_load_missing_state_returns_empty_bucket(tmp_path):
    state = load_paper_state(tmp_path / "missing.json")

    assert state == {"schema_version": 1, "papers": {}}


def test_merge_deduplicates_versions_and_stores_affiliations():
    state = {"schema_version": 1, "papers": {}}

    changed = merge_discovered_papers(state, [_paper("2406.12345v1")], "2026-06-03T12:00:00Z")
    set_affiliations(state, {"2406.12345": "University of Waterloo"})
    changed_again = merge_discovered_papers(
        state,
        [_paper("2406.12345v2", title="Paper revised")],
        "2026-06-04T12:00:00Z",
    )

    assert base_arxiv_id("2406.12345v2") == "2406.12345"
    assert changed == ["2406.12345"]
    assert changed_again == ["2406.12345"]
    assert len(state["papers"]) == 1
    record = state["papers"]["2406.12345"]
    assert record["latest_arxiv_id"] == "2406.12345v2"
    assert record["title"] == "Paper revised"
    assert record["affiliations"] == "University of Waterloo"
    assert record["status"] == "unsent"


def test_sent_paper_is_not_returned_as_unsent_or_changed():
    state = {"schema_version": 1, "papers": {}}
    merge_discovered_papers(state, [_paper("2406.12345v1")], "2026-06-03T12:00:00Z")
    mark_sent(state, "2406.12345", "2026-06-03T13:00:00Z", "run-1")

    changed = merge_discovered_papers(state, [_paper("2406.12345v2")], "2026-06-04T12:00:00Z")

    assert changed == []
    assert unsent_records(state) == []
    assert state["papers"]["2406.12345"]["latest_arxiv_id"] == "2406.12345v2"
    assert state["papers"]["2406.12345"]["status"] == "sent"


def test_unchanged_unsent_paper_with_affiliation_is_not_requeued():
    state = {"schema_version": 1, "papers": {}}
    merge_discovered_papers(state, [_paper("2406.12345v1")], "2026-06-03T12:00:00Z")
    set_affiliations(state, {"2406.12345": "University of Waterloo"})

    changed = merge_discovered_papers(state, [_paper("2406.12345v1")], "2026-06-04T12:00:00Z")

    assert changed == []
    assert state["papers"]["2406.12345"]["last_seen_at"] == "2026-06-04T12:00:00Z"


def test_records_to_papers_uses_bucket_ids_and_base_mapping():
    state = {"schema_version": 1, "papers": {}}
    merge_discovered_papers(
        state,
        [_paper("2406.10000v1"), _paper("2406.20000v1", paper_id="CL001")],
        "2026-06-03T12:00:00Z",
    )

    papers, mapping = records_to_papers(unsent_records(state))

    assert [paper.paper_id for paper in papers] == ["B001", "B002"]
    assert mapping == {"B001": "2406.10000", "B002": "2406.20000"}


def test_save_and_load_state_roundtrip(tmp_path):
    path = tmp_path / "state" / "discovered_papers.json"
    state = {"schema_version": 1, "papers": {}}
    merge_discovered_papers(state, [_paper("2406.12345v1")], "2026-06-03T12:00:00Z")

    save_paper_state(path, state)

    assert load_paper_state(path) == state
