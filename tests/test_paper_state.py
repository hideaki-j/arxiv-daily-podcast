from __future__ import annotations

from ir_arxiv_ranker.models import Paper
from ir_arxiv_ranker.paper_state import (
    base_arxiv_id,
    load_paper_state,
    mark_sent,
    merge_discovered_papers,
    needs_ranking_score,
    pooled_records,
    records_to_papers,
    save_paper_state,
    set_affiliations,
    set_ranking_scores,
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


def test_load_missing_state_returns_empty_pool(tmp_path):
    state = load_paper_state(tmp_path / "missing.json")

    assert state == {"schema_version": 1, "pooled_papers": {}}


def test_load_legacy_state_migrates_to_pool_with_sent_flag(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "papers": {
            "2406.1": {"base_arxiv_id": "2406.1", "status": "sent"},
            "2406.2": {"base_arxiv_id": "2406.2", "status": "unsent"}
          }
        }
        """
    )

    state = load_paper_state(path)

    assert set(state["pooled_papers"]) == {"2406.1", "2406.2"}
    assert state["pooled_papers"]["2406.1"]["sent"] is True
    assert state["pooled_papers"]["2406.2"]["sent"] is False


def test_load_two_bucket_state_migrates_sent_bucket_to_pool(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "sent_papers": {"2406.1": {"base_arxiv_id": "2406.1"}},
          "pooled_papers": {"2406.2": {"base_arxiv_id": "2406.2", "sent": false}}
        }
        """
    )

    state = load_paper_state(path)

    assert set(state["pooled_papers"]) == {"2406.1", "2406.2"}
    assert state["pooled_papers"]["2406.1"]["sent"] is True
    assert state["pooled_papers"]["2406.2"]["sent"] is False


def test_merge_pools_all_papers_and_stores_affiliations():
    state = {"schema_version": 1, "pooled_papers": {}}

    changed = merge_discovered_papers(
        state,
        [_paper("2406.12345v1"), _paper("2406.99999v1", paper_id="CL001")],
        scores_by_id={"IR001": 5, "CL001": 4},
        seen_at="2026-06-03T12:00:00Z",
    )
    set_affiliations(state, {"2406.12345": "University of Waterloo"})

    assert base_arxiv_id("2406.12345v1") == "2406.12345"
    assert changed == ["2406.12345", "2406.99999"]
    assert set(state["pooled_papers"]) == {"2406.12345", "2406.99999"}
    assert state["pooled_papers"]["2406.12345"]["influence_score"] == 5
    assert state["pooled_papers"]["2406.99999"]["influence_score"] == 4
    assert state["pooled_papers"]["2406.12345"]["in_pool"] is True
    assert state["pooled_papers"]["2406.99999"]["in_pool"] is True
    assert state["pooled_papers"]["2406.12345"]["affiliations"] == "University of Waterloo"


def test_merge_marks_below_threshold_papers_out_of_pool():
    state = {"schema_version": 1, "pooled_papers": {}}

    merge_discovered_papers(
        state,
        [_paper("2406.12345v1"), _paper("2406.99999v1", paper_id="CL001")],
        scores_by_id={"IR001": 3, "CL001": 2},
        seen_at="2026-06-03T12:00:00Z",
        influence_threshold=3,
    )

    assert state["pooled_papers"]["2406.12345"]["in_pool"] is True
    assert state["pooled_papers"]["2406.99999"]["in_pool"] is False
    assert [record["base_arxiv_id"] for record in pooled_records(state)] == ["2406.12345"]


def test_sent_paper_stays_in_pool_and_is_excluded_from_unsent_records():
    state = {"schema_version": 1, "pooled_papers": {}}
    merge_discovered_papers(
        state,
        [_paper("2406.12345v1")],
        scores_by_id={"IR001": 5},
        seen_at="2026-06-03T12:00:00Z",
    )
    mark_sent(state, "2406.12345", "2026-06-03T13:00:00Z", "run-1")

    changed = merge_discovered_papers(
        state,
        [_paper("2406.12345v2", title="Updated sent paper")],
        scores_by_id={"IR001": 5},
        seen_at="2026-06-04T12:00:00Z",
    )

    record = state["pooled_papers"]["2406.12345"]
    assert changed == ["2406.12345"]
    assert record["latest_arxiv_id"] == "2406.12345v2"
    assert record["title"] == "Updated sent paper"
    assert record["sent"] is True
    assert pooled_records(state) == []
    assert pooled_records(state, include_sent=True) == [record]


def test_missing_influence_score_does_not_wipe_existing_value():
    state = {"schema_version": 1, "pooled_papers": {}}
    merge_discovered_papers(
        state,
        [_paper("2406.12345v1")],
        scores_by_id={"IR001": 5},
        seen_at="2026-06-03T12:00:00Z",
    )

    changed = merge_discovered_papers(
        state,
        [_paper("2406.12345v1")],
        scores_by_id={},
        seen_at="2026-06-04T12:00:00Z",
    )

    assert changed == []
    assert state["pooled_papers"]["2406.12345"]["influence_score"] == 5


def test_existing_paper_influence_score_is_not_refreshed():
    state = {"schema_version": 1, "pooled_papers": {}}
    merge_discovered_papers(
        state,
        [_paper("2406.12345v1")],
        scores_by_id={"IR001": 5},
        seen_at="2026-06-03T12:00:00Z",
    )

    changed = merge_discovered_papers(
        state,
        [_paper("2406.12345v1")],
        scores_by_id={"IR001": 1},
        seen_at="2026-06-04T12:00:00Z",
    )

    assert changed == []
    assert state["pooled_papers"]["2406.12345"]["influence_score"] == 5


def test_records_to_papers_uses_bucket_ids_and_base_mapping():
    state = {"schema_version": 1, "pooled_papers": {}}
    merge_discovered_papers(
        state,
        [_paper("2406.10000v1"), _paper("2406.20000v1", paper_id="CL001")],
        scores_by_id={"IR001": 5, "CL001": 5},
        seen_at="2026-06-03T12:00:00Z",
    )

    papers, mapping = records_to_papers(pooled_records(state))

    assert [paper.paper_id for paper in papers] == ["B001", "B002"]
    assert set(mapping.values()) == {"2406.10000", "2406.20000"}


def test_set_ranking_scores_persists_all_scoring_outputs():
    state = {"schema_version": 1, "pooled_papers": {}}
    merge_discovered_papers(
        state,
        [_paper("2406.12345v1")],
        scores_by_id={"IR001": 5},
        seen_at="2026-06-03T12:00:00Z",
    )

    set_ranking_scores(
        state,
        paper_id_to_base_id={"B001": "2406.12345"},
        scores_by_id={"B001": {"legal_domain": 2, "survey": 1}},
        total_score_by_id={"B001": 1.0},
        tldr_by_id={"B001": "Short summary."},
    )

    record = state["pooled_papers"]["2406.12345"]
    assert record["ranking_scores"] == {"legal_domain": 2, "survey": 1}
    assert record["ranking_total_score"] == 1.0
    assert record["ranking_tldr"] == "Short summary."
    assert record["ranking_scored_input_hash"] == record["ranking_input_hash"]
    assert not needs_ranking_score(record, ["legal_domain", "survey"])


def test_needs_ranking_score_when_paper_input_changes():
    state = {"schema_version": 1, "pooled_papers": {}}
    merge_discovered_papers(
        state,
        [_paper("2406.12345v1")],
        scores_by_id={"IR001": 5},
        seen_at="2026-06-03T12:00:00Z",
    )
    set_ranking_scores(
        state,
        paper_id_to_base_id={"B001": "2406.12345"},
        scores_by_id={"B001": {"legal_domain": 2, "survey": 1}},
        total_score_by_id={"B001": 1.0},
        tldr_by_id={"B001": "Short summary."},
    )

    merge_discovered_papers(
        state,
        [_paper("2406.12345v2", title="Updated paper")],
        scores_by_id={"IR001": 5},
        seen_at="2026-06-04T12:00:00Z",
    )

    record = state["pooled_papers"]["2406.12345"]
    assert needs_ranking_score(record, ["legal_domain", "survey"])


def test_save_and_load_state_roundtrip(tmp_path):
    path = tmp_path / "state" / "discovered_papers.json"
    state = {"schema_version": 1, "pooled_papers": {}}
    merge_discovered_papers(
        state,
        [_paper("2406.12345v1")],
        scores_by_id={"IR001": 5},
        seen_at="2026-06-03T12:00:00Z",
    )

    save_paper_state(path, state)

    assert load_paper_state(path) == state
