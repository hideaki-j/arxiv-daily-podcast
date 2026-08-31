from __future__ import annotations

from datetime import datetime, timezone

from ir_arxiv_ranker.__main__ import (
    _cost_breakdown_rows,
    _count_record_sources,
    _meets_minimum_email_score,
    _records_after_sending,
    _unsent_pool_statistics,
)
from utils.costs import CostReport


def test_post_send_unsent_pool_stats_exclude_selected_paper():
    records = [
        {"base_arxiv_id": "2406.1", "source": "ir"},
        {"base_arxiv_id": "2406.2", "source": "cl"},
        {"base_arxiv_id": "2406.3", "source": "keywords"},
    ]

    post_send_records = _records_after_sending(records, {"2406.1"})

    assert len(post_send_records) == 2
    assert [record["base_arxiv_id"] for record in post_send_records] == [
        "2406.2",
        "2406.3",
    ]
    assert _count_record_sources(post_send_records) == {
        "ir": 0,
        "cl": 1,
        "keywords": 1,
        "total": 2,
    }


def test_unsent_pool_statistics_use_seen_windows():
    now = datetime(2026, 6, 4, 14, 0, tzinfo=timezone.utc)
    records = [
        {
            "base_arxiv_id": "2406.1",
            "first_seen_at": "2026-06-04T13:00:00Z",
            "last_seen_at": "2026-06-04T13:30:00Z",
            "influence_score": 4,
        },
        {
            "base_arxiv_id": "2406.2",
            "first_seen_at": "2026-06-01T12:00:00Z",
            "last_seen_at": "2026-06-04T13:15:00Z",
            "influence_score": 2,
        },
        {
            "base_arxiv_id": "2406.3",
            "first_seen_at": "2026-05-20T12:00:00Z",
            "last_seen_at": "2026-05-21T12:00:00Z",
            "influence_score": 3,
        },
    ]

    stats = _unsent_pool_statistics(records, now=now)

    assert stats == {
        "unsent_pool_total": 3,
        "fetched_24h": 2,
        "unique_added_24h": 1,
        "added_7d": 2,
    }


def test_cost_breakdown_rows_group_by_stage_and_model():
    report = CostReport()
    report.add("Scoring LLM 1", 0.01, "input 10, output 20", model="gpt-test")
    report.add("Scoring LLM 2", 0.02, "input 30, output 40", model="gpt-test")
    report.add("Transcript LLM 1", None, "usage unavailable", model="gemini-test")

    assert _cost_breakdown_rows(report) == [
        {"stage": "Scoring", "model": "gpt-test", "cost": "3.00¢"},
        {
            "stage": "Transcript",
            "model": "gemini-test",
            "cost": "unknown",
        },
    ]


def test_minimum_email_score_is_inclusive():
    assert _meets_minimum_email_score(9.0, 9.0)
    assert _meets_minimum_email_score(10.0, 9.0)
    assert not _meets_minimum_email_score(8.99, 9.0)
    assert not _meets_minimum_email_score(None, 9.0)
