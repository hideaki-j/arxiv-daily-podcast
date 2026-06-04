from __future__ import annotations

from ir_arxiv_ranker.__main__ import _count_record_sources, _records_after_sending


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
