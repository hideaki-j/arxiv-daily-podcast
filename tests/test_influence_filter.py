from __future__ import annotations

from ir_arxiv_ranker.influence_filter import _build_response_format, _validate_response


def test_priority_score_is_eight_and_gap_scores_are_rejected():
    assert _validate_response({"author_influence_score": 8}) == 8
    assert _validate_response({"author_influence_score": 4}) == 4
    assert _validate_response({"author_influence_score": 5}) is None
    assert _validate_response({"author_influence_score": 6}) is None
    assert _validate_response({"author_influence_score": 7}) is None


def test_structured_output_schema_exposes_only_allowed_scores():
    score_schema = _build_response_format()["schema"]["properties"][
        "author_influence_score"
    ]

    assert score_schema["enum"] == [0, 1, 2, 3, 4, 8]
