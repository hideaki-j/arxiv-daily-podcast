from __future__ import annotations

from types import SimpleNamespace

from ir_arxiv_ranker.manga_image import _estimate_image_cost


def test_estimate_image_cost_uses_text_and_image_token_details():
    usage = SimpleNamespace(
        input_tokens_details=SimpleNamespace(text_tokens=1000, image_tokens=0),
        output_tokens=2000,
        output_tokens_details=SimpleNamespace(text_tokens=0, image_tokens=2000),
    )
    pricing = {
        "text_input_usd_per_1m_tokens": 5.0,
        "image_input_usd_per_1m_tokens": 8.0,
        "image_output_usd_per_1m_tokens": 30.0,
    }

    assert _estimate_image_cost(usage, pricing) == 0.065
