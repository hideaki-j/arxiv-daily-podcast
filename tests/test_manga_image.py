from __future__ import annotations

import base64
from types import SimpleNamespace

from ir_arxiv_ranker.manga_image import _estimate_image_cost, generate_manga_image
from ir_arxiv_ranker.models import Paper


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


def test_generate_manga_image_omits_response_format_for_gpt_image_models(monkeypatch, tmp_path):
    class FakeImages:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                data=[SimpleNamespace(b64_json=base64.b64encode(b"image").decode("ascii"))],
                usage=None,
            )

    fake_images = FakeImages()
    fake_client = SimpleNamespace(images=fake_images)
    paper = Paper(
        paper_id="B001",
        arxiv_id="2406.12345v1",
        title="Test Paper",
        authors=["Ada Lovelace"],
        published="2026-06-01T00:00:00Z",
        updated="2026-06-01T00:00:00Z",
        summary="A test summary.",
        pdf_url="https://arxiv.org/pdf/2406.12345v1.pdf",
    )
    monkeypatch.setattr("ir_arxiv_ranker.manga_image._extract_pdf_text", lambda _: "paper text")

    image_path = generate_manga_image(
        client=fake_client,
        model="gpt-image-2",
        prompt_template="{{ paper_text }}",
        paper=paper,
        pdf_path=tmp_path / "paper.pdf",
        image_dir=tmp_path / "manga",
        rank=1,
        size="1536x1024",
        quality="high",
        output_format="png",
    )

    assert "response_format" not in fake_images.kwargs
    assert fake_images.kwargs["model"] == "gpt-image-2"
    assert image_path.read_bytes() == b"image"
