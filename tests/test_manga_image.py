from __future__ import annotations

import base64
from types import SimpleNamespace

from ir_arxiv_ranker.manga_image import (
    _estimate_image_cost,
    generate_manga_image,
    generate_manga_instruction,
)
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
    image_path = generate_manga_image(
        client=fake_client,
        model="gpt-image-2",
        prompt_template="{{ manga_instruction }}",
        manga_instruction="six panel manga plan",
        paper=paper,
        image_dir=tmp_path / "manga",
        rank=1,
        size="1536x1024",
        quality="high",
        output_format="png",
    )

    assert "response_format" not in fake_images.kwargs
    assert fake_images.kwargs["model"] == "gpt-image-2"
    assert fake_images.kwargs["prompt"] == "six panel manga plan"
    assert image_path.read_bytes() == b"image"


def test_generate_manga_image_applies_final_prompt_char_cutoff(tmp_path):
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

    generate_manga_image(
        client=fake_client,
        model="gpt-image-2",
        prompt_template="{{ manga_instruction }}",
        manga_instruction="abcdef",
        paper=paper,
        image_dir=tmp_path / "manga",
        rank=1,
        size="1536x1024",
        quality="high",
        output_format="png",
        char_cutoff=3,
    )

    assert fake_images.kwargs["prompt"] == "abc"


def test_generate_manga_image_renders_manga_style(tmp_path):
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

    generate_manga_image(
        client=fake_client,
        model="gpt-image-2",
        prompt_template="{{ manga_style }} {{ manga_instruction }}",
        manga_instruction="six panel manga plan",
        paper=paper,
        image_dir=tmp_path / "manga",
        rank=1,
        size="1536x1024",
        quality="high",
        output_format="png",
        manga_style="private style",
    )

    assert fake_images.kwargs["prompt"] == "private style six panel manga plan"


def test_generate_manga_instruction_applies_image_only_char_cutoff(monkeypatch, tmp_path):
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
    captured = {}
    monkeypatch.setattr(
        "ir_arxiv_ranker.manga_image._extract_pdf_text",
        lambda _: "abcdef",
    )

    def fake_call_llm_text(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "plan"

    monkeypatch.setattr("ir_arxiv_ranker.manga_image.call_llm_text", fake_call_llm_text)

    instruction = generate_manga_instruction(
        client=SimpleNamespace(),
        model="gpt-5.5",
        prompt_template="{{ paper_text }}",
        paper=paper,
        pdf_path=tmp_path / "paper.pdf",
        char_cutoff=3,
    )

    assert captured["prompt"] == "abc"
    assert instruction == "plan"


def test_generate_manga_instruction_renders_manga_style(monkeypatch, tmp_path):
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
    captured = {}
    monkeypatch.setattr(
        "ir_arxiv_ranker.manga_image._extract_pdf_text",
        lambda _: "paper text",
    )

    def fake_call_llm_text(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "plan"

    monkeypatch.setattr("ir_arxiv_ranker.manga_image.call_llm_text", fake_call_llm_text)

    instruction = generate_manga_instruction(
        client=SimpleNamespace(),
        model="gpt-5.5",
        prompt_template="{{ manga_style }} {{ paper_text }}",
        paper=paper,
        pdf_path=tmp_path / "paper.pdf",
        manga_style="private style",
    )

    assert captured["prompt"] == "private style paper text"
    assert instruction == "plan"
