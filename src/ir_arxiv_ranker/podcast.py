from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, StrictUndefined
from pypdf import PdfReader

from utils.call_llm import batch_call_llm_text, call_llm_text
from utils.costs import CostTracker
from utils.naming import build_file_stem

from .models import Paper


def load_podcast_prompt(path: Path) -> str:
    return path.read_text()


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _truncate_words(text: str, limit: int | None) -> str:
    if not limit:
        return text
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def _render_prompt(prompt_template: str, paper: Paper, paper_text: str) -> str:
    env = Environment(autoescape=False, undefined=StrictUndefined)
    return env.from_string(prompt_template).render(
        paper=paper,
        paper_text=paper_text,
    )


def generate_transcript(
    client,
    model: str,
    prompt_template: str,
    paper: Paper,
    pdf_path: Path,
    word_cutoff: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    label: str = "Podcast LLM",
    openai_timeout: int | None = None,
) -> str:
    paper_text = _truncate_words(_extract_pdf_text(pdf_path), word_cutoff)
    prompt = _render_prompt(prompt_template, paper, paper_text)
    return call_llm_text(
        client=client,
        model=model,
        prompt=prompt,
        pricing=pricing,
        cost_tracker=cost_tracker,
        label=label,
        timeout=openai_timeout,
    )


def generate_transcripts_batch(
    client,
    model: str,
    prompt_template: str,
    papers: list[Paper],
    pdf_paths: list[Path],
    word_cutoff: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    label: str = "Podcast LLM",
    openai_timeout: int | None = None,
    max_workers: int = 4,
) -> list[str]:
    prompts: list[str] = []
    for paper, pdf_path in zip(papers, pdf_paths):
        paper_text = _truncate_words(_extract_pdf_text(pdf_path), word_cutoff)
        prompt = _render_prompt(prompt_template, paper, paper_text)
        prompts.append(prompt)

    return batch_call_llm_text(
        client=client,
        model=model,
        prompts=prompts,
        timeout=openai_timeout,
        pricing=pricing,
        cost_tracker=cost_tracker,
        label=label,
        max_workers=max_workers,
    )


def write_transcript(transcript_dir: Path, paper: Paper, rank: int, transcript: str) -> Path:
    stem = build_file_stem(rank, paper.paper_id, paper.title)
    transcript_path = transcript_dir / f"{stem}.txt"
    transcript_path.write_text(transcript)
    return transcript_path
