from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from utils.call_llm import batch_call_llm_text, call_llm_text
from utils.costs import CostTracker

from .models import Paper


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _first_tokens(text: str, token_limit: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:token_limit])


def extract_affiliations(
    client,
    model: str,
    paper: Paper,
    pdf_path: Path,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    token_limit: int = 200,
    openai_timeout: int | None = None,
) -> str:
    full_text = _extract_pdf_text(pdf_path)
    snippet = _first_tokens(full_text, token_limit)
    prompt = (
        "Extract author affiliations from the paper snippet below.\n"
        "Return a single line listing affiliations separated by '; '.\n"
        "If affiliations are not present, return 'Not specified'.\n\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n\n"
        "Snippet:\n"
        f"{snippet}\n"
    )
    return call_llm_text(
        client=client,
        model=model,
        prompt=prompt,
        pricing=pricing,
        cost_tracker=cost_tracker,
        label=f"Affiliations {paper.paper_id}",
        timeout=openai_timeout,
    ).strip()


def extract_affiliations_batch(
    client,
    model: str,
    papers: list[Paper],
    pdf_paths: list[Path],
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    token_limit: int = 200,
    openai_timeout: int | None = None,
    max_workers: int = 4,
) -> dict[str, str]:
    prompts: list[str] = []
    for paper, pdf_path in zip(papers, pdf_paths):
        full_text = _extract_pdf_text(pdf_path)
        snippet = _first_tokens(full_text, token_limit)
        prompt = (
            "Extract author affiliations from the paper snippet below.\n"
            "Return a single line listing affiliations separated by '; '.\n"
            "If affiliations are not present, return 'Not specified'.\n\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n\n"
            "Snippet:\n"
            f"{snippet}\n"
        )
        prompts.append(prompt)

    outputs = batch_call_llm_text(
        client=client,
        model=model,
        prompts=prompts,
        timeout=openai_timeout,
        pricing=pricing,
        cost_tracker=cost_tracker,
        label="Affiliations",
        max_workers=max_workers,
    )
    return {paper.paper_id: output.strip() for paper, output in zip(papers, outputs)}
