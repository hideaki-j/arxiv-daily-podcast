from __future__ import annotations

from pathlib import Path
from typing import Dict

from jinja2 import Environment, StrictUndefined

from utils.call_llm import batch_call_llm_json
from utils.costs import CostReport, CostTracker

from .models import Paper
from .podcast import _extract_pdf_text, _truncate_words


SUMMARY_FIELDS = (
    "tldr",
    "background",
    "existing_problem",
    "proposed_method",
    "results",
)


def load_selected_summary_prompt(path: Path) -> str:
    return path.read_text()


def _build_selected_summary_response_format() -> dict:
    properties = {"id": {"type": "string"}}
    properties.update({field: {"type": "string"} for field in SUMMARY_FIELDS})
    return {
        "type": "json_schema",
        "name": "selected_paper_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": ["id", *SUMMARY_FIELDS],
            "additionalProperties": False,
        },
    }


def _validate_summary_payload(payload: dict, paper_id: str) -> dict[str, str]:
    if payload.get("id") != paper_id:
        raise ValueError(f"Summary payload id must be {paper_id}, got {payload.get('id')}")
    summary: dict[str, str] = {}
    for field in SUMMARY_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Summary payload for {paper_id} missing {field}")
        summary[field] = value.strip()
    return summary


def generate_selected_summaries_batch(
    client,
    model: str,
    prompt_template: str,
    papers: list[Paper],
    pdf_paths: list[Path],
    paper_text_word_cutoff: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    label: str = "Selected summary LLM",
    openai_timeout: int | None = None,
    max_workers: int = 4,
    provider: str = "openai",
) -> Dict[str, dict[str, str]]:
    if not papers:
        return {}
    if len(papers) != len(pdf_paths):
        raise ValueError("papers and pdf_paths must have the same length")

    env = Environment(autoescape=False, undefined=StrictUndefined)
    response_format = _build_selected_summary_response_format()
    prompts: list[str] = []
    response_formats: list[dict] = []
    for paper, pdf_path in zip(papers, pdf_paths):
        payload = paper.prompt_dict()
        paper_text = _truncate_words(_extract_pdf_text(pdf_path), paper_text_word_cutoff)
        prompts.append(
            env.from_string(prompt_template).render(
                paper=payload,
                paper_text=paper_text,
            )
        )
        response_formats.append(response_format)

    payloads = batch_call_llm_json(
        client=client,
        model=model,
        prompts=prompts,
        response_formats=response_formats,
        timeout=openai_timeout,
        pricing=pricing,
        cost_tracker=cost_tracker,
        cost_report=cost_report,
        label=label,
        max_workers=max_workers,
        provider=provider,
    )

    return {
        paper.paper_id: _validate_summary_payload(payload, paper.paper_id)
        for paper, payload in zip(papers, payloads)
    }
