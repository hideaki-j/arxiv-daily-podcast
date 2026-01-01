from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from jinja2 import Environment, StrictUndefined

from utils.call_llm import batch_call_llm_json
from utils.costs import CostTracker

from .models import Paper


INFLUENCE_SCORE_MIN = 0
INFLUENCE_SCORE_MAX = 4
DEFAULT_INFLUENCE_MAX_WORKERS = 150


@dataclass(frozen=True)
class InfluenceResult:
    scores_by_id: Dict[str, int]
    kept_papers: List[Paper]


def _build_response_format() -> dict:
    return {
        "type": "json_schema",
        "name": "author_influence_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "author_influence_score": {
                    "type": "integer",
                },
                "rationale": {"type": "string"},
            },
            "required": ["author_influence_score", "rationale"],
            "additionalProperties": False,
        },
    }


def _validate_response(payload: dict) -> int | None:
    """Validate and extract score from a single paper response."""
    score = payload.get("author_influence_score")
    if not isinstance(score, int):
        return None
    if score < INFLUENCE_SCORE_MIN or score > INFLUENCE_SCORE_MAX:
        return None
    return score


def filter_by_author_influence(
    client,
    model: str,
    prompt_template: str,
    papers: List[Paper],
    threshold: int,
    max_workers: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    openai_timeout: int | None = None,
) -> InfluenceResult:
    if threshold < INFLUENCE_SCORE_MIN or threshold > INFLUENCE_SCORE_MAX:
        raise ValueError(
            f"threshold must be between {INFLUENCE_SCORE_MIN} and {INFLUENCE_SCORE_MAX}"
        )

    if not papers:
        return InfluenceResult(scores_by_id={}, kept_papers=[])

    effective_max_workers = max_workers or DEFAULT_INFLUENCE_MAX_WORKERS

    env = Environment(autoescape=False, undefined=StrictUndefined)

    # Build one prompt per paper
    prompts: list[str] = []
    response_formats: list[dict] = []
    response_format = _build_response_format()
    for paper in papers:
        prompt = env.from_string(prompt_template).render(
            paper={
                "id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
            }
        )
        prompts.append(prompt)
        response_formats.append(response_format)

    # Process all papers in parallel with interactive progress
    payloads = batch_call_llm_json(
        client=client,
        model=model,
        prompts=prompts,
        response_formats=response_formats,
        timeout=openai_timeout,
        pricing=pricing,
        cost_tracker=cost_tracker,
        label="Author influence filter",
        max_workers=effective_max_workers,
    )

    # Extract scores from responses
    scores_by_id: Dict[str, int] = {}
    for paper, payload in zip(papers, payloads):
        score = _validate_response(payload)
        if score is not None:
            scores_by_id[paper.paper_id] = score

    # Report total cost
    if cost_tracker is not None:
        total_cost_cents = cost_tracker.total_cents()
        print(f"Author influence filter total cost: {total_cost_cents:.2f}¢")

    kept_papers: List[Paper] = []
    missing_ids: list[str] = []
    dropped_ids: list[str] = []
    for paper in papers:
        score = scores_by_id.get(paper.paper_id)
        if score is None:
            missing_ids.append(paper.paper_id)
            continue
        if score >= threshold:
            kept_papers.append(paper)
        else:
            dropped_ids.append(f"{paper.paper_id} ({score})")

    total = len(papers)
    kept = len(kept_papers)
    missing = len(missing_ids)
    dropped = len(dropped_ids)
    print(
        f"Author influence filter kept {kept}/{total} papers "
        f"(threshold >= {threshold}); dropped {dropped} below threshold "
        f"and {missing} with no usable score."
    )
    if dropped_ids:
        print("Below-threshold papers:", ", ".join(dropped_ids))
    if missing_ids:
        print("No score returned for:", ", ".join(missing_ids))

    return InfluenceResult(scores_by_id=scores_by_id, kept_papers=kept_papers)
