from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml
from jinja2 import Environment, StrictUndefined

from utils.call_llm import batch_call_llm_json
from utils.costs import CostTracker

from .models import Paper


DEFAULT_RANKING_MAX_WORKERS = 150


@dataclass(frozen=True)
class RankingAspect:
    key: str
    label: str
    weight: float
    polarity: str


@dataclass(frozen=True)
class Rankings:
    automatic_eval_ranking: List[str]
    user_simulator_ranking: List[str]
    final_ranking: List[str]
    tldr_by_id: Dict[str, str]
    scores_by_id: Dict[str, Dict[str, int]]
    total_score_by_id: Dict[str, float]


def load_ranking_aspects(path: Path) -> list[RankingAspect]:
    if not path.exists():
        raise SystemExit(f"Ranking aspects file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit("Ranking aspects file must contain a YAML object.")

    aspects: list[RankingAspect] = []
    for polarity in ("negative", "positive"):
        group = raw.get(polarity, {}) or {}
        if not isinstance(group, dict):
            raise SystemExit(f"ranking_aspects.{polarity} must be a mapping")
        for key, config in group.items():
            if not isinstance(key, str) or not key:
                raise SystemExit(f"ranking_aspects.{polarity} contains an invalid key")
            if isinstance(config, str):
                label = config
                weight = 1.0
            elif isinstance(config, dict):
                label = config.get("label")
                weight = config.get("weight", 1)
            else:
                raise SystemExit(f"ranking_aspects.{polarity}.{key} must be a mapping or string")
            if not isinstance(label, str) or not label:
                raise SystemExit(f"ranking_aspects.{polarity}.{key}.label must be a non-empty string")
            if not isinstance(weight, (int, float)) or weight < 0:
                raise SystemExit(f"ranking_aspects.{polarity}.{key}.weight must be >= 0")
            aspects.append(
                RankingAspect(key=key, label=label, weight=float(weight), polarity=polarity)
            )
    if not aspects:
        raise SystemExit("Ranking aspects file must define at least one aspect")
    return aspects


def _build_score_response_format(aspects: list[RankingAspect]) -> dict:
    score_properties = {
        aspect.key: {
            "type": "integer",
            "minimum": 0,
            "maximum": 2,
        }
        for aspect in aspects
    }
    return {
        "type": "json_schema",
        "name": "paper_aspect_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "scores": {
                    "type": "object",
                    "properties": score_properties,
                    "required": [aspect.key for aspect in aspects],
                    "additionalProperties": False,
                },
                "tldr": {"type": "string"},
            },
            "required": ["id", "scores", "tldr"],
            "additionalProperties": False,
        },
    }


def _validate_score_payload(payload: dict, paper_id: str, aspects: list[RankingAspect]) -> tuple[dict[str, int], str]:
    if payload.get("id") != paper_id:
        raise ValueError(f"Score payload id must be {paper_id}, got {payload.get('id')}")
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError(f"Score payload for {paper_id} missing scores object")

    scores: dict[str, int] = {}
    for aspect in aspects:
        value = raw_scores.get(aspect.key)
        if not isinstance(value, int) or value < 0 or value > 2:
            raise ValueError(f"Score for {paper_id}.{aspect.key} must be integer 0, 1, or 2")
        scores[aspect.key] = value

    tldr = str(payload.get("tldr", "")).strip()
    if not tldr:
        raise ValueError(f"Score payload for {paper_id} missing tldr")
    return scores, tldr


def aggregate_score(
    scores: dict[str, int],
    aspects: list[RankingAspect],
    author_influence_score: int | None = None,
) -> float:
    total = 0.0
    for aspect in aspects:
        value = scores.get(aspect.key, 0)
        if aspect.polarity == "negative":
            total -= aspect.weight * value
        else:
            total += aspect.weight * value
    if author_influence_score is not None:
        total += author_influence_score
    return total


def rank_from_scores(
    papers: List[Paper],
    top_n: int,
    scores_by_id: dict[str, dict[str, int]],
    total_score_by_id: dict[str, float],
    tldr_by_id: dict[str, str],
) -> Rankings:
    if not papers:
        return Rankings([], [], [], {}, {}, {})
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    papers_by_id = {paper.paper_id: paper for paper in papers}
    final_ranking = sorted(
        [paper.paper_id for paper in papers],
        key=lambda paper_id: (
            total_score_by_id.get(paper_id, 0.0),
            papers_by_id[paper_id].updated,
            paper_id,
        ),
        reverse=True,
    )[:top_n]

    return Rankings(
        automatic_eval_ranking=final_ranking,
        user_simulator_ranking=final_ranking,
        final_ranking=final_ranking,
        tldr_by_id=tldr_by_id,
        scores_by_id=scores_by_id,
        total_score_by_id=total_score_by_id,
    )


def rank_papers(
    client,
    model: str,
    scoring_prompt_template: str,
    papers: List[Paper],
    top_n: int,
    author_influence_by_id: Dict[str, int] | None = None,
    abstract_word_cutoff: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    openai_timeout: int | None = None,
    provider: str = "openai",
    include_keyword_papers: bool = True,
    aspects: list[RankingAspect] | None = None,
    max_workers: int = DEFAULT_RANKING_MAX_WORKERS,
) -> Rankings:
    del include_keyword_papers
    if not papers:
        return Rankings([], [], [], {}, {}, {})
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if not aspects:
        raise ValueError("aspects must be provided")

    env = Environment(autoescape=False, undefined=StrictUndefined)
    response_format = _build_score_response_format(aspects)
    prompts: list[str] = []
    response_formats: list[dict] = []
    aspects_payload = [
        {"key": aspect.key, "label": aspect.label, "polarity": aspect.polarity}
        for aspect in aspects
    ]

    for paper in papers:
        payload = paper.prompt_dict()
        if abstract_word_cutoff:
            words = payload["summary"].split()
            payload["summary"] = " ".join(words[:abstract_word_cutoff])
        prompts.append(
            env.from_string(scoring_prompt_template).render(
                paper=payload,
                aspects=aspects_payload,
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
        label="Ranking score LLM",
        max_workers=max_workers,
        provider=provider,
    )

    scores_by_id: dict[str, dict[str, int]] = {}
    tldr_by_id: dict[str, str] = {}
    total_score_by_id: dict[str, float] = {}
    for paper, payload in zip(papers, payloads):
        scores, tldr = _validate_score_payload(payload, paper.paper_id, aspects)
        author_influence_score = None
        if author_influence_by_id:
            raw_score = author_influence_by_id.get(paper.paper_id)
            if isinstance(raw_score, int):
                author_influence_score = raw_score
                scores["author_influence_score"] = raw_score
        scores_by_id[paper.paper_id] = scores
        tldr_by_id[paper.paper_id] = tldr
        total_score_by_id[paper.paper_id] = aggregate_score(
            scores,
            aspects,
            author_influence_score=author_influence_score,
        )

    return rank_from_scores(
        papers=papers,
        top_n=top_n,
        scores_by_id=scores_by_id,
        total_score_by_id=total_score_by_id,
        tldr_by_id=tldr_by_id,
    )
