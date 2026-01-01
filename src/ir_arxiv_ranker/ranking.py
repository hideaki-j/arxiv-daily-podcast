from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from jinja2 import Environment, StrictUndefined
from utils.call_llm import build_rankings_response_format, call_llm_json
from utils.costs import CostTracker

from .models import Paper


@dataclass(frozen=True)
class Rankings:
    automatic_eval_ranking: List[str]
    user_simulator_ranking: List[str]
    final_ranking: List[str]
    tldr_by_id: Dict[str, str]


def _validate_rankings(rankings: dict, valid_ids: List[str], top_n: int) -> Dict[str, str]:
    required_keys = {
        "automatic_eval_ranking",
        "user_simulator_ranking",
        "final_ranking",
        "tldr_list",
    }
    if set(rankings.keys()) != required_keys:
        raise ValueError(f"Rankings keys must be {sorted(required_keys)}")

    valid_set = set(valid_ids)
    for key in ("automatic_eval_ranking", "user_simulator_ranking", "final_ranking"):
        values = rankings[key]
        if len(values) != top_n:
            raise ValueError(f"{key} must contain exactly {top_n} ids")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} contains duplicate ids")
        invalid = [value for value in values if value not in valid_set]
        if invalid:
            raise ValueError(f"{key} contains invalid ids: {invalid}")

    tldr_list = rankings["tldr_list"]
    if len(tldr_list) != top_n:
        raise ValueError("tldr_list must contain exactly top_n entries")
    tldr_by_id: Dict[str, str] = {}
    for item in tldr_list:
        paper_id = item.get("id")
        tldr = item.get("tldr", "").strip()
        if not paper_id or paper_id not in valid_set:
            raise ValueError(f"tldr_list contains invalid id: {paper_id}")
        if paper_id in tldr_by_id:
            raise ValueError("tldr_list contains duplicate ids")
        if not tldr:
            raise ValueError(f"tldr_list missing tldr for {paper_id}")
        tldr_by_id[paper_id] = tldr

    if set(tldr_by_id.keys()) != set(rankings["final_ranking"]):
        raise ValueError("tldr_list ids must match final_ranking ids")
    return tldr_by_id


def rank_papers(
    client,
    model: str,
    prompt_template: str,
    papers: List[Paper],
    top_n: int,
    author_influence_by_id: Dict[str, int] | None = None,
    abstract_word_cutoff: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    openai_timeout: int | None = None,
) -> Rankings:
    author_scores = author_influence_by_id or {}
    papers_payload = []
    for paper in papers:
        payload = paper.prompt_dict()
        if abstract_word_cutoff:
            words = payload["summary"].split()
            payload["summary"] = " ".join(words[:abstract_word_cutoff])
        payload["author_influence_threshold"] = author_scores.get(paper.paper_id)
        papers_payload.append(payload)
    env = Environment(autoescape=False, undefined=StrictUndefined)
    prompt = env.from_string(prompt_template).render(
        top_n=top_n,
        papers=papers_payload,
    )

    payload = call_llm_json(
        client=client,
        model=model,
        prompt=prompt,
        response_format=build_rankings_response_format(top_n),
        pricing=pricing,
        cost_tracker=cost_tracker,
        label="Ranking LLM",
        timeout=openai_timeout,
    )
    tldr_by_id = _validate_rankings(payload, [paper.paper_id for paper in papers], top_n)

    return Rankings(
        automatic_eval_ranking=payload["automatic_eval_ranking"],
        user_simulator_ranking=payload["user_simulator_ranking"],
        final_ranking=payload["final_ranking"],
        tldr_by_id=tldr_by_id,
    )
