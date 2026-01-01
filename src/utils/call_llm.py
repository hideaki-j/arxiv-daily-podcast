from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import APITimeoutError, APIConnectionError, OpenAI
from tqdm import tqdm

from .costs import CostReport, CostTracker

DEFAULT_MAX_RETRIES = 2


def build_rankings_response_format(top_n: int) -> dict:
    return {
        "type": "json_schema",
        "name": "paper_rankings",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "automatic_eval_ranking": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": top_n,
                    "maxItems": top_n,
                },
                "user_simulator_ranking": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": top_n,
                    "maxItems": top_n,
                },
                "final_ranking": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": top_n,
                    "maxItems": top_n,
                },
                "tldr_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "tldr": {"type": "string"},
                        },
                        "required": ["id", "tldr"],
                        "additionalProperties": False,
                    },
                    "minItems": top_n,
                    "maxItems": top_n,
                },
            },
            "required": [
                "automatic_eval_ranking",
                "user_simulator_ranking",
                "final_ranking",
                "tldr_list",
            ],
            "additionalProperties": False,
        },
    }


def _extract_json_payload(response) -> dict:
    if hasattr(response, "output"):
        for item in response.output:
            contents = getattr(item, "content", None) or []
            for content in contents:
                content_type = getattr(content, "type", "")
                if content_type in {"output_json", "json"}:
                    payload = getattr(content, "json", None)
                    if payload:
                        return payload
                if content_type in {"output_text", "text"}:
                    text = getattr(content, "text", "")
                    if text:
                        return json.loads(text)
    if getattr(response, "output_text", ""):
        return json.loads(response.output_text)
    raise ValueError("No JSON payload found in response.")


def _extract_usage(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    cached_tokens = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached_tokens = getattr(details, "cached_tokens", 0)
    return {
        "input_tokens": usage.input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _estimate_cost(
    usage: dict | None,
    pricing: dict | None,
) -> float | None:
    if not usage:
        return None
    if not pricing:
        return None
    input_rate = pricing.get("input_usd_per_1m_tokens")
    cached_rate = pricing.get("cached_input_usd_per_1m_tokens")
    output_rate = pricing.get("output_usd_per_1m_tokens")

    cached_tokens = usage.get("cached_tokens", 0)
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    uncached_tokens = max(input_tokens - cached_tokens, 0)

    if input_rate is None and uncached_tokens > 0:
        return None
    if cached_rate is None and cached_tokens > 0:
        return None
    if output_rate is None and output_tokens > 0:
        return None

    cost = 0.0
    if uncached_tokens > 0:
        cost += (uncached_tokens / 1_000_000.0) * float(input_rate)
    if cached_tokens > 0:
        cost += (cached_tokens / 1_000_000.0) * float(cached_rate)
    if output_tokens > 0:
        cost += (output_tokens / 1_000_000.0) * float(output_rate)
    return cost


def _usage_detail(usage: dict | None) -> str:
    if not usage:
        return "usage unavailable"
    return (
        f"input {usage['input_tokens']}, cached {usage['cached_tokens']}, "
        f"output {usage['output_tokens']}"
    )


def _log_cost(label: str, usage: dict | None, cost: float | None) -> None:
    if not usage:
        print(f"{label} usage unavailable (cost unknown).")
        return
    detail = _usage_detail(usage)
    if cost is None:
        print(f"{label} usage: {detail}, total {usage['total_tokens']} (cost unknown).")
        return
    cost_cents = cost * 100.0
    print(f"{label} cost: {cost_cents:.2f}¢ ({detail}).")


def call_llm_json(
    client: OpenAI,
    model: str,
    prompt: str,
    response_format: dict,
    timeout: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    label: str = "LLM JSON",
    log_costs: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    attempt = 0
    while True:
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                text={"format": response_format},
                timeout=timeout,
            )
            payload = _extract_json_payload(response)
            usage = _extract_usage(response)
            cost = _estimate_cost(usage, pricing)
            if log_costs:
                _log_cost(label, usage, cost)
            if cost_tracker is not None:
                cost_tracker.add(cost)
            if cost_report is not None:
                cost_report.add(label, cost, _usage_detail(usage))
            return payload
        except (APITimeoutError, APIConnectionError, httpx.ReadTimeout):
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(2**attempt)


def _extract_text_payload(response) -> str:
    if hasattr(response, "output"):
        for item in response.output:
            contents = getattr(item, "content", None) or []
            for content in contents:
                content_type = getattr(content, "type", "")
                if content_type in {"output_text", "text"}:
                    text = getattr(content, "text", "")
                    if text:
                        return text
    if getattr(response, "output_text", ""):
        return response.output_text
    raise ValueError("No text payload found in response.")


def call_llm_text(
    client: OpenAI,
    model: str,
    prompt: str,
    timeout: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    label: str = "LLM Text",
    log_costs: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    attempt = 0
    while True:
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                timeout=timeout,
            )
            text = _extract_text_payload(response)
            usage = _extract_usage(response)
            cost = _estimate_cost(usage, pricing)
            if log_costs:
                _log_cost(label, usage, cost)
            if cost_tracker is not None:
                cost_tracker.add(cost)
            if cost_report is not None:
                cost_report.add(label, cost, _usage_detail(usage))
            return text
        except (APITimeoutError, APIConnectionError, httpx.ReadTimeout):
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(2**attempt)


def batch_call_llm_text(
    client: OpenAI,
    model: str,
    prompts: list[str],
    timeout: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    label: str = "LLM Text",
    max_workers: int = 4,
    show_cost_table: bool = True,
) -> list[str]:
    if not prompts:
        return []
    report = cost_report or CostReport()
    results: list[str | None] = [None] * len(prompts)
    workers = min(max_workers, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                call_llm_text,
                client,
                model,
                prompt,
                timeout=timeout,
                pricing=pricing,
                cost_tracker=cost_tracker,
                cost_report=report,
                label=f"{label} {idx + 1}",
                log_costs=False,
            ): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=label):
            idx = futures[future]
            results[idx] = future.result()
    if show_cost_table:
        print(report.render_psql(f"{label} costs"))
    return [result or "" for result in results]


def batch_call_llm_json(
    client: OpenAI,
    model: str,
    prompts: list[str],
    response_formats: list[dict],
    timeout: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    label: str = "LLM JSON",
    max_workers: int = 4,
) -> list[dict]:
    """Call LLM for multiple JSON prompts in parallel with interactive progress."""
    if not prompts:
        return []
    if len(prompts) != len(response_formats):
        raise ValueError("prompts and response_formats must have the same length")
    results: list[dict | None] = [None] * len(prompts)
    workers = min(max_workers, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                call_llm_json,
                client,
                model,
                prompt,
                response_format,
                timeout=timeout,
                pricing=pricing,
                cost_tracker=cost_tracker,
                label=f"{label} {idx + 1}",
                log_costs=False,
            ): idx
            for idx, (prompt, response_format) in enumerate(zip(prompts, response_formats))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=label):
            idx = futures[future]
            results[idx] = future.result()
    return [result or {} for result in results]
