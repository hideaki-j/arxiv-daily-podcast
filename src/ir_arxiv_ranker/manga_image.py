from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from utils.costs import CostReport, CostTracker
from utils.call_llm import call_llm_text
from utils.naming import build_file_stem

from .models import Paper
from .podcast import _extract_pdf_text


def load_manga_prompt(path: Path) -> str:
    return path.read_text()


def _truncate_chars(text: str, char_cutoff: int | None) -> str:
    if char_cutoff is None:
        return text
    return text[:char_cutoff]


def _render_prompt(
    prompt_template: str,
    paper: Paper,
    model: str,
    paper_text: str = "",
    manga_instruction: str = "",
    manga_style: str = "",
    manga_characters_list: str = "",
) -> str:
    env = Environment(autoescape=False, undefined=StrictUndefined)
    return env.from_string(prompt_template).render(
        paper=paper,
        paper_text=paper_text,
        manga_instruction=manga_instruction,
        manga_style=manga_style,
        manga_characters_list=manga_characters_list,
        model=model,
    )


def _usage_value(obj, name: str, default: int = 0) -> int:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return int(obj.get(name, default) or default)
    return int(getattr(obj, name, default) or default)


def _estimate_image_cost(usage, pricing: dict | None) -> float | None:
    if not usage or not pricing:
        return None

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    text_input_tokens = _usage_value(input_details, "text_tokens")
    image_input_tokens = _usage_value(input_details, "image_tokens")
    text_output_tokens = _usage_value(output_details, "text_tokens")
    image_output_tokens = _usage_value(output_details, "image_tokens")
    if not output_details:
        image_output_tokens = _usage_value(usage, "output_tokens")

    rates = {
        "text_input": pricing.get("text_input_usd_per_1m_tokens"),
        "image_input": pricing.get("image_input_usd_per_1m_tokens"),
        "text_output": pricing.get("text_output_usd_per_1m_tokens"),
        "image_output": pricing.get("image_output_usd_per_1m_tokens"),
    }
    if text_input_tokens and rates["text_input"] is None:
        return None
    if image_input_tokens and rates["image_input"] is None:
        return None
    if text_output_tokens and rates["text_output"] is None:
        return None
    if image_output_tokens and rates["image_output"] is None:
        return None

    cost = 0.0
    cost += (text_input_tokens / 1_000_000.0) * float(rates["text_input"] or 0)
    cost += (image_input_tokens / 1_000_000.0) * float(rates["image_input"] or 0)
    cost += (text_output_tokens / 1_000_000.0) * float(rates["text_output"] or 0)
    cost += (image_output_tokens / 1_000_000.0) * float(rates["image_output"] or 0)
    return cost


def _usage_detail(usage) -> str:
    if not usage:
        return "usage unavailable"
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return (
        f"text input {_usage_value(input_details, 'text_tokens')}, "
        f"image input {_usage_value(input_details, 'image_tokens')}, "
        f"text output {_usage_value(output_details, 'text_tokens')}, "
        f"image output {_usage_value(output_details, 'image_tokens', _usage_value(usage, 'output_tokens'))}"
    )


def generate_manga_instruction(
    client,
    model: str,
    prompt_template: str,
    paper: Paper,
    pdf_path: Path,
    char_cutoff: int | None = 30000,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    timeout: int | None = None,
    provider: str = "openai",
    manga_style: str = "",
    manga_characters_list: str = "",
) -> str:
    paper_text = _truncate_chars(_extract_pdf_text(pdf_path), char_cutoff)
    prompt = _render_prompt(
        prompt_template,
        paper,
        model=model,
        paper_text=paper_text,
        manga_style=manga_style,
        manga_characters_list=manga_characters_list,
    )
    instruction = call_llm_text(
        client=client,
        model=model,
        prompt=prompt,
        timeout=timeout,
        pricing=pricing,
        cost_tracker=cost_tracker,
        cost_report=cost_report,
        label="Manga planner LLM",
        provider=provider,
    ).strip()
    if not instruction:
        raise RuntimeError("Manga planner returned an empty instruction")
    return instruction


def generate_manga_image(
    client,
    model: str,
    prompt_template: str,
    manga_instruction: str,
    paper: Paper,
    image_dir: Path,
    rank: int,
    size: str,
    quality: str,
    output_format: str,
    char_cutoff: int | None = 30000,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    timeout: int | None = None,
    manga_style: str = "",
    manga_characters_list: str = "",
) -> Path:
    prompt = _truncate_chars(
        _render_prompt(
            prompt_template,
            paper,
            model=model,
            manga_instruction=manga_instruction,
            manga_style=manga_style,
            manga_characters_list=manga_characters_list,
        ),
        char_cutoff,
    )
    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=output_format,
        n=1,
        timeout=timeout,
    )
    if not result.data or not result.data[0].b64_json:
        raise RuntimeError("Image generation returned no image data")

    image_dir.mkdir(parents=True, exist_ok=True)
    stem = build_file_stem(rank, paper.paper_id, paper.title)
    image_path = image_dir / f"{stem}-manga.{output_format}"
    image_path.write_bytes(base64.b64decode(result.data[0].b64_json))

    cost = _estimate_image_cost(getattr(result, "usage", None), pricing)
    usage_detail = _usage_detail(getattr(result, "usage", None))
    if cost_tracker is not None:
        cost_tracker.add(cost)
    if cost_report is not None:
        cost_report.add("Manga image", cost, usage_detail, model=model)
    if cost is None:
        print(f"Manga image cost: ? ({usage_detail}).")
    else:
        print(
            f"Manga image cost: {cost * 100:.2f}¢ "
            f"({usage_detail})."
        )
    return image_path
