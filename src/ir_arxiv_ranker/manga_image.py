from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from utils.costs import CostTracker
from utils.naming import build_file_stem

from .models import Paper
from .podcast import _extract_pdf_text


def load_manga_prompt(path: Path) -> str:
    return path.read_text()


def _render_prompt(prompt_template: str, paper: Paper, paper_text: str, model: str) -> str:
    env = Environment(autoescape=False, undefined=StrictUndefined)
    return env.from_string(prompt_template).render(
        paper=paper,
        paper_text=paper_text,
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


def generate_manga_image(
    client,
    model: str,
    prompt_template: str,
    paper: Paper,
    pdf_path: Path,
    image_dir: Path,
    rank: int,
    size: str,
    quality: str,
    output_format: str,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    timeout: int | None = None,
) -> Path:
    paper_text = _extract_pdf_text(pdf_path)
    prompt = _render_prompt(prompt_template, paper, paper_text, model)
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
    if cost_tracker is not None:
        cost_tracker.add(cost)
    if cost is None:
        print(f"Manga image cost: ? ({_usage_detail(getattr(result, 'usage', None))}).")
    else:
        print(
            f"Manga image cost: {cost * 100:.2f}¢ "
            f"({_usage_detail(getattr(result, 'usage', None))})."
        )
    return image_path
