from __future__ import annotations

import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from openai import APITimeoutError, APIConnectionError, OpenAI
from tqdm import tqdm

from utils.costs import CostReport, CostTracker

DEFAULT_MAX_RETRIES = 2


def _estimate_tts_cost(text: str, pricing: dict | None) -> tuple[float | None, str]:
    if not pricing:
        return None, "unknown"
    per_minute = pricing.get("estimated_usd_per_minute")
    if per_minute is not None:
        minutes = max(len(text.split()) / 129.0, 0.1)
        return minutes * float(per_minute), f"est_minutes={minutes:.2f}"

    input_rate = pricing.get("input_usd_per_1m_text_tokens")
    output_rate = pricing.get("output_usd_per_1m_audio_tokens")
    if input_rate is None and output_rate is None:
        return None, "unknown"

    text_tokens_est = max(int(len(text) / 4), 1)
    cost = 0.0
    if input_rate is not None:
        cost += (text_tokens_est / 1_000_000.0) * float(input_rate)
    if output_rate is not None:
        cost += (text_tokens_est / 1_000_000.0) * float(output_rate)
    return cost, f"est_tokens={text_tokens_est}"


def _log_tts_cost(label: str, text: str, cost: float | None, detail: str) -> None:
    if cost is None:
        print(f"{label} usage: {len(text)} chars ({detail}, cost unknown).")
        return
    cost_cents = cost * 100.0
    print(f"{label} cost: {cost_cents:.2f}¢ ({len(text)} chars, {detail}).")


def _tts_detail(text: str, detail: str) -> str:
    return f"{len(text)} chars, {detail}"


def synthesize_podcast(
    client: OpenAI,
    model: str,
    voice: str,
    text: str,
    dest_path: Path,
    timeout: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    instructions: str | None = None,
    label: str = "TTS",
    log_costs: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Path:
    attempt = 0
    while True:
        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                instructions=instructions,
                response_format="mp3",
                timeout=timeout,
            )
            response.write_to_file(dest_path)
            cost, detail = _estimate_tts_cost(text, pricing)
            if log_costs:
                _log_tts_cost(label, text, cost, detail)
            if cost_tracker is not None:
                cost_tracker.add(cost)
            if cost_report is not None:
                cost_report.add(label, cost, _tts_detail(text, detail))
            return dest_path
        except (APITimeoutError, APIConnectionError, httpx.ReadTimeout):
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(2**attempt)


def compress_mp3_to_64kbps(path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; skipping mp3 compression.")
        return
    temp_path = path.with_suffix(".tmp.mp3")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-b:a",
            "64k",
            str(temp_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ffmpeg compression failed; keeping original mp3.")
        return
    temp_path.replace(path)


def batch_synthesize_podcast(
    client: OpenAI,
    model: str,
    voice: str,
    items: list[tuple[str, Path]],
    timeout: int | None = None,
    pricing: dict | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    instructions: str | None = None,
    label: str = "TTS",
    max_workers: int = 4,
    compress_to_64kbps: bool = True,
    show_cost_table: bool = True,
) -> list[Path]:
    if not items:
        return []
    report = cost_report or CostReport()
    results: list[Path | None] = [None] * len(items)
    workers = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                synthesize_podcast,
                client=client,
                model=model,
                voice=voice,
                text=text,
                dest_path=dest_path,
                timeout=timeout,
                pricing=pricing,
                cost_tracker=cost_tracker,
                cost_report=report,
                instructions=instructions,
                label=f"{label} {idx + 1}",
                log_costs=False,
            ): (idx, dest_path)
            for idx, (text, dest_path) in enumerate(items)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=label):
            idx, dest_path = futures[future]
            future.result()
            if compress_to_64kbps:
                compress_mp3_to_64kbps(dest_path)
            results[idx] = dest_path
    if show_cost_table:
        print(report.render_psql(f"{label} costs"))
    return [path for path in results if path is not None]
