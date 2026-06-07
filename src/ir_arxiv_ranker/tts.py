from __future__ import annotations

import os
import shutil
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import APITimeoutError, APIConnectionError, OpenAI
from tqdm import tqdm

from utils.costs import CostReport, CostTracker

DEFAULT_MAX_RETRIES = 2


def _estimate_tts_cost(text: str, pricing: dict | None, input_tokens: int | None = None, output_tokens: int | None = None) -> tuple[float | None, str]:
    if not pricing:
        return None, "unknown"
    
    # If explicit tokens are provided, use them for calculation
    if input_tokens is not None and output_tokens is not None:
        input_rate = pricing.get("input_usd_per_1m_text_tokens")
        output_rate = pricing.get("output_usd_per_1m_audio_tokens")
        cost = 0.0
        if input_rate is not None:
            cost += (input_tokens / 1_000_000.0) * float(input_rate)
        if output_rate is not None:
            cost += (output_tokens / 1_000_000.0) * float(output_rate)
        return cost, f"tokens_in={input_tokens}, tokens_out={output_tokens}"

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


def _synthesize_openai(
    client: OpenAI,
    model: str,
    voice: str,
    text: str,
    dest_path: Path,
    timeout: int | None,
    instructions: str | None = None,
) -> None:
    # OpenAI implementation
    # Pass instructions if provided, assuming client supports it.
    kwargs = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "mp3",
        "timeout": timeout,
    }
    if instructions:
         kwargs["instructions"] = instructions

    response = client.audio.speech.create(**kwargs)
    response.write_to_file(dest_path)


def _synthesize_gemini(
    model: str,
    text: str,
    dest_path: Path,
    voice: str | None = None,
    instructions: str | None = None,
) -> tuple[int, int]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    
    # Gemini 2.5 TTS works best with instructions in the prompt
    prompt = text
    if instructions:
        prompt = f"{instructions}\n\ntext to read: {text}"
    else:
        # Default prompt wrapper to ensure it reads the text
        prompt = f"Please read the following text: {text}"

    # Configure speech settings if voice is provided (SDK specific structure).
    speech_config = None
    if voice and voice.lower() != "n/a":
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice
                )
            )
        )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config
        )
    )

    audio_data = None
    sample_rate = 24000
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                audio_data = part.inline_data.data
                if "rate=" in part.inline_data.mime_type:
                    try:
                        sample_rate = int(part.inline_data.mime_type.split("rate=")[-1])
                    except ValueError:
                        pass
                break
    
    if not audio_data:
        raise RuntimeError("No audio data returned from Gemini TTS")
    
    input_tokens = 0
    output_tokens = 0
    if response.usage_metadata:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0

    # Save raw PCM as WAV.
    # If dest_path is mp3, compress later or rename if ffmpeg missing.
    
    # Write to a temporary WAV file
    wav_path = dest_path.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    # Convert to MP3 if requested path is MP3 (standard for this app)
    if dest_path.suffix.lower() == ".mp3":
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), str(dest_path)],
                check=True,
                capture_output=True
            )
            wav_path.unlink() # Remove temp wav
        else:
            print("Warning: ffmpeg not found for Gemini TTS conversion; saving wav as mp3.")
            wav_path.rename(dest_path)
    else:
        if wav_path != dest_path:
            wav_path.rename(dest_path)
            
    return input_tokens, output_tokens


def synthesize_podcast(
    text: str,
    dest_path: Path,
    primary_config: dict,
    openai_client: OpenAI | None = None, # Optional, used if provider is openai
    timeout: int | None = None,
    cost_tracker: CostTracker | None = None,
    cost_report: CostReport | None = None,
    instructions: str | None = None,
    label: str = "TTS",
    log_costs: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Path:
    
    # Assume pricing is for the primary model.
    # Future work: handle dynamic pricing per provider/model.

    # Helper to calculate and log cost
    def track_cost(model_pricing, txt, input_tokens=None, output_tokens=None):
        c, det = _estimate_tts_cost(txt, model_pricing, input_tokens, output_tokens)
        if log_costs:
            _log_tts_cost(label, txt, c, det)
        if cost_tracker is not None:
            cost_tracker.add(c)
        if cost_report is not None:
            cost_report.add(label, c, _tts_detail(txt, det), model=model)

    # Try Primary
    provider = primary_config["provider"]
    model = primary_config["model"]
    voice = primary_config["voice"]
    model_pricing = primary_config.get("pricing")
    
    input_tokens = None
    output_tokens = None

    if provider == "gemini":
        input_tokens, output_tokens = _synthesize_gemini(model, text, dest_path, voice, instructions)
    elif provider == "openai":
        if not openai_client:
             raise ValueError("OpenAI client required for OpenAI TTS")
        # OpenAI API doesn't return token usage for TTS (usually), so we rely on char estimation
        _synthesize_openai(openai_client, model, voice, text, dest_path, timeout, instructions)
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")

    # If success
    track_cost(model_pricing, text, input_tokens, output_tokens)
    return dest_path


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
    client: OpenAI, # Keep as is, mainly for OpenAI fallback/primary
    primary_config: dict, # {provider, model, voice, pricing}
    items: list[tuple[str, Path]],
    timeout: int | None = None,
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
    report_start = report.snapshot()
    results: list[Path | None] = [None] * len(items)
    workers = min(max_workers, len(items))
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                synthesize_podcast,
                text=text,
                dest_path=dest_path,
                primary_config=primary_config,
                openai_client=client,
                timeout=timeout,
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
            try:
                future.result()
                if compress_to_64kbps:
                    compress_mp3_to_64kbps(dest_path)
                results[idx] = dest_path
            except Exception as e:
                print(f"Failed to synthesize item {idx}: {e}")

    if show_cost_table:
        print(report.render_psql(f"{label} costs", entries=report.entries_since(report_start)))
    return [path for path in results if path is not None]
