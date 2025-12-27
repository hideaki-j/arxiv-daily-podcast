from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    ranking_model: str
    podcast_model: str
    tts_model: str | None
    tts_voice: str | None
    tts_instructions: str | None
    compress_to_64kbps: bool
    pricing_data: dict
    ir_limit: int
    nlp_limit: int
    others_limit: int
    keywords: list[str]
    top_n: int
    top_n_tts: int
    abst_word_cutoff: int
    transcript_word_cutoff: int | None
    generate_transcript: bool
    filter_since_last_schedule: bool
    use_tts: bool
    email_enabled: bool
    arxiv_timeout: int
    openai_timeout: int


def load_config(config_path: Path) -> Settings:
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    raw_config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw_config, dict):
        raise SystemExit("Config file must contain a YAML object at the top level.")

    ranking_model = raw_config.get("ranking_model")
    podcast_model = raw_config.get("podcast_model")
    ir_limit = raw_config.get("ir_limit")
    nlp_limit = raw_config.get("nlp_limit")
    others_limit = raw_config.get("others_limit")
    keywords_path = raw_config.get("keywords_path")
    top_n = raw_config.get("top_n")
    top_n_tts = raw_config.get("top_n_tts")
    abst_word_cutoff = raw_config.get("abst_word_cutoff")
    transcript_word_cutoff = raw_config.get("transcript_word_cutoff")
    generate_transcript = raw_config.get("generate_transcript", True)
    filter_since_last_schedule = raw_config.get("filter_since_last_schedule", False)
    use_tts = raw_config.get("use_tts", True)
    tts_model = raw_config.get("tts_model")
    tts_voice = raw_config.get("tts_voice")
    tts_instructions_path = raw_config.get("tts_instructions_path")
    compress_to_64kbps = raw_config.get("compress_to_64kbps", True)
    email_enabled = raw_config.get("email_enabled", False)
    pricing_path = raw_config.get("pricing_path")
    arxiv_timeout = raw_config.get("arxiv_timeout")
    openai_timeout = raw_config.get("openai_timeout")

    if not ranking_model:
        raise SystemExit("Config must include ranking_model")
    if not podcast_model:
        raise SystemExit("Config must include podcast_model")
    if not isinstance(use_tts, bool):
        raise SystemExit("use_tts must be a boolean")
    if not isinstance(filter_since_last_schedule, bool):
        raise SystemExit("filter_since_last_schedule must be a boolean")
    if not isinstance(email_enabled, bool):
        raise SystemExit("email_enabled must be a boolean")
    if not isinstance(ir_limit, int) or ir_limit < 1:
        raise SystemExit("ir_limit must be an integer >= 1")
    if not isinstance(nlp_limit, int) or nlp_limit < 1:
        raise SystemExit("nlp_limit must be an integer >= 1")
    if not isinstance(others_limit, int) or others_limit < 1:
        raise SystemExit("others_limit must be an integer >= 1")
    if not isinstance(top_n, int) or top_n < 1:
        raise SystemExit("top_n must be an integer >= 1")
    if not isinstance(top_n_tts, int) or top_n_tts < 0:
        raise SystemExit("top_n_tts must be an integer >= 0")
    if top_n_tts > top_n:
        raise SystemExit("top_n_tts must be <= top_n")
    if not isinstance(abst_word_cutoff, int) or abst_word_cutoff < 1:
        raise SystemExit("abst_word_cutoff must be an integer >= 1")
    if transcript_word_cutoff is not None:
        if not isinstance(transcript_word_cutoff, int) or transcript_word_cutoff < 1:
            raise SystemExit("transcript_word_cutoff must be an integer >= 1")
    if not isinstance(generate_transcript, bool):
        raise SystemExit("generate_transcript must be a boolean")
    if not isinstance(arxiv_timeout, int) or arxiv_timeout < 1:
        raise SystemExit("arxiv_timeout must be an integer >= 1")
    if not isinstance(openai_timeout, int) or openai_timeout < 1:
        raise SystemExit("openai_timeout must be an integer >= 1")
    if not pricing_path:
        raise SystemExit("pricing_path must be set in config")
    if not keywords_path:
        raise SystemExit("keywords_path must be set in config")

    if not generate_transcript and use_tts:
        print("use_tts ignored because generate_transcript is false.")
        use_tts = False

    if use_tts:
        if not tts_model:
            raise SystemExit("Config must include tts_model")
        if not tts_voice:
            raise SystemExit("Config must include tts_voice")
        if not isinstance(compress_to_64kbps, bool):
            raise SystemExit("compress_to_64kbps must be a boolean")
        if tts_instructions_path:
            tts_file = Path(tts_instructions_path)
            if not tts_file.exists():
                raise SystemExit(f"TTS instructions file not found: {tts_file}")
            tts_instructions = tts_file.read_text().strip()
            if not tts_instructions:
                raise SystemExit("tts_instructions_path must point to non-empty text")
        else:
            tts_instructions = raw_config.get(
                "tts_instructions",
                "Energetic, upbeat podcast host tone. Friendly and engaging, clear enunciation.",
            )
            if not isinstance(tts_instructions, str) or not tts_instructions.strip():
                raise SystemExit("tts_instructions must be a non-empty string")
    else:
        tts_model = None
        tts_voice = None
        tts_instructions = None
        if not isinstance(compress_to_64kbps, bool):
            raise SystemExit("compress_to_64kbps must be a boolean")

    pricing_file = Path(pricing_path)
    if not pricing_file.exists():
        raise SystemExit(f"Pricing file not found: {pricing_file}")
    pricing_data = json.loads(pricing_file.read_text() or "{}")
    if not isinstance(pricing_data, dict):
        raise SystemExit("Pricing file must contain a JSON object at the top level.")
    for model_name, pricing in pricing_data.items():
        if pricing is None:
            continue
        if not isinstance(pricing, dict):
            raise SystemExit(f"pricing.{model_name} must be a mapping")
        for key, value in pricing.items():
            if value is None:
                continue
            if not isinstance(value, (int, float)) or value < 0:
                raise SystemExit(f"pricing.{model_name}.{key} must be >= 0")

    keywords_file = Path(keywords_path)
    if not keywords_file.exists():
        raise SystemExit(f"Keywords file not found: {keywords_file}")
    keywords_data = yaml.safe_load(keywords_file.read_text()) or []
    if isinstance(keywords_data, dict):
        keywords = keywords_data.get("keywords", [])
    else:
        keywords = keywords_data
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise SystemExit("Keywords file must contain a list of strings or a 'keywords' list.")
    keywords = [k.strip() for k in keywords if k.strip()]
    if not keywords:
        raise SystemExit("Keywords list is empty.")

    return Settings(
        ranking_model=ranking_model,
        podcast_model=podcast_model,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_instructions=tts_instructions,
        compress_to_64kbps=compress_to_64kbps,
        pricing_data=pricing_data,
        ir_limit=ir_limit,
        nlp_limit=nlp_limit,
        others_limit=others_limit,
        keywords=keywords,
        top_n=top_n,
        top_n_tts=top_n_tts,
        abst_word_cutoff=abst_word_cutoff,
        transcript_word_cutoff=transcript_word_cutoff,
        generate_transcript=generate_transcript,
        filter_since_last_schedule=filter_since_last_schedule,
        use_tts=use_tts,
        email_enabled=email_enabled,
        arxiv_timeout=arxiv_timeout,
        openai_timeout=openai_timeout,
    )
