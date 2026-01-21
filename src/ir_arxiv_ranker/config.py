from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LLMCallConfig:
    provider: str
    model: str


@dataclass(frozen=True)
class TTSConfig:
    provider: str
    model: str
    voice: str


@dataclass(frozen=True)
class Settings:
    ranking: LLMCallConfig
    podcast: LLMCallConfig
    influence_filter: LLMCallConfig
    affiliation: LLMCallConfig
    tts: TTSConfig | None
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
    influence_score_threshold: int
    influence_max_workers: int | None


def _parse_llm_call_config(raw: dict | None, name: str) -> LLMCallConfig:
    if not raw or not isinstance(raw, dict):
        raise SystemExit(f"Config must include '{name}' section with provider and model")
    provider = raw.get("provider")
    model = raw.get("model")
    if not provider or not isinstance(provider, str):
        raise SystemExit(f"{name}.provider must be a non-empty string")
    if not model or not isinstance(model, str):
        raise SystemExit(f"{name}.model must be a non-empty string")
    return LLMCallConfig(provider=provider, model=model)


def _parse_tts_config(raw: dict | None) -> TTSConfig:
    if not raw or not isinstance(raw, dict):
        raise SystemExit("Config must include 'tts' section with provider, model, and voice")
    provider = raw.get("provider")
    model = raw.get("model")
    voice = raw.get("voice")
    if not provider or not isinstance(provider, str):
        raise SystemExit("tts.provider must be a non-empty string")
    if not model or not isinstance(model, str):
        raise SystemExit("tts.model must be a non-empty string")
    if not voice or not isinstance(voice, str):
        raise SystemExit("tts.voice must be a non-empty string")
    return TTSConfig(provider=provider, model=model, voice=voice)


def load_config(config_path: Path) -> Settings:
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    raw_config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw_config, dict):
        raise SystemExit("Config file must contain a YAML object at the top level.")

    # Parse LLM call configs
    ranking = _parse_llm_call_config(raw_config.get("ranking"), "ranking")
    podcast = _parse_llm_call_config(raw_config.get("podcast"), "podcast")
    influence_filter = _parse_llm_call_config(raw_config.get("influence_filter"), "influence_filter")
    affiliation = _parse_llm_call_config(raw_config.get("affiliation"), "affiliation")

    # Parse other settings
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
    compress_to_64kbps = raw_config.get("compress_to_64kbps", True)
    email_enabled = raw_config.get("email_enabled", False)
    pricing_path = raw_config.get("pricing_path")
    arxiv_timeout = raw_config.get("arxiv_timeout")
    openai_timeout = raw_config.get("openai_timeout")
    influence_score_threshold = raw_config.get("influence_score_threshold", 3)
    influence_max_workers = raw_config.get("influence_max_workers")

    # Validate boolean settings
    if not isinstance(use_tts, bool):
        raise SystemExit("use_tts must be a boolean")
    if not isinstance(filter_since_last_schedule, bool):
        raise SystemExit("filter_since_last_schedule must be a boolean")
    if not isinstance(email_enabled, bool):
        raise SystemExit("email_enabled must be a boolean")
    if not isinstance(generate_transcript, bool):
        raise SystemExit("generate_transcript must be a boolean")
    if not isinstance(compress_to_64kbps, bool):
        raise SystemExit("compress_to_64kbps must be a boolean")

    # Validate integer settings
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
    if not isinstance(arxiv_timeout, int) or arxiv_timeout < 1:
        raise SystemExit("arxiv_timeout must be an integer >= 1")
    if not isinstance(openai_timeout, int) or openai_timeout < 1:
        raise SystemExit("openai_timeout must be an integer >= 1")
    if not isinstance(influence_score_threshold, int):
        raise SystemExit("influence_score_threshold must be an integer")
    if influence_score_threshold < 0 or influence_score_threshold > 4:
        raise SystemExit("influence_score_threshold must be between 0 and 4")
    if influence_max_workers is not None:
        if not isinstance(influence_max_workers, int) or influence_max_workers < 1:
            raise SystemExit("influence_max_workers must be an integer >= 1")

    # Validate required paths
    if not pricing_path:
        raise SystemExit("pricing_path must be set in config")
    if not keywords_path:
        raise SystemExit("keywords_path must be set in config")

    # Handle TTS settings
    if not generate_transcript and use_tts:
        print("use_tts ignored because generate_transcript is false.")
        use_tts = False

    tts: TTSConfig | None = None
    if use_tts:
        tts = _parse_tts_config(raw_config.get("tts"))

    # Load pricing data
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
            if key == "provider":
                continue
            if value is None:
                continue
            if not isinstance(value, (int, float)) or value < 0:
                raise SystemExit(f"pricing.{model_name}.{key} must be >= 0")

    # Load keywords
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
        ranking=ranking,
        podcast=podcast,
        influence_filter=influence_filter,
        affiliation=affiliation,
        tts=tts,
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
        influence_score_threshold=influence_score_threshold,
        influence_max_workers=influence_max_workers,
    )
