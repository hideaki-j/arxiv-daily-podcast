from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jinja2 import Environment, StrictUndefined
from openai import OpenAI

from utils.costs import CostTracker

from .affiliations import extract_affiliations_batch
from .arxiv_client import fetch_keyword_papers, fetch_recent_papers
from .emailer import send_email
from .output import (
    create_run_dir,
    download_papers,
    write_csv,
    write_newsletter_html,
    write_results_json,
)
from .podcast import generate_transcripts_batch, load_podcast_prompt, write_transcript
from .tts import batch_synthesize_podcast
from .ranking import rank_papers


MAX_LIMIT = 50
MAX_EMAIL_ATTACHMENT_BYTES = 20 * 1024 * 1024
DEFAULT_CONFIG_PATH = Path("my_config") / "config.yaml"
DEFAULT_PROMPT_PATH = Path("prompt") / "prompt_ranking.j2"
DEFAULT_PODCAST_PROMPT_PATH = Path("prompt") / "prompt_podcast.j2"
DEFAULT_NEWSLETTER_TEMPLATE = Path("template") / "newsletter.j2"
AFFILIATION_MODEL = "gpt-5-mini-2025-08-07"
AFFILIATION_TOKEN_LIMIT = 200


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank recent arXiv cs.IR and cs.CL papers")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def _trim_attachments_by_size(attachments: list[Path], max_total_bytes: int) -> list[Path]:
    kept: list[Path] = []
    total = 0
    for path in attachments:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            print(f"Attachment missing, skipping: {path}")
            continue
        if total + size > max_total_bytes:
            break
        kept.append(path)
        total += size
    if len(kept) < len(attachments):
        print(f"Attachment limit reached; attaching {len(kept)} of {len(attachments)} files.")
    return kept


def _extract_version(arxiv_id: str) -> str:
    if "v" in arxiv_id:
        suffix = arxiv_id.rsplit("v", 1)[-1]
        if suffix.isdigit():
            return f"v{suffix}"
    return ""


def _date_only(value: str) -> str:
    if not value:
        return ""
    return value.split("T", 1)[0]


def main() -> None:
    load_dotenv()
    args = _parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")
    config = yaml.safe_load(args.config.read_text()) or {}
    if not isinstance(config, dict):
        raise SystemExit("Config file must contain a YAML object at the top level.")

    cost_tracker = CostTracker()

    ranking_model = config.get("ranking_model")
    podcast_model = config.get("podcast_model")
    ir_limit = config.get("ir_limit")
    nlp_limit = config.get("nlp_limit")
    others_limit = config.get("others_limit")
    keywords_path = config.get("keywords_path")
    top_n = config.get("top_n")
    top_n_tts = config.get("top_n_tts")
    abst_word_cutoff = config.get("abst_word_cutoff")
    transcript_word_cutoff = config.get("transcript_word_cutoff")
    generate_transcript_flag = config.get("generate_transcript", True)
    use_tts = config.get("use_tts", True)
    tts_model = config.get("tts_model")
    tts_voice = config.get("tts_voice")
    tts_instructions_path = config.get("tts_instructions_path")
    tts_instructions = None
    compress_to_64kbps = config.get("compress_to_64kbps", True)
    email_enabled = config.get("email_enabled", False)
    pricing_path = config.get("pricing_path")
    arxiv_timeout = config.get("arxiv_timeout")
    openai_timeout = config.get("openai_timeout")
    gmail_address = None
    gmail_password = None

    if not ranking_model:
        raise SystemExit("Config must include ranking_model")
    if not podcast_model:
        raise SystemExit("Config must include podcast_model")
    if not isinstance(use_tts, bool):
        raise SystemExit("use_tts must be a boolean")
    if not isinstance(email_enabled, bool):
        raise SystemExit("email_enabled must be a boolean")
    if not generate_transcript_flag and use_tts:
        print("use_tts ignored because generate_transcript is false.")
        use_tts = False
    if use_tts:
        if not tts_model:
            raise SystemExit("Config must include tts_model")
        if not tts_voice:
            raise SystemExit("Config must include tts_voice")
        if tts_instructions_path:
            tts_file = Path(tts_instructions_path)
            if not tts_file.exists():
                raise SystemExit(f"TTS instructions file not found: {tts_file}")
            tts_instructions = tts_file.read_text().strip()
            if not tts_instructions:
                raise SystemExit("tts_instructions_path must point to non-empty text")
        else:
            tts_instructions = config.get(
                "tts_instructions",
                "Energetic, upbeat podcast host tone. Friendly and engaging, clear enunciation.",
            )
            if not isinstance(tts_instructions, str) or not tts_instructions.strip():
                raise SystemExit("tts_instructions must be a non-empty string")
        if not isinstance(compress_to_64kbps, bool):
            raise SystemExit("compress_to_64kbps must be a boolean")
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
    if not isinstance(generate_transcript_flag, bool):
        raise SystemExit("generate_transcript must be a boolean")
    if not isinstance(arxiv_timeout, int) or arxiv_timeout < 1:
        raise SystemExit("arxiv_timeout must be an integer >= 1")
    if not isinstance(openai_timeout, int) or openai_timeout < 1:
        raise SystemExit("openai_timeout must be an integer >= 1")
    if not pricing_path:
        raise SystemExit("pricing_path must be set in config")
    if not keywords_path:
        raise SystemExit("keywords_path must be set in config")
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

    ranking_pricing = pricing_data.get(ranking_model, {}) or {}
    podcast_pricing = pricing_data.get(podcast_model, {}) or {}
    tts_pricing = pricing_data.get(tts_model, {}) or {}
    affiliation_pricing = pricing_data.get(AFFILIATION_MODEL, {}) or {}

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

    if email_enabled:
        gmail_address = os.getenv("GMAIL_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        if not gmail_address or not gmail_password:
            raise SystemExit("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env")

    capped_ir_limit = min(ir_limit, MAX_LIMIT)
    capped_nlp_limit = min(nlp_limit, MAX_LIMIT)
    capped_others_limit = min(others_limit, MAX_LIMIT)
    if capped_ir_limit != ir_limit:
        print(f"Capping ir_limit to {MAX_LIMIT}")
    if capped_nlp_limit != nlp_limit:
        print(f"Capping nlp_limit to {MAX_LIMIT}")
    if capped_others_limit != others_limit:
        print(f"Capping others_limit to {MAX_LIMIT}")

    print(
        "Fetching up to "
        f"{capped_ir_limit} cs.IR, {capped_nlp_limit} cs.CL, "
        f"and {capped_others_limit} keyword-matched papers from arXiv..."
    )
    prompt_template = DEFAULT_PROMPT_PATH.read_text()
    ir_papers = fetch_recent_papers(
        category="cs.IR",
        limit=capped_ir_limit,
        timeout=arxiv_timeout,
        id_prefix="IR",
        sort_by="lastUpdatedDate",
    )
    nlp_papers = fetch_recent_papers(
        category="cs.CL",
        limit=capped_nlp_limit,
        timeout=arxiv_timeout,
        id_prefix="CL",
        sort_by="lastUpdatedDate",
    )
    keyword_papers = fetch_keyword_papers(
        keywords=keywords,
        limit=capped_others_limit,
        timeout=arxiv_timeout,
        id_prefix="OTH",
        exclude_categories=["cs.IR", "cs.CL"],
        sort_by="lastUpdatedDate",
    )
    existing_ids = {paper.arxiv_id for paper in ir_papers + nlp_papers}
    filtered_keyword_papers = [
        paper for paper in keyword_papers if paper.arxiv_id not in existing_ids
    ]
    if len(filtered_keyword_papers) < len(keyword_papers):
        print(
            f"Removed {len(keyword_papers) - len(filtered_keyword_papers)} "
            "keyword papers that overlap with IR/CL."
        )
    if len(filtered_keyword_papers) < capped_others_limit:
        print(
            f"Only {len(filtered_keyword_papers)} keyword papers available after filtering."
        )
    papers = ir_papers + nlp_papers + filtered_keyword_papers
    print(
        f"Fetched {len(ir_papers)} cs.IR, {len(nlp_papers)} cs.CL, "
        f"and {len(filtered_keyword_papers)} keyword papers."
    )

    if len(papers) < top_n:
        raise SystemExit("Not enough papers fetched for the requested top-n")

    print("Ranking papers with LLM...")
    client = OpenAI()
    rankings = rank_papers(
        client=client,
        model=ranking_model,
        prompt_template=prompt_template,
        papers=papers,
        top_n=top_n,
        abstract_word_cutoff=abst_word_cutoff,
        pricing=ranking_pricing,
        cost_tracker=cost_tracker,
        openai_timeout=openai_timeout,
    )
    print("Ranking complete.")

    run_dir, papers_dir, transcript_dir, podcast_dir, newsletter_dir = create_run_dir()
    papers_by_id = {paper.paper_id: paper for paper in papers}

    write_csv(run_dir, papers_by_id, rankings, tldr_by_id=rankings.tldr_by_id)
    write_results_json(run_dir, papers, rankings, tldr_by_id=rankings.tldr_by_id)
    print("Downloading top papers...")
    pdf_paths = download_papers(papers_dir, papers, rankings.final_ranking)
    print("Download complete.")

    podcast_paths: list[Path] = []
    transcript_records: list[tuple[str, str, Path]] = []
    if generate_transcript_flag:
        podcast_prompt = load_podcast_prompt(DEFAULT_PODCAST_PROMPT_PATH)
        transcript_ids = rankings.final_ranking[:top_n_tts]
        if not transcript_ids:
            print("Transcript generation skipped (top_n_tts is 0).")
        else:
            print(f"Generating podcast transcripts for top {len(transcript_ids)} papers...")
            transcript_papers = [papers_by_id[paper_id] for paper_id in transcript_ids]
            transcript_pdf_paths = pdf_paths[: len(transcript_ids)]
            transcripts = generate_transcripts_batch(
                client=client,
                model=podcast_model,
                prompt_template=podcast_prompt,
                papers=transcript_papers,
                pdf_paths=transcript_pdf_paths,
                word_cutoff=transcript_word_cutoff,
                pricing=podcast_pricing,
                cost_tracker=cost_tracker,
                label="Transcript LLM",
                openai_timeout=openai_timeout,
                max_workers=min(4, len(transcript_papers)),
            )
            for rank, (paper, transcript) in enumerate(
                zip(transcript_papers, transcripts), start=1
            ):
                transcript_path = write_transcript(transcript_dir, paper, rank, transcript)
                transcript_records.append((paper.paper_id, transcript, transcript_path))
        if use_tts:
            tts_count = min(top_n_tts, len(transcript_records))
            if tts_count < top_n_tts:
                print(f"Only {tts_count} transcripts available for TTS.")
            tts_items = []
            for _, transcript, transcript_path in transcript_records[:tts_count]:
                audio_path = podcast_dir / transcript_path.with_suffix(".mp3").name
                tts_items.append((transcript, audio_path))
            podcast_paths = batch_synthesize_podcast(
                client=client,
                model=tts_model,
                voice=tts_voice,
                items=tts_items,
                timeout=openai_timeout,
                pricing=tts_pricing,
                cost_tracker=cost_tracker,
                instructions=tts_instructions,
                label="TTS",
                max_workers=min(4, len(tts_items)),
                compress_to_64kbps=compress_to_64kbps,
            )
            print("Transcripts and audio complete.")
        else:
            print("Transcripts complete (TTS disabled).")
    else:
        print("Transcript generation disabled.")

    if email_enabled:
        print("Extracting affiliations for email...")
        aff_papers = [papers_by_id[paper_id] for paper_id in rankings.final_ranking]
        affiliations_by_id = extract_affiliations_batch(
            client=client,
            model=AFFILIATION_MODEL,
            papers=aff_papers,
            pdf_paths=pdf_paths,
            pricing=affiliation_pricing,
            cost_tracker=cost_tracker,
            token_limit=AFFILIATION_TOKEN_LIMIT,
            openai_timeout=openai_timeout,
            max_workers=min(4, len(rankings.final_ranking)),
        )

        lines = [f"Run dir: {run_dir}", "", "Top papers:"]
        items: list[dict[str, str]] = []
        for rank, paper_id in enumerate(rankings.final_ranking, start=1):
            paper = papers_by_id[paper_id]
            tldr = rankings.tldr_by_id.get(paper_id, "")
            affiliations = affiliations_by_id.get(paper_id, "Not specified")
            authors = ", ".join(paper.authors)
            version = _extract_version(paper.arxiv_id)
            published_date = _date_only(paper.published)
            updated_date = _date_only(paper.updated)
            published_line = published_date
            if version and updated_date:
                published_line = f"{published_date} · Updated {version}: {updated_date}"

            lines.append(f"{rank}. {paper.title} ({paper.paper_id})")
            lines.append(f"Authors: {authors}")
            lines.append(f"Affiliations: {affiliations}")
            if published_line:
                lines.append(f"Published: {published_line}")
            if tldr:
                lines.append(f"TL;DR: {tldr}")
            lines.append("")
            items.append(
                {
                    "rank": str(rank),
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "arxiv_url": f"https://arxiv.org/abs/{paper.arxiv_id}",
                    "authors": authors,
                    "affiliations": affiliations,
                    "published_line": published_line,
                    "tldr": tldr,
                }
            )

        body = "\n".join(lines).strip()
        template_text = DEFAULT_NEWSLETTER_TEMPLATE.read_text()
        env = Environment(autoescape=True, undefined=StrictUndefined)
        html_body = env.from_string(template_text).render(
            run_name=run_dir.name,
            items=items,
        )
        write_newsletter_html(newsletter_dir, html_body, "newsletter.html")
        attachments = podcast_paths if podcast_paths else None
        if attachments:
            attachments = _trim_attachments_by_size(
                attachments, MAX_EMAIL_ATTACHMENT_BYTES
            )
            if not attachments:
                attachments = None
        send_email(
            smtp_user=gmail_address,
            smtp_password=gmail_password,
            to_addr=gmail_address,
            subject=f"arXiv update {run_dir.name}",
            body=body,
            html_body=html_body,
            attachments=attachments,
        )
        print("Sent update email.")

    print(f"Saved results to {run_dir}")
    total_cents = cost_tracker.total_cents()
    if cost_tracker.has_unknown:
        print(f"Total estimated cost: {total_cents:.2f}¢ (partial).")
    else:
        print(f"Total estimated cost: {total_cents:.2f}¢.")


if __name__ == "__main__":
    main()
