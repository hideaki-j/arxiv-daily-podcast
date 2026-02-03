from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from jinja2 import Environment, StrictUndefined
from openai import OpenAI

from utils.costs import CostTracker
from utils.timezone import format_toronto_time

from .affiliations import extract_affiliations_batch
from .arxiv_client import fetch_keyword_papers, fetch_recent_papers
from .config import load_config
from .influence_filter import filter_by_author_influence
from .emailer import send_email
from .output import (
    create_run_dir,
    download_papers,
    write_csv,
    write_newsletter_html,
    write_results_json,
)
from .podcast import generate_transcripts_batch, load_podcast_prompt, write_transcript
from .schedule import last_scheduled_run, load_workflow_cron_schedules
from .tts import batch_synthesize_podcast
from .ranking import rank_papers


MAX_LIMIT = 50
MAX_EMAIL_ATTACHMENT_BYTES = 20 * 1024 * 1024
DEFAULT_CONFIG_PATH = Path("my_config") / "config.yaml"
DEFAULT_PROMPT_PATH = Path("prompt") / "prompt_ranking.j2"
DEFAULT_PODCAST_PROMPT_PATH = Path("prompt") / "prompt_podcast.j2"
DEFAULT_NEWSLETTER_TEMPLATE = Path("template") / "newsletter.j2"
DEFAULT_WORKFLOW_PATH = Path(".github") / "workflows" / "arxiv-newsletter.yml"
DEFAULT_INFLUENCE_PROMPT_PATH = Path("prompt") / "prompt_influence_filter.j2"
DEFAULT_TTS_INSTRUCTIONS_PATH = Path("prompt") / "tts_instructions.txt"
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


def _load_tts_instructions() -> str:
    if DEFAULT_TTS_INSTRUCTIONS_PATH.exists():
        content = DEFAULT_TTS_INSTRUCTIONS_PATH.read_text().strip()
        if content:
            return content
    return "Energetic, upbeat podcast host tone. Friendly and engaging, clear enunciation."


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


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _paper_datetime(paper) -> datetime | None:
    for ts in (paper.updated, paper.published):
        dt = _parse_iso_datetime(ts)
        if dt:
            return dt
    return None


def _count_sources(papers: list) -> dict[str, int]:
    counts = {"ir": 0, "cl": 0, "keywords": 0, "total": len(papers)}
    for paper in papers:
        pid = paper.paper_id.upper()
        if pid.startswith("IR"):
            counts["ir"] += 1
        elif pid.startswith("CL"):
            counts["cl"] += 1
        elif pid.startswith("OTH"):
            counts["keywords"] += 1
    return counts


def _select_by_date_cascade(papers: list, min_count: int) -> tuple[list, tuple[str | None, str | None]]:
    date_buckets: dict[date, list] = {}
    undated: list = []
    for paper in papers:
        dt = _paper_datetime(paper)
        if dt:
            date_buckets.setdefault(dt.date(), []).append(paper)
        else:
            undated.append(paper)

    selected: list = []
    latest_date: date | None = None
    earliest_date: date | None = None
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    for day in sorted(date_buckets.keys(), reverse=True):
        bucket = sorted(
            date_buckets[day],
            key=lambda p: _paper_datetime(p) or min_dt,
            reverse=True,
        )
        selected.extend(bucket)
        if latest_date is None:
            latest_date = day
        earliest_date = day
        if len(selected) >= min_count:
            break

    if len(selected) < min_count and undated:
        selected.extend(undated)

    return selected, (
        latest_date.isoformat() if latest_date else None,
        earliest_date.isoformat() if earliest_date else None,
    )


def main() -> None:
    load_dotenv()
    args = _parse_args()

    settings = load_config(args.config)
    cost_tracker = CostTracker()

    ranking_model = settings.ranking.model
    ranking_provider = settings.ranking.provider
    podcast_model = settings.podcast.model
    podcast_provider = settings.podcast.provider
    influence_filter_model = settings.influence_filter.model
    affiliation_model = settings.affiliation.model
    ir_limit = settings.ir_limit
    nlp_limit = settings.nlp_limit
    others_limit = settings.others_limit
    keywords = settings.keywords
    include_keyword_papers = settings.include_keyword_papers
    top_n = settings.top_n
    top_n_tts = settings.top_n_tts
    abst_word_cutoff = settings.abst_word_cutoff
    transcript_word_cutoff = settings.transcript_word_cutoff
    generate_transcript_flag = settings.generate_transcript
    filter_since_last_schedule = settings.filter_since_last_schedule
    use_tts = settings.use_tts
    tts_provider = settings.tts.provider if settings.tts else None
    tts_model = settings.tts.model if settings.tts else None
    tts_voice = settings.tts.voice if settings.tts else None
    compress_to_64kbps = settings.compress_to_64kbps
    email_enabled = settings.email_enabled
    pricing_data = settings.pricing_data
    influence_prompt_path = DEFAULT_INFLUENCE_PROMPT_PATH
    influence_score_threshold = settings.influence_score_threshold
    influence_max_workers = settings.influence_max_workers
    arxiv_timeout = settings.arxiv_timeout
    openai_timeout = settings.openai_timeout
    gmail_address = None
    gmail_password = None
    ranking_pricing = pricing_data.get(ranking_model, {}) or {}
    podcast_pricing = pricing_data.get(podcast_model, {}) or {}
    tts_pricing = pricing_data.get(tts_model, {}) or {} if tts_model else {}
    influence_pricing = pricing_data.get(influence_filter_model, {}) or {}
    affiliation_pricing = pricing_data.get(affiliation_model, {}) or {}

    if email_enabled:
        gmail_address = os.getenv("GMAIL_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        if not gmail_address or not gmail_password:
            raise SystemExit("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env")

    last_scheduled_time: datetime | None = None
    fallback_note: str | None = None
    last_scheduled_datetime_str: str | None = None
    if filter_since_last_schedule:
        cron_entries = load_workflow_cron_schedules(DEFAULT_WORKFLOW_PATH)
        if not cron_entries:
            print(
                "filter_since_last_schedule is enabled but no cron schedules were found "
                f"in {DEFAULT_WORKFLOW_PATH}. Falling back to unfiltered results."
            )
        else:
            last_run = last_scheduled_run(
                cron_entries, now=datetime.now(timezone.utc), lookback_days=30
            )
            if last_run:
                last_scheduled_time = last_run
                last_scheduled_datetime_str = format_toronto_time(last_run)
                print(
                    "Schedule-based filtering enabled; will later keep papers updated after "
                    f"{last_scheduled_datetime_str} (last scheduled cron time, "
                    "not guaranteed to be the last successful run)."
                )
            else:
                print(
                    "filter_since_last_schedule is enabled but the last scheduled run "
                    "was not found in the recent window. Falling back to unfiltered results."
                )

    capped_ir_limit = min(ir_limit, MAX_LIMIT)
    capped_nlp_limit = min(nlp_limit, MAX_LIMIT)
    capped_others_limit = min(others_limit, MAX_LIMIT)
    if capped_ir_limit != ir_limit:
        print(f"Capping ir_limit to {MAX_LIMIT}")
    if capped_nlp_limit != nlp_limit:
        print(f"Capping nlp_limit to {MAX_LIMIT}")
    if capped_others_limit != others_limit:
        print(f"Capping others_limit to {MAX_LIMIT}")

    def fetch_all(updated_after_value: datetime | None, include_keywords: bool = True) -> tuple[list, dict[str, int]]:
        if include_keywords:
            print(
                "Fetching up to "
                f"{capped_ir_limit} cs.IR, {capped_nlp_limit} cs.CL, "
                f"and {capped_others_limit} keyword-matched papers from arXiv..."
            )
        else:
            print(
                "Fetching up to "
                f"{capped_ir_limit} cs.IR and {capped_nlp_limit} cs.CL papers from arXiv..."
            )
        ir_papers = fetch_recent_papers(
            category="cs.IR",
            limit=capped_ir_limit,
            timeout=arxiv_timeout,
            id_prefix="IR",
            sort_by="lastUpdatedDate",
            updated_after=updated_after_value,
        )
        nlp_papers = fetch_recent_papers(
            category="cs.CL",
            limit=capped_nlp_limit,
            timeout=arxiv_timeout,
            id_prefix="CL",
            sort_by="lastUpdatedDate",
            updated_after=updated_after_value,
        )
        filtered_keyword_papers = []
        if include_keywords:
            keyword_papers = fetch_keyword_papers(
                keywords=keywords,
                limit=capped_others_limit,
                timeout=arxiv_timeout,
                id_prefix="OTH",
                exclude_categories=["cs.IR", "cs.CL"],
                sort_by="lastUpdatedDate",
                updated_after=updated_after_value,
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
        papers_local = ir_papers + nlp_papers + filtered_keyword_papers
        if include_keywords:
            print(
                f"Fetched {len(ir_papers)} cs.IR, {len(nlp_papers)} cs.CL, "
                f"and {len(filtered_keyword_papers)} keyword papers."
            )
        else:
            print(f"Fetched {len(ir_papers)} cs.IR and {len(nlp_papers)} cs.CL papers (keywords disabled).")
        counts = _count_sources(papers_local)
        return papers_local, counts

    prompt_template = DEFAULT_PROMPT_PATH.read_text()
    influence_prompt_template = influence_prompt_path.read_text()

    # Create clients based on provider
    openai_client = OpenAI()
    gemini_client = None
    influence_provider = settings.influence_filter.provider
    ranking_provider = settings.ranking.provider
    affiliation_provider = settings.affiliation.provider

    if influence_provider == "gemini" or ranking_provider == "gemini" or affiliation_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY must be set for Gemini provider")
        gemini_client = genai.Client(api_key=api_key)

    influence_client = gemini_client if influence_provider == "gemini" else openai_client
    ranking_client = gemini_client if ranking_provider == "gemini" else openai_client
    affiliation_client = gemini_client if affiliation_provider == "gemini" else openai_client
    podcast_client = gemini_client if podcast_provider == "gemini" else openai_client

    # Stage 1: fetch without date filtering, then gate by author influence
    papers, _ = fetch_all(None, include_keywords=include_keyword_papers)
    fetched_count = len(papers)
    influence_result = filter_by_author_influence(
        client=influence_client,
        model=influence_filter_model,
        prompt_template=influence_prompt_template,
        papers=papers,
        threshold=influence_score_threshold,
        max_workers=influence_max_workers,
        pricing=influence_pricing,
        cost_tracker=cost_tracker,
        openai_timeout=openai_timeout,
        provider=influence_provider,
    )
    author_influence_by_id = influence_result.scores_by_id
    papers = influence_result.kept_papers
    influence_gate_note = (
        f"Author influence gate kept {len(papers)}/{fetched_count} papers "
        f"(threshold >= {influence_score_threshold})."
    )
    fetch_counts = _count_sources(papers)

    # Stage 2: apply date filtering (schedule-based) to already-filtered set
    if filter_since_last_schedule and last_scheduled_time:
        updated_after = last_scheduled_time
        recent_papers = []
        for paper in papers:
            dt = _paper_datetime(paper)
            if dt and dt > updated_after:
                recent_papers.append(paper)
        if len(recent_papers) < top_n:
            filtered_count = len(recent_papers)
            fallback_reason = (
                "no new papers found"
                if not recent_papers
                else f"only {len(recent_papers)} new paper(s) found"
            )
            print(
                f"{fallback_reason.capitalize()} after schedule filtering; "
                "falling back to the most recent dates from the influence-filtered set."
            )
            papers, (latest_date_iso, earliest_date_iso) = _select_by_date_cascade(
                papers, top_n
            )
            fetch_counts = _count_sources(papers)
            date_span = ""
            if latest_date_iso and earliest_date_iso:
                date_span = f"{latest_date_iso} to {earliest_date_iso}"
            elif latest_date_iso:
                date_span = latest_date_iso
            schedule_label = last_scheduled_datetime_str or (
                format_toronto_time(last_scheduled_time) if last_scheduled_time else "N/A"
            )
            fallback_note = (
                f"Schedule filter ({schedule_label}) found {filtered_count} papers; "
                f"fell back to recent dates ({date_span or 'unfiltered'}) after influence gate."
            )
        else:
            papers = recent_papers
            fetch_counts = _count_sources(papers)

    if len(papers) < top_n:
        raise SystemExit(
            f"Not enough papers ({len(papers)}) after author influence/date filters for requested top-n {top_n}"
        )
    if fallback_note:
        print(f"Fallback note: {fallback_note}")

    print("Ranking papers with LLM...")
    rankings = rank_papers(
        client=ranking_client,
        model=ranking_model,
        prompt_template=prompt_template,
        papers=papers,
        top_n=top_n,
        author_influence_by_id=author_influence_by_id,
        abstract_word_cutoff=abst_word_cutoff,
        pricing=ranking_pricing,
        cost_tracker=cost_tracker,
        openai_timeout=openai_timeout,
        provider=ranking_provider,
        include_keyword_papers=include_keyword_papers,
    )
    print("Ranking complete.")

    run_dir, papers_dir, transcript_dir, podcast_dir, newsletter_dir = create_run_dir()
    papers_by_id = {paper.paper_id: paper for paper in papers}

    write_csv(
        run_dir,
        papers_by_id,
        rankings,
        tldr_by_id=rankings.tldr_by_id,
        author_influence_by_id=author_influence_by_id,
    )
    write_results_json(
        run_dir,
        papers,
        rankings,
        tldr_by_id=rankings.tldr_by_id,
        author_influence_by_id=author_influence_by_id,
    )
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
            
            # Setup podcast client
            if podcast_provider == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    print("Warning: OPENROUTER_API_KEY not set, trying default client.")
                else:
                    print("Using dedicated podcast client (OpenRouter)")
                    podcast_client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_key
                    )
            elif podcast_provider and podcast_provider != "openai" and podcast_provider != "gemini":
                print(f"Warning: Unknown podcast_provider '{podcast_provider}', using default client.")

            transcript_papers = [papers_by_id[paper_id] for paper_id in transcript_ids]
            transcript_pdf_paths = pdf_paths[: len(transcript_ids)]
            transcripts = generate_transcripts_batch(
                client=podcast_client,
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
                provider=podcast_provider,
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
            primary_config = {
                "provider": tts_provider,
                "model": tts_model,
                "voice": tts_voice,
                "pricing": tts_pricing,
            }
            
            podcast_paths = batch_synthesize_podcast(
                client=openai_client,
                primary_config=primary_config,
                items=tts_items,
                timeout=openai_timeout,
                cost_tracker=cost_tracker,
                instructions=_load_tts_instructions(),
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
            client=affiliation_client,
            model=affiliation_model,
            papers=aff_papers,
            pdf_paths=pdf_paths,
            pricing=affiliation_pricing,
            cost_tracker=cost_tracker,
            token_limit=AFFILIATION_TOKEN_LIMIT,
            openai_timeout=openai_timeout,
            max_workers=min(4, len(rankings.final_ranking)),
            provider=affiliation_provider,
        )

        if include_keyword_papers:
            stats_line = (
                f"Stats: IR {fetch_counts.get('ir', 0)}, "
                f"CL {fetch_counts.get('cl', 0)}, "
                f"Keywords {fetch_counts.get('keywords', 0)} (final set)."
            )
        else:
            stats_line = (
                f"Stats: IR {fetch_counts.get('ir', 0)}, "
                f"CL {fetch_counts.get('cl', 0)} (final set, keywords disabled)."
            )

        lines: list[str] = []
        if fallback_note:
            lines.append(f"NOTE: {fallback_note}")
        if influence_gate_note:
            lines.append(influence_gate_note)
        lines.extend([stats_line, "", "Top papers:"])
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
        if include_keyword_papers:
            html_stats = (
                f"<p><strong>Stats:</strong> IR {fetch_counts.get('ir', 0)}, "
                f"CL {fetch_counts.get('cl', 0)}, "
                f"Keywords {fetch_counts.get('keywords', 0)} (final set).</p>"
            )
        else:
            html_stats = (
                f"<p><strong>Stats:</strong> IR {fetch_counts.get('ir', 0)}, "
                f"CL {fetch_counts.get('cl', 0)} (final set, keywords disabled).</p>"
            )
        html_prefix = ""
        if fallback_note:
            html_prefix += f"<p><strong>NOTE:</strong> {fallback_note}</p>"
        if influence_gate_note:
            html_prefix += f"<p><strong>Author filter:</strong> {influence_gate_note}</p>"
        html_body = html_prefix + html_stats + html_body
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
