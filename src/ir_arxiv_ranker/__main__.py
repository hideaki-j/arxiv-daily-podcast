from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Environment, StrictUndefined
from openai import OpenAI

from utils.costs import CostTracker

from .affiliations import extract_affiliations_batch
from .arxiv_client import fetch_keyword_papers, fetch_recent_papers
from .config import load_config
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

    ranking_model = settings.ranking_model
    podcast_model = settings.podcast_model
    ir_limit = settings.ir_limit
    nlp_limit = settings.nlp_limit
    others_limit = settings.others_limit
    keywords = settings.keywords
    top_n = settings.top_n
    top_n_tts = settings.top_n_tts
    abst_word_cutoff = settings.abst_word_cutoff
    transcript_word_cutoff = settings.transcript_word_cutoff
    generate_transcript_flag = settings.generate_transcript
    filter_since_last_schedule = settings.filter_since_last_schedule
    use_tts = settings.use_tts
    tts_model = settings.tts_model
    tts_voice = settings.tts_voice
    tts_instructions = settings.tts_instructions
    compress_to_64kbps = settings.compress_to_64kbps
    email_enabled = settings.email_enabled
    pricing_data = settings.pricing_data
    arxiv_timeout = settings.arxiv_timeout
    openai_timeout = settings.openai_timeout
    gmail_address = None
    gmail_password = None
    ranking_pricing = pricing_data.get(ranking_model, {}) or {}
    podcast_pricing = pricing_data.get(podcast_model, {}) or {}
    tts_pricing = pricing_data.get(tts_model, {}) or {}
    affiliation_pricing = pricing_data.get(AFFILIATION_MODEL, {}) or {}

    if email_enabled:
        gmail_address = os.getenv("GMAIL_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        if not gmail_address or not gmail_password:
            raise SystemExit("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env")

    updated_after = None
    last_scheduled_time: datetime | None = None
    fallback_note: str | None = None
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
                updated_after = last_run
                last_scheduled_time = last_run
                print(
                    "Schedule-based filtering enabled; only keeping papers updated after "
                    f"{updated_after.isoformat()} (last scheduled cron time, "
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

    def fetch_all(updated_after_value: datetime | None) -> tuple[list, dict[str, int]]:
        print(
            "Fetching up to "
            f"{capped_ir_limit} cs.IR, {capped_nlp_limit} cs.CL, "
            f"and {capped_others_limit} keyword-matched papers from arXiv..."
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
        print(
            f"Fetched {len(ir_papers)} cs.IR, {len(nlp_papers)} cs.CL, "
            f"and {len(filtered_keyword_papers)} keyword papers."
        )
        counts = _count_sources(papers_local)
        return papers_local, counts

    prompt_template = DEFAULT_PROMPT_PATH.read_text()
    papers, fetch_counts = fetch_all(updated_after)

    if len(papers) < top_n and filter_since_last_schedule and last_scheduled_time:
        filtered_count = len(papers)
        fallback_reason = (
            "no new papers found"
            if not papers
            else f"only {len(papers)} new paper(s) found"
        )
        print(
            f"{fallback_reason.capitalize()} after schedule filtering; "
            "falling back to the most recent dates from the unfiltered feed."
        )
        fallback_papers, _ = fetch_all(None)
        papers, (latest_date_iso, earliest_date_iso) = _select_by_date_cascade(
            fallback_papers, top_n
        )
        fetch_counts = _count_sources(papers)
        date_span = ""
        if latest_date_iso and earliest_date_iso:
            date_span = f"{latest_date_iso} to {earliest_date_iso}"
        elif latest_date_iso:
            date_span = latest_date_iso
        fallback_note = (
            "Schedule filter based on the last scheduled cron time "
            f"({last_scheduled_time.isoformat()}) returned {filtered_count} paper(s); "
            "fell back to the most recent dates "
            f"({date_span or 'unfiltered range'}) for this newsletter. "
            "Note this uses the scheduled time, not necessarily the last successful run."
        )

    if len(papers) < top_n:
        raise SystemExit("Not enough papers fetched for the requested top-n")
    if fallback_note:
        print(f"Fallback note: {fallback_note}")

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

        stats_line = (
            f"Stats: IR {fetch_counts.get('ir', 0)}, "
            f"CL {fetch_counts.get('cl', 0)}, "
            f"Keywords {fetch_counts.get('keywords', 0)} (final set)."
        )

        lines: list[str] = []
        if fallback_note:
            lines.append(f"NOTE: {fallback_note}")
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
        html_stats = (
            f"<p><strong>Stats:</strong> IR {fetch_counts.get('ir', 0)}, "
            f"CL {fetch_counts.get('cl', 0)}, "
            f"Keywords {fetch_counts.get('keywords', 0)} (final set).</p>"
        )
        html_prefix = ""
        if fallback_note:
            html_prefix += f"<p><strong>NOTE:</strong> {fallback_note}</p>"
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
