from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from jinja2 import Environment, StrictUndefined
from openai import OpenAI

from utils.costs import CostTracker
from .affiliations import extract_affiliations_batch
from .arxiv_client import fetch_keyword_papers, fetch_recent_papers
from .config import load_config
from .influence_filter import filter_by_author_influence
from .emailer import send_email
from .manga_image import generate_manga_image, generate_manga_instruction, load_manga_prompt
from .output import (
    create_run_dir,
    download_papers,
    write_csv,
    write_newsletter_html,
    write_results_json,
)
from .podcast import generate_transcripts_batch, load_podcast_prompt, write_transcript
from .paper_state import (
    base_arxiv_id,
    load_paper_state,
    mark_sent,
    merge_discovered_papers,
    needs_ranking_score,
    pooled_records,
    records_to_papers,
    save_paper_state,
    set_ranking_scores,
    set_affiliations,
    utc_now_iso,
)
from .selected_summary import generate_selected_summaries_batch, load_selected_summary_prompt
from .tts import batch_synthesize_podcast
from .ranking import aggregate_score, load_ranking_aspects, rank_from_scores, rank_papers


MAX_LIMIT = 50
MAX_EMAIL_ATTACHMENT_BYTES = 20 * 1024 * 1024
DEFAULT_CONFIG_PATH = Path("my_config") / "config.yaml"
DEFAULT_SCORING_PROMPT_PATH = Path("prompt") / "prompt_scoring.j2"
DEFAULT_PODCAST_PROMPT_PATH = Path("prompt") / "prompt_podcast.j2"
DEFAULT_MANGA_PROMPT_PATH = Path("prompt") / "prompt_manga.j2"
DEFAULT_MANGA_PLANNER_PROMPT_PATH = Path("prompt") / "prompt_manga_planner.j2"
DEFAULT_NEWSLETTER_TEMPLATE = Path("template") / "newsletter.j2"
DEFAULT_SELECTED_SUMMARY_PROMPT_PATH = Path("prompt") / "prompt_selected_summary.j2"
DEFAULT_INFLUENCE_PROMPT_PATH = Path("prompt") / "prompt_influence_filter.j2"
DEFAULT_TTS_INSTRUCTIONS_PATH = Path("prompt") / "tts_instructions.txt"
DEFAULT_STATE_PATH = Path("state") / "discovered_papers.json"
AFFILIATION_TOKEN_LIMIT = 200


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank recent arXiv cs.IR and cs.CL papers")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "fetch-score", "publish"),
        default="all",
        help=(
            "Pipeline stage to run: fetch-score discovers arXiv papers and stores "
            "ranking scores; publish reads stored papers and generates the newsletter, "
            "transcript/audio, and image."
        ),
    )
    return parser.parse_args()


def _load_tts_instructions() -> str:
    if DEFAULT_TTS_INSTRUCTIONS_PATH.exists():
        content = DEFAULT_TTS_INSTRUCTIONS_PATH.read_text().strip()
        if content:
            return content
    return "Energetic, upbeat podcast host tone. Friendly and engaging, clear enunciation."


def _state_path() -> Path:
    return Path(os.getenv("PAPER_STATE_PATH", str(DEFAULT_STATE_PATH)))


def _priority_authors() -> list[str]:
    raw = os.getenv("PRIORITY_AUTHORS", "")
    return [name.strip() for name in raw.split(";") if name.strip()]


def _manga_style_prompt() -> str:
    return os.getenv("MANGA_STYLE_PROMPT", "").strip()


def _manga_characters_list() -> str:
    teacher = os.getenv("CHARACTER_TEACHER", "").strip()
    student = os.getenv("CHARACTER_STUDENT", "").strip()
    lines = []
    if teacher:
        lines.append(f"Teacher: {teacher}")
    if student:
        lines.append(f"Student: {student}")
    return "\n".join(lines)


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


def _format_total_cost(cost_tracker: CostTracker) -> str:
    total_cents = cost_tracker.total_cents()
    suffix = " (partial)" if cost_tracker.has_unknown else ""
    return f"{total_cents:.2f}¢{suffix}"


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
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


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


def _count_record_sources(records: list[dict]) -> dict[str, int]:
    counts = {"ir": 0, "cl": 0, "keywords": 0, "total": len(records)}
    for record in records:
        source = record.get("source")
        if source == "ir":
            counts["ir"] += 1
        elif source == "cl":
            counts["cl"] += 1
        elif source == "keywords":
            counts["keywords"] += 1
    return counts


def _records_after_sending(records: list[dict], sent_base_ids: set[str]) -> list[dict]:
    return [
        record
        for record in records
        if record.get("base_arxiv_id") not in sent_base_ids
    ]


def _unsent_pool_statistics(
    records: list[dict],
    now: datetime | None = None,
) -> dict[str, int]:
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    cutoff_24h = current_time - timedelta(hours=24)
    cutoff_7d = current_time - timedelta(days=7)

    fetched_24h = 0
    unique_added_24h = 0
    added_7d = 0

    for record in records:
        last_seen_at = _parse_iso_datetime(record.get("last_seen_at", ""))
        first_seen_at = _parse_iso_datetime(record.get("first_seen_at", ""))

        fetched_recently = bool(last_seen_at and last_seen_at >= cutoff_24h)
        added_recently = bool(first_seen_at and first_seen_at >= cutoff_24h)
        if fetched_recently:
            fetched_24h += 1
        if fetched_recently and added_recently:
            unique_added_24h += 1
        if first_seen_at and first_seen_at >= cutoff_7d:
            added_7d += 1

    return {
        "unsent_pool_total": len(records),
        "fetched_24h": fetched_24h,
        "unique_added_24h": unique_added_24h,
        "added_7d": added_7d,
    }


def _score_contribution(score: int, weight: float, polarity: str) -> float:
    contribution = float(score) * weight
    if polarity == "negative":
        return -contribution
    return contribution


def _contribution_tone(contribution: float) -> str:
    if contribution >= 2.0:
        return "strong-positive"
    if contribution > 0:
        return "positive"
    if contribution <= -2.0:
        return "strong-negative"
    if contribution < 0:
        return "negative"
    return "neutral"


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
    run_fetch_score = args.stage in {"all", "fetch-score"}
    run_publish = args.stage in {"all", "publish"}

    settings = load_config(args.config)
    cost_tracker = CostTracker()

    ranking_model = settings.ranking.model
    ranking_provider = settings.ranking.provider
    podcast_model = settings.podcast.model
    podcast_provider = settings.podcast.provider
    generate_manga_image_flag = settings.generate_manga_image
    manga_planner_model = settings.manga_planner.model if settings.manga_planner else None
    manga_planner_provider = settings.manga_planner.provider if settings.manga_planner else None
    manga_image_model = settings.manga_image.model if settings.manga_image else None
    manga_image_size = settings.manga_image.size if settings.manga_image else None
    manga_image_quality = settings.manga_image.quality if settings.manga_image else None
    manga_image_output_format = settings.manga_image.output_format if settings.manga_image else None
    manga_image_char_cutoff = settings.manga_image.char_cutoff if settings.manga_image else None
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
    ranking_aspects_path = settings.ranking_aspects_path
    ranking_max_workers = settings.ranking_max_workers
    arxiv_timeout = settings.arxiv_timeout
    openai_timeout = settings.openai_timeout
    gmail_address = None
    gmail_password = None
    ranking_pricing = pricing_data.get(ranking_model, {}) or {}
    podcast_pricing = pricing_data.get(podcast_model, {}) or {}
    manga_planner_pricing = pricing_data.get(manga_planner_model, {}) or {} if manga_planner_model else {}
    manga_image_pricing = pricing_data.get(manga_image_model, {}) or {} if manga_image_model else {}
    tts_pricing = pricing_data.get(tts_model, {}) or {} if tts_model else {}
    influence_pricing = pricing_data.get(influence_filter_model, {}) or {}
    affiliation_pricing = pricing_data.get(affiliation_model, {}) or {}

    if email_enabled and run_publish:
        gmail_address = os.getenv("GMAIL_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        if not gmail_address or not gmail_password:
            raise SystemExit("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env")

    fallback_note: str | None = None
    if filter_since_last_schedule:
        print(
            "filter_since_last_schedule is ignored in bucket mode; "
            "paper-state deduplication controls repeat notifications."
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

    scoring_prompt_template = DEFAULT_SCORING_PROMPT_PATH.read_text()
    influence_prompt_template = influence_prompt_path.read_text()
    ranking_aspects = load_ranking_aspects(ranking_aspects_path)

    # Create clients based on provider
    openai_client = OpenAI()
    gemini_client = None
    influence_provider = settings.influence_filter.provider
    ranking_provider = settings.ranking.provider
    affiliation_provider = settings.affiliation.provider

    needs_gemini = (
        ranking_provider == "gemini"
        or (run_fetch_score and influence_provider == "gemini")
        or (run_fetch_score and affiliation_provider == "gemini")
        or (run_publish and podcast_provider == "gemini")
        or (run_publish and manga_planner_provider == "gemini")
    )
    if needs_gemini:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY must be set for Gemini provider")
        gemini_client = genai.Client(api_key=api_key)

    influence_client = gemini_client if influence_provider == "gemini" else openai_client
    ranking_client = gemini_client if ranking_provider == "gemini" else openai_client
    affiliation_client = gemini_client if affiliation_provider == "gemini" else openai_client
    podcast_client = gemini_client if podcast_provider == "gemini" else openai_client
    manga_planner_client = gemini_client if manga_planner_provider == "gemini" else openai_client
    manga_style_prompt = _manga_style_prompt()
    manga_characters_list = _manga_characters_list()

    state_path = _state_path()
    paper_state = load_paper_state(state_path)
    run_dir = None
    papers_dir = None
    transcript_dir = None
    podcast_dir = None
    newsletter_dir = None
    fetched_count = 0
    new_pool_count = 0
    author_influence_passed_count: int | None = None
    influence_gate_note: str | None = None

    if run_fetch_score:
        # Stage 1: fetch from arXiv, enrich/score the pool, and persist state.
        papers, _ = fetch_all(None, include_keywords=include_keyword_papers)
        fetched_count = len(papers)
        existing_base_ids = set(paper_state.get("pooled_papers", {}))
        new_papers = [
            paper
            for paper in papers
            if base_arxiv_id(paper.arxiv_id) not in existing_base_ids
        ]
        new_pool_count = len(new_papers)
        priority_authors = _priority_authors()
        influence_scores_by_id: dict[str, int] = {}
        if priority_authors and new_papers:
            influence_result = filter_by_author_influence(
                client=influence_client,
                model=influence_filter_model,
                prompt_template=influence_prompt_template,
                papers=new_papers,
                threshold=influence_score_threshold,
                max_workers=influence_max_workers,
                pricing=influence_pricing,
                cost_tracker=cost_tracker,
                openai_timeout=openai_timeout,
                provider=influence_provider,
                priority_authors=priority_authors,
            )
            influence_scores_by_id = influence_result.scores_by_id
            author_influence_passed_count = len(influence_result.kept_papers)
            influence_gate_note = (
                f"Author-influence scoring checked {new_pool_count} new/{fetched_count} "
                f"fetched papers and found {author_influence_passed_count} with score >= "
                f"{influence_score_threshold}."
            )
        elif priority_authors:
            influence_gate_note = (
                "Author-influence scoring skipped because all fetched papers already exist "
                "in the bucket; existing scores were reused."
            )
        else:
            influence_gate_note = (
                "Author-influence scoring skipped because PRIORITY_AUTHORS is empty; "
                "new papers were pooled without author-influence scores."
            )

        seen_at = utc_now_iso()
        changed_pooled_base_ids = merge_discovered_papers(
            paper_state,
            papers,
            scores_by_id=influence_scores_by_id,
            seen_at=seen_at,
            influence_threshold=influence_score_threshold,
        )
        save_paper_state(state_path, paper_state)

        if changed_pooled_base_ids:
            run_dir, papers_dir, transcript_dir, podcast_dir, newsletter_dir = create_run_dir()
            papers_by_base_id = {base_arxiv_id(paper.arxiv_id): paper for paper in papers}
            affiliation_papers = [
                papers_by_base_id[base_id]
                for base_id in changed_pooled_base_ids
                if base_id in papers_by_base_id
                and paper_state["pooled_papers"].get(base_id, {}).get("in_pool", True)
            ]
            if affiliation_papers:
                print(f"Extracting affiliations for {len(affiliation_papers)} bucket paper(s)...")
                affiliation_pdf_paths = download_papers(
                    papers_dir,
                    affiliation_papers,
                    [paper.paper_id for paper in affiliation_papers],
                )
                affiliations_by_paper_id = extract_affiliations_batch(
                    client=affiliation_client,
                    model=affiliation_model,
                    papers=affiliation_papers,
                    pdf_paths=affiliation_pdf_paths,
                    pricing=affiliation_pricing,
                    cost_tracker=cost_tracker,
                    token_limit=AFFILIATION_TOKEN_LIMIT,
                    openai_timeout=openai_timeout,
                    max_workers=min(4, len(affiliation_papers)),
                    provider=affiliation_provider,
                )
                set_affiliations(
                    paper_state,
                    {
                        base_arxiv_id(paper.arxiv_id): affiliations_by_paper_id.get(
                            paper.paper_id, "Not specified"
                        )
                        for paper in affiliation_papers
                    },
                )
        save_paper_state(state_path, paper_state)
        print(f"Saved paper bucket state to {state_path}")
    candidate_records = pooled_records(paper_state)
    if not candidate_records:
        print("No pooled papers available; skipping ranking and email.")
        if run_dir is not None:
            print(f"Saved results to {run_dir}")
        print(f"Total estimated cost: {_format_total_cost(cost_tracker)}.")
        return

    papers, paper_id_to_base_id = records_to_papers(candidate_records)
    author_influence_by_id = {}
    for paper, record in zip(papers, candidate_records):
        score = record.get("influence_score")
        if isinstance(score, int):
            author_influence_by_id[paper.paper_id] = score
    top_n_for_bucket = min(top_n, len(papers))

    aspect_keys = [aspect.key for aspect in ranking_aspects]
    scoring_items = [
        (paper, record)
        for paper, record in zip(papers, candidate_records)
        if run_fetch_score and needs_ranking_score(record, aspect_keys)
    ]
    if scoring_items:
        scoring_papers = [paper for paper, _ in scoring_items]
        scoring_paper_id_to_base_id = {
            paper.paper_id: paper_id_to_base_id[paper.paper_id]
            for paper in scoring_papers
        }
        scoring_author_influence_by_id = {
            paper.paper_id: author_influence_by_id[paper.paper_id]
            for paper in scoring_papers
            if paper.paper_id in author_influence_by_id
        }
        print(f"Scoring {len(scoring_papers)} new or changed pooled paper(s) with LLM...")
        scoring_rankings = rank_papers(
            client=ranking_client,
            model=ranking_model,
            scoring_prompt_template=scoring_prompt_template,
            papers=scoring_papers,
            top_n=len(scoring_papers),
            author_influence_by_id=scoring_author_influence_by_id,
            abstract_word_cutoff=abst_word_cutoff,
            pricing=ranking_pricing,
            cost_tracker=cost_tracker,
            openai_timeout=openai_timeout,
            provider=ranking_provider,
            include_keyword_papers=include_keyword_papers,
            aspects=ranking_aspects,
            max_workers=ranking_max_workers,
        )
        set_ranking_scores(
            paper_state,
            paper_id_to_base_id=scoring_paper_id_to_base_id,
            scores_by_id=scoring_rankings.scores_by_id,
            total_score_by_id=scoring_rankings.total_score_by_id,
        )
    else:
        if run_fetch_score:
            print("No pooled papers need ranking scoring; reusing existing scores.")
        else:
            print("Reusing stored ranking scores.")

    scores_by_id: dict[str, dict[str, int]] = {}
    total_score_by_id: dict[str, float] = {}
    for paper, record in zip(papers, candidate_records):
        scores = dict(record.get("ranking_scores", {}))
        author_influence_score = author_influence_by_id.get(paper.paper_id)
        if author_influence_score is not None:
            scores["author_influence_score"] = author_influence_score
        scores_by_id[paper.paper_id] = scores
        total_score_by_id[paper.paper_id] = aggregate_score(
            scores,
            ranking_aspects,
            author_influence_score=author_influence_score,
        )

    set_ranking_scores(
        paper_state,
        paper_id_to_base_id=paper_id_to_base_id,
        scores_by_id=scores_by_id,
        total_score_by_id=total_score_by_id,
    )
    save_paper_state(state_path, paper_state)
    if not run_publish:
        if run_dir is not None:
            print(f"Saved results to {run_dir}")
        print(f"Total estimated cost: {_format_total_cost(cost_tracker)}.")
        return

    rankings = rank_from_scores(
        papers=papers,
        top_n=top_n_for_bucket,
        scores_by_id=scores_by_id,
        total_score_by_id=total_score_by_id,
        tldr_by_id={},
    )
    print("Ranking complete.")

    if run_dir is None:
        run_dir, papers_dir, transcript_dir, podcast_dir, newsletter_dir = create_run_dir()

    papers_by_id = {paper.paper_id: paper for paper in papers}
    winner_ids = rankings.final_ranking[:1]
    winner_id = winner_ids[0]
    winner_base_id = paper_id_to_base_id[winner_id]
    winner_base_ids = {paper_id_to_base_id[paper_id] for paper_id in winner_ids}
    selected_papers = [papers_by_id[paper_id] for paper_id in winner_ids]

    print("Downloading winning paper...")
    pdf_paths = download_papers(papers_dir, papers, winner_ids)
    print("Download complete.")

    selected_summary_prompt = load_selected_summary_prompt(DEFAULT_SELECTED_SUMMARY_PROMPT_PATH)
    print(f"Generating selected-paper summary for {len(selected_papers)} paper(s)...")
    selected_summaries_by_id = generate_selected_summaries_batch(
        client=ranking_client,
        model=ranking_model,
        prompt_template=selected_summary_prompt,
        papers=selected_papers,
        pdf_paths=pdf_paths,
        paper_text_word_cutoff=transcript_word_cutoff,
        pricing=ranking_pricing,
        cost_tracker=cost_tracker,
        label="Selected summary LLM",
        openai_timeout=openai_timeout,
        max_workers=min(4, len(selected_papers)),
        provider=ranking_provider,
    )
    selected_tldr_by_id = {
        paper_id: summary.get("tldr", "")
        for paper_id, summary in selected_summaries_by_id.items()
    }

    write_csv(
        run_dir,
        papers_by_id,
        rankings,
        tldr_by_id=selected_tldr_by_id,
        author_influence_by_id=author_influence_by_id,
    )
    write_results_json(
        run_dir,
        papers,
        rankings,
        tldr_by_id=selected_tldr_by_id,
        summary_sections_by_id=selected_summaries_by_id,
        author_influence_by_id=author_influence_by_id,
    )

    podcast_paths: list[Path] = []
    manga_image_paths: list[Path] = []
    transcript_records: list[tuple[str, str, Path]] = []
    if generate_manga_image_flag:
        if (
            not manga_planner_model
            or not manga_planner_provider
            or not manga_image_model
            or not manga_image_size
            or not manga_image_quality
            or not manga_image_output_format
        ):
            raise RuntimeError("Manga image generation is enabled but manga config is incomplete")
        manga_planner_prompt = load_manga_prompt(DEFAULT_MANGA_PLANNER_PROMPT_PATH)
        manga_prompt = load_manga_prompt(DEFAULT_MANGA_PROMPT_PATH)
        manga_dir = run_dir / "manga"
        print("Planning manga image for winning paper...")
        try:
            manga_instruction = generate_manga_instruction(
                client=manga_planner_client,
                model=manga_planner_model,
                prompt_template=manga_planner_prompt,
                paper=selected_papers[0],
                pdf_path=pdf_paths[0],
                char_cutoff=manga_image_char_cutoff,
                pricing=manga_planner_pricing,
                cost_tracker=cost_tracker,
                timeout=openai_timeout,
                provider=manga_planner_provider,
                manga_style=manga_style_prompt,
                manga_characters_list=manga_characters_list,
            )
            print("Generating manga image for winning paper...")
            manga_image_paths.append(
                generate_manga_image(
                    client=openai_client,
                    model=manga_image_model,
                    prompt_template=manga_prompt,
                    manga_instruction=manga_instruction,
                    paper=selected_papers[0],
                    image_dir=manga_dir,
                    rank=1,
                    size=manga_image_size,
                    quality=manga_image_quality,
                    output_format=manga_image_output_format,
                    char_cutoff=manga_image_char_cutoff,
                    pricing=manga_image_pricing,
                    cost_tracker=cost_tracker,
                    timeout=openai_timeout,
                    manga_style=manga_style_prompt,
                    manga_characters_list=manga_characters_list,
                )
            )
            print("Manga image complete.")
        except Exception as exc:
            print(
                "Warning: manga image generation failed; "
                f"continuing without image attachment ({type(exc).__name__}: {exc})."
            )
    else:
        print("Manga image generation disabled.")

    if generate_transcript_flag:
        podcast_prompt = load_podcast_prompt(DEFAULT_PODCAST_PROMPT_PATH)
        transcript_ids = winner_ids if top_n_tts > 0 else []
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
            if tts_items:
                primary_config = {
                    "provider": tts_provider,
                    "model": tts_model,
                    "voice": tts_voice,
                    "pricing": tts_pricing,
                }

                try:
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
                except Exception as exc:
                    podcast_paths = []
                    print(
                        "Warning: TTS generation failed; "
                        f"continuing without audio attachment ({type(exc).__name__}: {exc})."
                    )
            else:
                print("TTS skipped (no transcripts available).")
        else:
            print("Transcripts complete (TTS disabled).")
    else:
        print("Transcript generation disabled.")

    if email_enabled:
        total_cost_summary = _format_total_cost(cost_tracker)
        pool_stats = _unsent_pool_statistics(
            candidate_records,
        )
        mail_stats = [
            {"label": "Unsent pool", "value": str(pool_stats["unsent_pool_total"])},
            {"label": "Fetched 24h", "value": str(pool_stats["fetched_24h"])},
            {
                "label": "Uniq/Fetched 24h",
                "value": f"{pool_stats['unique_added_24h']}/{pool_stats['fetched_24h']}",
            },
            {"label": "Added 7d", "value": str(pool_stats["added_7d"])},
        ]
        stats_line = (
            f"Stats: {pool_stats['unsent_pool_total']} unsent in pool; "
            f"{pool_stats['fetched_24h']} fetched in the last 24h; "
            f"{pool_stats['unique_added_24h']} unique additions from those 24h fetches; "
            f"{pool_stats['added_7d']} added in the last 7d."
        )

        lines: list[str] = []
        if fallback_note:
            lines.append(f"NOTE: {fallback_note}")
        if influence_gate_note:
            lines.append(influence_gate_note)
        lines.extend([stats_line, "", "Top paper:"])
        items: list[dict] = []
        for rank, paper_id in enumerate(winner_ids, start=1):
            paper = papers_by_id[paper_id]
            selected_summary = selected_summaries_by_id.get(paper_id, {})
            summary_sections = [
                ("TL;DR", selected_summary.get("tldr", "")),
                ("Background", selected_summary.get("background", "")),
                ("Existing problem", selected_summary.get("existing_problem", "")),
                ("Proposed method", selected_summary.get("proposed_method", "")),
                ("Results", selected_summary.get("results", "")),
            ]
            state_record = paper_state["pooled_papers"].get(paper_id_to_base_id[paper_id], {})
            affiliations = state_record.get("affiliations") or "Not specified"
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
            for section_label, section_text in summary_sections:
                if section_text:
                    lines.append(f"{section_label}: {section_text}")
            lines.append(f"Ranking score: {rankings.total_score_by_id.get(paper_id, 0):.1f}")
            ranking_scores = rankings.scores_by_id.get(paper_id, {})
            score_rows = []
            for aspect in ranking_aspects:
                score = ranking_scores.get(aspect.key)
                if not isinstance(score, int):
                    continue
                contribution = _score_contribution(score, aspect.weight, aspect.polarity)
                score_rows.append(
                    {
                        "label": aspect.label,
                        "score": str(score),
                        "range": "0-2",
                        "polarity": "minus" if aspect.polarity == "negative" else "plus",
                        "contribution": f"{contribution:+.1f}",
                        "contribution_tone": _contribution_tone(contribution),
                    }
                )
                lines.append(
                    f"- {aspect.label}: {score} (0-2) "
                    f"({aspect.polarity}, {contribution:+.1f})"
                )
            author_score = ranking_scores.get("author_influence_score")
            if isinstance(author_score, int):
                score_rows.append(
                    {
                        "label": "author influence",
                        "score": str(author_score),
                        "range": "0-5",
                        "polarity": "plus",
                        "contribution": f"{float(author_score):+.1f}",
                        "contribution_tone": _contribution_tone(float(author_score)),
                    }
                )
                lines.append(
                    f"- author influence: {author_score} (0-5) "
                    f"(plus, {float(author_score):+.1f})"
                )
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
                    "summary_sections": [
                        {"label": section_label, "text": section_text}
                        for section_label, section_text in summary_sections
                        if section_text
                    ],
                    "total_score": f"{rankings.total_score_by_id.get(paper_id, 0):.1f}",
                    "score_rows": score_rows,
                }
            )

        lines.extend(["", f"Total estimated cost: {total_cost_summary}."])
        body = "\n".join(lines).strip()
        template_text = DEFAULT_NEWSLETTER_TEMPLATE.read_text()
        env = Environment(autoescape=True, undefined=StrictUndefined)
        html_body = env.from_string(template_text).render(
            run_name=run_dir.name,
            items=items,
            stats=mail_stats,
            notices=[line for line in [fallback_note, influence_gate_note] if line],
            total_cost=total_cost_summary,
        )
        write_newsletter_html(newsletter_dir, html_body, "newsletter.html")
        all_attachments = manga_image_paths + podcast_paths
        attachments = all_attachments if all_attachments else None
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
        mark_sent(paper_state, winner_base_id, utc_now_iso(), run_dir.name)
        save_paper_state(state_path, paper_state)
        print(f"Marked {winner_base_id} as sent in paper bucket state.")

    print(f"Saved results to {run_dir}")
    print(f"Total estimated cost: {_format_total_cost(cost_tracker)}.")


if __name__ == "__main__":
    main()
