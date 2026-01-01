from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import httpx

from .models import Paper
from .ranking import Rankings
from utils.naming import build_file_stem


BASE_DATA_DIR = Path("data")


def _timestamp_dir() -> str:
    return datetime.now().strftime("%y%m%d-%H%M%S")


def create_run_dir(base_dir: Path = BASE_DATA_DIR) -> tuple[Path, Path, Path, Path, Path]:
    run_dir = base_dir / _timestamp_dir()
    run_dir.mkdir(parents=True, exist_ok=False)
    papers_dir = run_dir / "papers"
    transcript_dir = run_dir / "transcript"
    podcast_dir = run_dir / "podcast"
    newsletter_dir = run_dir / "newsletter"
    papers_dir.mkdir()
    transcript_dir.mkdir()
    podcast_dir.mkdir()
    newsletter_dir.mkdir()
    return run_dir, papers_dir, transcript_dir, podcast_dir, newsletter_dir


def write_newsletter_html(newsletter_dir: Path, html_body: str, filename: str) -> Path:
    path = newsletter_dir / filename
    path.write_text(html_body)
    return path


def _rank_position(ranking: List[str], paper_id: str) -> str:
    if paper_id in ranking:
        return str(ranking.index(paper_id) + 1)
    return ""


def write_csv(
    run_dir: Path,
    papers_by_id: Dict[str, Paper],
    rankings: Rankings,
    tldr_by_id: Dict[str, str] | None = None,
    author_influence_by_id: Dict[str, int] | None = None,
) -> Path:
    csv_path = run_dir / "rankings.csv"
    order = rankings.final_ranking

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "title",
                "authors",
                "uploaded_at",
                "author_influence_threshold",
                "tldr",
                "automatic_eval_ranking",
                "user_simulator_ranking",
                "final_ranking",
            ]
        )
        for paper_id in order:
            paper = papers_by_id[paper_id]
            authors = "; ".join(paper.authors)
            tldr = ""
            author_influence = ""
            if tldr_by_id:
                tldr = tldr_by_id.get(paper_id, "")
            if author_influence_by_id:
                score = author_influence_by_id.get(paper_id)
                if score is not None:
                    author_influence = str(score)
            writer.writerow(
                [
                    paper.paper_id,
                    paper.title,
                    authors,
                    paper.published,
                    author_influence,
                    tldr,
                    _rank_position(rankings.automatic_eval_ranking, paper_id),
                    _rank_position(rankings.user_simulator_ranking, paper_id),
                    _rank_position(rankings.final_ranking, paper_id),
                ]
            )

    return csv_path


def write_results_json(
    run_dir: Path,
    papers: Iterable[Paper],
    rankings: Rankings,
    tldr_by_id: Dict[str, str] | None = None,
    author_influence_by_id: Dict[str, int] | None = None,
) -> Path:
    payload = {
        "rankings": {
            "automatic_eval_ranking": rankings.automatic_eval_ranking,
            "user_simulator_ranking": rankings.user_simulator_ranking,
            "final_ranking": rankings.final_ranking,
        },
        "papers": [
            {
                "paper_id": paper.paper_id,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "published": paper.published,
                "updated": paper.updated,
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "tldr": (tldr_by_id or {}).get(paper.paper_id, ""),
                "author_influence_threshold": (
                    (author_influence_by_id or {}).get(paper.paper_id)
                ),
            }
            for paper in papers
        ],
    }

    json_path = run_dir / "results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    return json_path


def _download_pdf(client: httpx.Client, url: str, dest: Path) -> None:
    with client.stream("GET", url, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)


def download_papers(papers_dir: Path, papers: List[Paper], paper_ids: List[str]) -> List[Path]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    downloaded: List[Path] = []

    with httpx.Client() as client:
        for rank, paper_id in enumerate(paper_ids, start=1):
            paper = papers_by_id[paper_id]
            stem = build_file_stem(rank, paper.paper_id, paper.title)
            filename = f"{stem}.pdf"
            dest = papers_dir / filename
            _download_pdf(client, paper.pdf_url, dest)
            downloaded.append(dest)

    return downloaded
