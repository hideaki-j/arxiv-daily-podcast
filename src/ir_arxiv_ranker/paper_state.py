from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import Paper


SCHEMA_VERSION = 1
UNSENT_STATUS = "unsent"
SENT_STATUS = "sent"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def base_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def load_paper_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "papers": {}}
    payload = json.loads(path.read_text() or "{}")
    if not isinstance(payload, dict):
        raise ValueError(f"Paper state must be a JSON object: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported paper state schema version in {path}")
    papers = payload.get("papers")
    if not isinstance(papers, dict):
        raise ValueError(f"Paper state must contain a papers object: {path}")
    return payload


def save_paper_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _source_from_paper_id(paper_id: str) -> str:
    upper = paper_id.upper()
    if upper.startswith("IR"):
        return "ir"
    if upper.startswith("CL"):
        return "cl"
    if upper.startswith("OTH"):
        return "keywords"
    return "unknown"


def _paper_record(paper: Paper, seen_at: str, existing: dict | None = None) -> dict:
    previous = existing or {}
    base_id = base_arxiv_id(paper.arxiv_id)
    status = previous.get("status", UNSENT_STATUS)
    record = {
        "base_arxiv_id": base_id,
        "latest_arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "affiliations": previous.get("affiliations", ""),
        "published": paper.published,
        "updated": paper.updated,
        "summary": paper.summary,
        "pdf_url": paper.pdf_url,
        "source": previous.get("source") or _source_from_paper_id(paper.paper_id),
        "first_seen_at": previous.get("first_seen_at", seen_at),
        "last_seen_at": seen_at,
        "status": status,
        "sent_at": previous.get("sent_at"),
        "sent_run_id": previous.get("sent_run_id"),
    }
    return record


def merge_discovered_papers(state: dict, papers: Iterable[Paper], seen_at: str) -> list[str]:
    changed_unsent_ids: list[str] = []
    changed_unsent_seen: set[str] = set()
    state_papers = state.setdefault("papers", {})

    for paper in papers:
        base_id = base_arxiv_id(paper.arxiv_id)
        existing = state_papers.get(base_id)
        record = _paper_record(paper, seen_at, existing)
        state_papers[base_id] = record
        if record["status"] == UNSENT_STATUS:
            metadata_fields = (
                "latest_arxiv_id",
                "title",
                "authors",
                "published",
                "updated",
                "summary",
                "pdf_url",
                "source",
            )
            metadata_changed = existing is None or any(
                existing.get(field) != record.get(field) for field in metadata_fields
            )
            missing_affiliations = not record.get("affiliations")
            if (metadata_changed or missing_affiliations) and base_id not in changed_unsent_seen:
                changed_unsent_ids.append(base_id)
                changed_unsent_seen.add(base_id)

    return changed_unsent_ids


def set_affiliations(state: dict, affiliations_by_base_id: dict[str, str]) -> None:
    state_papers = state.setdefault("papers", {})
    for base_id, affiliations in affiliations_by_base_id.items():
        if base_id in state_papers:
            state_papers[base_id]["affiliations"] = affiliations or "Not specified"


def unsent_records(state: dict) -> list[dict]:
    return [
        record
        for record in state.get("papers", {}).values()
        if record.get("status", UNSENT_STATUS) != SENT_STATUS
    ]


def records_to_papers(records: Iterable[dict]) -> tuple[list[Paper], dict[str, str]]:
    papers: list[Paper] = []
    paper_id_to_base_id: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        paper_id = f"B{index:03d}"
        paper_id_to_base_id[paper_id] = record["base_arxiv_id"]
        papers.append(
            Paper(
                paper_id=paper_id,
                arxiv_id=record["latest_arxiv_id"],
                title=record["title"],
                authors=list(record.get("authors", [])),
                published=record.get("published", ""),
                updated=record.get("updated", ""),
                summary=record.get("summary", ""),
                pdf_url=record.get("pdf_url", ""),
            )
        )
    return papers, paper_id_to_base_id


def mark_sent(state: dict, base_id: str, sent_at: str, run_id: str) -> None:
    record = state.setdefault("papers", {}).get(base_id)
    if not record:
        raise KeyError(f"Paper not found in state: {base_id}")
    record["status"] = SENT_STATUS
    record["sent_at"] = sent_at
    record["sent_run_id"] = run_id
