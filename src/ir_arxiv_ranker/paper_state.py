from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from .models import Paper


SCHEMA_VERSION = 1
SENT_STATUS = "sent"
POOLED_STATUS = "pooled"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def base_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def load_paper_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "pooled_papers": {}}
    payload = json.loads(path.read_text() or "{}")
    if not isinstance(payload, dict):
        raise ValueError(f"Paper state must be a JSON object: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported paper state schema version in {path}")
    return _normalize_state(payload)


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


def _ranking_input_hash(paper: Paper) -> str:
    payload = {
        "latest_arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "published": paper.published,
        "updated": paper.updated,
        "summary": paper.summary,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _normalize_state(state: dict) -> dict:
    if "sent_papers" in state or "pooled_papers" in state:
        pooled = state.get("pooled_papers", {})
        sent = state.get("sent_papers", {})
        if not isinstance(pooled, dict) or not isinstance(sent, dict):
            raise ValueError("Paper state pooled_papers and sent_papers must be objects")
        merged = {
            base_id: {
                **record,
                "status": POOLED_STATUS,
                "sent": bool(record.get("sent", record.get("status") == SENT_STATUS)),
            }
            for base_id, record in pooled.items()
        }
        for base_id, record in sent.items():
            merged[base_id] = {
                **record,
                "status": POOLED_STATUS,
                "sent": True,
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "pooled_papers": merged,
        }

    legacy_papers = state.get("papers", {})
    if not isinstance(legacy_papers, dict):
        raise ValueError("Paper state must contain papers or sent_papers/pooled_papers objects")

    pooled_papers = {}
    for base_id, record in legacy_papers.items():
        pooled_papers[base_id] = {
            **record,
            "status": POOLED_STATUS,
            "sent": record.get("status") == SENT_STATUS,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "pooled_papers": pooled_papers,
    }


def _paper_record(
    paper: Paper,
    seen_at: str,
    existing: dict | None = None,
    influence_score: int | None = None,
) -> dict:
    previous = existing or {}
    base_id = base_arxiv_id(paper.arxiv_id)
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
        "status": POOLED_STATUS,
        "sent": bool(previous.get("sent", previous.get("status") == SENT_STATUS)),
        "sent_at": previous.get("sent_at"),
        "sent_run_id": previous.get("sent_run_id"),
        "influence_score": (
            influence_score if influence_score is not None else previous.get("influence_score")
        ),
        "ranking_scores": previous.get("ranking_scores", {}),
        "ranking_total_score": previous.get("ranking_total_score"),
        "ranking_tldr": previous.get("ranking_tldr", ""),
        "ranking_input_hash": _ranking_input_hash(paper),
        "ranking_scored_input_hash": previous.get("ranking_scored_input_hash"),
    }
    return record


def merge_discovered_papers(
    state: dict,
    papers: Iterable[Paper],
    scores_by_id: dict[str, int],
    seen_at: str,
) -> list[str]:
    changed_pooled_ids: list[str] = []
    changed_pooled_seen: set[str] = set()
    pooled_papers = state.setdefault("pooled_papers", {})

    for paper in papers:
        base_id = base_arxiv_id(paper.arxiv_id)
        score = scores_by_id.get(paper.paper_id)
        existing = pooled_papers.get(base_id)
        record = _paper_record(
            paper,
            seen_at,
            existing,
            influence_score=score,
        )
        pooled_papers[base_id] = record

        metadata_fields = (
            "latest_arxiv_id",
            "title",
            "authors",
            "published",
            "updated",
            "summary",
            "pdf_url",
            "source",
            "influence_score",
        )
        metadata_changed = existing is None or any(
            existing.get(field) != record.get(field) for field in metadata_fields
        )
        missing_affiliations = not record.get("affiliations")
        if (metadata_changed or missing_affiliations) and base_id not in changed_pooled_seen:
            changed_pooled_ids.append(base_id)
            changed_pooled_seen.add(base_id)

    return changed_pooled_ids


def _record_bucket(state: dict) -> dict:
    state = _normalize_state(state)
    return state["pooled_papers"]


def _target_bucket(state: dict, base_id: str) -> dict | None:
    pooled_papers = _record_bucket(state)
    if base_id in pooled_papers:
        return pooled_papers
    return None


def set_affiliations(state: dict, affiliations_by_base_id: dict[str, str]) -> None:
    for base_id, affiliations in affiliations_by_base_id.items():
        bucket = _target_bucket(state, base_id)
        if bucket is not None:
            bucket[base_id]["affiliations"] = affiliations or "Not specified"


def set_ranking_scores(
    state: dict,
    paper_id_to_base_id: dict[str, str],
    scores_by_id: dict[str, dict[str, int]],
    total_score_by_id: dict[str, float],
    tldr_by_id: dict[str, str] | None = None,
) -> None:
    pooled_papers = state.setdefault("pooled_papers", {})
    for paper_id, base_id in paper_id_to_base_id.items():
        if base_id in pooled_papers:
            record = pooled_papers[base_id]
            record["ranking_scores"] = scores_by_id.get(paper_id, {})
            record["ranking_total_score"] = total_score_by_id.get(paper_id)
            if tldr_by_id is not None:
                record["ranking_tldr"] = tldr_by_id.get(paper_id, "")
            record["ranking_scored_input_hash"] = record.get("ranking_input_hash")


def needs_ranking_score(record: dict, aspect_keys: Iterable[str]) -> bool:
    scores = record.get("ranking_scores")
    if not isinstance(scores, dict):
        return True
    if any(not isinstance(scores.get(key), int) for key in aspect_keys):
        return True
    if record.get("ranking_total_score") is None:
        return True
    return record.get("ranking_scored_input_hash") != record.get("ranking_input_hash")


def pooled_records(state: dict, include_sent: bool = False) -> list[dict]:
    records = [
        record
        for record in state.get("pooled_papers", {}).values()
        if include_sent or not record.get("sent")
    ]

    def sort_key(record: dict) -> tuple[float, str]:
        return (
            float(record.get("ranking_total_score") or 0),
            record.get("base_arxiv_id", ""),
        )

    return sorted(records, key=sort_key, reverse=True)


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
    pooled_papers = state.setdefault("pooled_papers", {})
    record = pooled_papers.get(base_id)
    if not record:
        raise KeyError(f"Paper not found in state: {base_id}")
    record["status"] = POOLED_STATUS
    record["sent"] = True
    record["sent_at"] = sent_at
    record["sent_run_id"] = run_id
