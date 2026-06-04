from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import datetime
from typing import Iterable


def parse_priority_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [name.strip() for name in raw.split(";") if name.strip()]


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower().replace(".", " ")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.split())


def _first_name_compatible(candidate: str, priority: str) -> bool:
    candidate_first = candidate.split()[0]
    priority_first = priority.split()[0]
    if candidate_first == priority_first:
        return True
    if len(candidate_first) == 1 and priority_first.startswith(candidate_first):
        return True
    if len(priority_first) == 1 and candidate_first.startswith(priority_first):
        return True
    return False


def _name_matches(candidate: str, priority: str) -> bool:
    candidate_norm = _normalize_name(candidate)
    priority_norm = _normalize_name(priority)
    if not candidate_norm or not priority_norm:
        return False
    if candidate_norm == priority_norm:
        return True

    candidate_tokens = candidate_norm.split()
    priority_tokens = priority_norm.split()
    if len(candidate_tokens) < 2 or len(priority_tokens) < 2:
        return False
    if candidate_tokens[-1] != priority_tokens[-1]:
        return False
    if not _first_name_compatible(candidate_norm, priority_norm):
        return False

    ratio = difflib.SequenceMatcher(None, candidate_norm, priority_norm).ratio()
    return ratio >= 0.86


def match_priority_authors(authors: Iterable[str], priority_authors: Iterable[str]) -> list[str]:
    matches: list[str] = []
    for priority_author in priority_authors:
        if any(_name_matches(author, priority_author) for author in authors):
            matches.append(priority_author)
    return matches


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def sort_records_by_priority(records: list[dict], priority_authors: list[str]) -> list[dict]:
    def sort_key(record: dict) -> tuple[int, datetime, str]:
        matches = match_priority_authors(record.get("authors", []), priority_authors)
        updated = _parse_dt(record.get("updated", "") or record.get("published", ""))
        return (1 if matches else 0, updated, record.get("base_arxiv_id", ""))

    return sorted(records, key=sort_key, reverse=True)
