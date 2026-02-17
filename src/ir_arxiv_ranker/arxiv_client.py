from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from typing import List
from urllib.parse import urlencode

import feedparser
import httpx

from .models import Paper

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_REQUEST_INTERVAL = 3   # seconds between requests (arXiv policy)
MAX_RETRIES = 5
INITIAL_BACKOFF = 5          # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


def _arxiv_get(url: str, timeout: int) -> httpx.Response:
    """HTTP GET with inter-request delay and retry/backoff for rate limits and timeouts."""
    time.sleep(ARXIV_REQUEST_INTERVAL)
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.get(url, timeout=timeout)
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise
            delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "arXiv request failed (%s, attempt %d/%d), retrying in %.1fs",
                type(exc).__name__, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue
        if response.status_code not in RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            return response
        if attempt == MAX_RETRIES:
            response.raise_for_status()  # raise on final failure
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            delay = float(retry_after)
        else:
            delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
        logger.warning(
            "arXiv returned %s (attempt %d/%d), retrying in %.1fs",
            response.status_code, attempt + 1, MAX_RETRIES, delay,
        )
        time.sleep(delay)
    # Unreachable, but keeps type checkers happy
    raise last_exc  # type: ignore[misc]


def _build_query(search_query: str, limit: int, sort_by: str = "submittedDate") -> str:
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": limit,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    return f"{ARXIV_API_URL}?{urlencode(params)}"


def _extract_arxiv_id(entry_id: str) -> str:
    return entry_id.rsplit("/", 1)[-1]


def _extract_pdf_url(entry, arxiv_id: str) -> str:
    for link in getattr(entry, "links", []):
        if getattr(link, "type", "") == "application/pdf":
            return link.href
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _parse_arxiv_datetime(value: str) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def fetch_recent_papers(
    category: str,
    limit: int,
    timeout: int = 60,
    id_prefix: str = "P",
    sort_by: str = "submittedDate",
    updated_after: datetime | None = None,
) -> List[Paper]:
    url = _build_query(f"cat:{category}", limit, sort_by=sort_by)
    response = _arxiv_get(url, timeout)

    feed = feedparser.parse(response.text)
    papers: List[Paper] = []

    for entry in feed.entries:
        arxiv_id = _extract_arxiv_id(entry.id)
        title = " ".join(entry.title.split())
        authors = [author.name for author in getattr(entry, "authors", [])]
        published = getattr(entry, "published", "")
        updated = getattr(entry, "updated", "")
        summary = " ".join(getattr(entry, "summary", "").split())
        pdf_url = _extract_pdf_url(entry, arxiv_id)

        if updated_after:
            updated_dt = _parse_arxiv_datetime(updated) or _parse_arxiv_datetime(published)
            if updated_dt and updated_dt <= updated_after:
                continue

        paper_id = f"{id_prefix}{len(papers) + 1:03d}"

        papers.append(
            Paper(
                paper_id=paper_id,
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                published=published,
                updated=updated,
                summary=summary,
                pdf_url=pdf_url,
            )
        )

    return papers


def _sanitize_keyword(term: str) -> str:
    return term.replace('"', "").strip()


def fetch_keyword_papers(
    keywords: List[str],
    limit: int,
    timeout: int = 60,
    id_prefix: str = "OTH",
    exclude_categories: List[str] | None = None,
    sort_by: str = "submittedDate",
    updated_after: datetime | None = None,
) -> List[Paper]:
    terms = [_sanitize_keyword(term) for term in keywords if term.strip()]
    query_terms = [f'all:"{term}"' for term in terms]
    search_query = " OR ".join(query_terms)
    if exclude_categories:
        exclusion_query = " OR ".join(f"cat:{cat}" for cat in exclude_categories)
        search_query = f"({search_query}) ANDNOT ({exclusion_query})"
    url = _build_query(search_query, limit, sort_by=sort_by)
    response = _arxiv_get(url, timeout)

    feed = feedparser.parse(response.text)
    papers: List[Paper] = []

    for entry in feed.entries:
        arxiv_id = _extract_arxiv_id(entry.id)
        title = " ".join(entry.title.split())
        authors = [author.name for author in getattr(entry, "authors", [])]
        published = getattr(entry, "published", "")
        updated = getattr(entry, "updated", "")
        summary = " ".join(getattr(entry, "summary", "").split())
        pdf_url = _extract_pdf_url(entry, arxiv_id)

        if updated_after:
            updated_dt = _parse_arxiv_datetime(updated) or _parse_arxiv_datetime(published)
            if updated_dt and updated_dt <= updated_after:
                continue

        paper_id = f"{id_prefix}{len(papers) + 1:03d}"

        papers.append(
            Paper(
                paper_id=paper_id,
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                published=published,
                updated=updated,
                summary=summary,
                pdf_url=pdf_url,
            )
        )

    return papers
