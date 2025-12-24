from __future__ import annotations

from typing import List
from urllib.parse import urlencode

import feedparser
import httpx

from .models import Paper


ARXIV_API_URL = "https://export.arxiv.org/api/query"


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


def fetch_recent_papers(
    category: str,
    limit: int,
    timeout: int = 30,
    id_prefix: str = "P",
    sort_by: str = "submittedDate",
) -> List[Paper]:
    url = _build_query(f"cat:{category}", limit, sort_by=sort_by)
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()

    feed = feedparser.parse(response.text)
    papers: List[Paper] = []

    for idx, entry in enumerate(feed.entries, start=1):
        arxiv_id = _extract_arxiv_id(entry.id)
        paper_id = f"{id_prefix}{idx:03d}"
        title = " ".join(entry.title.split())
        authors = [author.name for author in getattr(entry, "authors", [])]
        published = getattr(entry, "published", "")
        updated = getattr(entry, "updated", "")
        summary = " ".join(getattr(entry, "summary", "").split())
        pdf_url = _extract_pdf_url(entry, arxiv_id)

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
    timeout: int = 30,
    id_prefix: str = "OTH",
    exclude_categories: List[str] | None = None,
    sort_by: str = "submittedDate",
) -> List[Paper]:
    terms = [_sanitize_keyword(term) for term in keywords if term.strip()]
    query_terms = [f'all:"{term}"' for term in terms]
    search_query = " OR ".join(query_terms)
    if exclude_categories:
        exclusion_query = " OR ".join(f"cat:{cat}" for cat in exclude_categories)
        search_query = f"({search_query}) ANDNOT ({exclusion_query})"
    url = _build_query(search_query, limit, sort_by=sort_by)
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()

    feed = feedparser.parse(response.text)
    papers: List[Paper] = []

    for idx, entry in enumerate(feed.entries, start=1):
        arxiv_id = _extract_arxiv_id(entry.id)
        paper_id = f"{id_prefix}{idx:03d}"
        title = " ".join(entry.title.split())
        authors = [author.name for author in getattr(entry, "authors", [])]
        published = getattr(entry, "published", "")
        updated = getattr(entry, "updated", "")
        summary = " ".join(getattr(entry, "summary", "").split())
        pdf_url = _extract_pdf_url(entry, arxiv_id)

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
