from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Paper:
    paper_id: str
    arxiv_id: str
    title: str
    authors: List[str]
    published: str
    updated: str
    summary: str
    pdf_url: str

    def prompt_dict(self) -> dict:
        return {
            "id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "updated": self.updated,
            "summary": self.summary,
        }
