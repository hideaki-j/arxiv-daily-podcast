"""Deterministic, conservative matching of registered author names."""
from __future__ import annotations

import unicodedata
from collections.abc import Iterable


def normalize_author_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name).casefold()
    # Ignore accents and punctuation, but retain word boundaries. Never use
    # substrings, surnames alone, or LLM guesses to authorize delivery.
    letters = "".join(
        c for c in decomposed
        if not unicodedata.combining(c) and (c.isalnum() or c.isspace())
    )
    return " ".join(letters.split())


def matched_priority_authors(authors: Iterable[str], priority_authors: Iterable[str]) -> list[str]:
    paper_names = {normalize_author_name(name) for name in authors}
    return list(dict.fromkeys(
        name for name in priority_authors
        if normalize_author_name(name) and normalize_author_name(name) in paper_names
    ))


def author_display_names(authors: Iterable[str], priority_authors: Iterable[str]) -> list[dict]:
    targets = list(priority_authors)
    return [
        {"name": name, "matched": bool(matched_priority_authors([name], targets))}
        for name in authors
    ]
