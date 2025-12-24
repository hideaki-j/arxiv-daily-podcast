from __future__ import annotations


def _normalize_title(title: str) -> list[str]:
    cleaned = []
    for char in title:
        if char.isascii() and char.isalnum():
            cleaned.append(char.lower())
        else:
            cleaned.append(" ")
    words = "".join(cleaned).split()
    return words


def build_file_stem(rank: int, paper_id: str, title: str, max_words: int = 5) -> str:
    words = _normalize_title(title)[:max_words]
    slug = "_".join(words) if words else "paper"
    return f"{rank}_{paper_id}_{slug}"
