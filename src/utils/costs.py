from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class CostTracker:
    total_usd: float = 0.0
    has_unknown: bool = False
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def add(self, cost: float | None) -> None:
        with self._lock:
            if cost is None:
                self.has_unknown = True
                return
            self.total_usd += cost

    def total_cents(self) -> float:
        with self._lock:
            return self.total_usd * 100.0


@dataclass(frozen=True)
class CostEntry:
    label: str
    cost_cents: float | None
    detail: str


@dataclass
class CostReport:
    entries: list[CostEntry] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def add(self, label: str, cost_usd: float | None, detail: str) -> None:
        cost_cents = None if cost_usd is None else cost_usd * 100.0
        with self._lock:
            self.entries.append(CostEntry(label=label, cost_cents=cost_cents, detail=detail))

    def render_psql(self, title: str | None = None) -> str:
        with self._lock:
            rows = [
                (
                    entry.label,
                    f"{entry.cost_cents:.2f}¢" if entry.cost_cents is not None else "?",
                    entry.detail,
                )
                for entry in self.entries
            ]
        headers = ("label", "cost", "detail")
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
            for i in range(3)
        ]
        border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
        lines = [border]
        header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(3)) + " |"
        lines.append(header_line)
        lines.append(border)
        for row in rows:
            lines.append(
                "| "
                + " | ".join(row[i].ljust(widths[i]) for i in range(3))
                + " |"
            )
        lines.append(border)
        table = "\n".join(lines)
        if title:
            return f"{title}\n{table}"
        return table
