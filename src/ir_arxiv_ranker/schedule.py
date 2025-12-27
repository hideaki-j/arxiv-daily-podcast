from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import yaml


_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_WEEKDAY_NAMES = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}


@dataclass(frozen=True)
class CronSpec:
    minutes: set[int]
    hours: set[int]
    days: set[int]
    months: set[int]
    weekdays: set[int]
    dom_is_all: bool
    dow_is_all: bool


def _parse_value(token: str, name_map: dict[str, int], min_value: int, max_value: int) -> int:
    key = token.lower()
    if key in name_map:
        return name_map[key]
    value = int(token)
    if value < min_value or value > max_value:
        raise ValueError(f"Value {value} outside range {min_value}-{max_value}")
    return value


def _parse_field(
    field: str, min_value: int, max_value: int, name_map: dict[str, int] | None = None
) -> tuple[set[int], bool]:
    field = field.strip()
    if field == "*":
        return set(range(min_value, max_value + 1)), True

    allowed: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            range_part, step_part = part.split("/", 1)
            step = int(step_part)
            if step < 1:
                raise ValueError("Step must be >= 1")
        else:
            range_part = part

        if range_part == "*":
            start, end = min_value, max_value
        elif "-" in range_part:
            start_str, end_str = range_part.split("-", 1)
            start = _parse_value(start_str, name_map or {}, min_value, max_value)
            end = _parse_value(end_str, name_map or {}, min_value, max_value)
        else:
            start = end = _parse_value(range_part, name_map or {}, min_value, max_value)

        if start > end:
            raise ValueError(f"Invalid range {start}-{end}")

        for value in range(start, end + 1, step):
            allowed.add(value)

    if not allowed:
        raise ValueError(f"No allowed values parsed from field '{field}'")
    return allowed, False


def parse_cron_expression(expr: str) -> CronSpec:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError("Cron expression must have 5 fields")

    minute_f, hour_f, dom_f, month_f, dow_f = parts
    minutes, _ = _parse_field(minute_f, 0, 59)
    hours, _ = _parse_field(hour_f, 0, 23)
    days, dom_is_all = _parse_field(dom_f, 1, 31)
    months, _ = _parse_field(month_f, 1, 12, _MONTH_NAMES)
    weekdays_raw, dow_is_all = _parse_field(dow_f, 0, 7, _WEEKDAY_NAMES)
    weekdays = {(value % 7) for value in weekdays_raw}

    return CronSpec(
        minutes=minutes,
        hours=hours,
        days=days,
        months=months,
        weekdays=weekdays,
        dom_is_all=dom_is_all,
        dow_is_all=dow_is_all,
    )


def _cron_matches(spec: CronSpec, dt: datetime) -> bool:
    if (
        dt.minute not in spec.minutes
        or dt.hour not in spec.hours
        or dt.month not in spec.months
    ):
        return False

    cron_weekday = (dt.weekday() + 1) % 7  # Convert Python weekday (Mon=0) to cron (Sun=0)
    dom_match = spec.dom_is_all or dt.day in spec.days
    dow_match = spec.dow_is_all or cron_weekday in spec.weekdays
    if spec.dom_is_all or spec.dow_is_all:
        return dom_match and dow_match
    return dom_match or dow_match


def previous_schedule_hit(
    spec: CronSpec, now: datetime, lookback_days: int = 30
) -> datetime | None:
    current = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
    cutoff = current - timedelta(days=lookback_days)
    while current >= cutoff:
        if _cron_matches(spec, current):
            return current
        current -= timedelta(minutes=1)
    return None


def last_scheduled_run(
    cron_expressions: Iterable[str], now: datetime | None = None, lookback_days: int = 30
) -> datetime | None:
    now = now or datetime.now(timezone.utc)
    candidates: list[datetime] = []
    for expr in cron_expressions:
        try:
            spec = parse_cron_expression(expr)
        except ValueError:
            continue
        hit = previous_schedule_hit(spec, now, lookback_days=lookback_days)
        if hit:
            candidates.append(hit)
    if not candidates:
        return None
    return max(candidates)


def load_workflow_cron_schedules(workflow_path: Path) -> list[str]:
    if not workflow_path.exists():
        return []
    try:
        workflow = yaml.safe_load(workflow_path.read_text()) or {}
    except Exception:
        return []

    on_section = workflow.get("on", {})
    if not isinstance(on_section, dict):
        return []
    schedule_entries = on_section.get("schedule", [])
    if not isinstance(schedule_entries, list):
        return []

    crons: list[str] = []
    for entry in schedule_entries:
        if isinstance(entry, dict):
            cron_value = entry.get("cron")
            if isinstance(cron_value, str) and cron_value.strip():
                crons.append(cron_value.strip())
    return crons
