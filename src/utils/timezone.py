"""Timezone utilities for converting UTC to local time."""

from datetime import datetime
from zoneinfo import ZoneInfo

TORONTO_TZ = ZoneInfo("America/Toronto")


def utc_to_toronto(dt: datetime) -> datetime:
    """Convert a UTC datetime to Toronto time (handles EST/EDT automatically).

    Args:
        dt: A datetime object. If naive, it's assumed to be UTC.
            If aware, it will be converted from its timezone to Toronto.

    Returns:
        A timezone-aware datetime in America/Toronto timezone.
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(TORONTO_TZ)


def format_toronto_time(dt: datetime, fmt: str = "%Y-%m-%d %H:%M %Z") -> str:
    """Convert a UTC datetime to Toronto time and format as string.

    Args:
        dt: A datetime object (UTC or timezone-aware).
        fmt: strftime format string. Default includes timezone abbreviation.

    Returns:
        Formatted string in Toronto time (e.g., "2025-12-29 08:00 EST").
    """
    toronto_dt = utc_to_toronto(dt)
    return toronto_dt.strftime(fmt)
