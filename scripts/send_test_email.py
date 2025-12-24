from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


def main() -> None:
    _ensure_repo_on_path()
    from ir_arxiv_ranker.emailer import send_email

    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail_address or not gmail_password:
        raise SystemExit("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set")
    to_addr = gmail_address

    subject = os.getenv("TEST_EMAIL_SUBJECT", "test")
    body = os.getenv("TEST_EMAIL_BODY", "test")
    send_email(gmail_address, gmail_password, to_addr, subject, body)


if __name__ == "__main__":
    main()
