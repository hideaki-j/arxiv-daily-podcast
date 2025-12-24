from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
import mimetypes


def _normalize_recipients(to_addr: str) -> list[str]:
    if "," in to_addr:
        return [addr.strip() for addr in to_addr.split(",") if addr.strip()]
    return [to_addr.strip()]


def send_email(
    smtp_user: str,
    smtp_password: str,
    to_addr: str,
    subject: str,
    body: str,
    html_body: str | None = None,
    attachments: list[Path] | None = None,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> None:
    message = EmailMessage()
    message["From"] = smtp_user
    message["To"] = ", ".join(_normalize_recipients(to_addr))
    message["Subject"] = subject
    message.set_content(body)
    if html_body:
        message.add_alternative(html_body, subtype="html")
    if attachments:
        for path in attachments:
            data = path.read_bytes()
            content_type, _ = mimetypes.guess_type(path.name)
            if content_type:
                maintype, subtype = content_type.split("/", 1)
            else:
                maintype, subtype = "application", "octet-stream"
            message.add_attachment(
                data,
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls(context=context)
        server.login(smtp_user, smtp_password)
        server.send_message(message)
