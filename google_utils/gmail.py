"""
Generic Gmail retrieval utilities.

Works for any use case: submission responses, Evite invitations, Paperless Post,
calendar-related emails, etc. Nothing here is domain-specific.

Usage:
    from google_utils import get_service, build_query, fetch_messages, message_text
    from google_utils.auth import get_creds

    service = get_service(token_path=..., credentials_path=...)
    msgs = fetch_messages(service, build_query(domains=["evite.com"], after_date="2026-01-01"))
    for m in msgs:
        subject, body = message_text(service, m["id"])
"""

import base64
import re
from pathlib import Path

from googleapiclient.discovery import build

from .auth import get_creds

GMAIL_READONLY = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_service(
    token_path: Path,
    credentials_path: Path,
    scopes: list[str] = GMAIL_READONLY,
):
    """Return an authenticated Gmail API service."""
    creds = get_creds(scopes=scopes, token_path=token_path, credentials_path=credentials_path)
    return build("gmail", "v1", credentials=creds)


def build_query(
    domains: list[str] | None = None,
    after_date: str | None = None,
    subject_contains: str | None = None,
    extra: str | None = None,
) -> str:
    """
    Build a Gmail search query string.

    domains: sender domains to match (partial match on from: field)
    after_date: ISO date string YYYY-MM-DD — only emails after this date
    subject_contains: keyword that must appear in subject
    extra: any additional raw Gmail search clause to append
    """
    parts = []
    if domains:
        domain_clauses = " OR ".join(f"from:{d}" for d in domains)
        parts.append(f"({domain_clauses})")
    if after_date:
        parts.append(f"after:{after_date.replace('-', '/')}")
    if subject_contains:
        parts.append(f"subject:{subject_contains}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def fetch_messages(service, query: str, max_results: int = 50) -> list[dict]:
    """
    Return a list of message metadata dicts matching the query.
    Each dict has at minimum an 'id' key — pass to message_text() to get content.
    """
    result = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    return result.get("messages", [])


def message_text(service, msg_id: str, body_limit: int = 3000) -> tuple[str, str, str]:
    """
    Fetch a message and return (subject, body, received_date).
    received_date: ISO YYYY-MM-DD from Gmail's internalDate (when the email arrived).
    body_limit: max characters of body to return (keeps LLM calls cheap).
    """
    from datetime import datetime, timezone
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    subject = next(
        (h["value"] for h in msg["payload"].get("headers", []) if h["name"].lower() == "subject"),
        "",
    )
    body = _extract_body(msg["payload"])
    received_date = datetime.fromtimestamp(
        int(msg["internalDate"]) / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d")
    return subject, body[:body_limit], received_date


def send_email(service, to: str, subject: str, body_html: str) -> str:
    """
    Send an HTML email from the authenticated Gmail account.
    Requires gmail.compose or gmail.send scope.
    Returns the sent message ID.
    """
    import base64
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["To"] = to
    msg.attach(MIMEText(body_html, "html"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    return result["id"]


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text from a Gmail message payload."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = _extract_body(part)
            if text:
                return text

    if mime == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
            return re.sub(r"<[^>]+>", " ", html)

    return ""
