"""
SMTP email sending — no OAuth, no token expiry.

Usage:
    from smtp_utils import send_email

    send_email(
        to="you@example.com",
        subject="Hello",
        body_html="<p>Hello</p>",
        from_addr="you@gmail.com",
        password_path=Path("~/.credentials/gmail_smtp_password").expanduser(),
    )

Setup (one-time):
    1. myaccount.google.com → Security → 2-Step Verification → App passwords
    2. Generate a password for "Mail"
    3. Save the 16-char password (no spaces) to password_path
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587

ICLOUD_SMTP_HOST = "smtp.mail.me.com"
ICLOUD_SMTP_PORT = 587


def send_email(
    to: str,
    subject: str,
    body_html: str,
    from_addr: str,
    password_path: Path,
    smtp_host: str = GMAIL_SMTP_HOST,
    smtp_port: int = GMAIL_SMTP_PORT,
) -> None:
    """Send an HTML email via SMTP using an app password."""
    password = password_path.read_text().strip()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    msg.attach(MIMEText(body_html, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(from_addr, password)
        server.send_message(msg)
