"""
Google OAuth2 helper — reusable across any Google API project.

Each project keeps its own credentials.json and token.json locally (never committed).
This module provides only the auth flow.

Setup (one-time per project):
    1. console.cloud.google.com → enable the APIs you need
    2. Credentials → Create → OAuth client ID → Desktop app → download JSON
    3. Save to credentials_path (never commit this file)
    4. Run once — browser opens for authorization, token cached to token_path automatically

Scopes reference:
    gmail.readonly          read emails
    gmail.compose           draft / send
    calendar.events         add / edit calendar events
    drive.readonly          read Drive files
    spreadsheets.readonly   read Sheets
"""

import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


def get_creds(
    scopes: list[str],
    token_path: Path,
    credentials_path: Path,
) -> Credentials:
    """
    Return valid Google credentials, refreshing or re-authorizing as needed.

    token_path: where to cache the access/refresh token (created on first run)
    credentials_path: the OAuth client secret JSON from Google Cloud Console
    """
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                print(f"Google credentials not found: {credentials_path}")
                print("Download an OAuth client secret (Desktop app) from console.cloud.google.com")
                print("and save it to that path.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes)
            creds = flow.run_local_server(host="127.0.0.1", port=8080)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())

    return creds
