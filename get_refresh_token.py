#!/usr/bin/env python3
"""Run this once, locally, to mint a Google Drive OAuth refresh token.

This authorizes the app to act as *your* Google account (so files it
creates count against your own Drive storage quota, unlike a service
account which has none). You'll need an OAuth 2.0 Client ID of type
"Desktop app" from the same Google Cloud project that has the Drive API
enabled - create one at https://console.cloud.google.com/apis/credentials.

Usage:
    pip install google-auth-oauthlib
    python get_refresh_token.py

A browser window opens for you to sign in and grant Drive access. The
resulting refresh token is printed at the end - set it as
GOOGLE_OAUTH_REFRESH_TOKEN (alongside GOOGLE_OAUTH_CLIENT_ID and
GOOGLE_OAUTH_CLIENT_SECRET) wherever the app runs. This script never
sends your credentials anywhere but Google's own OAuth servers.
"""

import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]


def main() -> int:
    client_id = input("OAuth client ID: ").strip()
    client_secret = input("OAuth client secret: ").strip()

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    # access_type=offline requests a refresh token at all; prompt=consent
    # forces Google to re-issue one even if you've authorized this app
    # before (by default a repeat authorization returns none).
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

    if not creds.refresh_token:
        print(
            "\nNo refresh token was returned. Revoke this app's access at "
            "https://myaccount.google.com/permissions and run this script again.",
            file=sys.stderr,
        )
        return 1

    print("\nSuccess. Set these environment variables wherever the app runs:\n")
    print(f"GOOGLE_OAUTH_CLIENT_ID={client_id}")
    print(f"GOOGLE_OAUTH_CLIENT_SECRET={client_secret}")
    print(f"GOOGLE_OAUTH_REFRESH_TOKEN={creds.refresh_token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
