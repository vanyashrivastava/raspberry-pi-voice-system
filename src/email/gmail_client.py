"""
Simple Gmail client using OAuth2 to fetch latest messages.

Place your Google OAuth `credentials.json` (OAuth client ID) in the
same folder as this file or set the `GOOGLE_CREDENTIALS` env var to the path.
This module writes/reads `token.json` to persist credentials after first login.

Usage: call `fetch_latest_messages(n)` to get the latest `n` messages.
"""
from __future__ import print_function
import os
import json
import webbrowser
from typing import List, Dict

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def _credentials_paths():
    # prefer env var, else look in the same directory
    file_dir = os.path.dirname(__file__)
    cred_env = os.environ.get('GOOGLE_CREDENTIALS')
    cred_path = cred_env or os.path.join(file_dir, 'credentials.json')
    token_path = os.path.join(file_dir, 'token.json')
    # If an env var was provided but the file doesn't exist, try a few common
    # filename patterns in the same directory (e.g., client_secret_*.json).
    if cred_path and not os.path.exists(cred_path):
        # look for candidate files
        import glob
        patterns = [
            os.path.join(file_dir, 'client_secret_*.json'),
            os.path.join(file_dir, 'client_secret-*.json'),
            os.path.join(file_dir, 'client_secret.json'),
            os.path.join(file_dir, 'credentials.json'),
            os.path.join(file_dir, '*.credentials.json'),
            os.path.join(file_dir, '*.json'),
        ]
        candidates = []
        for p in patterns:
            candidates.extend(glob.glob(p))
        # prefer credentials.json if present
        if os.path.exists(os.path.join(file_dir, 'credentials.json')):
            return os.path.join(file_dir, 'credentials.json'), token_path
        if candidates:
            # choose the first reasonable candidate
            return candidates[0], token_path
        # fall through to returning original (missing) path
    return cred_path, token_path


def _console_oauth_flow(cred_path: str, token_path: str):
    """Manual console-based OAuth flow for headless environments.
    
    Prints authorization URL, waits for user to paste the auth code,
    then exchanges it for credentials.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        import google.auth.oauthlib.flow
    except Exception as e:
        raise RuntimeError(
            "Missing Google API libraries. Install: \n"
            "pip install google-auth google-auth-oauthlib google-api-python-client"
        ) from e

    # Load client secrets
    with open(cred_path, 'r') as f:
        client_config = json.load(f)

    # Create flow manually (not using InstalledAppFlow which expects localhost)
    flow = google.auth.oauthlib.flow.Flow.from_client_config(
        client_config, 
        scopes=SCOPES,
        redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # out-of-band (console) mode
    )

    # Generate authorization URL
    auth_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )

    print("\n" + "="*70)
    print("AUTHORIZATION REQUIRED")
    print("="*70)
    print("\nOpen this URL in your browser and authorize access:\n")
    print(auth_url)
    print("\n" + "="*70)
    
    # Try to open browser automatically
    try:
        webbrowser.open(auth_url)
        print("Browser should open automatically. If not, copy the URL above.")
    except Exception:
        print("Could not open browser automatically.")
    
    print("\nAfter authorizing, you will see a code. Copy it and paste below.\n")
    
    # Wait for authorization code
    auth_code = input("Paste authorization code here: ").strip()
    
    if not auth_code:
        raise ValueError("No authorization code provided")
    
    # Exchange code for credentials
    flow.fetch_token(code=auth_code)
    creds = flow.credentials
    
    # Save credentials
    with open(token_path, 'w') as token_file:
        token_file.write(creds.to_json())
    
    print("✓ Authorization successful. Token saved.\n")
    return creds


def get_gmail_service(use_console=False):
    """Return an authorized Gmail API service object.

    Requires `google-auth-oauthlib` and `google-api-python-client` packages.
    
    Args:
        use_console: If True, use console-based OAuth flow (no localhost needed).
                     Useful for headless/dev container environments.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except Exception as e:
        raise RuntimeError(
            "Missing Google API libraries. Install: \n"
            "pip install google-auth google-auth-oauthlib google-api-python-client"
        ) from e

    cred_path, token_path = _credentials_paths()

    creds = None
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            if not os.path.exists(cred_path):
                raise FileNotFoundError(
                    f"Google credentials not found. Put your OAuth client file at {cred_path} "
                    "or set GOOGLE_CREDENTIALS environment variable."
                )
            
            # Use console-based flow
            if use_console:
                creds = _console_oauth_flow(cred_path, token_path)
            else:
                # Try local server first; fall back to console if it fails
                try:
                    from google_auth_oauthlib.flow import InstalledAppFlow
                    flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                    # Save the credentials for the next run
                    with open(token_path, 'w') as token_file:
                        token_file.write(creds.to_json())
                except Exception as e:
                    print(f"\nLocal server OAuth failed ({e}). Falling back to console mode...")
                    creds = _console_oauth_flow(cred_path, token_path)

    service = build('gmail', 'v1', credentials=creds)
    return service


def fetch_latest_messages(n: int = 3) -> List[Dict]:
    """Fetch latest `n` messages from the user's mailbox and return simple dicts.

    Each dict contains: `id`, `from`, `subject`, `date`, `snippet`.
    """
    service = get_gmail_service(use_console=True)

    results = service.users().messages().list(userId='me', maxResults=n).execute()
    messages = results.get('messages', [])

    out = []
    for m in messages:
        msg = service.users().messages().get(userId='me', id=m['id'], format='metadata',
                                             metadataHeaders=['From', 'Subject', 'Date']).execute()
        headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
        snippet = msg.get('snippet', '')
        out.append({
            'id': msg.get('id'),
            'from': headers.get('From', ''),
            'subject': headers.get('Subject', '(No Subject)'),
            'date': headers.get('Date', ''),
            'snippet': snippet,
        })

    return out


if __name__ == '__main__':
    # small demo when run directly
    try:
        msgs = fetch_latest_messages(3)
        print('\nLatest messages:')
        for i, m in enumerate(msgs, 1):
            print(f"\n[{i}] From: {m['from']}")
            print(f"    Subject: {m['subject']}")
            print(f"    Date: {m['date']}")
            print(f"    Snippet: {m['snippet'][:140]}")
    except Exception as e:
        print('Error fetching messages:', e)

