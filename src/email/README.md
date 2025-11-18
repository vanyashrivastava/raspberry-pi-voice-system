Gmail Fetcher + Local Auth
==========================

This folder contains a minimal example to authenticate a local user and
then fetch the latest 3 Gmail messages via OAuth.

Files:
- `gmail_client.py` - handles Google OAuth and fetching messages.
- `email_auth.py` - simple local username/password register & login, then calls `gmail_client`.

Setup:
1. Install dependencies (from project root):

```bash
pip install -r requirements.txt
```

2. Create OAuth credentials for a Desktop app in Google Cloud Console and
   download the `credentials.json` file. Place it next to these files or set
   the `GOOGLE_CREDENTIALS` env var to point to it.

3. Run the script:

```bash
python src/email/email_auth.py
```

Follow prompts to register/login. On first OAuth run a browser window will
open to authorize access; a `token.json` file will be written to persist
credentials for future runs.

Security note:
- This is a simple demo. The local user store uses PBKDF2 hashing but is
  intended for testing only. Do not use as-is for production.
