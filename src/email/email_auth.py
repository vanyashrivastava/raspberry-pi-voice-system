"""
Simple local username/password wrapper that gates access to the Gmail fetcher.

Usage:
  - Place your Google OAuth `credentials.json` in this folder (or set `GOOGLE_CREDENTIALS`).
  - Run: `python src/email/email_auth.py`

This script stores a minimal user database in `users.json` (PBKDF2-hashed passwords).
On successful login, it will start the OAuth flow (if necessary) and print the latest 3 emails.
"""
import os
import json
import getpass
import hashlib
import hmac
from typing import Dict

USER_DB = os.path.join(os.path.dirname(__file__), 'users.json')


def _load_users() -> Dict[str, Dict]:
    if not os.path.exists(USER_DB):
        return {}
    try:
        with open(USER_DB, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(data: Dict[str, Dict]):
    with open(USER_DB, 'w') as f:
        json.dump(data, f, indent=2)


def _hash_password(password: str, salt: bytes = None) -> Dict:
    if salt is None:
        salt = os.urandom(16)
    iterations = 100_000
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return {'salt': salt.hex(), 'iterations': iterations, 'hash': dk.hex()}


def _verify_password(stored: Dict, password: str) -> bool:
    salt = bytes.fromhex(stored['salt'])
    iters = stored.get('iterations', 100_000)
    expected = bytes.fromhex(stored['hash'])
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iters)
    return hmac.compare_digest(dk, expected)


def register_user(username: str, password: str) -> bool:
    users = _load_users()
    if username in users:
        return False
    users[username] = _hash_password(password)
    _save_users(users)
    return True


def authenticate_user(username: str, password: str) -> bool:
    users = _load_users()
    stored = users.get(username)
    if not stored:
        return False
    return _verify_password(stored, password)


def prompt_register():
    print('--- Register new user ---')
    username = input('Username: ').strip()
    if not username:
        print('Username cannot be empty')
        return
    pw1 = getpass.getpass('Password: ')
    pw2 = getpass.getpass('Confirm: ')
    if pw1 != pw2:
        print('Passwords do not match')
        return
    if register_user(username, pw1):
        print('User registered. You can now login.')
    else:
        print('User already exists.')


def prompt_login():
    print('--- Login ---')
    username = input('Username: ').strip()
    pw = getpass.getpass('Password: ')
    ok = authenticate_user(username, pw)
    if not ok:
        print('Authentication failed')
        return False
    print('Authentication successful')
    return True


def google_login_and_fetch(use_console=True):
    try:
        try:
            import gmail_client as gc
        except Exception:
            from src.email import gmail_client as gc

        # ensure service works and retrieve profile email
        service = gc.get_gmail_service(use_console=use_console)
        profile = service.users().getProfile(userId='me').execute()
        acct = profile.get('emailAddress', 'me')
        print(f"Authenticated as: {acct}")
        msgs = gc.fetch_latest_messages(3)
        print('\nLatest 3 emails:')
        for i, m in enumerate(msgs, 1):
            print(f"\n[{i}] From: {m['from']}")
            print(f"    Subject: {m['subject']}")
            print(f"    Date: {m['date']}")
            print(f"    Snippet: {m['snippet'][:200]}")
    except Exception as e:
        print('Google login / fetch failed:', e)


def main():
    print('Simple local auth wrapper for Gmail fetcher')
    while True:
        print('\nOptions: [L]ogin  [R]egister  [G]oogle Login  [Q]uit')
        choice = input('> ').strip().lower()
        if choice in ('q', 'quit'):
            print('Goodbye')
            return
        if choice in ('r', 'register'):
            prompt_register()
            continue
        if choice in ('g', 'google'):
            google_login_and_fetch(use_console=True)
            return
        if choice in ('l', 'login'):
            if prompt_login():
                # proceed to fetch emails using gmail client
                try:
                    google_login_and_fetch(use_console=True)
                except Exception as e:
                    print('Failed to fetch emails after login:', e)
                return
        print('Unknown option')


if __name__ == '__main__':
    main()
