#!/usr/bin/env python3
"""
Raspberry Pi Gmail Monitor - Polling Method
Continuously monitors Gmail for new emails
"""

import os
import pickle
import time
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CLIENT_ID = '132594459985-t6gve02973gl5h2f1og12qcvgf22efv3.apps.googleusercontent.com'
CLIENT_SECRET = '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'

# Configuration
CHECK_INTERVAL = 30  # Check every 30 seconds
LAST_CHECK_FILE = 'last_check.txt'

def get_gmail_service():
    """Authenticates and returns Gmail API service object"""
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_config = {
                "installed": {
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": ["http://localhost"]
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def get_last_check_time():
    """Get the last time we checked for emails"""
    if os.path.exists(LAST_CHECK_FILE):
        with open(LAST_CHECK_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_last_check_time():
    """Save current time as last check time"""
    with open(LAST_CHECK_FILE, 'w') as f:
        f.write(str(int(time.time())))

def get_message_details(service, msg_id):
    """Get full message details"""
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()

        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')

        # Get message body
        body = ""
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
        else:
            if 'body' in message['payload'] and 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')

        return {
            'id': msg_id,
            'subject': subject,
            'from': sender,
            'date': date,
            'body': body
        }
    except Exception as e:
        print(f"Error getting message {msg_id}: {str(e)}")
        return None

def process_new_email(email_data):
    """
    Process a new email - CUSTOMIZE THIS FUNCTION
    Add your own logic here for what to do with new emails
    """
    print(f"\n{'='*80}")
    print(f"NEW EMAIL RECEIVED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"From: {email_data['from']}")
    print(f"Subject: {email_data['subject']}")
    print(f"Date: {email_data['date']}")
    print(f"\nBody Preview:")
    print(email_data['body'][:300] + '...' if len(email_data['body']) > 300 else email_data['body'])
    print(f"{'='*80}\n")

    # Add your custom logic here:
    # - Save to database
    # - Send notifications
    # - Trigger other actions
    # - Parse specific content
    # etc.

def check_for_new_emails(service):
    """Check for new emails since last check"""
    try:
        last_check = get_last_check_time()

        # Query for recent emails (last hour to be safe)
        query = f'after:{int(time.time()) - 3600}'

        results = service.users().messages().list(
            userId='me',
            maxResults=50,
            q=query
        ).execute()

        messages = results.get('messages', [])

        new_emails = []
        for message in messages:
            # Get internal date
            msg = service.users().messages().get(userId='me', id=message['id'], format='minimal').execute()
            msg_time = int(msg['internalDate']) // 1000  # Convert to seconds

            if msg_time > last_check:
                details = get_message_details(service, message['id'])
                if details:
                    new_emails.append(details)

        # Process new emails
        for email in new_emails:
            process_new_email(email)

        if new_emails:
            print(f"Processed {len(new_emails)} new email(s)")

        # Update last check time
        save_last_check_time()

        return len(new_emails)

    except HttpError as error:
        print(f"An error occurred: {error}")
        return 0

def monitor_loop():
    """Main monitoring loop"""
    print("Starting Gmail monitor...")
    print(f"Checking every {CHECK_INTERVAL} seconds")
    print("Press Ctrl+C to stop\n")

    service = get_gmail_service()

    # Initialize last check time if first run
    if not os.path.exists(LAST_CHECK_FILE):
        save_last_check_time()
        print("Initialized. Future checks will detect new emails.\n")

    try:
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Checking for new emails...")

            new_count = check_for_new_emails(service)

            if new_count == 0:
                print(f"[{timestamp}] No new emails\n")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError in monitoring loop: {str(e)}")
        print("Restarting in 60 seconds...")
        time.sleep(60)
        monitor_loop()  # Restart on error

if __name__ == '__main__':
    monitor_loop()
