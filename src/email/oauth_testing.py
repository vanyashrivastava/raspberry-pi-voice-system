#!/usr/bin/env python3
"""
Gmail Email Scanner
Scans emails from a Gmail account using Google OAuth 2.0
"""

import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText

# If modifying these scopes, delete the file token.pickle
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# OAuth credentials
CLIENT_ID = '132594459985-cne8lmpvhtjlvjaameqdnanmcqqonum2.apps.googleusercontent.com'
CLIENT_SECRET = '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'

def get_gmail_service():
    """Authenticates and returns Gmail API service object"""
    creds = None
    
    # Token file stores user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Create client config dictionary
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
        
        # Save credentials for next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

def get_message_details(service, msg_id):
    """Get full message details including body"""
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
            'body': body[:500] + '...' if len(body) > 500 else body  # Truncate long bodies
        }
    except Exception as e:
        print(f"Error getting message {msg_id}: {str(e)}")
        return None

def scan_emails(max_results=10, query=''):
    """
    Scan emails from Gmail account
    
    Args:
        max_results: Maximum number of emails to retrieve (default: 10)
        query: Gmail search query (e.g., 'is:unread', 'from:someone@example.com')
    """
    try:
        service = get_gmail_service()
        
        print(f"\n{'='*80}")
        print(f"Scanning emails for: rbpatel@usc.edu")
        print(f"{'='*80}\n")
        
        # Get list of messages
        results = service.users().messages().list(
            userId='me',
            maxResults=max_results,
            q=query
        ).execute()
        
        messages = results.get('messages', [])
        
        if not messages:
            print("No messages found.")
            return
        
        print(f"Found {len(messages)} message(s)\n")
        
        # Get details for each message
        for idx, message in enumerate(messages, 1):
            details = get_message_details(service, message['id'])
            if details:
                print(f"\n{'─'*80}")
                print(f"Email #{idx}")
                print(f"{'─'*80}")
                print(f"From: {details['from']}")
                print(f"Date: {details['date']}")
                print(f"Subject: {details['subject']}")
                print(f"\nBody Preview:")
                print(details['body'])
                print(f"{'─'*80}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Scan complete. Retrieved {len(messages)} email(s)")
        print(f"{'='*80}\n")
        
    except HttpError as error:
        print(f"An error occurred: {error}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    # Scan last 10 emails
    scan_emails(max_results=10)
    
    # Examples of other queries you can use:
    # scan_emails(max_results=20, query='is:unread')  # Only unread emails
    # scan_emails(max_results=50, query='from:example@example.com')  # From specific sender
    # scan_emails(max_results=30, query='subject:invoice')  # Specific subject
    # scan_emails(max_results=15, query='after:2025/11/01')  # After specific date
