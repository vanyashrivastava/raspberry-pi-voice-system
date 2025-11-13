
# Email Monitor - IMAP Response Examples and Testing Guide

## Table of Contents
1. [IMAP Server Response Formats](#imap-server-response-formats)
2. [Mock Data for Testing](#mock-data-for-testing)
3. [Authentication Handling](#authentication-handling)
4. [Testing Without Real Email](#testing-without-real-email)

---

## 1. IMAP Server Response Formats

### 1.1 Connecting to IMAP Server

**Request (Client to Server):**
```python
import imaplib
import ssl

# Create SSL connection
ssl_context = ssl.create_default_context()
connection = imaplib.IMAP4_SSL('imap.gmail.com', 993, ssl_context=ssl_context)
```

**Response (Server to Client):**
```
* OK [CAPABILITY IMAP4REV1 UNSELECT IDLE NAMESPACE QUOTA ID XLIST 
      CHILDREN X-GM-EXT-1 UIDPLUS COMPRESS=DEFLATE ENABLE MOVE 
      CONDSTORE ESEARCH UTF8=ACCEPT LIST-EXTENDED LIST-STATUS 
      LITERAL+ APPENDLIMIT=35651584] Gimap ready
```

**Parsed Response:**
```python
# Connection successful - no explicit response needed
# The connection object is created and ready
```

---

### 1.2 Authentication (Login)

**Request:**
```python
response_code, response_data = connection.login('user@gmail.com', 'app_password')
```

**Response (Success):**
```
Response Code: 'OK'
Response Data: [b'user@gmail.com authenticated (Success)']
```

**Response (Failure - Bad Credentials):**
```
Response Code: 'NO'
Error: imaplib.IMAP4.error: [AUTHENTICATIONFAILED] Invalid credentials
```

**Response (Failure - App Password Required):**
```
Response Code: 'NO'
Error: [WEBALERT] Please log in via your web browser
```

**Mock Response for Testing:**
```python
def mock_connect_response():
    return {
        'status': 'OK',
        'response': [b'user@example.com authenticated'],
        'capabilities': ['IMAP4REV1', 'IDLE', 'UIDPLUS', 'NAMESPACE']
    }
```

---

### 1.3 Selecting Mailbox (INBOX)

**Request:**
```python
status, message_count = connection.select('INBOX', readonly=False)
```

**Response:**
```
Status: 'OK'
Data: [b'42']  # Number of messages in inbox

Full Server Response:
* 42 EXISTS
* 0 RECENT  
* OK [UNSEEN 12] Message 12 is first unseen
* OK [UIDVALIDITY 1698765432] UIDs valid
* OK [UIDNEXT 43] Predicted next UID
* FLAGS (\Answered \Flagged \Deleted \Seen \Draft)
* OK [PERMANENTFLAGS (\Answered \Flagged \Deleted \Seen \Draft \*)] Flags permitted
OK [READ-WRITE] INBOX selected
```

**Mock Response:**
```python
def mock_select_response():
    return {
        'status': 'OK',
        'message_count': [b'42'],
        'exists': 42,
        'recent': 0,
        'unseen': 12,
        'uidvalidity': 1698765432,
        'uidnext': 43,
        'flags': ['\\Answered', '\\Flagged', '\\Deleted', '\\Seen', '\\Draft']
    }
```

---

### 1.4 Searching for Unread Emails

**Request:**
```python
status, email_ids = connection.search(None, 'UNSEEN')
```

**Response (Has Unread Emails):**
```
Status: 'OK'
Data: [b'12 15 23 37 41']  # Space-separated email IDs

Full Server Response:
* SEARCH 12 15 23 37 41
OK Search completed
```

**Response (No Unread Emails):**
```
Status: 'OK'
Data: [b'']  # Empty bytes object

Full Server Response:
* SEARCH
OK Search completed (0.001 secs)
```

**Mock Response:**
```python
def mock_search_unseen_response():
    return {
        'status': 'OK',
        'email_ids': [b'12 15 23 37 41'],  # Has unread emails
        'parsed_ids': ['12', '15', '23', '37', '41']
    }

def mock_search_no_unseen_response():
    return {
        'status': 'OK',
        'email_ids': [b''],  # No unread emails
        'parsed_ids': []
    }
```

---

### 1.5 Fetching Specific Email

**Request:**
```python
status, email_data = connection.fetch('12', '(RFC822)')
```

**Response:**
```
Status: 'OK'
Data: [
    (
        b'12 (RFC822 {2048}',  # Email ID and size in bytes
        b'Return-Path: <sender@example.com>\r\n'
        b'Delivered-To: recipient@example.com\r\n'
        b'Received: from mail.example.com ...\r\n'
        b'Date: Sun, 26 Oct 2025 19:00:00 +0000\r\n'
        b'From: sender@example.com\r\n'
        b'To: recipient@example.com\r\n'
        b'Subject: Test Email Subject\r\n'
        b'Message-ID: <abc123@example.com>\r\n'
        b'Content-Type: text/plain; charset=UTF-8\r\n'
        b'Content-Transfer-Encoding: 7bit\r\n'
        b'\r\n'
        b'This is the email body content.\r\n'
        b'It can span multiple lines.\r\n'
    ),
    b')'
]

Full Server Response:
* 12 FETCH (RFC822 {2048}
... raw email data ...
)
OK Fetch completed
```

**Mock Response:**
```python
def mock_fetch_response():
    raw_email = (
        b'Return-Path: <sender@example.com>\r\n'
        b'Date: Sun, 26 Oct 2025 19:00:00 +0000\r\n'
        b'From: John Doe <sender@example.com>\r\n'
        b'To: recipient@example.com\r\n'
        b'Subject: Test Email\r\n'
        b'Content-Type: text/plain; charset=UTF-8\r\n'
        b'\r\n'
        b'This is the email body.\r\n'
    )

    return {
        'status': 'OK',
        'data': [(b'12 (RFC822 {1024}', raw_email), b')'],
        'parsed': {
            'id': '12',
            'from': 'sender@example.com',
            'to': 'recipient@example.com',
            'subject': 'Test Email',
            'date': 'Sun, 26 Oct 2025 19:00:00 +0000',
            'body': 'This is the email body.'
        }
    }
```

---

## 2. Mock Data for Testing

### 2.1 Complete Mock Email Dataset

Here are three sample emails representing different scenarios:

```python
MOCK_EMAILS = {
    # Phishing/Scam Email Example
    '1': {
        'id': '1',
        'from': 'security@suspicious-bank.com',
        'to': 'user@example.com',
        'subject': 'Urgent: Verify Your Account Now!',
        'date': 'Sun, 26 Oct 2025 19:00:00 +0000',
        'body_text': '''Dear Valued Customer,

We have detected suspicious activity on your account. Please verify 
your identity immediately by clicking the link below:

http://totally-legit-bank.suspicious-domain.ru/verify?id=12345

Failure to verify within 24 hours will result in account suspension.

Regards,
Security Team''',
        'body_html': '',
        'headers': {
            'Return-Path': '<security@suspicious-bank.com>',
            'Received': 'from mail.suspicious-bank.com ...',
            'Content-Type': 'text/plain; charset=UTF-8'
        }
    },

    # Lottery Scam Email Example
    '2': {
        'id': '2',
        'from': 'lottery@scam-international.org',
        'to': 'user@example.com',
        'subject': 'You Won $1,000,000 - Claim Now!',
        'date': 'Sun, 26 Oct 2025 18:45:00 +0000',
        'body_text': '''CONGRATULATIONS!!!

You have been selected as the winner of our international lottery!

To claim your prize of ONE MILLION DOLLARS, please send your:
- Full name
- Address  
- Social Security Number
- Bank account details

Reply to this email within 48 hours or forfeit your prize!

Best regards,
International Lottery Commission''',
        'body_html': '',
        'headers': {
            'Return-Path': '<lottery@scam-international.org>',
            'X-Spam-Score': '9.5',
            'Content-Type': 'text/plain; charset=UTF-8'
        }
    },

    # Legitimate Email Example
    '3': {
        'id': '3',
        'from': 'john.smith@company.com',
        'to': 'user@example.com',
        'subject': 'Team Meeting Tomorrow at 10 AM',
        'date': 'Sun, 26 Oct 2025 17:30:00 +0000',
        'body_text': '''Hi Team,

Just a reminder that we have our weekly team meeting tomorrow at 10 AM 
in Conference Room B.

Agenda:
- Q4 project updates
- Resource allocation
- Upcoming deadlines

Please review the attached documents before the meeting.

Thanks,
John Smith
Project Manager''',
        'body_html': '',
        'headers': {
            'Return-Path': '<john.smith@company.com>',
            'Content-Type': 'text/plain; charset=UTF-8'
        }
    }
}
```

---

## 3. Authentication Handling

### 3.1 App-Specific Passwords (Recommended)

**Gmail:**
1. Enable 2-Factor Authentication on your Google account
2. Go to: https://myaccount.google.com/apppasswords
3. Select "Mail" and your device
4. Copy the 16-character password (format: xxxx xxxx xxxx xxxx)
5. Use this password instead of your regular password

**Example:**
```python
EMAIL = "your-email@gmail.com"
APP_PASSWORD = "abcd efgh ijkl mnop"  # 16 characters from Google

monitor = EmailMonitor(EMAIL, APP_PASSWORD, "imap.gmail.com", 993)
success, error = monitor.connect_to_imap(EMAIL, APP_PASSWORD)
```

**Microsoft/Outlook:**
1. Go to: https://account.microsoft.com/security
2. Enable two-step verification
3. Create app password under "App passwords"
4. Use generated password

**Yahoo:**
1. Go to: https://login.yahoo.com/account/security
2. Enable two-step verification  
3. Generate app password
4. Use for IMAP connection

### 3.2 OAuth2 Authentication (Production)

For production systems, OAuth2 is more secure:

```python
import imaplib
from imapclient.oauth2_util import build_oauth2_string

def connect_with_oauth2(email, access_token):
    auth_string = build_oauth2_string(email, access_token)

    connection = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    connection.authenticate('XOAUTH2', lambda x: auth_string)

    return connection
```

**Note:** OAuth2 setup requires:
- Registering your app with the email provider
- Obtaining client ID and secret
- Implementing token refresh logic

---

## 4. Testing Without Real Email

### 4.1 Using MockEmailMonitor

```python
from email_monitor import MockEmailMonitor

# Create mock monitor
monitor = MockEmailMonitor()

# Connect (simulated)
success, error = monitor.connect_to_imap("test@example.com", "fake_password")
# Output: [MOCK] Connected successfully

# Check for emails (simulated)  
email_ids = monitor.check_for_new_emails()
# Output: [MOCK] Found 3 unread email(s)
# Returns: ['1', '2', '3']

# Fetch email (simulated)
email_data = monitor.fetch_email_by_id('1')
# Returns mock email data dictionary

# Close connection
monitor.close_connection()
```

### 4.2 Expected Console Output

When running with MockEmailMonitor:

```
======================================================================
EMAIL MONITOR - EXAMPLE USAGE
======================================================================

NOTE: Using MockEmailMonitor for demonstration
      Replace with real EmailMonitor for production

======================================================================
STEP 1: CONNECTING TO IMAP SERVER
======================================================================

[MOCK] Connecting to IMAP server...
[MOCK] Server: imap.mock-server.com:993
[MOCK] User: your-email@gmail.com
[MOCK] ✓ Connected successfully
[MOCK] Response: ('OK', [b'user your-email@gmail.com authenticated'])

======================================================================
STEP 2: CHECKING FOR NEW EMAILS
======================================================================

[MOCK] Checking for unread emails...
[MOCK] SELECT INBOX response:
[MOCK]   * 3 EXISTS
[MOCK]   * 3 RECENT
[MOCK]   * OK [UNSEEN 1] First unseen
[MOCK] SEARCH UNSEEN response:
[MOCK]   * SEARCH 1 2 3
[MOCK] ✓ Found 3 unread email(s)

Processing 3 new email(s)...

======================================================================
STEP 3: FETCHING EMAIL DETAILS
======================================================================

──────────────────────────────────────────────────────────────────────
Processing Email ID: 1
──────────────────────────────────────────────────────────────────────

[MOCK] Fetching email ID 1...
[MOCK] FETCH 1 (RFC822) response:
[MOCK]   * 1 FETCH (RFC822 {2048})
[MOCK]   Return-Path: <security@suspicious-bank.com>
[MOCK]   Subject: Urgent: Verify Your Account Now!
[MOCK]   From: security@suspicious-bank.com
[MOCK]   To: user@example.com
[MOCK]   Date: Sun, 26 Oct 2025 19:00:00 +0000
[MOCK]   Content-Type: text/plain; charset=utf-8
[MOCK]   ... (email body follows)
[MOCK] ✓ Successfully fetched email 1

📧 Email Details:
   From: security@suspicious-bank.com
   To: user@example.com
   Subject: Urgent: Verify Your Account Now!
   Date: Sun, 26 Oct 2025 19:00:00 +0000

📝 Body Preview:
   Dear Valued Customer,

We have detected suspicious activity on your account. Please verify 
your identity immediately by clicking the link below:

http://totally-legit-bank.suspicious-domain...
```

### 4.3 Unit Testing Examples

```python
import unittest
from email_monitor import MockEmailMonitor

class TestEmailMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = MockEmailMonitor()

    def test_connection(self):
        success, error = self.monitor.connect_to_imap(
            "test@example.com", 
            "password"
        )
        self.assertTrue(success)
        self.assertIsNone(error)

    def test_check_emails(self):
        self.monitor.connect_to_imap("test@example.com", "password")
        email_ids = self.monitor.check_for_new_emails()
        self.assertEqual(len(email_ids), 3)
        self.assertIn('1', email_ids)
        self.assertIn('2', email_ids)
        self.assertIn('3', email_ids)

    def test_fetch_email(self):
        self.monitor.connect_to_imap("test@example.com", "password")
        email_data = self.monitor.fetch_email_by_id('1')
        self.assertIsNotNone(email_data)
        self.assertEqual(email_data['id'], '1')
        self.assertIn('Urgent', email_data['subject'])
        self.assertIn('suspicious-bank.com', email_data['from'])

    def test_scam_detection_integration(self):
        self.monitor.connect_to_imap("test@example.com", "password")
        email_ids = self.monitor.check_for_new_emails()

        for email_id in email_ids:
            email_data = self.monitor.fetch_email_by_id(email_id)

            # Test scam indicators
            body = email_data['body_text'].lower()

            # Email 1 & 2 should have scam indicators
            if email_id in ['1', '2']:
                has_scam_indicator = (
                    'urgent' in body or
                    'verify' in body or  
                    'million' in body or
                    'account suspension' in body or
                    'social security' in body
                )
                self.assertTrue(has_scam_indicator)

            # Email 3 should be legitimate
            elif email_id == '3':
                self.assertIn('meeting', body)
                self.assertIn('team', body)

if __name__ == '__main__':
    unittest.main()
```

---

## 5. Error Handling Examples

### 5.1 Connection Errors

```python
# Timeout Error
try:
    connection = imaplib.IMAP4_SSL('imap.gmail.com', 993, timeout=10)
except socket.timeout:
    print("Connection timed out")
    # Retry with exponential backoff

# SSL Error  
try:
    connection = imaplib.IMAP4_SSL('imap.example.com', 993)
except ssl.SSLError as e:
    print(f"SSL Error: {e}")
    # Check SSL certificate validity

# Network Error
try:
    connection = imaplib.IMAP4_SSL('imap.gmail.com', 993)
except socket.gaierror:
    print("Network error - check internet connection")
```

### 5.2 Authentication Errors

```python
try:
    response_code, response_data = connection.login(email, password)
except imaplib.IMAP4.error as e:
    error_msg = str(e)

    if 'AUTHENTICATIONFAILED' in error_msg:
        print("Invalid credentials")
        print("Tip: Use app-specific password if 2FA is enabled")

    elif 'WEBALERT' in error_msg:
        print("Please enable 'Less Secure Apps' or use app password")

    elif 'UNAVAILABLE' in error_msg:
        print("Server temporarily unavailable")
        # Retry after delay
```

### 5.3 Connection Maintenance

```python
def monitor_emails_continuously():
    monitor = EmailMonitor(EMAIL, PASSWORD)

    # Connect
    success, error = monitor.connect_to_imap(EMAIL, PASSWORD)
    if not success:
        print(f"Failed to connect: {error}")
        return

    while True:
        try:
            # Maintain connection
            if not monitor.maintain_connection():
                print("Failed to maintain connection, exiting...")
                break

            # Check for new emails
            email_ids = monitor.check_for_new_emails()

            for email_id in email_ids:
                email_data = monitor.fetch_email_by_id(email_id)
                # Process email...

            # Wait before checking again
            time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            print("Stopping email monitor...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)  # Brief pause before retry

    monitor.close_connection()
```

---

## 6. Common IMAP Server Addresses

| Provider | IMAP Server | Port | SSL |
|----------|-------------|------|-----|
| Gmail | imap.gmail.com | 993 | Yes |
| Outlook/Hotmail | outlook.office365.com | 993 | Yes |
| Yahoo | imap.mail.yahoo.com | 993 | Yes |
| iCloud | imap.mail.me.com | 993 | Yes |
| AOL | imap.aol.com | 993 | Yes |
| ProtonMail | imap.protonmail.com | 993 | Yes |

---

## 7. Security Best Practices

1. **Never hardcode credentials** - Use environment variables:
   ```python
   import os
   EMAIL = os.getenv('EMAIL_ADDRESS')
   PASSWORD = os.getenv('EMAIL_PASSWORD')
   ```

2. **Use app-specific passwords** - Don't use your main account password

3. **Enable 2FA** - Always use two-factor authentication

4. **Secure credential storage** - Consider using:
   - Environment variables
   - Secure key management systems (AWS Secrets Manager, Azure Key Vault)
   - Encrypted configuration files

5. **OAuth2 for production** - Use OAuth2 tokens instead of passwords

6. **Rate limiting** - Don't check emails too frequently (respect server limits)

7. **Error logging** - Log errors securely without exposing credentials

---

## 8. Integration with Scam Detection

```python
# Example integration flow

def process_emails_for_scam_detection(monitor, scam_detector):
    """
    Main processing loop for email scam detection.

    Args:
        monitor: EmailMonitor instance
        scam_detector: Your scam detection AI/system
    """
    # Get new emails
    email_ids = monitor.check_for_new_emails()

    for email_id in email_ids:
        # Fetch email
        email_data = monitor.fetch_email_by_id(email_id)

        if email_data:
            # Extract relevant features for scam detection
            features = {
                'sender': email_data['from'],
                'subject': email_data['subject'],
                'body': email_data['body_text'],
                'date': email_data['date']
            }

            # Run scam detection
            is_scam, confidence, reasons = scam_detector.analyze(features)

            # Take action based on result
            if is_scam and confidence > 0.8:
                print(f"⚠️  SCAM DETECTED (confidence: {confidence:.2%})")
                print(f"   Email ID: {email_id}")
                print(f"   From: {features['sender']}")
                print(f"   Subject: {features['subject']}")
                print(f"   Reasons: {', '.join(reasons)}")

                # Mark as spam, alert user, etc.
                # mark_as_spam(email_id)
                # send_alert_to_user(features, reasons)
            else:
                print(f"✓ Email appears legitimate")
```

---

This documentation provides everything needed to test the email monitor component
without connecting to real email servers, and shows exactly what responses to expect
from actual IMAP servers.
