
"""
Email System Testing File - No OAuth Required
==============================================
Simple test file to work with email scanning and scam detection
without dealing with real IMAP connections or OAuth.
"""

import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ParsedEmail:
    """Represents a parsed email message."""
    message_id: str
    sender: str
    recipient: str
    subject: str
    body_text: str
    body_html: str
    date: str
    headers: Dict[str, str]

    def __repr__(self):
        return f"ParsedEmail(id={self.message_id}, from={self.sender}, subject={self.subject[:30]}...)"


# ============================================================================
# MOCK IMAP CONNECTOR (replaces your imap_connector)
# ============================================================================

class MockImapConnector:
    """
    Mock IMAP connector that returns fake emails for testing.
    Replace this with real ImapConnector when ready for production.
    """

    def __init__(self, host, port, username, password, mailbox='INBOX'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.mailbox = mailbox
        self.connected = False
        self.email_index = 0

        # Mock email database
        self.mock_emails = self._generate_mock_emails()

    def connect(self):
        """Simulate IMAP connection."""
        print(f"[MOCK IMAP] Connecting to {self.host}:{self.port}")
        print(f"[MOCK IMAP] User: {self.username}")
        time.sleep(0.3)  # Simulate network delay
        self.connected = True
        print("[MOCK IMAP] ✓ Connected successfully")

    def disconnect(self):
        """Simulate disconnection."""
        print("[MOCK IMAP] Disconnecting...")
        self.connected = False
        print("[MOCK IMAP] ✓ Disconnected")

    def fetch_unseen(self, limit=10) -> List[bytes]:
        """
        Return mock raw email data (RFC822 format).
        Simulates fetching unread emails from IMAP server.
        """
        if not self.connected:
            print("[MOCK IMAP] ✗ Not connected")
            return []

        # Return a subset of mock emails each time
        emails = []
        for i in range(min(limit, len(self.mock_emails))):
            idx = (self.email_index + i) % len(self.mock_emails)
            emails.append(self.mock_emails[idx])

        self.email_index = (self.email_index + 1) % len(self.mock_emails)

        if emails:
            print(f"[MOCK IMAP] Fetched {len(emails)} unseen email(s)")

        return emails

    def _generate_mock_emails(self) -> List[bytes]:
        """Generate mock RFC822 formatted emails."""

        # Email 1: Obvious phishing scam
        email1 = b'''Return-Path: <security@suspicious-bank.com>
Date: Mon, 27 Oct 2025 11:00:00 +0000
From: Bank Security <security@suspicious-bank.com>
To: victim@example.com
Subject: URGENT: Your Account Will Be Closed!
Message-ID: <scam001@suspicious-bank.com>
Content-Type: text/plain; charset=UTF-8

URGENT ACTION REQUIRED!!!

Dear Valued Customer,

We have detected SUSPICIOUS activity on your account. Your account will be 
PERMANENTLY CLOSED within 24 hours unless you verify your identity immediately.

Click here to verify: http://totally-real-bank.ru/verify?user=12345

You must provide:
- Full name
- Social Security Number
- Account number
- Password
- Mother's maiden name

Failure to comply will result in PERMANENT account closure and legal action.

Regards,
Security Department
DO NOT REPLY TO THIS EMAIL
'''

        # Email 2: Lottery scam
        email2 = b'''Return-Path: <winner@international-lottery.org>
Date: Mon, 27 Oct 2025 10:30:00 +0000
From: Lottery Commission <winner@international-lottery.org>
To: lucky@example.com
Subject: CONGRATULATIONS! You Won $2,500,000!!!
Message-ID: <scam002@international-lottery.org>
Content-Type: text/plain; charset=UTF-8

CONGRATULATIONS!!! YOU ARE A WINNER!!!

You have been randomly selected as the GRAND PRIZE winner of the 
International Email Lottery!

PRIZE AMOUNT: $2,500,000.00 USD

To claim your prize, you must:
1. Send $500 processing fee via Western Union
2. Provide your bank account details
3. Send copy of your passport/driver's license
4. Reply within 48 hours or forfeit prize

Send processing fee to:
Name: Ahmed Mohammed
Location: Lagos, Nigeria
Reference: WINNER2025

This is a LIMITED TIME offer. ACT NOW!!!

Best regards,
International Lottery Commission
'''

        # Email 3: Legitimate work email
        email3 = b'''Return-Path: <sarah.jones@techcorp.com>
Date: Mon, 27 Oct 2025 09:15:00 +0000
From: Sarah Jones <sarah.jones@techcorp.com>
To: team@techcorp.com
Subject: Q4 Project Status Meeting - Wednesday 2PM
Message-ID: <work001@techcorp.com>
Content-Type: text/plain; charset=UTF-8

Hi Team,

Quick reminder about our Q4 project status meeting this Wednesday at 2 PM 
in Conference Room B.

Please come prepared to discuss:
- Current sprint progress
- Blockers and dependencies  
- Resource allocation for next quarter

I've shared the agenda doc in our team drive. Please review before the meeting.

Thanks,
Sarah Jones
Senior Project Manager
TechCorp Inc.
'''

        # Email 4: CEO fraud / Business Email Compromise
        email4 = b'''Return-Path: <ceo@company-urgent.com>
Date: Mon, 27 Oct 2025 08:45:00 +0000
From: John Smith CEO <ceo@company-urgent.com>
To: finance@example.com
Subject: URGENT: Wire Transfer Needed Today
Message-ID: <scam003@company-urgent.com>
Content-Type: text/plain; charset=UTF-8

Hi,

I'm in an important meeting with investors and need you to process an urgent 
wire transfer immediately.

Amount: $50,000
Bank: International Trust Bank
Account: 1234567890
Routing: 987654321
Recipient: Global Business Partners LLC

This is time-sensitive for closing a major deal. Please handle this before 
end of day and confirm once sent.

Do NOT discuss this with anyone else - confidential acquisition in progress.

Thanks,
John Smith
CEO

Sent from my iPhone
'''

        # Email 5: Tech support scam
        email5 = b'''Return-Path: <support@microsoft-security.net>
Date: Mon, 27 Oct 2025 08:00:00 +0000
From: Microsoft Support <support@microsoft-security.net>
To: user@example.com
Subject: Your Computer Has Been Infected - Call Now
Message-ID: <scam004@microsoft-security.net>
Content-Type: text/plain; charset=UTF-8

SECURITY ALERT - IMMEDIATE ACTION REQUIRED

Dear Microsoft User,

Our security systems have detected that your computer (IP: 192.168.1.1) has 
been infected with a CRITICAL VIRUS that is stealing your personal data.

Your Windows license will expire in 24 hours and your files will be encrypted.

CALL OUR TOLL-FREE SUPPORT IMMEDIATELY:
1-800-FAKE-NUM

Our certified technicians are standing by 24/7 to help you.

Do not ignore this warning. Your data is at risk.

Microsoft Security Team
'''

        # Email 6: Legitimate newsletter
        email6 = b'''Return-Path: <newsletter@nytimes.com>
Date: Mon, 27 Oct 2025 07:00:00 +0000
From: The New York Times <newsletter@nytimes.com>
To: subscriber@example.com
Subject: Morning Briefing: Today's Top Stories
Message-ID: <newsletter001@nytimes.com>
Content-Type: text/plain; charset=UTF-8

Good morning,

Here are today's top stories:

1. Markets Rally on Economic Data
   Stock markets reached new highs following positive employment figures...

2. Climate Summit Reaches Agreement
   World leaders announced commitments to reduce carbon emissions...

3. Tech Innovation Spotlight
   New developments in AI technology are transforming industries...

Read the full stories at nytimes.com

To unsubscribe, click here.

The New York Times
'''

        return [email1, email2, email3, email4, email5, email6]


# ============================================================================
# EMAIL PARSER (replaces your email.email_parser)
# ============================================================================

class EmailParser:
    """
    Parses raw RFC822 email data into structured ParsedEmail objects.
    """

    def parse(self, raw_email: bytes) -> ParsedEmail:
        """
        Parse raw email bytes into ParsedEmail object.

        Args:
            raw_email: Raw RFC822 formatted email as bytes

        Returns:
            ParsedEmail object with parsed data
        """
        try:
            # Decode bytes to string
            email_text = raw_email.decode('utf-8', errors='ignore')

            # Split headers and body
            parts = email_text.split('\n\n', 1)
            headers_text = parts[0]
            body_text = parts[1] if len(parts) > 1 else ''

            # Parse headers
            headers = {}
            for line in headers_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()

            # Extract common fields
            message_id = headers.get('Message-ID', f'<mock-{time.time()}@example.com>')
            sender = headers.get('From', 'unknown@example.com')
            recipient = headers.get('To', 'unknown@example.com')
            subject = headers.get('Subject', '(No Subject)')
            date = headers.get('Date', datetime.now().isoformat())

            # Clean body text
            body_text = body_text.strip()

            parsed = ParsedEmail(
                message_id=message_id,
                sender=sender,
                recipient=recipient,
                subject=subject,
                body_text=body_text,
                body_html='',  # Mock parser doesn't handle HTML
                date=date,
                headers=headers
            )

            print(f"[PARSER] Parsed: {subject[:50]}")
            return parsed

        except Exception as e:
            print(f"[PARSER] Error parsing email: {e}")
            # Return a dummy email on parse failure
            return ParsedEmail(
                message_id='<error@example.com>',
                sender='error@example.com',
                recipient='user@example.com',
                subject='Parse Error',
                body_text='Failed to parse email',
                body_html='',
                date=datetime.now().isoformat(),
                headers={}
            )


# ============================================================================
# EMAIL QUEUE MANAGER (replaces your email.email_queue_manager)
# ============================================================================

class EmailQueueManager:
    """
    Thread-safe queue for managing parsed emails.
    """

    def __init__(self, maxsize=100):
        self.queue = Queue(maxsize=maxsize)
        self.processed_count = 0

    def push(self, email: ParsedEmail):
        """Add email to queue."""
        try:
            self.queue.put(email, block=False)
            print(f"[QUEUE] Added to queue: {email.subject[:50]}")
        except:
            print(f"[QUEUE] Queue full, dropping email: {email.subject[:50]}")

    def get(self, timeout=5) -> Optional[ParsedEmail]:
        """Get email from queue with timeout."""
        try:
            email = self.queue.get(timeout=timeout)
            self.processed_count += 1
            return email
        except Empty:
            return None

    def size(self) -> int:
        """Return current queue size."""
        return self.queue.qsize()


# ============================================================================
# MOCK SCAM DETECTOR (your inference logic)
# ============================================================================

class MockScamDetector:
    """
    Mock scam detection system.
    Replace with your actual ML model or rule-based system.
    """

    def __init__(self, alert_threshold=0.7):
        self.alert_threshold = alert_threshold

        # Simple keyword-based detection for testing
        self.scam_keywords = [
            'urgent', 'verify your account', 'suspended', 'click here',
            'social security', 'bank account details', 'wire transfer',
            'congratulations', 'you won', 'lottery', 'million dollars',
            'processing fee', 'limited time', 'act now', 'call immediately',
            'virus detected', 'security alert', 'expire', 'infected'
        ]

        self.urgency_keywords = [
            'urgent', 'immediate', 'act now', 'within 24 hours', 
            'expires', 'last chance', 'limited time'
        ]

    def classify_email(self, email: ParsedEmail) -> Dict:
        """
        Classify email as scam or legitimate.

        Returns:
            Dict with 'scam_probability', 'reasons', 'category'
        """
        text = (email.subject + ' ' + email.body_text).lower()

        # Count scam indicators
        scam_score = 0
        reasons = []

        # Check for scam keywords
        keyword_matches = [kw for kw in self.scam_keywords if kw in text]
        if keyword_matches:
            scam_score += len(keyword_matches) * 0.15
            reasons.append(f"Scam keywords detected: {', '.join(keyword_matches[:3])}")

        # Check for urgency tactics
        urgency_matches = [kw for kw in self.urgency_keywords if kw in text]
        if urgency_matches:
            scam_score += 0.2
            reasons.append("Uses urgency tactics")

        # Check for suspicious sender domain
        sender = email.sender.lower()
        suspicious_domains = ['.ru', '.org', '-urgent.com', '-security.']
        if any(domain in sender for domain in suspicious_domains):
            scam_score += 0.25
            reasons.append(f"Suspicious sender domain: {sender}")

        # Check for requests for personal info
        personal_info_requests = ['password', 'social security', 'bank account', 
                                  'credit card', "mother's maiden"]
        if any(req in text for req in personal_info_requests):
            scam_score += 0.3
            reasons.append("Requests personal/financial information")

        # Check for money requests
        money_keywords = ['wire transfer', 'send money', 'processing fee', 
                         'pay now', '$', 'usd']
        if any(kw in text for kw in money_keywords):
            scam_score += 0.2
            reasons.append("Requests money or payment")

        # Cap at 1.0
        scam_probability = min(scam_score, 1.0)

        # Determine category
        if scam_probability >= 0.8:
            category = 'HIGH_RISK_SCAM'
        elif scam_probability >= 0.5:
            category = 'SUSPICIOUS'
        elif scam_probability >= 0.3:
            category = 'LOW_RISK'
        else:
            category = 'LEGITIMATE'

        return {
            'scam_probability': scam_probability,
            'reasons': reasons,
            'category': category,
            'confidence': 0.85  # Mock confidence score
        }


# ============================================================================
# MOCK CONFIG CLASSES
# ============================================================================

class MockModelConfig:
    """Mock model configuration."""
    ALERT_THRESHOLD_PERCENT = 0.7  # 70% threshold for alerts


class MockEmailConfig:
    """Mock email configuration."""
    IMAP_HOST = 'imap.gmail.com'
    IMAP_PORT = 993
    USERNAME = 'test@example.com'
    PASSWORD = 'mock_password'
    MAILBOX = 'INBOX'
    POLL_INTERVAL_S = 5  # Check every 5 seconds


# ============================================================================
# MAIN TESTING LOGIC
# ============================================================================

def imap_poller(imap, email_parser, email_q, config):
    """Poll IMAP server for new emails."""
    try:
        imap.connect()
        print('[POLLER] IMAP connected')

        # Simple poll loop
        while True:
            raws = imap.fetch_unseen(limit=10)
            for raw in raws:
                parsed = email_parser.parse(raw)
                email_q.push(parsed)
            time.sleep(config.POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print('[POLLER] Stopped by user')
    except Exception as e:
        print(f'[POLLER] IMAP poller error: {e}')
    finally:
        imap.disconnect()


def email_consumer(email_q, scam_detector, model_config):
    """Consume emails from queue and run scam detection."""
    print('[CONSUMER] Email consumer started')

    try:
        while True:
            item = email_q.get(timeout=5)
            if not item:
                continue

            print(f'\n{"="*70}')
            print(f'[CONSUMER] Processing: {item.subject}')
            print(f'[CONSUMER] From: {item.sender}')
            print(f'{"="*70}')

            # Run scam detection
            result = scam_detector.classify_email(item)
            prob = result.get('scam_probability', 0.0)
            category = result.get('category', 'UNKNOWN')
            reasons = result.get('reasons', [])

            print(f'\n[DETECTION] Category: {category}')
            print(f'[DETECTION] Scam Probability: {prob:.1%}')

            if reasons:
                print(f'[DETECTION] Reasons:')
                for reason in reasons:
                    print(f'  - {reason}')

            # Check if alert threshold exceeded
            if prob >= model_config.ALERT_THRESHOLD_PERCENT:
                print(f'\n🚨 [ALERT] SCAM DETECTED!')
                print(f'   Confidence: {prob:.0%}')
                print(f'   Message ID: {item.message_id}')
                print(f'   Subject: {item.subject}')
                # In production: trigger LED, play audio alert, etc.
            else:
                print(f'\n✓ [OK] Email appears legitimate')

            print()

    except KeyboardInterrupt:
        print('[CONSUMER] Stopped by user')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EMAIL SCAM DETECTION SYSTEM - TEST MODE")
    print("="*70)
    print()

    # Initialize components
    print("[INIT] Initializing components...")
    mcfg = MockModelConfig()
    ecfg = MockEmailConfig()

    # Use mock IMAP connector (no real email connection)
    imap = MockImapConnector(
        ecfg.IMAP_HOST, 
        ecfg.IMAP_PORT, 
        ecfg.USERNAME, 
        ecfg.PASSWORD, 
        mailbox=ecfg.MAILBOX
    )

    email_parser = EmailParser()
    email_q = EmailQueueManager()
    scam_detector = MockScamDetector(alert_threshold=mcfg.ALERT_THRESHOLD_PERCENT)

    print("[INIT] ✓ All components initialized")
    print()

    # Start threads
    print("[INIT] Starting worker threads...")
    t_imap = threading.Thread(
        target=imap_poller, 
        args=(imap, email_parser, email_q, ecfg),
        daemon=True
    )
    t_email_consumer = threading.Thread(
        target=email_consumer,
        args=(email_q, scam_detector, mcfg),
        daemon=True
    )

    t_imap.start()
    t_email_consumer.start()

    print("[INIT] ✓ Threads started")
    print()
    print("="*70)
    print("System running. Press Ctrl+C to stop.")
    print("="*70)
    print()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        print("✓ System stopped")
