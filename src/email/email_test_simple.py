
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
        
                # Email 7: Spear-phishing (targeted, appears from colleague)
        
        email7 = b'''Return-Path: <liaison@partners-company.com>
Date: Mon, 27 Oct 2025 12:05:00 +0000
From: Mark Allen <mark.allen@partners-company.com>
To: rohan@example.com
Subject: Fwd: Updated invoice - please approve
Message-ID: <scam007@partners-company.com>
Content-Type: text/plain; charset=UTF-8

Rohan,

Can you quickly approve this updated invoice for the vendor? I'm in a meeting 
and can't access Concur right now.

Invoice: INV-2025-9987
Amount: $12,450.00
Approve here: http://partners-approve-payments.net/approve?id=9987

Thanks,
Mark
Sent from my mobile
'''

        # Email 8: Fake courier / delivery scam (click link to "reschedule")
        email8 = b'''Return-Path: <updates@fastship-delivery.com>
Date: Mon, 27 Oct 2025 11:50:00 +0000
From: FastShip Delivery <updates@fastship-delivery.com>
To: customer@example.com
Subject: ACTION REQUIRED: Delivery Attempt Failed - Reschedule Now
Message-ID: <scam008@fastship-delivery.com>
Content-Type: text/plain; charset=UTF-8

Dear Customer,

We attempted to deliver a package but were unable to reach you. Please reschedule 
delivery or provide additional address details to avoid return to sender.

Tracking Number: FS123456789US
Reschedule: http://fastship-delivery-reschedule.ru/track?tn=FS123456789US

If we do not receive a response within 48 hours the package will be returned.

FastShip Customer Support
'''

        # Email 9: Romance scam / advance-fee request
        email9 = b'''Return-Path: <lovelysender@mailbox.com>
Date: Mon, 27 Oct 2025 11:30:00 +0000
From: Anna Marie <anna.marie@mailbox.com>
To: lonely@example.com
Subject: I need your help - urgent travel expenses
Message-ID: <scam009@mailbox.com>
Content-Type: text/plain; charset=UTF-8

Hi there,

I've really enjoyed our chats these past few weeks. I have an unexpected 
medical bill and a plane ticket issue while traveling abroad. I hate to ask 
but could you send $1,200 to help cover the emergency so I can come see you?

Send via MoneyGram to:
Name: Anna M.
City: Accra
Reference: HELP2025

I promise to pay you back and I can't thank you enough.

Love,
Anna
'''

        # Email 10: Fake job offer with attachment link (malicious .exe disguised)
        email10 = b'''Return-Path: <recruiting@innovatehiring.com>
Date: Mon, 27 Oct 2025 10:55:00 +0000
From: Talent Acquisition <recruiting@innovatehiring.com>
To: candidate@example.com
Subject: Offer Letter & Onboarding Documents (Action Required)
Message-ID: <scam010@innovatehiring.com>
Content-Type: text/plain; charset=UTF-8

Congratulations!

We are pleased to extend you an offer for the position of Junior Analyst. 
Please download the offer packet and complete the onboarding form.

Download offer packet: http://innovatehiring-docs.com/offer_AnnaMalware.exe

Start date: November 10, 2025
Salary: $68,000/year

Welcome aboard,
Talent Acquisition
Innovate Hiring Ltd.
'''

        # Email 11: Legitimate bank notification
        email11 = b'''Return-Path: <alerts@trustbank.com>
Date: Mon, 27 Oct 2025 10:10:00 +0000
From: TrustBank Alerts <alerts@trustbank.com>
To: customer@example.com
Subject: Your Monthly Statement is Ready
Message-ID: <notice011@trustbank.com>
Content-Type: text/plain; charset=UTF-8

Hello,

Your October 2025 statement is now available online. No action is required 
if you have already reviewed it.

To view your statement, please sign in to your secure account at our official 
website or use the TrustBank mobile app.

Thank you for banking with TrustBank.

TrustBank Customer Service
'''

        # Email 12: Investment/crypto scam
        email12 = b'''Return-Path: <invest@crypto-profits.io>
Date: Mon, 27 Oct 2025 09:50:00 +0000
From: Crypto Profits <invest@crypto-profits.io>
To: investor@example.com
Subject: Double Your Crypto in 7 Days - Exclusive Offer
Message-ID: <scam012@crypto-profits.io>
Content-Type: text/plain; charset=UTF-8

EXCLUSIVE - LIMITED SLOTS AVAILABLE

Invest $5,000 in our proprietary trading bot and receive 100 percent returns within 7 days.
This opportunity is only offered to select clients.

Register now to secure your slot:
https://crypto-profits-quick.com/signup?ref=elite

Risk-free guarantee included. Minimum deposit: $500.

Sincerely,
Crypto Profits Team
'''

        # Email 13: Invoice scam
        email13 = b'''Return-Path: <billing@acme-supplies.co>
Date: Mon, 27 Oct 2025 09:20:00 +0000
From: ACME Supplies Billing <billing@acme-supplies.co>
To: accounting@example.com
Subject: Past Due Invoice - Urgent Payment Required
Message-ID: <scam013@acme-supplies.co>
Content-Type: text/plain; charset=UTF-8

Attention Accounts Payable,

Our records indicate Invoice #A-55432 is past due. Immediate payment is required 
to avoid a late fee.

Invoice: A-55432
Amount Due: $7,880.00
Pay here: http://acme-billing-payments.net/pay?inv=A-55432

Please confirm payment once processed.

Regards,
ACME Supplies - Billing Department
'''

        # Email 14: Fake government tax notice
        email14 = b'''Return-Path: <taxnotice@irs-tax.gov>
Date: Mon, 27 Oct 2025 08:40:00 +0000
From: IRS Tax <taxnotice@irs-tax.gov>
To: taxpayer@example.com
Subject: Immediate Action Required: Unpaid Taxes Owed
Message-ID: <scam014@irs-tax.gov>
Content-Type: text/plain; charset=UTF-8

NOTICE OF TAX DELINQUENCY

Our records show outstanding taxes for tax year 2024. A payment or response 
is required within 7 days to avoid legal action.

Amount Due: $6,300.00
Pay online now: http://irs-payment-portal-secure.com/pay?uid=TX2024

For questions call: 1-800-FAKE-IRS

This is an automated message.
'''

        # Email 15: Legitimate HR email
        email15 = b'''Return-Path: <hr@greenworks.com>
Date: Mon, 27 Oct 2025 08:15:00 +0000
From: GreenWorks HR <hr@greenworks.com>
To: all-employees@greenworks.com
Subject: Open Enrollment for 2026 Benefits Starts Nov 1
Message-ID: <hr015@greenworks.com>
Content-Type: text/plain; charset=UTF-8

Hello Team,

Open enrollment for 2026 benefits begins on November 1. Please review your 
options for medical, dental, and retirement plan changes. An information session 
will be held on Oct 29 at 3 PM in the main auditorium.

Resources and forms are available on the HR portal.

Best,
GreenWorks HR
'''

        # Email 16: Subscription renewal scam
        email16 = b'''Return-Path: <billing@streamflix-premium.com>
Date: Mon, 27 Oct 2025 07:55:00 +0000
From: StreamFlix Billing <billing@streamflix-premium.com>
To: subscriber@example.com
Subject: Payment Failed - Reactivate Your Account
Message-ID: <scam016@streamflix-premium.com>
Content-Type: text/plain; charset=UTF-8

We were unable to process your monthly payment. Your StreamFlix account will be 
suspended within 24 hours unless you update your payment details.

Update payment: http://streamflix-billing-update.com/card

Recent transaction: $14.99 on Oct 27, 2025

StreamFlix Support
'''

        # Email 17: Legitimate university admin notice
        email17 = b'''Return-Path: <registrar@usc.edu>
Date: Mon, 27 Oct 2025 07:30:00 +0000
From: USC Registrar <registrar@usc.edu>
To: rohan@usc.edu
Subject: Reminder: Course Enrollment Deadline Nov 5
Message-ID: <uscreg017@usc.edu>
Content-Type: text/plain; charset=UTF-8

Dear Student,

This is a reminder that the enrollment add/drop deadline for the Fall 2025 term 
is November 5. Please finalize your schedule and contact your academic advisor 
with any questions.

Regards,
Office of the Registrar
University of Southern California
'''

        # Email 18: Fake charity scam
        email18 = b'''Return-Path: <donations@reliefnow-global.org>
Date: Mon, 27 Oct 2025 07:10:00 +0000
From: ReliefNow Global <donations@reliefnow-global.org>
To: donor@example.com
Subject: Help Victims of Recent Disaster - Donate Now
Message-ID: <scam018@reliefnow-global.org>
Content-Type: text/plain; charset=UTF-8

URGENT RELIEF NEEDED

A major disaster has struck and thousands are affected. Donate now to provide 
food, shelter, and medical care. Every dollar helps.

Donate securely: http://reliefnow-global-donate.org/secure?campaign=urgent

ReliefNow Global - Registered Charity (EIN: 00-0000000)
'''

        # Email 19: Business partnership inquiry (legitimate)
        email19 = b'''Return-Path: <partnerships@mediacollab.io>
Date: Mon, 27 Oct 2025 06:50:00 +0000
From: MediaCollab Partnerships <partnerships@mediacollab.io>
To: partnerships@example.com
Subject: Potential Partnership Opportunity with MediaCollab
Message-ID: <biz019@mediacollab.io>
Content-Type: text/plain; charset=UTF-8

Hello,

I'm reaching out from MediaCollab to explore a potential content partnership 
with your team. We produce sponsored short-form content and are interested in 
collaborating on a pilot campaign next quarter.

If interested, please reply and we can set a 20-minute introductory call.

Best,
Leah Kim
Head of Partnerships
MediaCollab
'''

        # Email 20: Malware-laden attachment disguised as resume
        email20 = b'''Return-Path: <applicant@jobs-mail.com>
Date: Mon, 27 Oct 2025 06:30:00 +0000
From: Job Applicant <applicant@jobs-mail.com>
To: hiring@example.com
Subject: Application for Software Engineer - Resume Attached
Message-ID: <scam020@jobs-mail.com>
Content-Type: text/plain; charset=UTF-8

Dear Hiring Team,

Please find my resume attached for the Software Engineer position.

Attachment: resume_final_2025.docm

I look forward to the opportunity to discuss my qualifications.

Sincerely,
A. Candidate
'''

        # Email 21: Friendly personal email
        email21 = b'''Return-Path: <friendship@mail.example.com>
Date: Mon, 27 Oct 2025 06:00:00 +0000
From: Priya Patel <priya.patel@mail.example.com>
To: rohan@example.com
Subject: Weekend hike photos and dinner this Friday?
Message-ID: <friend021@mail.example.com>
Content-Type: text/plain; charset=UTF-8

Hey Rohan,

Great hike this weekend - attached are a few of the photos I took. Want to grab 
dinner this Friday to catch up and plan another trip?

Let me know what works for you.

Cheers,
Priya
'''

        return [email1, email2, email3, email4, email5, email6, email7, email8, email9, email10, 
                email11, email12, email13, email14, email15, email16, email17, email18, 
                email19, email20, email21]


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
