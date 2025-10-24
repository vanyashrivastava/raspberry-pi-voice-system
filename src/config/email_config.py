"""
Owner: Config Team
Responsibility: Example IMAP/email configuration values and polling/IDLE parameters.

Adjust credentials and hostnames before deploying. For production, prefer using environment
variables or a secrets manager instead of hardcoding credentials.

Integration:
 - `email.imap_connector.ImapConnector` will consume these settings

Example usage:
    from src.config.email_config import EmailConfig
    cfg = EmailConfig()
    connector = ImapConnector(cfg.IMAP_HOST, cfg.IMAP_PORT, cfg.USERNAME, cfg.PASSWORD)

"""

from dataclasses import dataclass


@dataclass
class EmailConfig:
    IMAP_HOST: str = 'imap.example.com'
    IMAP_PORT: int = 993
    USERNAME: str = 'user@example.com'
    PASSWORD: str = 'change-me'
    MAILBOX: str = 'INBOX'

    # Polling interval in seconds if IDLE is not available
    POLL_INTERVAL_S: int = 30

    # Maximum size of emails to download (bytes) to avoid OOM on Pi
    MAX_EMAIL_BYTES: int = 5 * 1024 * 1024


if __name__ == '__main__':
    print('EmailConfig example', EmailConfig())
