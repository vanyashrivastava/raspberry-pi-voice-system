# Owner: Rohan
# Responsibility: Connect to IMAP servers and stream new emails into the processing pipeline.
# Goals:
# - Maintain secure IMAP connection (SSL/TLS), periodically IDLE or poll for new messages
# - Download raw RFC822 messages and metadata
# Integration points:
# - Push parsed messages to `email.email_queue_manager.EmailQueueManager`
# - Provide hooks for `detection.model_inference` to request message bodies for classification
# Testing requirements:
# - Mock IMAP server tests verifying login, folder selection, and new message retrieval logic

import imaplib
import email as py_email
import ssl
import typing as t

# Dependencies noted in requirements.txt: imaplib, email (stdlib)


class ImapConnector:
    """
    Simple IMAP connector that can poll or use IDLE to receive new messages.

    Constructor args:
        host: str, port: int, username: str, password: str, mailbox: str

    Public methods:
        - connect(): establish the IMAP connection
        - disconnect(): logout and close connection
        - fetch_unseen(limit=None) -> List[raw_message_bytes]
        - idle_loop(callback): optional long-running IDLE loop calling callback(raw_bytes)

    TODOs:
        - Implement robust IDLE handling using IMAP4_SSL or an async library
        - Provide reconnection/backoff
        - Support OAuth2 tokens if required for modern providers
    """

    def __init__(self, host: str, port: int = 993, username: str = '', password: str = '', mailbox: str = 'INBOX'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.mailbox = mailbox
        self.conn: t.Optional[imaplib.IMAP4_SSL] = None

    def connect(self) -> None:
        """Establish IMAP SSL connection and select mailbox.

        Raises imaplib.IMAP4.error on failure.
        """
        context = ssl.create_default_context()
        self.conn = imaplib.IMAP4_SSL(self.host, self.port, ssl_context=context)
        self.conn.login(self.username, self.password)
        self.conn.select(self.mailbox)

    def disconnect(self) -> None:
        """Logout and close connection cleanly."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            try:
                self.conn.logout()
            except Exception:
                pass
            self.conn = None

    def fetch_unseen(self, limit: t.Optional[int] = None) -> t.List[bytes]:
        """Fetch unseen messages as raw RFC822 bytes.

        Returns list of raw message bytes.
        """
        if not self.conn:
            raise RuntimeError('Not connected')
        typ, data = self.conn.search(None, 'UNSEEN')
        if typ != 'OK':
            return []
        ids = data[0].split()
        if limit:
            ids = ids[:limit]
        messages = []
        for mid in ids:
            typ, msg_data = self.conn.fetch(mid, '(RFC822)')
            if typ == 'OK' and msg_data:
                # msg_data is a list of tuples
                raw = msg_data[0][1]
                messages.append(raw)
        return messages

    def idle_loop(self, callback, timeout: int = 300):
        """Long-running loop that calls callback(raw_message_bytes) for new messages.

        Note: imaplib doesn't have a native IDLE helper; a production implementation
        should use libs like `imapclient` or implement the IDLE protocol.
        """
        # TODO: Implement IDLE or efficient polling with exponential backoff
        import time
        while True:
            try:
                msgs = self.fetch_unseen()
                for raw in msgs:
                    callback(raw)
            except Exception as e:
                # TODO: add logging and backoff
                print('imap idle loop error', e)
            time.sleep(10)


if __name__ == '__main__':
    # Example usage (requires real credentials):
    # conn = ImapConnector('imap.example.com', username='user', password='pw')
    # conn.connect()
    # msgs = conn.fetch_unseen()
    # for raw in msgs:
    #     print(raw[:200])
    pass
