# Owner: Rohan
# Responsibility: Parse raw RFC822 bytes into structured email objects and extract text for classification.
# Goals:
# - Decode multipart emails and attachments safely
# - Provide plaintext and HTML fallback extraction
# - Sanitize and normalize text for ML consumption
# Integration points:
# - Consumes raw bytes from `ImapConnector`
# - Provides structured messages to `email_queue_manager` and `detection.model_inference`
# Testing requirements:
# - Unit tests for various MIME types, charsets, and malformed payloads

import typing as t
import email as py_email
from email.policy import default

class ParsedEmail(t.NamedTuple):
    message_id: str
    from_addr: str
    to_addrs: t.List[str]
    subject: str
    date: str
    text: str
    html: t.Optional[str]
    attachments: t.List[t.Tuple[str, bytes]]  # (filename, bytes)


class EmailParser:
    """
    Parse raw RFC822 bytes into a ParsedEmail namedtuple ready for analysis.

    Public methods:
        - parse(raw_bytes) -> ParsedEmail

    TODOs:
        - Improve HTML-to-text sanitization
        - Implement attachment scanning limits to avoid OOM
    """

    def parse(self, raw_bytes: bytes) -> ParsedEmail:
        """Parse raw RFC822 message bytes.

        Args:
            raw_bytes: raw message bytes from IMAP FETCH RFC822

        Returns:
            ParsedEmail with extracted fields and text suitable for classification.
        """
        msg = py_email.message_from_bytes(raw_bytes, policy=default)
        subject = msg.get('subject', '')
        from_addr = msg.get('from', '')
        to_addrs = msg.get_all('to', []) or []
        date = msg.get('date', '')
        message_id = msg.get('message-id', '')

        text_parts = []
        html_part = None
        attachments = []

        for part in msg.walk():
            ctype = part.get_content_type()
            disp = part.get_content_disposition()
            if disp == 'attachment':
                filename = part.get_filename()
                payload = part.get_payload(decode=True) or b''
                attachments.append((filename or 'unknown', payload))
            elif ctype == 'text/plain' and part.get_content():
                text_parts.append(part.get_content())
            elif ctype == 'text/html' and part.get_content():
                # Keep one HTML fallback
                html_part = part.get_content()

        text = '\n'.join(text_parts).strip()
        if not text and html_part:
            # TODO: sanitize and convert HTML to text (e.g., using BeautifulSoup)
            text = html_part  # placeholder; implement sanitizer

        return ParsedEmail(
            message_id=message_id,
            from_addr=from_addr,
            to_addrs=to_addrs,
            subject=subject,
            date=date,
            text=text,
            html=html_part,
            attachments=attachments,
        )


if __name__ == '__main__':
    # Example parser usage skeleton
    p = EmailParser()
    # raw = open('sample_email.eml','rb').read()
    # parsed = p.parse(raw)
    # print(parsed.subject, len(parsed.text))
    pass
