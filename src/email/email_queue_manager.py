# Owner: Rohan
# Responsibility: Buffer parsed emails and expose them to the inference pipeline in FIFO order.
# Goals:
# - Provide thread-safe queueing of ParsedEmail objects
# - Deduplicate repeated messages and support priority (e.g., suspected scam flag)
# Integration points:
# - Receives ParsedEmail from `EmailParser` or `ImapConnector` callback
# - Consumer: `detection.model_inference.inference_engine` polls for ready items
# Testing requirements:
# - Tests for concurrency, deduplication, and priority behavior

import queue
import typing as t
from .email_parser import ParsedEmail


class EmailQueueManager:
    """
    Thread-safe queue manager for parsed emails.

    Methods:
        - push(parsed_email)
        - get(timeout=None) -> ParsedEmail or None
        - size() -> int

    TODOs:
        - Implement dedupe by message-id
        - Allow priority re-queueing for suspicious messages
    """

    def __init__(self, maxsize: int = 100):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._seen_ids = set()

    def push(self, parsed_email: ParsedEmail) -> bool:
        """Push parsed email into queue; return True if enqueued, False if dropped/deduped."""
        mid = parsed_email.message_id
        if mid and mid in self._seen_ids:
            return False
        try:
            self._q.put(parsed_email, block=False)
            if mid:
                self._seen_ids.add(mid)
            return True
        except queue.Full:
            # TODO: implement overflow policy (drop oldest, expand queue, etc.)
            return False

    def get(self, timeout: t.Optional[float] = None) -> t.Optional[ParsedEmail]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def size(self) -> int:
        return self._q.qsize()


if __name__ == '__main__':
    # Simple manual test
    from .email_parser import ParsedEmail
    q = EmailQueueManager()
    sample = ParsedEmail(message_id='<1>', from_addr='a@b', to_addrs=['c@d'], subject='hi', date='', text='hello', html=None, attachments=[])
    print('pushed', q.push(sample))
    print('size', q.size())
    print('got', q.get())
