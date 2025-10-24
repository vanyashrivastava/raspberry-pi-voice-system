# email package initializer
# Owner: Rohan
# Purpose: Expose IMAP connector and parsing utilities to the rest of the system.

from .imap_connector import ImapConnector
from .email_parser import EmailParser
from .email_queue_manager import EmailQueueManager

__all__ = ['ImapConnector', 'EmailParser', 'EmailQueueManager']
