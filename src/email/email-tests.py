from imap_connector import ImapConnector
from email.email_parser import EmailParser
from email.email_queue_manager import EmailQueueManager

from src.config.model_config import ModelConfig
from src.config.email_config import EmailConfig

print("hello world")

mcfg = ModelConfig()
ecfg = EmailConfig()

imap = ImapConnector(ecfg.IMAP_HOST, ecfg.IMAP_PORT, ecfg.USERNAME, ecfg.PASSWORD, mailbox=ecfg.MAILBOX)
email_parser = EmailParser()
email_q = EmailQueueManager()

import threading
import time

def imap_poller():
    try:
        imap.connect()
        print('IMAP connected')
        # Simple poll loop; replace with IMAP IDLE in production
        while True:
            raws = imap.fetch_unseen(limit=10)
            for raw in raws:
                parsed = email_parser.parse(raw)
                email_q.push(parsed)
            time.sleep(ecfg.POLL_INTERVAL_S)
    except Exception as e:
        print('IMAP poller error', e)
    finally:
        imap.disconnect()

def email_consumer():
    while True:
        item = email_q.get(timeout=5)
        if not item:
            continue
        res = infer.classify_email(item)
        prob = res.get('scam_probability', 0.0)
        if prob >= mcfg.ALERT_THRESHOLD_PERCENT:
            event = {'source': 'email', 'scam_probability': prob, 'message_id': item.message_id, 'type':'email'}            
            # audio_alert.play_message_tts(f'Scams detected with {prob:.0f} percent confidence')
            print("ALERT DETECTED")            
        else:
            # leds.set_ok()
            print("All clear")

# Start threads
t_imap = threading.Thread(target=imap_poller, daemon=True)
t_email_consumer = threading.Thread(target=email_consumer, daemon=True)
t_imap.start()
t_email_consumer.start()
    
    