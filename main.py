"""
Main orchestrator for the Raspberry Pi Scam Detection System.

Purpose:
 - Wire audio capture, email ingestion, inference, and alerts into a running application.
 - Provide example startup/shutdown flow and integration notes.

Responsibilities for the owner (orchestrator):
 - Ensure each module's `start()` / `connect()` methods are called in correct order
 - Provide safe shutdown and resource cleanup (GPIO, audio devices, IMAP sessions)

This file is intentionally a high-level orchestrator with detailed comments and TODOs.

Example run (development):
    python main.py

NOTE: This file does not implement production-grade process supervision. For deployment,
use systemd or a container and ensure the app runs as an unprivileged user.
"""

import threading
import time
from src.audio.voip_audio_capture import VoipAudioCapture
from src.audio.audio_preprocessor import AudioPreprocessor
from src.audio.audio_stream_handler import AudioStreamHandler
from src.email.imap_connector import ImapConnector
from src.email.email_parser import EmailParser
from src.email.email_queue_manager import EmailQueueManager
from src.detection.model_inference.inference_engine import InferenceEngine
from src.alerts.alert_logger import AlertLogger
from src.alerts.audio_alert_player import AudioAlertPlayer
from src.alerts.visual_indicators import VisualIndicators
from src.config.model_config import ModelConfig
from src.config.email_config import EmailConfig


def run_orchestrator():
    """Set up components and run the main loop.

    The orchestrator demonstrates how modules integrate. Each team member should
    implement real logic in their modules; this script shows one pattern for wiring them.

    Contract:
      - Inputs: configuration objects
      - Outputs: logs, alerts via speakers/LEDs, web dashboard data
      - Error modes: components may raise; orchestrator should catch and restart where sensible

    Success criteria:
      - IMAP connector downloads new emails, parser enqueues them
      - Audio capture provides frames, preprocessor forms segments for inference
      - InferenceEngine returns probabilities; if >= ALERT_THRESHOLD_PERCENT, alerts are logged + played

    """

    # Configuration
    mcfg = ModelConfig()
    ecfg = EmailConfig()

    # Components
    imap = ImapConnector(ecfg.IMAP_HOST, ecfg.IMAP_PORT, ecfg.USERNAME, ecfg.PASSWORD, mailbox=ecfg.MAILBOX)
    email_parser = EmailParser()
    email_q = EmailQueueManager()

    audio_cap = VoipAudioCapture(sip_config={'server': 'sip.example'}, use_twilio=False)
    pre = AudioPreprocessor(target_sr=16000)
    stream_handler = AudioStreamHandler(segment_seconds=5.0)

    infer = InferenceEngine(text_model_name=mcfg.TEXT_MODEL_EN, device=mcfg.DEVICE)
    infer.load_model(mcfg.TEXT_MODEL_EN)

    alert_logger = AlertLogger()
    audio_alert = AudioAlertPlayer()
    leds = VisualIndicators()

    # Example threads: IMAP poller and email consumer
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
                alert_logger.log_alert(event)
                audio_alert.play_message_tts(f'Scams detected with {prob:.0f} percent confidence')
                leds.set_alert()
            else:
                leds.set_ok()

    # Start threads
    t_imap = threading.Thread(target=imap_poller, daemon=True)
    t_email_consumer = threading.Thread(target=email_consumer, daemon=True)
    t_imap.start()
    t_email_consumer.start()

    # Audio capture example (skeleton) - in production this should be event/callback driven
    audio_cap.start()
    stream_handler.start()

    try:
        print('Orchestrator running. Press Ctrl+C to stop.')
        while True:
            # Example check: poll audio segments and run inference
            seg = stream_handler.get_segment(timeout=2.0)
            if seg:
                start_ts, end_ts, pcm_bytes, sr = seg
                audio_array, sr = pre.normalize(pcm_bytes, sr, 1)
                # TODO: call STT engine to get transcript, here we use a placeholder
                transcript = '<transcript placeholder>'
                r = infer.classify_transcript(transcript)
                prob = r.get('scam_probability', 0.0)
                if prob >= mcfg.ALERT_THRESHOLD_PERCENT:
                    event = {'source': 'call', 'scam_probability': prob, 'type': 'voice', 'note': transcript}
                    alert_logger.log_alert(event)
                    audio_alert.play_beep('warning')
                    leds.set_alert()
                else:
                    leds.set_ok()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('Shutting down')
    finally:
        audio_cap.stop()
        stream_handler.stop()
        leds.cleanup()


if __name__ == '__main__':
    run_orchestrator()
