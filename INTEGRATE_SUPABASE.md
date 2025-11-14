# How to Update Your main.py for Supabase

This guide shows exactly where to add Supabase integration to your existing `main.py`.

## Option 1: Simple Integration (Recommended)

Replace your existing AlertLogger with SupabaseAlertLogger:

### Current Code (in main.py)
```python
from src.alerts.alert_logger import AlertLogger

def run_orchestrator():
    # ... other code ...
    alert_logger = AlertLogger()  # ← This line
    # ... rest of code ...
```

### New Code (with Supabase)
```python
from src.supabase.supabase_alert_logger import SupabaseAlertLogger

def run_orchestrator():
    # ... other code ...
    alert_logger = SupabaseAlertLogger(use_supabase=True)  # ← Updated line
    # ... rest of code ...
```

**That's it!** The alert logger now:
- ✅ Logs to local file (same as before)
- ✅ Automatically syncs to Supabase in background
- ✅ Falls back gracefully if Supabase is unavailable

### Usage (no changes needed!)
```python
# Use exactly the same way:
alert_logger.log_alert({
    'type': 'email',
    'scam_probability': 87.5,
    'source': 'inference_engine',
    'message_id': 'msg_123'
})
```

---

## Option 2: Advanced Integration

If you want direct database access for transcripts and email events:

### Add Database Initialization
```python
from src.supabase.supabase_alert_logger import SupabaseAlertLogger
from src.supabase.db_manager import SupabaseDB

def run_orchestrator():
    # Initialize Supabase
    db = SupabaseDB()
    
    # Test connection (optional, logs a warning if fails)
    if not db.test_connection():
        print("Warning: Could not connect to Supabase (will use local fallback)")
    
    # Initialize enhanced alert logger
    alert_logger = SupabaseAlertLogger(use_supabase=True)
    
    # ... rest of code ...
```

### Sync Transcripts
In your audio processing, after getting a transcript:

```python
# Existing code
transcript = transcription_engine.transcribe(audio_bytes)

# NEW: Sync to Supabase
try:
    db.insert_transcript({
        'transcript': transcript,
        'audio_duration_seconds': segment_duration,
        'confidence_score': confidence,
        'source': 'voip',
        'call_id': call_id,
        'language': 'en'
    })
except Exception as e:
    logger.error(f'Failed to sync transcript: {e}')
    # Continue - local logs still work
```

### Sync Email Events
In your email classification:

```python
# Existing code
prob = res.get('scam_probability', 0.0)
classification = 'scam' if prob >= threshold else 'legitimate'

# NEW: Sync to Supabase
try:
    db.insert_email_event({
        'email_subject': item.subject,
        'email_from': item.sender,
        'email_to': item.recipient,
        'scam_probability': prob,
        'classification': classification,
        'message_id': item.message_id,
        'processing_time_ms': elapsed_ms
    })
except Exception as e:
    logger.error(f'Failed to sync email: {e}')
```

---

## Complete main.py Example

Here's what your full main.py might look like:

```python
"""
Main orchestrator for the Raspberry Pi Scam Detection System.
WITH SUPABASE INTEGRATION
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
from src.alerts.audio_alert_player import AudioAlertPlayer
from src.alerts.visual_indicators import VisualIndicators
from src.config.model_config import ModelConfig
from src.config.email_config import EmailConfig

# NEW: Supabase imports
from src.supabase.supabase_alert_logger import SupabaseAlertLogger
from src.supabase.db_manager import SupabaseDB


def run_orchestrator():
    """Set up components and run the main loop."""

    # Configuration
    mcfg = ModelConfig()
    ecfg = EmailConfig()

    # NEW: Initialize Supabase
    db = SupabaseDB()
    if db.test_connection():
        print('✅ Supabase connected')
    else:
        print('⚠️  Supabase unavailable (will use local fallback)')

    # Components
    imap = ImapConnector(ecfg.IMAP_HOST, ecfg.IMAP_PORT, ecfg.USERNAME, ecfg.PASSWORD, mailbox=ecfg.MAILBOX)
    email_parser = EmailParser()
    email_q = EmailQueueManager()

    audio_cap = VoipAudioCapture(sip_config={'server': 'sip.example'}, use_twilio=False)
    pre = AudioPreprocessor(target_sr=16000)
    stream_handler = AudioStreamHandler(segment_seconds=5.0)

    infer = InferenceEngine(text_model_name=mcfg.TEXT_MODEL_EN, device=mcfg.DEVICE)
    infer.load_model(mcfg.TEXT_MODEL_EN)

    # NEW: Enhanced alert logger with auto-sync
    alert_logger = SupabaseAlertLogger(use_supabase=True)
    
    audio_alert = AudioAlertPlayer()
    leds = VisualIndicators()

    # IMAP poller
    def imap_poller():
        try:
            imap.connect()
            print('IMAP connected')
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

    # Email consumer
    def email_consumer():
        while True:
            item = email_q.get(timeout=5)
            if not item:
                continue
            
            res = infer.classify_email(item)
            prob = res.get('scam_probability', 0.0)
            
            # NEW: Sync email event to Supabase
            try:
                db.insert_email_event({
                    'email_subject': item.subject,
                    'email_from': item.sender,
                    'email_to': item.recipient,
                    'scam_probability': prob,
                    'classification': 'scam' if prob >= 70 else 'legitimate',
                    'message_id': item.message_id
                })
            except Exception as e:
                print(f'Email sync error: {e}')
            
            # Alert if suspicious
            if prob >= mcfg.ALERT_THRESHOLD_PERCENT:
                event = {
                    'type': 'email',
                    'scam_probability': prob,
                    'message_id': item.message_id,
                    'source': 'inference_engine'
                }
                alert_logger.log_alert(event)  # Auto-syncs to Supabase
                audio_alert.play_message_tts(f'Scam detected with {prob:.0f} percent confidence')
                leds.set_alert()
            else:
                leds.set_ok()

    # Start threads
    t_imap = threading.Thread(target=imap_poller, daemon=True)
    t_email_consumer = threading.Thread(target=email_consumer, daemon=True)
    t_imap.start()
    t_email_consumer.start()

    # Audio capture
    audio_cap.start()
    stream_handler.start()

    try:
        print('Orchestrator running. Press Ctrl+C to stop.')
        while True:
            seg = stream_handler.get_segment(timeout=2.0)
            if seg:
                start_ts, end_ts, pcm_bytes, sr = seg
                audio_array, sr = pre.normalize(pcm_bytes, sr, 1)
                
                # Placeholder transcript (replace with real STT)
                transcript = '<transcript placeholder>'
                
                # NEW: Sync transcript to Supabase
                try:
                    db.insert_transcript({
                        'transcript': transcript,
                        'audio_duration_seconds': (end_ts - start_ts) / 1000.0,
                        'confidence_score': 0.95,
                        'source': 'voip'
                    })
                except Exception as e:
                    print(f'Transcript sync error: {e}')
                
                # Check for scams
                r = infer.classify_transcript(transcript)
                prob = r.get('scam_probability', 0.0)
                
                if prob >= mcfg.ALERT_THRESHOLD_PERCENT:
                    event = {
                        'type': 'voice',
                        'scam_probability': prob,
                        'source': 'inference_engine',
                        'note': transcript
                    }
                    alert_logger.log_alert(event)  # Auto-syncs to Supabase
                    audio_alert.play_beep('warning')
                    leds.set_alert()
                else:
                    leds.set_ok()
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print('Shutting down')
    
    finally:
        # NEW: Graceful shutdown
        alert_logger.shutdown()
        
        audio_cap.stop()
        stream_handler.stop()
        leds.cleanup()


if __name__ == '__main__':
    run_orchestrator()
```

---

## Testing Your Integration

After updating main.py:

### 1. Test locally
```bash
python test_supabase_query.py
```

### 2. Push to GitHub
```bash
git add main.py src/
git commit -m "Add Supabase integration to main.py"
git push origin main
```

### 3. Watch GitHub Actions
- Go to Actions tab
- Watch workflows run
- See your data being synced!

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'src.supabase'"**
- Make sure you're in the project root directory
- Run: `pip install -r requirements.txt`

**"Connection failed" when running main.py**
- Check `.env` has correct credentials
- Run: `python test_supabase_query.py` to debug

**Data not appearing in Supabase**
- Check GitHub Actions logs for errors
- Verify tables exist in Supabase Dashboard
- Run local test to verify connection first

---

## Key Points

✅ **Backward compatible** - Works with your existing code
✅ **Graceful fallback** - Uses local logs if Supabase unavailable  
✅ **Non-blocking** - Sync happens in background thread
✅ **Easy integration** - Just swap AlertLogger class
✅ **Production ready** - Error handling and retry logic included

You're ready to start syncing data to the cloud! 🚀
