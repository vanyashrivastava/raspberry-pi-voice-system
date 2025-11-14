"""
Example integration showing how to wire Supabase into existing components.

This demonstrates how to modify your audio stream handler, email parser,
and alert logger to automatically sync to Supabase.
"""

import logging
from datetime import datetime

# Example 1: Integrate with AudioStreamHandler
# =============================================

def example_audio_transcript_sync():
    """
    Show how to modify audio processing to sync transcripts to Supabase.
    
    Add this to your AudioStreamHandler.handle_segment() method:
    """
    
    from src.supabase.db_manager import SupabaseDB
    from src.audio.audio_stream_handler import AudioStreamHandler
    
    code_example = """
    # In src/audio/audio_stream_handler.py, in handle_segment() method:
    
    async def handle_segment(self, segment):
        # ... existing processing ...
        transcript = transcription_engine.transcribe(audio_bytes)
        
        # NEW: Sync transcript to Supabase
        try:
            db = SupabaseDB()
            db.insert_transcript({
                'transcript': transcript,
                'audio_duration_seconds': segment_duration,
                'confidence_score': confidence,
                'source': 'voip',
                'call_id': call_id,
                'language': 'en',
                'metadata': {
                    'sample_rate': sample_rate,
                    'channels': num_channels,
                    'processing_time_ms': processing_time
                }
            })
        except Exception as e:
            logger.error(f'Failed to sync transcript: {e}')
            # Continue processing even if sync fails
    """
    
    print(code_example)


# Example 2: Integrate with EmailParser
# ======================================

def example_email_event_sync():
    """
    Show how to modify email parsing to sync events to Supabase.
    
    Add this to your EmailParser or InferenceEngine classification:
    """
    
    from src.supabase.db_manager import SupabaseDB
    from src.email.email_parser import EmailParser
    
    code_example = """
    # In src/detection/model_inference/inference_engine.py, in classify_email():
    
    def classify_email(self, email_obj):
        # ... existing classification logic ...
        
        # Get classification result
        classification = 'scam' if scam_prob >= threshold else 'legitimate'
        
        # NEW: Sync to Supabase
        try:
            db = SupabaseDB()
            db.insert_email_event({
                'email_subject': email_obj.subject,
                'email_from': email_obj.sender,
                'email_to': email_obj.recipient,
                'email_body': email_obj.body[:1000],  # First 1000 chars
                'scam_probability': scam_prob,
                'classification': classification,
                'message_id': email_obj.message_id,
                'processing_time_ms': elapsed_time,
                'flags': {
                    'has_links': has_links,
                    'has_attachments': has_attachments,
                    'suspicious_keywords': suspicious_kw
                },
                'metadata': {
                    'model_version': self.model_version,
                    'model_name': self.text_model_name
                }
            })
        except Exception as e:
            logger.error(f'Failed to sync email event: {e}')
        
        return {
            'scam_probability': scam_prob,
            'classification': classification
        }
    """
    
    print(code_example)


# Example 3: Integrate with AlertLogger
# =====================================

def example_alert_sync():
    """
    Show how to use the new SupabaseAlertLogger (recommended approach).
    
    This handles both local and cloud sync automatically:
    """
    
    from src.supabase.supabase_alert_logger import SupabaseAlertLogger
    
    code_example = """
    # In main.py, replace:
    #   alert_logger = AlertLogger()
    # With:
    
    alert_logger = SupabaseAlertLogger(
        local_path='/var/log/raspi_scam_alerts.log',
        use_supabase=True,
        sync_interval_seconds=5
    )
    
    # Then in your detection code, use it the same way:
    alert_logger.log_alert({
        'type': 'email',
        'scam_probability': 87.5,
        'source': 'inference_engine',
        'message_id': 'msg_12345',
        'note': 'Detected phishing attempt',
        'metadata': {
            'alert_level': 'high',
            'recipient_email': 'victim@example.com'
        }
    })
    
    # The alert is:
    # 1. Written to local file immediately (reliable)
    # 2. Queued for async Supabase sync
    # 3. Automatically retried if Supabase is down
    """
    
    print(code_example)


# Example 4: Complete Integration in main.py
# ==========================================

def example_complete_main_py():
    """
    Show the complete modified main.py with Supabase integration.
    """
    
    code_example = '''
    """
    Main orchestrator for Raspberry Pi Scam Detection System - Supabase Edition
    """
    
    import threading
    import time
    from src.supabase.supabase_alert_logger import SupabaseAlertLogger
    from src.supabase.db_manager import SupabaseDB
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
    
    
    def run_orchestrator():
        # Initialize Supabase
        db = SupabaseDB()
        if not db.test_connection():
            print("Warning: Could not connect to Supabase")
        
        # Replace AlertLogger with Supabase version
        alert_logger = SupabaseAlertLogger(use_supabase=True)
        
        # ... rest of existing code ...
        
        # In your email consumer:
        def email_consumer():
            while True:
                item = email_q.get(timeout=5)
                if not item:
                    continue
                
                res = infer.classify_email(item)
                prob = res.get('scam_probability', 0.0)
                
                # Sync email event to Supabase
                try:
                    db.insert_email_event({
                        'email_subject': item.subject,
                        'email_from': item.sender,
                        'scam_probability': prob,
                        'classification': 'scam' if prob >= 70 else 'legitimate',
                        'message_id': item.message_id
                    })
                except Exception as e:
                    print(f"Supabase sync error: {e}")
                
                # Log alert (automatically syncs to Supabase)
                if prob >= 70:
                    alert_logger.log_alert({
                        'type': 'email',
                        'scam_probability': prob,
                        'message_id': item.message_id,
                        'source': 'inference_engine'
                    })
        
        # ... rest of code ...
        
        # On shutdown
        try:
            alert_logger.shutdown()
        except:
            pass
    
    
    if __name__ == '__main__':
        run_orchestrator()
    '''
    
    print(code_example)


# Example 5: Querying Data
# ========================

def example_querying_data():
    """
    Show how to query data from Supabase.
    """
    
    from src.supabase.db_manager import SupabaseDB
    
    code_example = """
    # Fetch recent alerts from Supabase
    db = SupabaseDB()
    recent_alerts = db.fetch_alerts(limit=50, days_back=7)
    
    for alert in recent_alerts:
        print(f"Alert: {alert['alert_type']} - {alert['scam_probability']}%")
    
    # Get from local log too
    from src.supabase.supabase_alert_logger import SupabaseAlertLogger
    logger = SupabaseAlertLogger()
    local_alerts = logger.recent(n=100)
    """
    
    print(code_example)


if __name__ == '__main__':
    print("=" * 70)
    print("SUPABASE INTEGRATION EXAMPLES")
    print("=" * 70)
    
    print("\n1. Audio Transcript Sync:")
    print("-" * 70)
    example_audio_transcript_sync()
    
    print("\n2. Email Event Sync:")
    print("-" * 70)
    example_email_event_sync()
    
    print("\n3. Alert Sync (Recommended):")
    print("-" * 70)
    example_alert_sync()
    
    print("\n4. Complete main.py Integration:")
    print("-" * 70)
    example_complete_main_py()
    
    print("\n5. Querying Data:")
    print("-" * 70)
    example_querying_data()
    
    print("\n" + "=" * 70)
    print("See SUPABASE_SETUP.md for detailed setup instructions")
    print("=" * 70)
