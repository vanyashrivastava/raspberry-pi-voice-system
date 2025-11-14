"""
Supabase-integrated Alert Logger
Owner: Siddhant
Responsibility: Log alerts to local storage AND sync to Supabase automatically.

Integration:
- Called by inference engine when suspicious events detected
- Automatically forwards to Supabase in background
- Falls back to local file if Supabase is unavailable
"""

import json
import logging
import os
import threading
import time
import typing as t
from datetime import datetime
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class SupabaseAlertLogger:
    """
    Persist alert events locally and sync to Supabase cloud database.

    Features:
    - Local file backup for reliability
    - Background thread for async Supabase sync
    - Automatic retry logic
    - Graceful degradation if Supabase unavailable
    """

    def __init__(
        self,
        local_path: str = '/var/log/raspi_scam_alerts.log',
        use_supabase: bool = True,
        supabase_db=None,
        sync_interval_seconds: int = 5,
    ):
        """
        Initialize alert logger.

        Args:
            local_path: Path to local log file
            use_supabase: Whether to enable Supabase sync
            supabase_db: SupabaseDB instance (lazy-loaded if None)
            sync_interval_seconds: How often to flush to Supabase
        """
        self.local_path = local_path
        self.use_supabase = use_supabase
        self.supabase_db = supabase_db
        self.sync_interval = sync_interval_seconds
        self.logger = logger

        # Ensure local directory exists
        d = os.path.dirname(local_path)
        if d and not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                self.local_path = './raspi_scam_alerts.log'

        # Queue for async Supabase syncing
        self.sync_queue = Queue()
        self.stop_sync = threading.Event()
        
        if self.use_supabase:
            self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self.sync_thread.start()

    def _get_supabase_db(self):
        """Lazy load Supabase DB if not provided."""
        if self.supabase_db is None and self.use_supabase:
            try:
                from src.supabase.db_manager import SupabaseDB
                self.supabase_db = SupabaseDB()
            except Exception as e:
                self.logger.warning(f'Failed to initialize Supabase: {e}')
                self.use_supabase = False
        return self.supabase_db

    def log_alert(self, event: dict) -> None:
        """
        Log an alert event locally and queue for Supabase sync.

        Args:
            event: Dictionary with alert data
                - type: 'email', 'voice', 'call'
                - scam_probability: float 0-100
                - source: string
                - message_id: optional
                - note: optional transcript
                - timestamp: optional (auto-added if missing)
        """
        # Add timestamp if missing
        if 'ts' not in event:
            event['ts'] = time.time()

        # Log locally (always reliable)
        entry = {'ts': event.get('ts'), **event}
        try:
            with open(self.local_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f'Failed to write local alert log: {e}')

        # Queue for Supabase sync if enabled
        if self.use_supabase:
            self.sync_queue.put(event)

    def _sync_worker(self):
        """Background thread that syncs queued alerts to Supabase."""
        self.logger.info('Alert sync worker started')
        
        while not self.stop_sync.is_set():
            try:
                db = self._get_supabase_db()
                if not db:
                    time.sleep(self.sync_interval)
                    continue

                # Collect batch of alerts from queue
                batch = []
                try:
                    while len(batch) < 50:  # Max 50 per batch
                        event = self.sync_queue.get(timeout=self.sync_interval)
                        # Convert to Supabase format
                        record = {
                            'alert_type': event.get('type', 'unknown'),
                            'scam_probability': event.get('scam_probability', 0),
                            'source': event.get('source', 'system'),
                            'message_id': event.get('message_id'),
                            'description': event.get('note'),
                            'metadata': {
                                'original_ts': event.get('ts'),
                                'extra': event.get('metadata'),
                            },
                        }
                        batch.append(record)
                except Empty:
                    pass

                # Sync batch if we have records
                if batch:
                    result = db.batch_insert('alerts', batch)
                    if result.get('success'):
                        self.logger.info(f'Synced {len(batch)} alerts to Supabase')
                    else:
                        self.logger.warning(f'Failed to sync alerts: {result.get("error")}')
                        # Put alerts back in queue to retry
                        for record in batch:
                            self.sync_queue.put(record)

            except Exception as e:
                self.logger.error(f'Sync worker error: {e}')
                time.sleep(self.sync_interval)

    def recent(self, n: int = 100) -> t.List[dict]:
        """
        Get recent alerts from local file.

        Args:
            n: Number of recent records to return

        Returns:
            List of alert dictionaries
        """
        try:
            with open(self.local_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-n:]
            return [json.loads(l) for l in lines]
        except FileNotFoundError:
            return []

    def fetch_cloud_alerts(self, limit: int = 100) -> t.List[dict]:
        """
        Fetch alerts from Supabase cloud database.

        Args:
            limit: Maximum records to fetch

        Returns:
            List of alerts from cloud
        """
        db = self._get_supabase_db()
        if db:
            return db.fetch_alerts(limit=limit)
        return []

    def shutdown(self):
        """Gracefully shut down the sync worker."""
        if self.use_supabase:
            self.logger.info('Shutting down alert sync worker')
            self.stop_sync.set()
            self.sync_thread.join(timeout=5)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = SupabaseAlertLogger(local_path='./test_alerts.log')
    
    # Test logging
    logger.log_alert({
        'type': 'email',
        'scam_probability': 85.5,
        'source': 'inference_engine',
        'message_id': 'msg_123',
        'note': 'Suspicious sender detected',
    })
    
    print('Logged alert to local file and queued for Supabase')
    time.sleep(2)  # Wait for sync
    
    logger.shutdown()
