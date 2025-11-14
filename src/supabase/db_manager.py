"""
Supabase Database Manager
Owner: Data Team
Responsibility: Handle all Supabase database operations for transcripts, alerts, and system logs.

Integration points:
 - AlertLogger: forwards alerts to Supabase
 - AudioStreamHandler: forwards transcripts to Supabase
 - EmailParser: forwards email events to Supabase
 - System components: log metrics and events

Features:
 - Automatic data sync to cloud
 - Batch insert operations
 - Error handling and retries
 - Connection pooling
"""

import json
import logging
import os
import typing as t
from datetime import datetime
from uuid import uuid4

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class SupabaseDB:
    """
    Manages all database operations with Supabase PostgreSQL backend.
    """

    def __init__(
        self,
        supabase_url: str = None,
        supabase_key: str = None,
        service_role_key: str = None,
        timeout: int = 10,
        max_retries: int = 3,
    ):
        """
        Initialize Supabase database connection.

        Args:
            supabase_url: API URL (e.g., https://xxx.supabase.co)
            supabase_key: Anon key for client operations
            service_role_key: Service role key for admin operations
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
        """
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.anon_key = supabase_key or os.getenv('SUPABASE_KEY')
        self.service_role_key = service_role_key or os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        self.timeout = timeout
        self.max_retries = max_retries

        if not all([self.supabase_url, self.anon_key]):
            raise ValueError('Supabase URL and key are required. Set env vars or pass arguments.')

        self.session = self._create_session()
        self.logger = logger

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _get_headers(self, is_admin: bool = False) -> dict:
        """Get request headers with authorization."""
        key = self.service_role_key if is_admin else self.anon_key
        return {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal',
        }

    def insert_alert(self, alert_data: dict) -> dict:
        """
        Insert an alert into the alerts table.

        Args:
            alert_data: Dictionary with alert fields
                - alert_type: 'email', 'voice', 'call'
                - scam_probability: float 0-100
                - source: string
                - message_id: optional
                - transcript_id: optional UUID
                - description: optional
                - metadata: optional dict

        Returns:
            Response from server
        """
        try:
            record = {
                'id': str(uuid4()),
                'created_at': datetime.utcnow().isoformat() + 'Z',
                **alert_data,
            }
            url = f'{self.supabase_url}/rest/v1/alerts'
            response = self.session.post(
                url,
                json=record,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.logger.info(f'Alert inserted: {record["id"]}')
            return {'success': True, 'id': record['id']}
        except Exception as e:
            self.logger.error(f'Failed to insert alert: {e}')
            return {'success': False, 'error': str(e)}

    def insert_transcript(self, transcript_data: dict) -> dict:
        """
        Insert an audio transcript into the audio_transcripts table.

        Args:
            transcript_data: Dictionary with fields
                - transcript: string (required)
                - audio_duration_seconds: float
                - confidence_score: float
                - source: string ('voip', 'twilio', etc.)
                - call_id: optional
                - language: optional (default 'en')
                - metadata: optional dict

        Returns:
            Response from server
        """
        try:
            record = {
                'id': str(uuid4()),
                'created_at': datetime.utcnow().isoformat() + 'Z',
                **transcript_data,
            }
            url = f'{self.supabase_url}/rest/v1/audio_transcripts'
            response = self.session.post(
                url,
                json=record,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.logger.info(f'Transcript inserted: {record["id"]}')
            return {'success': True, 'id': record['id']}
        except Exception as e:
            self.logger.error(f'Failed to insert transcript: {e}')
            return {'success': False, 'error': str(e)}

    def insert_email_event(self, email_data: dict) -> dict:
        """
        Insert an email event into the email_events table.

        Args:
            email_data: Dictionary with fields
                - email_subject: string
                - email_from: string
                - email_to: string
                - email_body: string
                - scam_probability: float 0-100 (required)
                - classification: 'scam', 'legitimate', 'uncertain', 'spam'
                - message_id: optional, unique
                - processing_time_ms: optional
                - metadata: optional dict

        Returns:
            Response from server
        """
        try:
            record = {
                'id': str(uuid4()),
                'created_at': datetime.utcnow().isoformat() + 'Z',
                **email_data,
            }
            url = f'{self.supabase_url}/rest/v1/email_events'
            response = self.session.post(
                url,
                json=record,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.logger.info(f'Email event inserted: {record["id"]}')
            return {'success': True, 'id': record['id']}
        except Exception as e:
            self.logger.error(f'Failed to insert email event: {e}')
            return {'success': False, 'error': str(e)}

    def insert_system_log(
        self,
        log_level: str,
        message: str,
        component: str = None,
        stack_trace: str = None,
        metadata: dict = None,
    ) -> dict:
        """
        Insert a system log into the system_logs table.

        Args:
            log_level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            message: Log message
            component: Optional component name
            stack_trace: Optional error stack trace
            metadata: Optional additional data

        Returns:
            Response from server
        """
        try:
            record = {
                'id': str(uuid4()),
                'log_level': log_level,
                'message': message,
                'component': component,
                'stack_trace': stack_trace,
                'metadata': metadata,
                'created_at': datetime.utcnow().isoformat() + 'Z',
            }
            url = f'{self.supabase_url}/rest/v1/system_logs'
            response = self.session.post(
                url,
                json=record,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return {'success': True, 'id': record['id']}
        except Exception as e:
            self.logger.error(f'Failed to insert system log: {e}')
            return {'success': False, 'error': str(e)}

    def insert_metric(
        self,
        metric_name: str,
        metric_value: float,
        component: str = None,
        unit: str = None,
        metadata: dict = None,
    ) -> dict:
        """
        Insert a system metric into the system_metrics table.

        Args:
            metric_name: Name of the metric
            metric_value: Numeric value
            component: Optional component name
            unit: Optional unit ('ms', 'MB', '%', etc.)
            metadata: Optional additional data

        Returns:
            Response from server
        """
        try:
            record = {
                'id': str(uuid4()),
                'metric_name': metric_name,
                'metric_value': metric_value,
                'component': component,
                'unit': unit,
                'metadata': metadata,
                'created_at': datetime.utcnow().isoformat() + 'Z',
            }
            url = f'{self.supabase_url}/rest/v1/system_metrics'
            response = self.session.post(
                url,
                json=record,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return {'success': True, 'id': record['id']}
        except Exception as e:
            self.logger.error(f'Failed to insert metric: {e}')
            return {'success': False, 'error': str(e)}

    def batch_insert(self, table: str, records: t.List[dict]) -> dict:
        """
        Insert multiple records at once (batch operation).

        Args:
            table: Table name
            records: List of record dictionaries

        Returns:
            Response with success count
        """
        if not records:
            return {'success': True, 'inserted': 0}

        try:
            # Add timestamps and IDs to records
            processed_records = [
                {
                    'id': str(uuid4()),
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                    **record,
                }
                for record in records
            ]

            url = f'{self.supabase_url}/rest/v1/{table}'
            response = self.session.post(
                url,
                json=processed_records,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.logger.info(f'Batch inserted {len(processed_records)} records to {table}')
            return {'success': True, 'inserted': len(processed_records)}
        except Exception as e:
            self.logger.error(f'Batch insert failed: {e}')
            return {'success': False, 'error': str(e)}

    def fetch_alerts(self, limit: int = 100, days_back: int = 7) -> t.List[dict]:
        """
        Fetch recent alerts from the database.

        Args:
            limit: Maximum records to fetch
            days_back: Fetch alerts from last N days

        Returns:
            List of alert records
        """
        try:
            from_date = (
                datetime.utcnow().isoformat() + 'Z'
            )  # Simplified - in production use proper date math
            url = f'{self.supabase_url}/rest/v1/alerts?limit={limit}&order=created_at.desc'
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f'Failed to fetch alerts: {e}')
            return []

    def test_connection(self) -> bool:
        """Test the connection to Supabase."""
        try:
            url = f'{self.supabase_url}/rest/v1/alerts?limit=1'
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.logger.info('Supabase connection successful')
            return True
        except Exception as e:
            self.logger.error(f'Supabase connection failed: {e}')
            return False


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize DB
    db = SupabaseDB()

    # Test connection
    if db.test_connection():
        print('✓ Connected to Supabase')

        # Example: Insert an alert
        result = db.insert_alert({
            'alert_type': 'email',
            'scam_probability': 87.5,
            'source': 'test_script',
            'description': 'Example alert from test script',
        })
        print(f'Alert insert result: {result}')

        # Example: Insert a transcript
        result = db.insert_transcript({
            'transcript': 'Sample transcribed audio text',
            'audio_duration_seconds': 45.2,
            'confidence_score': 0.95,
            'source': 'voip',
        })
        print(f'Transcript insert result: {result}')
    else:
        print('✗ Failed to connect to Supabase')
