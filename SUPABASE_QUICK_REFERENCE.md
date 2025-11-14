# Supabase Quick Reference

## Essential Commands

### 1. Test Supabase Connection
```bash
python -c "from src.supabase.db_manager import SupabaseDB; db = SupabaseDB(); print('Connected!' if db.test_connection() else 'Failed')"
```

### 2. Run Integration Tests
```bash
python src/supabase/db_manager.py
```

### 3. View Integration Examples
```bash
python src/supabase/integration_examples.py
```

## Environment Setup

### Create `.env` file
```bash
cp .env.example .env
# Edit with your Supabase credentials
```

### Required Environment Variables
```
SUPABASE_URL=https://your_project_id.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

## Code Snippets

### Insert Alert
```python
from src.supabase.db_manager import SupabaseDB

db = SupabaseDB()
db.insert_alert({
    'alert_type': 'email',
    'scam_probability': 87.5,
    'source': 'inference_engine',
    'description': 'Phishing attempt detected'
})
```

### Insert Transcript
```python
db.insert_transcript({
    'transcript': 'Can you help me with my account?',
    'audio_duration_seconds': 45.2,
    'confidence_score': 0.95,
    'source': 'voip',
    'call_id': 'call_12345'
})
```

### Insert Email Event
```python
db.insert_email_event({
    'email_subject': 'Urgent: Update Payment',
    'email_from': 'attacker@fake.com',
    'email_to': 'victim@example.com',
    'scam_probability': 92.0,
    'classification': 'scam',
    'message_id': 'msg_xyz'
})
```

### Batch Insert
```python
alerts = [
    {'alert_type': 'email', 'scam_probability': 85.5, 'source': 'test'},
    {'alert_type': 'voice', 'scam_probability': 92.0, 'source': 'test'},
]
db.batch_insert('alerts', alerts)
```

### Insert System Log
```python
db.insert_system_log(
    log_level='ERROR',
    message='Failed to process audio',
    component='audio_preprocessor'
)
```

### Insert Metric
```python
db.insert_metric(
    metric_name='inference_latency',
    metric_value=145.5,
    unit='ms',
    component='inference_engine'
)
```

## Using SupabaseAlertLogger (Recommended)

```python
from src.supabase.supabase_alert_logger import SupabaseAlertLogger

# Create logger with automatic Supabase sync
logger = SupabaseAlertLogger(use_supabase=True)

# Log alert (syncs to both local file and Supabase)
logger.log_alert({
    'type': 'email',
    'scam_probability': 85.5,
    'source': 'inference_engine',
    'message_id': 'msg_123'
})

# Fetch recent local alerts
local_alerts = logger.recent(n=50)

# Fetch recent cloud alerts
cloud_alerts = logger.fetch_cloud_alerts(limit=50)

# Shutdown
logger.shutdown()
```

## Database Tables

| Table | Purpose | Key Fields |
|-------|---------|-----------|
| `audio_transcripts` | Audio transcriptions | transcript, source, call_id, confidence_score |
| `alerts` | Scam detection alerts | alert_type, scam_probability, source |
| `email_events` | Email analysis results | email_from, scam_probability, classification |
| `system_logs` | System events & errors | log_level, component, message |
| `system_metrics` | Performance metrics | metric_name, metric_value, unit |

## Useful SQL Queries

### Recent High-Confidence Alerts
```sql
SELECT * FROM alerts 
WHERE scam_probability > 80 
ORDER BY created_at DESC 
LIMIT 20;
```

### Transcripts from Last 24 Hours
```sql
SELECT * FROM audio_transcripts 
WHERE created_at >= NOW() - INTERVAL '1 day'
ORDER BY created_at DESC;
```

### Email Detection Statistics
```sql
SELECT 
  classification,
  COUNT(*) as count,
  AVG(scam_probability) as avg_confidence
FROM email_events 
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY classification;
```

### System Performance
```sql
SELECT 
  component,
  metric_name,
  AVG(metric_value) as avg_value,
  MAX(metric_value) as max_value,
  MIN(metric_value) as min_value
FROM system_metrics
WHERE created_at >= NOW() - INTERVAL '1 day'
GROUP BY component, metric_name;
```

## GitHub Actions

### Trigger Sync Manually
```bash
git add supabase/migrations/
git commit -m "Update database schema"
git push origin main
```

## Troubleshooting

### Test Connection Failed
```bash
# Check environment variables
python -c "import os; print('URL:', os.getenv('SUPABASE_URL'))"

# Verify they're set in .env
cat .env
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Data Not Appearing
```bash
# Check local logs
tail -f ./raspi_scam_alerts.log

# Check Supabase logs
# Go to Supabase Dashboard > Logs
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│            Raspberry Pi Scam Detection System                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   VOIP       │  │   Email      │  │   System     │      │
│  │   Audio      │  │   Parsing    │  │   Metrics    │      │
│  └────────┬─────┘  └────────┬─────┘  └────────┬─────┘      │
│           │                 │                 │             │
│           ▼                 ▼                 ▼             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Supabase Database Module (src/)             │   │
│  │    - SupabaseDB: Core database operations           │   │
│  │    - SupabaseAlertLogger: Auto-sync alerts          │   │
│  └────────────┬────────────────────────────────────────┘   │
│               │                                              │
└───────────────┼──────────────────────────────────────────────┘
                │
                │ HTTPS (REST API)
                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Supabase Cloud                            │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL Database                                        │
│  ├─ audio_transcripts  (transcribed voice)                 │
│  ├─ alerts             (detection alerts)                  │
│  ├─ email_events       (email analysis)                    │
│  ├─ system_logs        (system events)                     │
│  └─ system_metrics     (performance data)                  │
│                                                             │
│  Backup & Monitoring                                       │
│  ├─ Automatic backups                                      │
│  ├─ Real-time sync                                         │
│  └─ Query analytics                                        │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Checklist

- [ ] Set Supabase credentials in GitHub Secrets
- [ ] Run database migration in Supabase
- [ ] Test connection locally
- [ ] Update main.py with Supabase integration
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test on Raspberry Pi
- [ ] Monitor Supabase logs
- [ ] Set up email/Slack alerts for high-confidence detections
- [ ] Document custom configurations

## Performance Tips

1. **Batch Inserts**: Use `batch_insert()` for multiple records
2. **Async Sync**: Use `SupabaseAlertLogger` for non-blocking inserts
3. **Index Queries**: Frequently queried columns are already indexed
4. **Rate Limits**: Supabase has limits; monitor usage in dashboard
5. **Local Fallback**: All data backed up locally before cloud sync

---

**Last Updated**: 2025-11-14
**Status**: Production Ready ✅
