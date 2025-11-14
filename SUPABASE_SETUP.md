# Supabase Setup & Integration Guide

This guide walks you through setting up Supabase with your GitHub repository to automatically populate cloud tables with audio transcripts, timestamps, and alerts.

## Quick Start (5 minutes)

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Sign up or log in
3. Click **"New Project"**
4. Fill in:
   - **Name**: `raspberry-pi-scam-detection`
   - **Password**: Save this securely
   - **Region**: Choose closest to you
5. Click **Create new project**

### 2. Get Your Credentials

1. Go to **Settings** → **API** (left sidebar)
2. Copy these values:
   - `Project URL` → `SUPABASE_URL`
   - `anon key` → `SUPABASE_KEY`
   - `service_role key` → `SUPABASE_SERVICE_ROLE_KEY`

### 3. Setup GitHub Secrets

1. Go to your GitHub repo: **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add:
   - Name: `SUPABASE_URL` | Value: (your project URL)
   - Name: `SUPABASE_KEY` | Value: (your anon key)
   - Name: `SUPABASE_SERVICE_ROLE_KEY` | Value: (your service role key)

### 4. Create Database Tables

1. In Supabase dashboard, go to **SQL Editor**
2. Click **New Query**
3. Paste the entire content of `supabase/migrations/001_init_schema.sql`
4. Click **Run**

### 5. Update Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in:
   ```bash
   SUPABASE_URL=https://your_project_id.supabase.co
   SUPABASE_KEY=your_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Flow & Automatic Syncing

### Audio Transcripts

When audio is processed and transcribed:

```
Audio Capture 
  ↓
Preprocessor 
  ↓
Transcription Engine 
  ↓
AudioStreamHandler.handle_segment() 
  ↓
SupabaseDB.insert_transcript() 
  ↓
📊 Supabase: audio_transcripts table
```

**Example record:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "transcript": "Can you hear me clearly?",
  "audio_duration_seconds": 45.2,
  "confidence_score": 0.95,
  "source": "voip",
  "call_id": "call_12345",
  "created_at": "2025-11-14T10:30:45Z"
}
```

### Alerts & Detections

When scam detected:

```
Inference Engine 
  ↓
Classification Result (scam_probability >= threshold) 
  ↓
AlertLogger.log_alert() 
  ↓
SupabaseAlertLogger (async sync) 
  ↓
📊 Supabase: alerts table
```

**Example record:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "alert_type": "voice",
  "scam_probability": 87.5,
  "source": "inference_engine",
  "description": "Suspicious caller detected",
  "created_at": "2025-11-14T10:32:15Z"
}
```

### Email Events

When emails analyzed:

```
IMAP Connector 
  ↓
EmailParser 
  ↓
InferenceEngine.classify_email() 
  ↓
SupabaseDB.insert_email_event() 
  ↓
📊 Supabase: email_events table
```

## Integration in Your Code

### Option 1: Using the Enhanced AlertLogger (Recommended)

The new `SupabaseAlertLogger` automatically syncs to both local files AND Supabase:

```python
from src.supabase.supabase_alert_logger import SupabaseAlertLogger

# Initialize with Supabase enabled
logger = SupabaseAlertLogger(use_supabase=True)

# Log alerts - automatically synced in background
logger.log_alert({
    'type': 'email',
    'scam_probability': 85.5,
    'source': 'inference_engine',
    'message_id': 'msg_123',
    'note': 'Suspicious sender'
})
```

### Option 2: Direct Database Access

For more control, use the `SupabaseDB` class directly:

```python
from src.supabase.db_manager import SupabaseDB

db = SupabaseDB()

# Insert transcript
db.insert_transcript({
    'transcript': 'Your transcribed text here',
    'audio_duration_seconds': 45.2,
    'confidence_score': 0.95,
    'source': 'voip',
    'call_id': 'call_12345'
})

# Insert alert
db.insert_alert({
    'alert_type': 'voice',
    'scam_probability': 87.5,
    'source': 'inference_engine',
    'description': 'Suspicious audio detected'
})

# Insert email event
db.insert_email_event({
    'email_subject': 'Urgent: Update Payment',
    'email_from': 'attacker@fake.com',
    'scam_probability': 92.0,
    'classification': 'scam'
})
```

### Option 3: Batch Insert for Performance

For high-volume data:

```python
db = SupabaseDB()

alerts = [
    {'alert_type': 'email', 'scam_probability': 85.5, ...},
    {'alert_type': 'voice', 'scam_probability': 92.0, ...},
    {'alert_type': 'call', 'scam_probability': 78.3, ...},
]

result = db.batch_insert('alerts', alerts)
print(f"Inserted {result['inserted']} records")
```

## Updating main.py

To integrate Supabase into your existing orchestrator:

```python
from src.supabase.supabase_alert_logger import SupabaseAlertLogger

def run_orchestrator():
    # ... existing code ...
    
    # Replace the old AlertLogger with Supabase version
    alert_logger = SupabaseAlertLogger(use_supabase=True)
    
    # ... rest of code ...
    
    # On shutdown
    alert_logger.shutdown()
```

## Dashboard & Querying

### View Data in Supabase

1. Go to **Supabase Dashboard** → **Data Editor**
2. Select table: `audio_transcripts`, `alerts`, `email_events`, etc.
3. Filter, sort, and explore your data
4. Use **SQL Editor** for advanced queries

### Example Queries

**Get recent high-confidence scam alerts:**
```sql
SELECT * FROM alerts 
WHERE scam_probability > 80 
ORDER BY created_at DESC 
LIMIT 20;
```

**Transcripts from today:**
```sql
SELECT * FROM audio_transcripts 
WHERE created_at >= NOW() - INTERVAL '1 day' 
ORDER BY created_at DESC;
```

**Alert statistics by type:**
```sql
SELECT 
  alert_type,
  COUNT(*) as count,
  AVG(scam_probability) as avg_confidence,
  MAX(scam_probability) as max_confidence
FROM alerts 
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY alert_type;
```

## Testing

### Test Connection

```bash
python -c "
from src.supabase.db_manager import SupabaseDB
db = SupabaseDB()
if db.test_connection():
    print('✓ Connected to Supabase!')
else:
    print('✗ Connection failed')
"
```

### Test Insert

```bash
python src/supabase/db_manager.py
```

This will insert test data and show results.

## Monitoring & Logs

### System Logs

All errors are automatically logged to Supabase `system_logs` table:

```python
db = SupabaseDB()
db.insert_system_log(
    log_level='ERROR',
    message='Failed to process email',
    component='email_parser',
    stack_trace=traceback.format_exc()
)
```

### Metrics

Track performance metrics:

```python
db.insert_metric(
    metric_name='inference_latency',
    metric_value=145.5,
    unit='ms',
    component='inference_engine'
)
```

## Troubleshooting

### "Connection failed" Error

- Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
- Check that Supabase project is active (not paused)
- Check firewall rules allow outbound HTTPS

### "Table not found" Error

- Run the migration in **Supabase → SQL Editor**
- Verify migration file was copied correctly
- Check table names match (they're lowercase)

### Data Not Syncing

- Check logs in Supabase **Logs** section
- Verify GitHub Actions workflows are enabled
- Manually trigger sync: `git push`

### Too Many Requests (429 Error)

- Supabase has rate limits on free tier
- Batch inserts to reduce requests
- Implement exponential backoff (already built-in)

## Security

### Row Level Security (RLS)

Tables have basic RLS enabled. For production:

1. Go to **Authentication** → **Policies**
2. Define stricter access rules per user/role
3. Example: Only app can insert, users can only read

### API Keys

- **Anon Key**: Use for client-side, public operations
- **Service Role Key**: Use for server-side, sensitive operations
- Never commit keys to GitHub (use secrets)

## Next Steps

1. ✅ Deploy Supabase integration
2. ✅ Test with sample data
3. ✅ Monitor via GitHub Actions
4. ✅ Add Row Level Security policies
5. ✅ Create custom dashboards in Supabase
6. ✅ Set up alerts (e.g., Slack notifications on high-confidence detections)

## Resources

- [Supabase Documentation](https://supabase.com/docs)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [Supabase Python Client](https://github.com/supabase-community/supabase-py)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
