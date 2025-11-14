# Supabase Integration

This directory contains Supabase configuration, database migrations, and integration modules for the Raspberry Pi Scam Detection System.

## Overview

Supabase is a PostgreSQL-based Backend-as-a-Service (BaaS) that automatically syncs audio transcripts, alerts, and system logs with cloud tables.

## Setup

### 1. Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign up
2. Create a new project
3. Copy your project ID, API URL, and keys
4. Update `config.json` with your credentials

### 2. Set Environment Variables

Add to your `.env` file:

```bash
SUPABASE_URL=https://your_project_id.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### 3. Run Database Migrations

Apply migrations to create tables:

```bash
python src/supabase/db_manager.py --migrate
```

## Database Schema

### Tables

- **audio_transcripts** - Stores audio transcriptions with timestamps
- **alerts** - Stores detected scam alerts
- **system_logs** - Stores system events and monitoring data
- **email_events** - Stores email analysis and detection results

## Integration

The system automatically syncs data to Supabase via:

1. **AlertLogger** → `alerts` table
2. **AudioStreamHandler** → `audio_transcripts` table
3. **EmailParser** → `email_events` table
4. **System Events** → `system_logs` table

## GitHub Actions Integration

The `.github/workflows/supabase-sync.yml` workflow automatically:
- Pushes migrations to Supabase
- Validates schema consistency
- Triggers on pull requests and main branch changes

## Local Development

Run local Supabase instance:

```bash
supabase start
```

View the database:

```bash
supabase dashboard
```

## Production Deployment

Ensure Row Level Security (RLS) policies are enabled for sensitive data.
