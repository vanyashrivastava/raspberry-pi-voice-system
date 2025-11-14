-- Migration: 001_init_schema.sql
-- Description: Initialize database schema for Scam Detection System
-- Created: 2025-11-14

-- Table: audio_transcripts
-- Stores transcribed audio from voice calls with timestamps and metadata
CREATE TABLE IF NOT EXISTS audio_transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript TEXT NOT NULL,
    audio_duration_seconds FLOAT,
    confidence_score FLOAT,
    language VARCHAR(10) DEFAULT 'en',
    source VARCHAR(50) NOT NULL, -- 'voip', 'twilio', etc.
    call_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT audio_transcripts_source_check CHECK (source IN ('voip', 'twilio', 'other'))
);

CREATE INDEX idx_audio_transcripts_created_at ON audio_transcripts(created_at DESC);
CREATE INDEX idx_audio_transcripts_source ON audio_transcripts(source);
CREATE INDEX idx_audio_transcripts_call_id ON audio_transcripts(call_id);

-- Table: alerts
-- Stores detected scam alerts with probabilities and metadata
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type VARCHAR(50) NOT NULL, -- 'email', 'voice', 'call'
    scam_probability FLOAT NOT NULL CHECK (scam_probability >= 0 AND scam_probability <= 100),
    source VARCHAR(100),
    message_id VARCHAR(255),
    transcript_id UUID REFERENCES audio_transcripts(id) ON DELETE SET NULL,
    alert_level VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    description TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT alerts_type_check CHECK (alert_type IN ('email', 'voice', 'call', 'sms', 'other'))
);

CREATE INDEX idx_alerts_created_at ON alerts(created_at DESC);
CREATE INDEX idx_alerts_alert_type ON alerts(alert_type);
CREATE INDEX idx_alerts_scam_probability ON alerts(scam_probability DESC);
CREATE INDEX idx_alerts_processed ON alerts(processed);
CREATE INDEX idx_alerts_transcript_id ON alerts(transcript_id);

-- Table: email_events
-- Stores email analysis and detection results
CREATE TABLE IF NOT EXISTS email_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email_subject VARCHAR(500),
    email_from VARCHAR(255),
    email_to VARCHAR(255),
    email_body TEXT,
    scam_probability FLOAT NOT NULL CHECK (scam_probability >= 0 AND scam_probability <= 100),
    classification VARCHAR(50) NOT NULL, -- 'scam', 'legitimate', 'uncertain'
    message_id VARCHAR(255) UNIQUE,
    flags JSONB,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT email_events_classification_check CHECK (classification IN ('scam', 'legitimate', 'uncertain', 'spam'))
);

CREATE INDEX idx_email_events_created_at ON email_events(created_at DESC);
CREATE INDEX idx_email_events_classification ON email_events(classification);
CREATE INDEX idx_email_events_scam_probability ON email_events(scam_probability DESC);
CREATE INDEX idx_email_events_from ON email_events(email_from);

-- Table: system_logs
-- Stores system events, health checks, and monitoring data
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    log_level VARCHAR(20) NOT NULL, -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component VARCHAR(100), -- 'audio_processor', 'inference_engine', 'email_poller', etc.
    message TEXT NOT NULL,
    stack_trace TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT system_logs_level_check CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'))
);

CREATE INDEX idx_system_logs_created_at ON system_logs(created_at DESC);
CREATE INDEX idx_system_logs_level ON system_logs(log_level);
CREATE INDEX idx_system_logs_component ON system_logs(component);

-- Table: system_metrics
-- Stores performance metrics and health monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    unit VARCHAR(50), -- 'ms', 'MB', 'count', '%', etc.
    component VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_system_metrics_created_at ON system_metrics(created_at DESC);
CREATE INDEX idx_system_metrics_component ON system_metrics(component);
CREATE INDEX idx_system_metrics_metric_name ON system_metrics(metric_name);

-- Enable Row Level Security (RLS)
ALTER TABLE audio_transcripts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE email_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_metrics ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access (adjust as needed for your security model)
CREATE POLICY "Enable read access for all users" ON audio_transcripts
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON alerts
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON email_events
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON system_logs
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON system_metrics
    FOR SELECT USING (true);
