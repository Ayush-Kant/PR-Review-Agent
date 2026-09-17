-- ============================================================================
-- 001_initial_tiger_schema.sql
-- Initial Tiger Cloud / PostgreSQL Schema Migration for PR-Review-Agent (W1-05)
--
-- Lanes:
-- 1. Memory Lane: code_chunks (embeddings, FTS, chunk index) & repo_file_index
-- 2. Time Lane: agent_events (time-series event spine, OTel spans, cost/latency)
-- 3. Truth Lane: pr_review_records, finding_records, hitl_reviews, hitl_feedback, github_review_effects
-- ============================================================================

-- ============================================================================
-- 0. SCHEMA MIGRATIONS TABLE (Tracking applied versions)
-- ============================================================================
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 1. EXTENSIONS (Handled idempotently)
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 2. MEMORY LANE
-- ============================================================================
CREATE TABLE IF NOT EXISTS code_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id VARCHAR(255) NOT NULL,
    revision VARCHAR(64) NOT NULL,
    file_path TEXT NOT NULL,
    symbol VARCHAR(255),
    chunk_index INTEGER NOT NULL DEFAULT 0,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    token_count INTEGER,
    embedding vector(1536),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    index_version VARCHAR(32) NOT NULL DEFAULT 'v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_code_chunks_ordering UNIQUE (repository_id, revision, file_path, chunk_index),
    CONSTRAINT chk_code_chunks_lines CHECK (end_line >= start_line),
    CONSTRAINT chk_code_chunks_index CHECK (chunk_index >= 0)
);

CREATE INDEX IF NOT EXISTS idx_code_chunks_lookup 
    ON code_chunks (repository_id, revision, file_path);

CREATE INDEX IF NOT EXISTS idx_code_chunks_hash 
    ON code_chunks (repository_id, content_hash);

CREATE INDEX IF NOT EXISTS idx_code_chunks_tsv 
    ON code_chunks USING gin (tsv);

-- Freshness and repository revision indexing table
CREATE TABLE IF NOT EXISTS repo_file_index (
    repository_id VARCHAR(255) NOT NULL,
    revision VARCHAR(64) NOT NULL,
    file_path TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    index_version VARCHAR(32) NOT NULL DEFAULT 'v1',
    is_fresh BOOLEAN NOT NULL DEFAULT TRUE,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (repository_id, revision, file_path)
);

CREATE INDEX IF NOT EXISTS idx_repo_freshness 
    ON repo_file_index (repository_id, is_fresh);

-- ============================================================================
-- 3. TIME LANE (Observability, agent traces, event spine)
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_events (
    event_id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    correlation_id VARCHAR(128) NOT NULL,
    event_name VARCHAR(128) NOT NULL,
    step VARCHAR(64) NOT NULL,
    repository_id VARCHAR(255),
    pull_number INTEGER,
    head_sha VARCHAR(64),
    run_id VARCHAR(128),
    agent VARCHAR(64),
    span_id VARCHAR(64),
    parent_span_id VARCHAR(64),
    model VARCHAR(64),
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd NUMERIC(10, 6) DEFAULT 0.0,
    latency_ms DOUBLE PRECISION,
    outcome VARCHAR(32),
    confidence DOUBLE PRECISION,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_agent_events_correlation 
    ON agent_events (correlation_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_repo_pr 
    ON agent_events (repository_id, pull_number, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_name 
    ON agent_events (event_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_run 
    ON agent_events (run_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_payload 
    ON agent_events USING gin (payload);

-- ============================================================================
-- 4. TRUTH LANE (Review runs, finding lifecycle, HITL, effects)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pr_review_records (
    run_id VARCHAR(128) PRIMARY KEY,
    repository_id VARCHAR(255) NOT NULL,
    pull_number INTEGER NOT NULL,
    head_sha VARCHAR(64) NOT NULL,
    base_sha VARCHAR(64) NOT NULL,
    delivery_id VARCHAR(128) NOT NULL,
    state VARCHAR(32) NOT NULL,
    policy_version VARCHAR(32) NOT NULL DEFAULT 'v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_pr_review_delivery UNIQUE (delivery_id)
);

CREATE INDEX IF NOT EXISTS idx_pr_review_records_lookup 
    ON pr_review_records (repository_id, pull_number, head_sha);

CREATE TABLE IF NOT EXISTS finding_records (
    record_id VARCHAR(128) PRIMARY KEY,
    canonical_id VARCHAR(128) NOT NULL,
    sequence_id INTEGER NOT NULL,
    repository_id VARCHAR(255) NOT NULL,
    head_sha VARCHAR(64) NOT NULL,
    delivery_id VARCHAR(128) NOT NULL,
    run_id VARCHAR(128) NOT NULL REFERENCES pr_review_records(run_id) ON DELETE RESTRICT,
    state VARCHAR(32) NOT NULL,
    actor VARCHAR(128),
    actor_role VARCHAR(64),
    rationale TEXT,
    finding_json JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_finding_records_version UNIQUE (canonical_id, sequence_id)
);

CREATE INDEX IF NOT EXISTS idx_finding_records_canonical 
    ON finding_records (canonical_id, sequence_id ASC);

CREATE INDEX IF NOT EXISTS idx_finding_records_repo_head 
    ON finding_records (repository_id, head_sha);

CREATE INDEX IF NOT EXISTS idx_finding_records_run 
    ON finding_records (run_id);

CREATE INDEX IF NOT EXISTS idx_finding_records_state 
    ON finding_records (state);

CREATE TABLE IF NOT EXISTS hitl_reviews (
    hitl_id VARCHAR(128) PRIMARY KEY,
    canonical_id VARCHAR(128) NOT NULL,
    run_id VARCHAR(128) NOT NULL REFERENCES pr_review_records(run_id) ON DELETE RESTRICT,
    actor VARCHAR(128) NOT NULL,
    actor_role VARCHAR(64) NOT NULL,
    action VARCHAR(32) NOT NULL,
    rationale TEXT NOT NULL,
    previous_state VARCHAR(32) NOT NULL,
    resulting_state VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_hitl_rationale CHECK (length(trim(rationale)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_hitl_canonical 
    ON hitl_reviews (canonical_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_hitl_actor 
    ON hitl_reviews (actor, created_at DESC);

CREATE TABLE IF NOT EXISTS hitl_feedback (
    feedback_id VARCHAR(128) PRIMARY KEY,
    repository_id VARCHAR(255) NOT NULL,
    canonical_id VARCHAR(128) NOT NULL,
    run_id VARCHAR(128) NOT NULL REFERENCES pr_review_records(run_id) ON DELETE RESTRICT,
    policy_version VARCHAR(32) NOT NULL,
    prompt_version VARCHAR(64),
    model_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    disposition VARCHAR(32) NOT NULL,
    actor VARCHAR(128) NOT NULL,
    actor_role VARCHAR(64) NOT NULL,
    rationale TEXT,
    confidence_at_review DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hitl_fb_repo_time 
    ON hitl_feedback (repository_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_hitl_fb_policy 
    ON hitl_feedback (policy_version);

CREATE INDEX IF NOT EXISTS idx_hitl_fb_canonical 
    ON hitl_feedback (canonical_id);

CREATE TABLE IF NOT EXISTS github_review_effects (
    idempotency_key VARCHAR(128) PRIMARY KEY,
    repository_id VARCHAR(255) NOT NULL,
    pull_number INTEGER NOT NULL,
    head_sha VARCHAR(64) NOT NULL,
    canonical_id VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL,
    review_id VARCHAR(64),
    comment_id VARCHAR(64),
    html_url TEXT,
    published_inline BOOLEAN NOT NULL DEFAULT FALSE,
    reason TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_gh_effects_lookup 
    ON github_review_effects (repository_id, pull_number, canonical_id);

-- ============================================================================
-- 5. IMMUTABILITY & APPEND-ONLY INVARIANTS (Triggers)
-- ============================================================================
CREATE OR REPLACE FUNCTION prevent_modification_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Table % is append-only: UPDATE and DELETE operations are prohibited', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_agent_events_prevent_mod') THEN
        CREATE TRIGGER trg_agent_events_prevent_mod
            BEFORE UPDATE OR DELETE ON agent_events
            FOR EACH ROW EXECUTE FUNCTION prevent_modification_append_only();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_finding_records_prevent_mod') THEN
        CREATE TRIGGER trg_finding_records_prevent_mod
            BEFORE UPDATE OR DELETE ON finding_records
            FOR EACH ROW EXECUTE FUNCTION prevent_modification_append_only();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_hitl_reviews_prevent_mod') THEN
        CREATE TRIGGER trg_hitl_reviews_prevent_mod
            BEFORE UPDATE OR DELETE ON hitl_reviews
            FOR EACH ROW EXECUTE FUNCTION prevent_modification_append_only();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_hitl_feedback_prevent_mod') THEN
        CREATE TRIGGER trg_hitl_feedback_prevent_mod
            BEFORE UPDATE OR DELETE ON hitl_feedback
            FOR EACH ROW EXECUTE FUNCTION prevent_modification_append_only();
    END IF;
END $$;
