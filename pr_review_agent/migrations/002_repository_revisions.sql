-- ============================================================================
-- 002_repository_revisions.sql
-- Revision-Level Freshness & Repository Snapshot Tracking (W1-06)
--
-- Preserves exact SQLite parity for repository-level freshness semantics:
-- - Primary Key: (repository_id, revision) ensuring multi-repo isolation
-- - is_fresh: Boolean flag updated atomically on new revision arrival
-- - indexed_at: Timestamp of index creation / refresh
-- - chunk_count: Number of chunks in the revision snapshot
--
-- Supports:
-- 1. Empty snapshots: Repositories with 0 chunks are recorded with chunk_count=0
--    and is_fresh=TRUE, avoiding false stale detection.
-- 2. Deleted files: Revision N+1 cleanly supersedes revision N without needing
--    to reconcile historical file deletion tombstones in repo_file_index.
-- 3. Partial file sets: Incomplete indexing runs fail closed before marking
--    is_fresh=TRUE.
-- 4. Revision N -> N+1 transitions: Single atomic statement can mark older
--    revisions is_fresh=FALSE and the new revision is_fresh=TRUE.
-- 5. Cross-repository isolation: Primary key is partitioned by repository_id.
-- ============================================================================

CREATE TABLE IF NOT EXISTS repository_revisions (
    repository_id VARCHAR(255) NOT NULL,
    revision VARCHAR(64) NOT NULL,
    is_fresh BOOLEAN NOT NULL DEFAULT TRUE,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (repository_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_repo_revisions_lookup
    ON repository_revisions (repository_id, revision, is_fresh);

CREATE INDEX IF NOT EXISTS idx_repo_revisions_freshness
    ON repository_revisions (repository_id, is_fresh);
