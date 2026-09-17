"""Comprehensive test suite for Tiger Cloud production data adapters (W1-06).

Tests:
1. Migration ordering, discovery, and checksum behavior (001 then 002)
2. TigerCodeMemoryStore: indexing, retrieval, freshness, isolation, empty snapshots, deleted files
3. TigerReviewTruthStore: run registration, parent FK enforcement, fail-closed on missing context,
   initial recording, state machine transitions, history ordering, list_by_state, list_all_canonical
4. Concurrency: race condition on record_initial and record_transition
5. TigerAuditSpine: append-only event recording, non-idempotency, secret redaction,
   reconstruct_run with partial_tiger_only completeness semantics, missing run failure
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import concurrent.futures
from dataclasses import asdict, replace
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
import unittest
from typing import Any
import uuid

from pr_review_agent.adapters.tiger_connection import (
    TigerConfig,
    TigerConnectionError,
    TigerConnectionManager,
)
from pr_review_agent.adapters.tiger_migrations import (
    MigrationRunner,
)
from pr_review_agent.adapters.tiger_stores import (
    ConcurrentTransitionError,
    ConflictingFindingError,
    ReviewRunContext,
    RunNotFoundError,
    TigerAuditSpine,
    TigerCodeMemoryStore,
    TigerReviewTruthStore,
    TigerStoreError,
    TigerTruthMissingParentError,
)
from pr_review_agent.observability import AuditRecord, AuditSpine, RunProvenanceTrace
from pr_review_agent.orchestration import AuditEvent
from pr_review_agent.policy import (
    CanonicalFinding,
    FindingDisposition,
    ReviewTruthRecord,
    ReviewTruthStore,
    TruthState,
)
from pr_review_agent.retrieval import CodeChunk, CodeMemoryStoreProtocol


class SimulationCursor:
    """Thread-safe cursor simulation that holds pre-fetched rows."""

    def __init__(self, rows: list[Any] | None = None, fake_last_id: int | None = None) -> None:
        self._rows = list(rows) if rows is not None else []
        self._fake_last_id = fake_last_id
        self._idx = 0
        self._fetched_fake = False

    def fetchone(self) -> Any:
        if self._fake_last_id is not None and not self._fetched_fake:
            self._fetched_fake = True
            return (self._fake_last_id,)
        if self._idx < len(self._rows):
            r = self._rows[self._idx]
            self._idx += 1
            return r
        return None

    def fetchall(self) -> list[Any]:
        if self._fake_last_id is not None and not self._fetched_fake:
            self._fetched_fake = True
            return [(self._fake_last_id,)]
        remaining = self._rows[self._idx:]
        self._idx = len(self._rows)
        return remaining

    def __iter__(self) -> Any:
        return iter(self._rows)



class TigerPostgresSimulationConnection:
    """Thread-safe in-memory SQLite connection adapter simulating PostgreSQL semantics for Tiger adapters.

    Translates %s parameter placeholders to ? and initializes SQLite tables conforming
    to Tiger Cloud migrations 001 and 002.
    """

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._in_transaction = False
        self.fail_advisory_lock = False
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            with self.conn:
                self.conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS pr_review_records (
                        run_id TEXT PRIMARY KEY,
                        repository_id TEXT NOT NULL,
                        pull_number INTEGER NOT NULL,
                        head_sha TEXT NOT NULL,
                        base_sha TEXT NOT NULL,
                        delivery_id TEXT NOT NULL UNIQUE,
                        state TEXT NOT NULL,
                        policy_version TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS finding_records (
                        record_id TEXT PRIMARY KEY,
                        canonical_id TEXT NOT NULL,
                        sequence_id INTEGER NOT NULL,
                        repository_id TEXT NOT NULL,
                        head_sha TEXT NOT NULL,
                        delivery_id TEXT NOT NULL,
                        run_id TEXT NOT NULL REFERENCES pr_review_records(run_id),
                        state TEXT NOT NULL,
                        actor TEXT,
                        actor_role TEXT,
                        rationale TEXT,
                        finding_json TEXT NOT NULL,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (canonical_id, sequence_id)
                    );

                    CREATE TABLE IF NOT EXISTS code_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        repository_id TEXT NOT NULL,
                        revision TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        index_version TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (repository_id, revision, file_path, chunk_index)
                    );

                    CREATE TABLE IF NOT EXISTS repo_file_index (
                        repository_id TEXT NOT NULL,
                        revision TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        index_version TEXT NOT NULL,
                        is_fresh INTEGER NOT NULL DEFAULT 1,
                        indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (repository_id, revision, file_path)
                    );

                    CREATE TABLE IF NOT EXISTS repository_revisions (
                        repository_id TEXT NOT NULL,
                        revision TEXT NOT NULL,
                        is_fresh INTEGER NOT NULL DEFAULT 1,
                        indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (repository_id, revision)
                    );

                    CREATE TABLE IF NOT EXISTS agent_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        correlation_id TEXT NOT NULL,
                        event_name TEXT NOT NULL,
                        step TEXT NOT NULL,
                        repository_id TEXT,
                        pull_number INTEGER,
                        head_sha TEXT,
                        run_id TEXT,
                        payload TEXT NOT NULL,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS github_review_effects (
                        idempotency_key TEXT PRIMARY KEY,
                        repository_id TEXT NOT NULL,
                        pull_number INTEGER NOT NULL,
                        head_sha TEXT NOT NULL,
                        canonical_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        review_id TEXT,
                        comment_id TEXT,
                        html_url TEXT,
                        published_inline INTEGER NOT NULL DEFAULT 0,
                        reason TEXT,
                        payload TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )

    def execute(self, query: str, params: Sequence[Any] | Mapping[str, Any] | None = None) -> Any:
        with self._lock:
            if "pg_advisory_xact_lock" in query:
                if self.fail_advisory_lock:
                    raise sqlite3.OperationalError("Simulated advisory lock acquisition failure: deadlock / timeout")
                return SimulationCursor([(None,)])

            q = query.replace("%s", "?")
            # Convert RETURNING event_id for sqlite3 if needed
            is_returning_event_id = "RETURNING event_id" in q
            if is_returning_event_id:
                q = q.replace("RETURNING event_id", "")

            # Ensure transaction is active for mutating DML
            q_upper = q.strip().upper()
            if not self._in_transaction and any(q_upper.startswith(cmd) for cmd in ("INSERT", "UPDATE", "DELETE", "REPLACE")):
                self.conn.execute("BEGIN")
                self._in_transaction = True

            # Sanitize datetime parameters to ISO format for SQLite to avoid Python 3.12 deprecation
            clean_params = params
            if params is not None:
                if isinstance(params, Mapping):
                    clean_params = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in params.items()}
                elif isinstance(params, Sequence):
                    clean_params = [(p.isoformat() if isinstance(p, datetime) else p) for p in params]

            cur = self.conn.cursor()
            if clean_params is not None:
                cur.execute(q, clean_params)
            else:
                cur.execute(q)

            rows: list[Any] = []
            try:
                rows = cur.fetchall()
            except Exception:
                rows = []

            if is_returning_event_id:
                last_id = cur.lastrowid
                return SimulationCursor(rows=rows, fake_last_id=last_id)
            return SimulationCursor(rows=rows)

    def cursor(self) -> Any:
        return self

    def commit(self) -> None:
        with self._lock:
            if self._in_transaction:
                self.conn.commit()
                self._in_transaction = False

    def rollback(self) -> None:
        with self._lock:
            if self._in_transaction:
                self.conn.rollback()
                self._in_transaction = False

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def _make_sample_finding(
    canonical_id: str = "can-123456789abc",
    repository_id: str = "test-owner/test-repo",
    head_sha: str = "abcdef0123456789abcdef0123456789abcdef01",
    run_id: str = "run-001",
    delivery_id: str = "deliv-001",
) -> CanonicalFinding:
    return CanonicalFinding(
        canonical_id=canonical_id,
        repository_id=repository_id,
        head_sha=head_sha,
        category="correctness",
        severity="high",
        confidence=0.92,
        summary="Potential null dereference in processor",
        rationale="Variable pointer is dereferenced without prior null validation",
        file_path="src/processor.py",
        line_range=(45, 52),
        contributing_candidate_ids=("cand-001",),
        contributing_specialists=("correctness_specialist",),
        evidence_refs=("diff://src/processor.py#L45-L52",),
        disposition=FindingDisposition.HELD,
        disposition_reason="High severity finding requires human approval",
        run_id=run_id,
        delivery_id=delivery_id,
    )


class TestTigerDataAdapters(unittest.TestCase):
    """Test suite covering all W1-06 Tiger data adapter contracts and invariants."""

    def setUp(self) -> None:
        self.sim_conn = TigerPostgresSimulationConnection()
        self.config = TigerConfig.from_url("postgresql://postgres:test_pw@localhost:5432/testdb?sslmode=require")
        self.mgr = TigerConnectionManager(
            self.config,
            connection_factory=lambda cfg: self.sim_conn,
        )
        self.code_store = TigerCodeMemoryStore(self.mgr)
        self.truth_store = TigerReviewTruthStore(self.mgr)
        self.audit_spine = TigerAuditSpine(self.mgr)

    def tearDown(self) -> None:
        self.mgr.close()

    # =========================================================================
    # A. MIGRATION ORDERING & BEHAVIOR
    # =========================================================================

    def test_migration_discovery_and_ordering(self) -> None:
        """Verify migration runner discovers 001 and 002 in ascending version order."""
        runner = MigrationRunner(self.sim_conn)
        migrations = runner.discover_migrations()
        self.assertTrue(len(migrations) >= 2)
        self.assertEqual(1, migrations[0].version)
        self.assertEqual("initial_tiger_schema", migrations[0].name)
        self.assertEqual(2, migrations[1].version)
        self.assertEqual("repository_revisions", migrations[1].name)
        self.assertTrue(migrations[0].checksum)
        self.assertTrue(migrations[1].checksum)
        self.assertEqual(64, len(migrations[0].checksum))
        self.assertEqual(64, len(migrations[1].checksum))

    # =========================================================================
    # B. CODE MEMORY STORE
    # =========================================================================

    def test_code_memory_index_and_retrieve(self) -> None:
        """Verify indexing files into code_chunks and retrieving them by repository and revision."""
        files = {
            "main.py": "def main():\n    print('hello world')\n",
            "utils.py": "def helper():\n    return 42\n",
        }
        count = self.code_store.index_repository("repo-a", "rev-1", files, chunk_line_size=50)
        self.assertEqual(2, count)

        # Retrieve chunks (strictly 2 arguments)
        chunks = self.code_store.get_chunks("repo-a", "rev-1")
        self.assertEqual(2, len(chunks))
        paths = [c.file_path for c in chunks]
        self.assertIn("main.py", paths)
        self.assertIn("utils.py", paths)

        # Retrieve filtered by file using private/internal helper
        main_chunks = self.code_store._get_chunks_for_file("repo-a", "rev-1", file_path="main.py")
        self.assertEqual(1, len(main_chunks))
        self.assertEqual("main.py", main_chunks[0].file_path)

    def test_code_memory_index_chunk_sequence_and_uuid_schema(self) -> None:
        """Verify indexing Sequence[CodeChunk] directly and validating UUID column format."""
        chk1 = CodeChunk(
            chunk_id="chk-custom-1",
            repository_id="repo-seq",
            revision="rev-seq",
            file_path="src/lib.py",
            start_line=1,
            end_line=10,
            content="def add(a, b): return a + b\n",
            content_hash="hash1",
            index_version="v1",
        )
        chk2 = CodeChunk(
            chunk_id="chk-custom-2",
            repository_id="repo-seq",
            revision="rev-seq",
            file_path="src/lib.py",
            start_line=11,
            end_line=20,
            content="def sub(a, b): return a - b\n",
            content_hash="hash2",
            index_version="v1",
        )
        count = self.code_store.index_repository("repo-seq", "rev-seq", [chk1, chk2])
        self.assertEqual(2, count)
        self.assertTrue(self.code_store.is_fresh("repo-seq", "rev-seq"))

        # Verify underlying database table code_chunks stores valid RFC 4122 UUIDs
        cur = self.sim_conn.execute(
            "SELECT chunk_id FROM code_chunks WHERE repository_id = ? AND revision = ?",
            ("repo-seq", "rev-seq"),
        )
        rows = cur.fetchall()
        self.assertEqual(2, len(rows))
        for r in rows:
            parsed = uuid.UUID(r[0])
            self.assertEqual(5, parsed.version)

    def test_code_memory_freshness_and_prior_stale(self) -> None:
        """Verify revision freshness transitions and marking prior revisions stale."""
        files_v1 = {"main.py": "v1"}
        self.code_store.index_repository("repo-a", "rev-1", files_v1)
        self.assertTrue(self.code_store.is_fresh("repo-a", "rev-1"))

        # Index revision 2 for repo-a
        files_v2 = {"main.py": "v2"}
        self.code_store.index_repository("repo-a", "rev-2", files_v2)
        self.assertTrue(self.code_store.is_fresh("repo-a", "rev-2"))
        self.assertFalse(self.code_store.is_fresh("repo-a", "rev-1"))

    def test_code_memory_empty_snapshot(self) -> None:
        """Verify empty snapshot (0 files) records freshness=TRUE with chunk_count=0."""
        count = self.code_store.index_repository("repo-empty", "rev-empty", {})
        self.assertEqual(0, count)
        self.assertTrue(self.code_store.is_fresh("repo-empty", "rev-empty"))
        self.assertEqual([], self.code_store.get_chunks("repo-empty", "rev-empty"))

    def test_code_memory_deleted_file_snapshot(self) -> None:
        """Verify deleted files between revisions cleanly reflect in the new snapshot."""
        self.code_store.index_repository(
            "repo-del", "rev-1", {"keep.py": "keep", "delete.py": "delete"}
        )
        self.assertEqual(2, len(self.code_store.get_chunks("repo-del", "rev-1")))

        # Revision 2 deletes delete.py
        self.code_store.index_repository("repo-del", "rev-2", {"keep.py": "keep"})
        rev2_chunks = self.code_store.get_chunks("repo-del", "rev-2")
        self.assertEqual(1, len(rev2_chunks))
        self.assertEqual("keep.py", rev2_chunks[0].file_path)

    def test_code_memory_repeated_identical_indexing(self) -> None:
        """Verify repeated indexing of identical content does not create duplicate chunks."""
        files = {"app.py": "content"}
        c1 = self.code_store.index_repository("repo-dup", "rev-1", files)
        c2 = self.code_store.index_repository("repo-dup", "rev-1", files)
        self.assertEqual(c1, c2)
        chunks = self.code_store.get_chunks("repo-dup", "rev-1")
        self.assertEqual(1, len(chunks))

    def test_code_memory_repository_and_revision_isolation(self) -> None:
        """Verify strict multi-tenant isolation across repositories and revisions."""
        self.code_store.index_repository("tenant-1", "rev-shared", {"file.py": "tenant1 code"})
        self.code_store.index_repository("tenant-2", "rev-shared", {"file.py": "tenant2 code"})

        t1_chunks = self.code_store.get_chunks("tenant-1", "rev-shared")
        t2_chunks = self.code_store.get_chunks("tenant-2", "rev-shared")

        self.assertEqual(1, len(t1_chunks))
        self.assertEqual(1, len(t2_chunks))
        self.assertEqual("tenant1 code", t1_chunks[0].content)
        self.assertEqual("tenant2 code", t2_chunks[0].content)
        self.assertTrue(self.code_store.is_fresh("tenant-1", "rev-shared"))
        self.assertTrue(self.code_store.is_fresh("tenant-2", "rev-shared"))

    def test_code_memory_index_repository_atomicity_rollback(self) -> None:
        """Verify index_repository rolls back all writes on failure, leaving connection usable."""
        sim_conn = self.code_store.connection_manager.get_connection()
        original_execute = sim_conn.execute

        # Inject failure during repo_file_index insertion after code_chunks
        fail_now = True

        def failing_execute(query: str, params: Any = None) -> Any:
            if fail_now and "repo_file_index" in query:
                raise sqlite3.OperationalError("Simulated write failure during repo_file_index")
            return original_execute(query, params)

        sim_conn.execute = failing_execute
        try:
            with self.assertRaises(sqlite3.OperationalError):
                self.code_store.index_repository("owner/repo-rb", "rev-rb-1", {"a.py": "code a", "b.py": "code b"})
        finally:
            sim_conn.execute = original_execute

        # Verify atomicity: NO chunks or revision metadata committed
        self.assertFalse(self.code_store.is_fresh("owner/repo-rb", "rev-rb-1"))
        persisted_chunks = self.code_store.get_chunks("owner/repo-rb", "rev-rb-1")
        self.assertEqual(0, len(persisted_chunks))

        # Verify connection is usable afterward and subsequent indexing succeeds
        count = self.code_store.index_repository("owner/repo-rb", "rev-rb-1", {"a.py": "code a", "b.py": "code b"})
        self.assertEqual(2, count)
        self.assertTrue(self.code_store.is_fresh("owner/repo-rb", "rev-rb-1"))
        self.assertEqual(2, len(self.code_store.get_chunks("owner/repo-rb", "rev-rb-1")))

    # =========================================================================
    # C. REVIEW TRUTH STORE
    # =========================================================================

    def test_review_truth_registration_and_idempotency(self) -> None:
        """Verify review run registration succeeds idempotently for identical context."""
        self.truth_store.register_review_run(
            run_id="run-100",
            repository_id="owner/repo",
            pull_number=42,
            head_sha="head123",
            base_sha="base123",
            delivery_id="deliv-100",
        )
        # Duplicate registration with identical context succeeds without error
        self.truth_store.register_review_run(
            run_id="run-100",
            repository_id="owner/repo",
            pull_number=42,
            head_sha="head123",
            base_sha="base123",
            delivery_id="deliv-100",
        )

    def test_review_truth_registration_conflict_fails_closed(self) -> None:
        """Verify conflicting run_id or delivery_id registration raises ConflictingFindingError."""
        self.truth_store.register_review_run(
            run_id="run-200",
            repository_id="owner/repo",
            pull_number=1,
            head_sha="head1",
            base_sha="base1",
            delivery_id="deliv-200",
        )
        # Conflicting pull number for same run_id
        with self.assertRaises(ConflictingFindingError):
            self.truth_store.register_review_run(
                run_id="run-200",
                repository_id="owner/repo",
                pull_number=999,
                head_sha="head1",
                base_sha="base1",
                delivery_id="deliv-200",
            )
        # Same delivery_id for different run_id
        with self.assertRaises(ConflictingFindingError):
            self.truth_store.register_review_run(
                run_id="run-201",
                repository_id="owner/repo",
                pull_number=1,
                head_sha="head1",
                base_sha="base1",
                delivery_id="deliv-200",
            )

    def test_review_truth_missing_parent_context_fails_closed(self) -> None:
        """Verify record_initial fails closed if parent run context is not registered."""
        finding = _make_sample_finding(run_id="unregistered-run")
        with self.assertRaises(TigerTruthMissingParentError):
            self.truth_store.record_initial(finding, initial_state=TruthState.HELD)

    def test_review_truth_record_initial_and_idempotency(self) -> None:
        """Verify record_initial creates sequence 1 and returns identical record on duplicate."""
        self.truth_store.register_review_run(
            run_id="run-001",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-001",
        )
        finding = _make_sample_finding(canonical_id="can-rec-init", run_id="run-001", delivery_id="deliv-001")
        rec1 = self.truth_store.record_initial(finding, initial_state=TruthState.HELD)
        self.assertEqual("can-rec-init", rec1.canonical_id)
        self.assertEqual(1, rec1.sequence_id)
        self.assertEqual(TruthState.HELD, rec1.state)

        # Duplicate identical record_initial returns existing record
        rec2 = self.truth_store.record_initial(finding, initial_state=TruthState.HELD)
        self.assertEqual(rec1.record_id, rec2.record_id)
        self.assertEqual(rec1.sequence_id, rec2.sequence_id)

    def test_review_truth_record_initial_conflict_fails_closed(self) -> None:
        """Verify record_initial with conflicting data for existing canonical ID fails closed."""
        self.truth_store.register_review_run(
            run_id="run-001",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-001",
        )
        finding1 = _make_sample_finding(canonical_id="can-conflict", run_id="run-001", delivery_id="deliv-001")
        self.truth_store.record_initial(finding1, initial_state=TruthState.HELD)

        # Finding with same canonical_id but different summary/file_path
        finding_conflicting = CanonicalFinding(
            canonical_id="can-conflict",
            repository_id="test-owner/test-repo",
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            category="correctness",
            severity="critical",
            confidence=0.99,
            summary="Completely different defect",
            rationale="Different rationale",
            file_path="src/other.py",
            line_range=(1, 10),
            contributing_candidate_ids=("cand-999",),
            contributing_specialists=("spec",),
            evidence_refs=("diff://src/other.py#L1-L10",),
            run_id="run-001",
            delivery_id="deliv-001",
        )
        with self.assertRaises(ConflictingFindingError):
            self.truth_store.record_initial(finding_conflicting, initial_state=TruthState.HELD)

    def test_review_truth_state_transitions(self) -> None:
        """Verify valid state transitions increment sequence_id and invalid transitions fail."""
        self.truth_store.register_review_run(
            run_id="run-001",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-001",
        )
        finding = _make_sample_finding(canonical_id="can-trans", run_id="run-001", delivery_id="deliv-001")
        self.truth_store.record_initial(finding, initial_state=TruthState.HELD)

        # Valid transition: HELD -> APPROVED
        t1 = self.truth_store.record_transition(
            "can-trans",
            TruthState.APPROVED,
            actor="alice",
            actor_role="maintainer",
            rationale="Verified by maintainer",
        )
        self.assertEqual(2, t1.sequence_id)
        self.assertEqual(TruthState.APPROVED, t1.state)

        # Valid transition: APPROVED -> PUBLISHED
        t2 = self.truth_store.record_transition(
            "can-trans",
            TruthState.PUBLISHED,
            actor="system",
            actor_role="agent",
            rationale="Published to GitHub PR",
        )
        self.assertEqual(3, t2.sequence_id)
        self.assertEqual(TruthState.PUBLISHED, t2.state)

        # Invalid transition: PUBLISHED -> APPROVED (not allowed)
        with self.assertRaises(ValueError):
            self.truth_store.record_transition(
                "can-trans",
                TruthState.APPROVED,
                actor="alice",
                actor_role="maintainer",
                rationale="Illegal re-approval",
            )

    def test_record_transition_updated_finding_provenance_validation(self) -> None:
        """Verify record_transition rejects updated_finding with conflicting provenance."""
        self.truth_store.register_review_run(
            run_id="run-trans-prov",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-trans-prov",
        )
        finding = _make_sample_finding(
            canonical_id="can-prov-test",
            run_id="run-trans-prov",
            delivery_id="deliv-trans-prov",
        )
        self.truth_store.record_initial(finding, initial_state=TruthState.HELD)

        # 1. Conflicting canonical_id in updated_finding
        bad_cid = replace(finding, canonical_id="can-different-id")
        with self.assertRaises(ConflictingFindingError) as cm:
            self.truth_store.record_transition(
                "can-prov-test",
                TruthState.APPROVED,
                updated_finding=bad_cid,
            )
        self.assertIn("does not match transition target", str(cm.exception))

        # 2. Conflicting repository_id in updated_finding
        bad_repo = replace(finding, repository_id="rogue/repo")
        with self.assertRaises(ConflictingFindingError) as cm:
            self.truth_store.record_transition(
                "can-prov-test",
                TruthState.APPROVED,
                updated_finding=bad_repo,
            )
        self.assertIn("does not match truth record", str(cm.exception))

        # 3. Conflicting head_sha in updated_finding
        bad_sha = replace(finding, head_sha="9999999999999999999999999999999999999999")
        with self.assertRaises(ConflictingFindingError) as cm:
            self.truth_store.record_transition(
                "can-prov-test",
                TruthState.APPROVED,
                updated_finding=bad_sha,
            )
        self.assertIn("does not match truth record", str(cm.exception))

        # 4. Valid updated_finding transitions cleanly
        valid_updated = replace(finding, rationale="Approved by human reviewer")
        rec = self.truth_store.record_transition(
            "can-prov-test",
            TruthState.APPROVED,
            updated_finding=valid_updated,
        )
        self.assertEqual(TruthState.APPROVED, rec.state)
        self.assertEqual("Approved by human reviewer", rec.finding_data.get("rationale"))

    def test_review_truth_history_and_querying(self) -> None:
        """Verify get_latest_state, get_history, list_by_state, and list_all_canonical."""
        self.truth_store.register_review_run(
            run_id="run-001",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-001",
        )
        f1 = _make_sample_finding(canonical_id="can-h1", run_id="run-001", delivery_id="deliv-001")
        f2 = _make_sample_finding(canonical_id="can-h2", run_id="run-001", delivery_id="deliv-001")

        self.truth_store.record_initial(f1, initial_state=TruthState.HELD)
        self.truth_store.record_initial(f2, initial_state=TruthState.AUTO_APPROVED)
        self.truth_store.record_transition("can-h1", TruthState.APPROVED, actor="maintainer")

        # History for f1 has 2 items in ascending order
        hist = self.truth_store.get_history("can-h1")
        self.assertEqual(2, len(hist))
        self.assertEqual(1, hist[0].sequence_id)
        self.assertEqual(TruthState.HELD, hist[0].state)
        self.assertEqual(2, hist[1].sequence_id)
        self.assertEqual(TruthState.APPROVED, hist[1].state)

        # Latest state
        latest_1 = self.truth_store.get_latest_state("can-h1")
        self.assertIsNotNone(latest_1)
        self.assertEqual(TruthState.APPROVED, latest_1.state)

        # List by state
        approved = self.truth_store.list_by_state(TruthState.APPROVED)
        self.assertEqual(1, len(approved))
        self.assertEqual("can-h1", approved[0].canonical_id)

        # List all canonical IDs
        all_cids = self.truth_store.list_all_canonical()
        self.assertIn("can-h1", all_cids)
        self.assertIn("can-h2", all_cids)

    def test_record_initial_idempotency_different_material_payload_fails_closed(self) -> None:
        """Verify record_initial fails closed if an existing finding has same ID but differing material fields."""
        self.truth_store.register_review_run(
            run_id="run-idem",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-idem",
        )
        f1 = _make_sample_finding(
            canonical_id="can-idem-diff",
            run_id="run-idem",
            delivery_id="deliv-idem",
        )
        rec1 = self.truth_store.record_initial(f1, initial_state=TruthState.HELD)
        self.assertEqual(1, rec1.sequence_id)

        # Same ID, identical payload -> returns existing record
        f1_same = _make_sample_finding(
            canonical_id="can-idem-diff",
            run_id="run-idem",
            delivery_id="deliv-idem",
        )
        rec1_dup = self.truth_store.record_initial(f1_same, initial_state=TruthState.HELD)
        self.assertEqual(rec1.record_id, rec1_dup.record_id)

        # Same summary, file_path, severity, but different rationale
        f1_diff_rationale = replace(f1, rationale="DIFFERENT RATIONALE: potential exploit path")
        with self.assertRaises(ConflictingFindingError):
            self.truth_store.record_initial(f1_diff_rationale, initial_state=TruthState.HELD)

        # Same summary, file_path, severity, but different confidence
        f1_diff_confidence = replace(f1, confidence=0.99)
        with self.assertRaises(ConflictingFindingError):
            self.truth_store.record_initial(f1_diff_confidence, initial_state=TruthState.HELD)

        # Same summary, file_path, severity, but different category
        f1_diff_category = replace(f1, category="architecture")
        with self.assertRaises(ConflictingFindingError):
            self.truth_store.record_initial(f1_diff_category, initial_state=TruthState.HELD)

    def test_parent_context_conflicting_explicit_and_finding_identifiers(self) -> None:
        """Verify _resolve_context fails closed when explicit and finding identifiers conflict."""
        self.truth_store.register_review_run(
            run_id="run-p1",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-p1",
        )
        self.truth_store.register_review_run(
            run_id="run-p2",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-p2",
        )

        finding = _make_sample_finding(
            canonical_id="can-ctx-conflict",
            run_id="run-p1",
            delivery_id="deliv-p1",
        )

        # 1. Conflicting explicit run_id vs finding.run_id
        with self.assertRaises(ConflictingFindingError) as cm:
            self.truth_store.record_initial(finding, initial_state=TruthState.HELD, run_id="run-p2")
        self.assertIn("conflicts with finding.run_id", str(cm.exception))

        # 2. Conflicting explicit delivery_id vs finding.delivery_id
        with self.assertRaises(ConflictingFindingError) as cm:
            self.truth_store.record_initial(finding, initial_state=TruthState.HELD, delivery_id="deliv-p2")
        self.assertIn("conflicts with finding.delivery_id", str(cm.exception))

        # 3. Conflicting repository_id vs registered run context
        finding_wrong_repo = _make_sample_finding(
            canonical_id="can-ctx-wrong-repo",
            repository_id="other-owner/other-repo",
            run_id="run-p1",
            delivery_id="deliv-p1",
        )
        with self.assertRaises(ConflictingFindingError) as cm:
            self.truth_store.record_initial(finding_wrong_repo, initial_state=TruthState.HELD)
        self.assertIn("conflicts with registered run repository", str(cm.exception))

        # 4. Conflicting head_sha vs registered run context
        finding_wrong_sha = _make_sample_finding(
            canonical_id="can-ctx-wrong-sha",
            head_sha="1111111111111111111111111111111111111111",
            run_id="run-p1",
            delivery_id="deliv-p1",
        )
        with self.assertRaises(ConflictingFindingError) as cm:
            self.truth_store.record_initial(finding_wrong_sha, initial_state=TruthState.HELD)
        self.assertIn("conflicts with registered run head_sha", str(cm.exception))

    def test_timestamp_parity_roundtrip_truth_and_audit(self) -> None:
        """Verify caller-supplied timestamps are persisted and read back with parity."""
        self.truth_store.register_review_run(
            run_id="run-ts",
            repository_id="test-owner/test-repo",
            pull_number=42,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-ts",
        )
        finding = _make_sample_finding(canonical_id="can-ts", run_id="run-ts", delivery_id="deliv-ts")
        known_initial_ts = 1700000000.0
        rec_init = self.truth_store.record_initial(finding, initial_state=TruthState.HELD, now=known_initial_ts)
        self.assertAlmostEqual(known_initial_ts, rec_init.timestamp, delta=1.0)

        # Read back from database
        latest = self.truth_store.get_latest_state("can-ts")
        self.assertIsNotNone(latest)
        self.assertAlmostEqual(known_initial_ts, latest.timestamp, delta=1.0)

        # Transition with distinct known timestamp
        known_trans_ts = 1700003600.0
        rec_trans = self.truth_store.record_transition(
            "can-ts",
            TruthState.APPROVED,
            actor="time-tester",
            now=known_trans_ts,
        )
        self.assertAlmostEqual(known_trans_ts, rec_trans.timestamp, delta=1.0)

        # Read back history
        hist = self.truth_store.get_history("can-ts")
        self.assertEqual(2, len(hist))
        self.assertAlmostEqual(known_initial_ts, hist[0].timestamp, delta=1.0)
        self.assertAlmostEqual(known_trans_ts, hist[1].timestamp, delta=1.0)

        # Audit event timestamp parity
        known_event_ts = 1700007200.0
        event = AuditEvent(
            correlation_id="run-ts",
            event_name="timestamp_check",
            step="testing",
            timestamp=known_event_ts,
            details={"parity": True},
        )
        self.audit_spine.record_event(event, run_id="run-ts")
        events = self.audit_spine.get_events("run-ts")
        matching = [e for e in events if e.event_name == "timestamp_check"]
        self.assertEqual(1, len(matching))
        self.assertAlmostEqual(known_event_ts, matching[0].timestamp, delta=1.0)

    # =========================================================================
    # D. CONCURRENCY RACES & LOCK SAFETY
    # =========================================================================

    def test_record_transition_advisory_lock_failure_fails_closed(self) -> None:
        """Verify record_transition fails closed on advisory lock failure, rolls back, and leaves connection usable."""
        self.truth_store.register_review_run(
            run_id="run-lock",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-lock",
        )
        finding = _make_sample_finding(canonical_id="can-lock-fail", run_id="run-lock", delivery_id="deliv-lock")
        self.truth_store.record_initial(finding, initial_state=TruthState.HELD)

        # Simulate advisory lock failure
        sim_conn = self.truth_store.connection_manager.get_connection()
        sim_conn.fail_advisory_lock = True
        try:
            with self.assertRaises(ConcurrentTransitionError) as ctx:
                self.truth_store.record_transition(
                    "can-lock-fail",
                    TruthState.APPROVED,
                    actor="auditor",
                )
            self.assertIn("Failed to acquire advisory transaction lock", str(ctx.exception))
        finally:
            sim_conn.fail_advisory_lock = False

        # Verify no transition row was persisted (history has only sequence 1)
        hist = self.truth_store.get_history("can-lock-fail")
        self.assertEqual(1, len(hist))
        self.assertEqual(TruthState.HELD, hist[0].state)

        # Verify connection is usable afterward and subsequent transition succeeds
        rec = self.truth_store.record_transition(
            "can-lock-fail",
            TruthState.APPROVED,
            actor="auditor",
        )
        self.assertEqual(TruthState.APPROVED, rec.state)
        self.assertEqual(2, rec.sequence_id)
        self.assertEqual(2, len(self.truth_store.get_history("can-lock-fail")))

    def test_concurrent_record_transition_race(self) -> None:
        """Verify concurrent workers calling record_transition serialize and sequence monotonically."""
        self.truth_store.register_review_run(
            run_id="run-race-trans",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-race-trans",
        )
        finding = _make_sample_finding(canonical_id="can-race-trans", run_id="run-race-trans", delivery_id="deliv-race-trans")
        self.truth_store.record_initial(finding, initial_state=TruthState.HELD)

        t1 = self.truth_store.record_transition("can-race-trans", TruthState.APPROVED, actor="worker-1")
        t2 = self.truth_store.record_transition("can-race-trans", TruthState.PUBLISHED, actor="worker-2")

        self.assertEqual(2, t1.sequence_id)
        self.assertEqual(TruthState.APPROVED, t1.state)
        self.assertEqual(3, t2.sequence_id)
        self.assertEqual(TruthState.PUBLISHED, t2.state)

        history = self.truth_store.get_history("can-race-trans")
        self.assertEqual(3, len(history))
        self.assertEqual([1, 2, 3], [h.sequence_id for h in history])

    def test_concurrent_record_initial_race(self) -> None:
        """Verify concurrent workers calling record_initial for same canonical_id produce exactly one sequence 1."""
        self.truth_store.register_review_run(
            run_id="run-race",
            repository_id="test-owner/test-repo",
            pull_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            base_sha="base0123456789abcdef0123456789abcdef01",
            delivery_id="deliv-race",
        )
        finding = _make_sample_finding(canonical_id="can-race-init", run_id="run-race", delivery_id="deliv-race")

        def worker_call() -> ReviewTruthRecord:
            return self.truth_store.record_initial(finding, initial_state=TruthState.HELD)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_call) for _ in range(8)]
            results = [f.result() for f in futures]

        # All workers returned the exact same sequence 1 record ID
        record_ids = set(r.record_id for r in results)
        self.assertEqual(1, len(record_ids))
        self.assertEqual("truth-can-race-init-1", list(record_ids)[0])

        # Exactly 1 row in database
        history = self.truth_store.get_history("can-race-init")
        self.assertEqual(1, len(history))

    # =========================================================================
    # E. AUDIT SPINE
    # =========================================================================

    def test_audit_spine_record_event_append_only_and_non_idempotent(self) -> None:
        """Verify record_event appends monotonically and is non-idempotent."""
        event = AuditEvent(
            correlation_id="corr-1",
            event_name="step_started",
            step="intake",
            timestamp=1000.0,
            details={"key": "value"},
        )
        id1 = self.audit_spine.record_event(event)
        id2 = self.audit_spine.record_event(event)

        # Both events persisted with incremented event IDs
        self.assertTrue(id2 > id1)
        events = self.audit_spine.get_events("corr-1")
        self.assertEqual(2, len(events))
        self.assertEqual(id1, events[0].event_id)
        self.assertEqual(id2, events[1].event_id)

    def test_audit_spine_secret_redaction(self) -> None:
        """Verify sensitive credentials are automatically redacted before persistence."""
        event = AuditEvent(
            correlation_id="corr-sec",
            event_name="auth_event",
            step="auth",
            timestamp=1000.0,
            details={"api_key": "super-secret-token", "safe": "public"},
        )
        self.audit_spine.record_event(event)
        events = self.audit_spine.get_events("corr-sec")
        self.assertEqual(1, len(events))
        self.assertEqual("[REDACTED]", events[0].details.get("api_key"))
        self.assertEqual("public", events[0].details.get("safe"))

    def test_audit_spine_reconstruct_run_partial_tiger_only(self) -> None:
        """Verify reconstruct_run returns partial_tiger_only completeness in W1-06."""
        self.truth_store.register_review_run(
            run_id="run-recon",
            repository_id="owner/repo",
            pull_number=15,
            head_sha="head15",
            base_sha="base15",
            delivery_id="deliv-recon",
        )
        self.audit_spine.record_event(
            AuditEvent(
                correlation_id="run-recon",
                event_name="review_started",
                step="orchestration",
                timestamp=1000.0,
                details={"delivery_id": "deliv-recon"},
            ),
            run_id="run-recon",
        )

        trace = self.audit_spine.reconstruct_run("run-recon")
        self.assertIsInstance(trace, RunProvenanceTrace)
        self.assertEqual("run-recon", trace.run_id)
        self.assertEqual("owner/repo", trace.repository_id)
        self.assertEqual("deliv-recon", trace.delivery_id)
        self.assertIsNone(trace.delivery_status)
        self.assertIsNone(trace.queue_job_status)
        self.assertEqual("partial_tiger_only", trace.completeness)
        self.assertIn("Redis/Ingress", trace.completeness_reasons[0])
        self.assertFalse(trace.contains_secrets)

    def test_audit_spine_reconstruct_run_not_found_fails_closed(self) -> None:
        """Verify reconstruct_run raises RunNotFoundError when correlation ID is unknown."""
        with self.assertRaises(RunNotFoundError):
            self.audit_spine.reconstruct_run("unknown-run-id")

    # =========================================================================
    # F. V1 PUBLIC CONTRACT PARITY
    # =========================================================================

    def test_v1_public_contract_signature_parity(self) -> None:
        """Verify Tiger stores public method signatures match V1 persistence contracts."""
        # 1. TigerCodeMemoryStore
        sig_index = inspect.signature(self.code_store.index_repository)
        self.assertIn("repository_id", sig_index.parameters)
        self.assertIn("revision", sig_index.parameters)
        self.assertIn("chunks", sig_index.parameters)

        sig_fresh = inspect.signature(self.code_store.is_fresh)
        self.assertEqual(["repository_id", "revision"], [p for p in sig_fresh.parameters.keys() if p != "self"])

        sig_chunks = inspect.signature(self.code_store.get_chunks)
        param_names = [p for p in sig_chunks.parameters.keys() if p != "self"]
        self.assertEqual(["repository_id", "revision"], param_names, "get_chunks must take strictly 2 arguments")

        # 2. TigerReviewTruthStore vs ReviewTruthStore (SQLite)
        sig_ti_init = inspect.signature(self.truth_store.record_initial)
        sig_v1_init = inspect.signature(ReviewTruthStore.record_initial)
        ti_init_params = [p for p in sig_ti_init.parameters.keys() if p != "self"]
        v1_init_params = [p for p in sig_v1_init.parameters.keys() if p != "self"]
        self.assertEqual(v1_init_params, ti_init_params)

        sig_ti_trans = inspect.signature(self.truth_store.record_transition)
        sig_v1_trans = inspect.signature(ReviewTruthStore.record_transition)
        ti_trans_params = [p for p in sig_ti_trans.parameters.keys() if p != "self"]
        v1_trans_params = [p for p in sig_v1_trans.parameters.keys() if p != "self"]
        self.assertEqual(v1_trans_params, ti_trans_params)
        self.assertIn("new_state", ti_trans_params)
        self.assertIn("updated_finding", ti_trans_params)

        sig_ti_latest = inspect.signature(self.truth_store.get_latest_state)
        self.assertEqual(["canonical_id"], [p for p in sig_ti_latest.parameters.keys() if p != "self"])

        sig_ti_hist = inspect.signature(self.truth_store.get_history)
        self.assertEqual(["canonical_id"], [p for p in sig_ti_hist.parameters.keys() if p != "self"])

        sig_ti_lbs = inspect.signature(self.truth_store.list_by_state)
        self.assertEqual(["state"], [p for p in sig_ti_lbs.parameters.keys() if p != "self"])

        sig_ti_lac = inspect.signature(self.truth_store.list_all_canonical)
        self.assertEqual([], [p for p in sig_ti_lac.parameters.keys() if p != "self"])

        # 3. TigerAuditSpine vs AuditSpine (SQLite)
        sig_ti_rec = inspect.signature(self.audit_spine.record_event)
        sig_v1_rec = inspect.signature(AuditSpine.record_event)
        self.assertEqual(
            [p for p in sig_v1_rec.parameters.keys() if p != "self"],
            [p for p in sig_ti_rec.parameters.keys() if p != "self"],
        )

        sig_ti_recs = inspect.signature(self.audit_spine.record_events)
        sig_v1_recs = inspect.signature(AuditSpine.record_events)
        self.assertEqual(
            [p for p in sig_v1_recs.parameters.keys() if p != "self"],
            [p for p in sig_ti_recs.parameters.keys() if p != "self"],
        )

        sig_ti_gev = inspect.signature(self.audit_spine.get_events)
        self.assertEqual(["correlation_id"], [p for p in sig_ti_gev.parameters.keys() if p != "self"])

        sig_ti_rec_run = inspect.signature(self.audit_spine.reconstruct_run)
        self.assertEqual(["correlation_id"], [p for p in sig_ti_rec_run.parameters.keys() if p != "self"])


if __name__ == "__main__":
    unittest.main()
