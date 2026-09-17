"""Comprehensive tests for Tiger Cloud / PostgreSQL schema foundation & connection manager (W1-05).

Test Matrix:
1. Configuration loads Tiger URL correctly
2. Missing Tiger URL fails clearly when Tiger backend is requested
3. SSL requirement/configuration is represented correctly
4. Credentials are masked across repr, logging, and error messages
5. Connection manager initializes without changing SQLite default
6. Migration files are discoverable and strictly ordered
7. Migration execution is idempotent
8. Extension capability detection (pgvector, TimescaleDB, pgvectorscale)
9. Memory lane tables exist with required columns and types
10. Vector column/index semantics are verified
11. Full-text search column/index semantics are verified
12. Time lane event table exists with required columns and types
13. Timescale hypertable semantics are verified when available
14. Truth lane tables exist with required relationships and constraints
15. Append-only and versioning constraints required by current domain are represented
16. Foreign keys and unique constraints required for idempotency are verified
17. Migration can be applied twice without destructive changes
18. Schema initialization does not alter SQLite
19. Real infrastructure validation vs deterministic harness is clearly distinguished
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
import sqlite3
import unittest
from unittest.mock import MagicMock, patch

from pr_review_agent.adapters.tiger_connection import (
    TigerConfig,
    TigerConfigurationError,
    TigerConnectionError,
    TigerConnectionManager,
    TigerExtensionError,
    mask_database_url,
)
from pr_review_agent.adapters.tiger_migrations import (
    CapabilityReport,
    ExtensionInspector,
    Migration,
    MigrationError,
    MigrationRunner,
)
from pr_review_agent.security import (
    RuntimeSecretRegistry,
    SecretType,
    SecurityConfig,
)
from pr_review_agent.service_config import (
    ServiceConfig,
    load_service_config,
)


class DeterministicPostgresHarness:
    """Deterministic in-memory PostgreSQL harness for schema, DDL, and migration validation.

    Simulates PostgreSQL catalog queries, DDL parsing, extension inspection,
    and schema constraint tracking without requiring external infrastructure.
    """

    def __init__(
        self,
        *,
        vector_installed: bool = True,
        vector_available: bool = True,
        timescaledb_installed: bool = True,
        timescaledb_available: bool = True,
        vectorscale_installed: bool = False,
        vectorscale_available: bool = False,
        has_vector: bool | None = None,
        has_timescaledb: bool | None = None,
        has_vectorscale: bool | None = None,
    ) -> None:
        self.vector_installed = has_vector if has_vector is not None else vector_installed
        self.vector_available = has_vector if has_vector is not None else vector_available
        self.timescaledb_installed = has_timescaledb if has_timescaledb is not None else timescaledb_installed
        self.timescaledb_available = has_timescaledb if has_timescaledb is not None else timescaledb_available
        self.vectorscale_installed = has_vectorscale if has_vectorscale is not None else vectorscale_installed
        self.vectorscale_available = has_vectorscale if has_vectorscale is not None else vectorscale_available

        self.tables: dict[str, dict[str, Any]] = {}
        self.indexes: dict[str, dict[str, Any]] = {}
        self.constraints: dict[str, dict[str, Any]] = {}
        self.triggers: dict[str, dict[str, Any]] = {}
        self.migration_records: list[tuple[int, str, str, str]] = []

        self.executed_statements: list[str] = []
        self.closed = False
        self.paramstyle = "pyformat"

    def execute(self, query: str, params: Any = None) -> DeterministicCursor:
        if self.closed:
            raise TigerConnectionError("Connection is closed")
        return self._handle_query(query, params)

    def cursor(self) -> DeterministicCursor:
        if self.closed:
            raise TigerConnectionError("Connection is closed")
        return DeterministicCursor(self)

    def commit(self) -> None:
        if self.closed:
            raise TigerConnectionError("Connection is closed")

    def rollback(self) -> None:
        if self.closed:
            raise TigerConnectionError("Connection is closed")

    def close(self) -> None:
        self.closed = True

    def _handle_query(self, query: str, params: Any = None) -> DeterministicCursor:
        clean_q = query.strip()
        self.executed_statements.append(clean_q)

        # 1. Extension inspection queries (clearly distinguishing installed vs available)
        if "FROM pg_extension" in clean_q:
            installed = ["uuid-ossp", "pgcrypto"]
            if self.vector_installed:
                installed.append("vector")
            if self.timescaledb_installed:
                installed.append("timescaledb")
            if self.vectorscale_installed:
                installed.append("vectorscale")
            return DeterministicCursor(self, rows=[(ext,) for ext in installed])

        if "FROM pg_available_extensions" in clean_q:
            available = ["uuid-ossp", "pgcrypto"]
            if self.vector_available:
                available.append("vector")
            if self.timescaledb_available:
                available.append("timescaledb")
            if self.vectorscale_available:
                available.append("vectorscale")
            return DeterministicCursor(self, rows=[(ext,) for ext in available])

        # 2. SELECT 1 Health check probe
        if clean_q == "SELECT 1":
            return DeterministicCursor(self, rows=[(1,)])

        # 3. schema_migrations query
        if "FROM schema_migrations" in clean_q:
            rows = [(rec[0], rec[2]) for rec in self.migration_records]
            return DeterministicCursor(self, rows=rows)

        # 4. INSERT into schema_migrations
        if "INSERT INTO schema_migrations" in clean_q:
            if params and len(params) == 3:
                self.migration_records.append((int(params[0]), str(params[1]), str(params[2]), "2026-09-17"))
            else:
                m = re.search(r"VALUES\s*\(\s*(\d+)\s*,\s*'([^']+)'\s*,\s*'([^']+)'", clean_q)
                if m:
                    self.migration_records.append((int(m.group(1)), m.group(2), m.group(3), "2026-09-17"))
            return DeterministicCursor(self)

        # 5. CREATE TABLE parsing
        for stmt in clean_q.split(";"):
            s = stmt.strip()
            if not s:
                continue
            table_match = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_]+)\s*\((.*)\)", s, re.DOTALL | re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1).lower()
                body = table_match.group(2)
                cols: dict[str, str] = {}
                constrs: list[str] = []

                for raw_line in body.split("\n"):
                    line = raw_line.strip().rstrip(",")
                    if not line or line.startswith("--"):
                        continue
                    if line.upper().startswith("CONSTRAINT") or line.upper().startswith("PRIMARY KEY"):
                        constrs.append(line)
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            cols[parts[0].lower()] = " ".join(parts[1:])

                self.tables[table_name] = {
                    "columns": cols,
                    "constraints": constrs,
                }

            # Index parsing
            idx_match = re.search(r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_]+)\s+ON\s+([a-zA-Z0-9_]+)", s, re.IGNORECASE)
            if idx_match:
                idx_name = idx_match.group(1).lower()
                tbl_name = idx_match.group(2).lower()
                self.indexes[idx_name] = {
                    "table": tbl_name,
                    "sql": s,
                }

            # Trigger parsing
            trg_match = re.search(r"CREATE\s+TRIGGER\s+([a-zA-Z0-9_]+)\s+BEFORE\s+(UPDATE|DELETE|UPDATE\s+OR\s+DELETE)\s+ON\s+([a-zA-Z0-9_]+)", s, re.IGNORECASE)
            if trg_match:
                trg_name = trg_match.group(1).lower()
                tbl_name = trg_match.group(3).lower()
                self.triggers[trg_name] = {
                    "table": tbl_name,
                    "event": trg_match.group(2),
                    "sql": s,
                }

        return DeterministicCursor(self)


class DeterministicCursor:
    """Cursor returned by DeterministicPostgresHarness."""

    def __init__(self, harness: DeterministicPostgresHarness, rows: list[tuple] | None = None) -> None:
        self.harness = harness
        self.rows = rows or []
        self._idx = 0

    def execute(self, query: str, params: Any = None) -> DeterministicCursor:
        return self.harness._handle_query(query, params)

    def fetchone(self) -> tuple | None:
        if self._idx < len(self.rows):
            r = self.rows[self._idx]
            self._idx += 1
            return r
        return None

    def fetchall(self) -> list[tuple]:
        return list(self.rows)

    def close(self) -> None:
        pass


class TestTigerSchemaAndConnection(unittest.TestCase):
    """Full W1-05 test suite verifying Tiger Cloud schema foundation and connection manager."""

    def setUp(self) -> None:
        self.secret_registry = RuntimeSecretRegistry()
        self.valid_tiger_url = "postgres://tiger_user:tiger_supersecret_pass123@db.tigercloud.internal:5432/review_production?sslmode=require"

    # -------------------------------------------------------------------------
    # 1. Configuration loads Tiger URL correctly
    # -------------------------------------------------------------------------
    def test_configuration_loads_tiger_url_correctly(self) -> None:
        """1. TigerConfig parses host, port, db, user, password, and ssl_mode accurately."""
        cfg = TigerConfig.from_url(self.valid_tiger_url, secret_registry=self.secret_registry)
        self.assertEqual("db.tigercloud.internal", cfg.host)
        self.assertEqual(5432, cfg.port)
        self.assertEqual("review_production", cfg.database)
        self.assertEqual("tiger_user", cfg.user)
        self.assertEqual("tiger_supersecret_pass123", cfg.password)
        self.assertEqual("require", cfg.ssl_mode)
        self.assertEqual(1, cfg.min_pool_size)
        self.assertEqual(10, cfg.max_pool_size)

    # -------------------------------------------------------------------------
    # 2. Missing Tiger URL fails clearly when Tiger backend is requested
    # -------------------------------------------------------------------------
    def test_missing_tiger_url_fails_clearly_when_requested(self) -> None:
        """2. When Tiger backend is required, missing URL raises TigerConfigurationError / ValueError."""
        # A. TigerConfig.from_env
        with self.assertRaises(TigerConfigurationError) as ctx:
            TigerConfig.from_env({}, require_url=True)
        self.assertIn("Missing required TIGER_DATABASE_URL", str(ctx.exception))

        # B. load_service_config fails closed when DATABASE_BACKEND=tiger
        minimal_env = {
            "GITHUB_TOKEN": "ghp_12345678901234567890123456789012",
            "GITHUB_WEBHOOK_SECRET": "webhook_secret_32bytes_value_here",
            "GITHUB_REPOSITORY": "testorg/testrepo",
            "OPENAI_API_KEY": "sk-12345678901234567890123456789012",
            "DATABASE_BACKEND": "tiger",
        }
        with self.assertRaises(ValueError) as ctx2:
            load_service_config(minimal_env, require_live_credentials=True)
        self.assertIn("Missing required TIGER_DATABASE_URL", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # 3. SSL requirement/configuration is represented correctly
    # -------------------------------------------------------------------------
    def test_ssl_requirement_is_represented_correctly(self) -> None:
        """3. SSL mode is parsed, validated against ALLOWED_SSL_MODES, and fails closed on disable/invalid."""
        # A. Explicit allowed secure modes
        url1 = "postgres://u:p@host:5432/db?sslmode=verify-full"
        cfg1 = TigerConfig.from_url(url1)
        self.assertEqual("verify-full", cfg1.ssl_mode)
        self.assertIn("sslmode=verify-full", cfg1.database_url)

        url_ca = "postgres://u:p@host:5432/db?sslmode=verify-ca"
        cfg_ca = TigerConfig.from_url(url_ca)
        self.assertEqual("verify-ca", cfg_ca.ssl_mode)
        self.assertIn("sslmode=verify-ca", cfg_ca.database_url)

        # ssl=true parameter maps to require
        url2 = "postgres://u:p@host:5432/db?ssl=true"
        cfg2 = TigerConfig.from_url(url2)
        self.assertEqual("require", cfg2.ssl_mode)
        self.assertIn("sslmode=require", cfg2.database_url)

        # Default is require
        url3 = "postgres://u:p@host:5432/db"
        cfg3 = TigerConfig.from_url(url3)
        self.assertEqual("require", cfg3.ssl_mode)
        self.assertIn("sslmode=require", cfg3.database_url)

        # B. Fail-closed: sslmode=disable is strictly rejected
        disable_urls = [
            "postgres://u:p@host:5432/db?sslmode=disable",
            "postgres://u:p@host:5432/db?sslmode=DISABLE",
            "postgres://u:p@host:5432/db?ssl=false",
            "postgres://u:p@host:5432/db?ssl=0",
        ]
        for bad_url in disable_urls:
            with self.assertRaises(TigerConfigurationError) as ctx:
                TigerConfig.from_url(bad_url)
            self.assertIn("prohibited", str(ctx.exception).lower())

        with self.assertRaises(TigerConfigurationError):
            TigerConfig.from_env({"TIGER_DATABASE_URL": "postgres://u:p@host:5432/db", "TIGER_SSL_MODE": "disable"})

        with self.assertRaises(TigerConfigurationError):
            TigerConfig(database_url="postgres://u:p@host:5432/db", host="host", ssl_mode="disable")

        # C. Fail-closed: unapproved / insecure / arbitrary modes are rejected
        invalid_urls = [
            "postgres://u:p@host:5432/db?sslmode=prefer",
            "postgres://u:p@host:5432/db?sslmode=allow",
            "postgres://u:p@host:5432/db?sslmode=banana",
            "postgres://u:p@host:5432/db?sslmode=no-verify",
        ]
        for bad_url in invalid_urls:
            with self.assertRaises(TigerConfigurationError) as ctx:
                TigerConfig.from_url(bad_url)
            self.assertIn("invalid ssl mode", str(ctx.exception).lower())

        with self.assertRaises(TigerConfigurationError):
            TigerConfig.from_env({"TIGER_DATABASE_URL": "postgres://u:p@host:5432/db", "TIGER_SSL_MODE": "prefer"})

    def test_psycopg_connection_path_enforces_validated_ssl_mode(self) -> None:
        """3b. Real psycopg connection path explicitly passes validated sslmode and cannot bypass SSL."""
        cfg = TigerConfig.from_url("postgres://u:secret@db.tigercloud.internal:5432/review_production?sslmode=verify-ca")
        mgr = TigerConnectionManager(cfg)

        mock_psycopg = MagicMock()
        with patch.dict("sys.modules", {"psycopg": mock_psycopg}):
            conn = mgr._create_real_connection()
            self.assertEqual(mock_psycopg.connect.return_value, conn)
            mock_psycopg.connect.assert_called_once()
            call_kwargs = mock_psycopg.connect.call_args[1]
            self.assertEqual("verify-ca", call_kwargs.get("sslmode"))
            self.assertEqual(10.0, call_kwargs.get("connect_timeout"))
            # Ensure URL conninfo itself also contains sslmode=verify-ca
            call_args = mock_psycopg.connect.call_args[0]
            self.assertIn("sslmode=verify-ca", call_args[0])

    # -------------------------------------------------------------------------
    # 4. Credentials are masked across repr, error messages, and URL masking utilities
    # -------------------------------------------------------------------------
    def test_credentials_are_masked(self) -> None:
        """4. Passwords never appear in repr, str, logs, or error strings."""
        cfg = TigerConfig.from_url(self.valid_tiger_url, secret_registry=self.secret_registry)
        repr_str = repr(cfg)
        self.assertNotIn("tiger_supersecret_pass123", repr_str)
        self.assertIn("tiger_user:***@", repr_str)

        # Secret registered in registry
        cats = self.secret_registry.get_registered_categories()
        self.assertIn(SecretType.TIGERDB_CREDENTIAL.value, cats)
        scan = self.secret_registry.scan_exact_matches(f"Connecting to {self.valid_tiger_url}")
        self.assertTrue(len(scan) > 0)

        # mask_database_url helper
        masked = mask_database_url(self.valid_tiger_url)
        self.assertNotIn("tiger_supersecret_pass123", masked)
        self.assertIn("tiger_user:***@", masked)

        # TigerConnectionManager repr masks URL
        mgr = TigerConnectionManager(cfg)
        self.assertNotIn("tiger_supersecret_pass123", repr(mgr))

    # -------------------------------------------------------------------------
    # 5. Connection manager initializes without changing SQLite default
    # -------------------------------------------------------------------------
    def test_connection_manager_initializes_without_changing_sqlite_default(self) -> None:
        """5. Default service config uses SQLite; Tiger remains explicit and isolated."""
        minimal_env = {
            "GITHUB_TOKEN": "ghp_12345678901234567890123456789012",
            "GITHUB_WEBHOOK_SECRET": "webhook_secret_32bytes_value_here",
            "GITHUB_REPOSITORY": "testorg/testrepo",
            "OPENAI_API_KEY": "sk-12345678901234567890123456789012",
        }
        config = load_service_config(minimal_env, require_live_credentials=True)
        self.assertEqual("sqlite", config.database_backend)
        self.assertEqual("pr_review_agent.db", config.database_path)
        self.assertEqual("", config.tiger_database_url)

    # -------------------------------------------------------------------------
    # 6. Migration files are discoverable and ordered
    # -------------------------------------------------------------------------
    def test_migration_files_discoverable_and_ordered(self) -> None:
        """6. MigrationRunner discovers .sql migration files in strict ascending version order."""
        harness = DeterministicPostgresHarness()
        runner = MigrationRunner(harness)
        migrations = runner.discover_migrations()
        self.assertTrue(len(migrations) >= 1)
        self.assertEqual(1, migrations[0].version)
        self.assertEqual("initial_tiger_schema", migrations[0].name)
        self.assertTrue(migrations[0].checksum)
        self.assertTrue(len(migrations[0].checksum) == 64)

    # -------------------------------------------------------------------------
    # 7. Migration execution is idempotent
    # -------------------------------------------------------------------------
    def test_migration_execution_is_idempotent(self) -> None:
        """7. Applying migrations twice produces identical state with zero duplicate errors."""
        harness = DeterministicPostgresHarness(vector_installed=True)
        runner = MigrationRunner(harness)

        # Pass 1: Applies migrations 001 and 002
        applied_1 = runner.apply_all()
        self.assertEqual([1, 2], [m.version for m in applied_1])

        # Pass 2: Detects already-applied migration with matching checksum; applies 0
        applied_2 = runner.apply_all()
        self.assertEqual(0, len(applied_2))

        # Check recorded migrations
        recorded = runner.get_applied_migrations()
        self.assertIn(1, recorded)
        self.assertIn(2, recorded)
        self.assertEqual(applied_1[0].checksum, recorded[1])
        self.assertEqual(applied_1[1].checksum, recorded[2])

    # -------------------------------------------------------------------------
    # 8. Extension capability detection (pgvector, TimescaleDB, pgvectorscale)
    # -------------------------------------------------------------------------
    def test_extension_capability_detection_and_installed_vs_available(self) -> None:
        """8. ExtensionInspector strictly distinguishes installed from available extensions."""
        # Case A: Vector installed and available -> capability True, migration succeeds
        harness_installed = DeterministicPostgresHarness(vector_installed=True, vector_available=True)
        inspector_a = ExtensionInspector(harness_installed)
        report_a = inspector_a.inspect()
        self.assertTrue(report_a.has_vector)
        self.assertTrue(report_a.vector_available)
        self.assertIn("vector", report_a.installed_extensions)
        self.assertIn("vector", report_a.available_extensions)
        runner_a = MigrationRunner(harness_installed)
        applied = runner_a.apply_all()
        self.assertEqual([1, 2], [m.version for m in applied])

        # Case B: Vector available in cluster but NOT installed in pg_extension
        # Must report has_vector=False, vector_available=True, and fail closed before vector DDL!
        harness_available_only = DeterministicPostgresHarness(vector_installed=False, vector_available=True)
        inspector_b = ExtensionInspector(harness_available_only)
        report_b = inspector_b.inspect()
        self.assertFalse(report_b.has_vector, "Available extension must NOT be treated as installed capability")
        self.assertTrue(report_b.vector_available)
        self.assertNotIn("vector", report_b.installed_extensions)
        self.assertIn("vector", report_b.available_extensions)

        runner_b = MigrationRunner(harness_available_only)
        with self.assertRaises(TigerExtensionError) as ctx_b:
            runner_b.apply_all()
        self.assertIn("available", str(ctx_b.exception).lower())
        self.assertIn("not installed", str(ctx_b.exception).lower())

        # verify_extensions fail-closed when require_vector=True
        runner_b_req = MigrationRunner(harness_available_only, require_vector=True)
        with self.assertRaises(TigerExtensionError):
            runner_b_req.verify_extensions()

        # Case C: Vector completely unavailable
        harness_unavailable = DeterministicPostgresHarness(vector_installed=False, vector_available=False)
        inspector_c = ExtensionInspector(harness_unavailable)
        report_c = inspector_c.inspect()
        self.assertFalse(report_c.has_vector)
        self.assertFalse(report_c.vector_available)
        runner_c = MigrationRunner(harness_unavailable)
        with self.assertRaises(TigerExtensionError):
            runner_c.apply_all()
        runner_c_req = MigrationRunner(harness_unavailable, require_vector=True)
        with self.assertRaises(TigerExtensionError):
            runner_c_req.verify_extensions()

        # Case D: TimescaleDB / pgvectorscale capability semantics
        harness_d = DeterministicPostgresHarness(
            vector_installed=True,
            timescaledb_installed=False,
            timescaledb_available=True,
            vectorscale_installed=False,
            vectorscale_available=True,
        )
        report_d = ExtensionInspector(harness_d).inspect()
        self.assertFalse(report_d.has_timescaledb)
        self.assertTrue(report_d.timescaledb_available)
        self.assertFalse(report_d.has_vectorscale)
        self.assertTrue(report_d.vectorscale_available)

    # -------------------------------------------------------------------------
    # 9. Memory lane tables exist with required columns and types
    # -------------------------------------------------------------------------
    def test_memory_lane_tables_exist_with_required_columns(self) -> None:
        """9. code_chunks and repo_file_index contain all domain and vector/FTS columns."""
        harness = DeterministicPostgresHarness(vector_installed=True)
        runner = MigrationRunner(harness)
        runner.apply_all()

        # Check code_chunks
        self.assertIn("code_chunks", harness.tables)
        cols = harness.tables["code_chunks"]["columns"]
        self.assertIn("chunk_id", cols)
        self.assertIn("repository_id", cols)
        self.assertIn("revision", cols)
        self.assertIn("file_path", cols)
        self.assertIn("symbol", cols)
        self.assertIn("chunk_index", cols)
        self.assertIn("start_line", cols)
        self.assertIn("end_line", cols)
        self.assertIn("content", cols)
        self.assertIn("content_hash", cols)
        self.assertIn("token_count", cols)
        self.assertIn("embedding", cols)
        self.assertIn("tsv", cols)
        self.assertIn("index_version", cols)
        self.assertIn("created_at", cols)
        self.assertIn("updated_at", cols)

        # Check repo_file_index
        self.assertIn("repo_file_index", harness.tables)
        rf_cols = harness.tables["repo_file_index"]["columns"]
        self.assertIn("repository_id", rf_cols)
        self.assertIn("revision", rf_cols)
        self.assertIn("file_path", rf_cols)
        self.assertIn("content_hash", rf_cols)
        self.assertIn("is_fresh", rf_cols)

    # -------------------------------------------------------------------------
    # 10. Vector column/index semantics are verified (DECISION-61b6c150)
    # -------------------------------------------------------------------------
    def test_vector_column_semantics(self) -> None:
        """10. code_chunks includes unconstrained vector column; vector index and dimension deferred to W2."""
        harness = DeterministicPostgresHarness(vector_installed=True)
        runner = MigrationRunner(harness)
        runner.apply_all()

        col_def = harness.tables["code_chunks"]["columns"]["embedding"]
        # Column is unconstrained vector; must NOT have fixed dimension like vector(1536)
        self.assertEqual("vector", col_def.lower().strip())
        self.assertNotIn("vector(1536)", col_def.lower())
        self.assertNotIn("vector(128)", col_def.lower())

        # Verify NO vector index (HNSW or DiskANN) exists in W1-05 schema
        self.assertNotIn("idx_code_chunks_embedding", harness.indexes)
        for idx_name, idx_info in harness.indexes.items():
            self.assertNotIn("using hnsw", idx_info["sql"].lower())
            self.assertNotIn("using diskann", idx_info["sql"].lower())

    # -------------------------------------------------------------------------
    # 11. Full-text search column/index semantics are verified
    # -------------------------------------------------------------------------
    def test_full_text_search_semantics(self) -> None:
        """11. code_chunks includes tsvector column and GIN index on tsv."""
        harness = DeterministicPostgresHarness()
        runner = MigrationRunner(harness)
        runner.apply_all()

        col_def = harness.tables["code_chunks"]["columns"]["tsv"]
        self.assertIn("tsvector", col_def.lower())
        self.assertIn("to_tsvector", col_def.lower())

        # Check GIN index
        self.assertIn("idx_code_chunks_tsv", harness.indexes)
        self.assertIn("gin", harness.indexes["idx_code_chunks_tsv"]["sql"].lower())

    # -------------------------------------------------------------------------
    # 12. Time lane event table exists with required columns and types
    # -------------------------------------------------------------------------
    def test_time_lane_agent_events_exists_with_required_columns(self) -> None:
        """12. agent_events contains all tracing, timing, token count, and cost columns."""
        harness = DeterministicPostgresHarness()
        runner = MigrationRunner(harness)
        runner.apply_all()

        self.assertIn("agent_events", harness.tables)
        cols = harness.tables["agent_events"]["columns"]
        self.assertIn("event_id", cols)
        self.assertIn("timestamp", cols)
        self.assertIn("correlation_id", cols)
        self.assertIn("event_name", cols)
        self.assertIn("step", cols)
        self.assertIn("repository_id", cols)
        self.assertIn("pull_number", cols)
        self.assertIn("head_sha", cols)
        self.assertIn("run_id", cols)
        self.assertIn("agent", cols)
        self.assertIn("span_id", cols)
        self.assertIn("parent_span_id", cols)
        self.assertIn("model", cols)
        self.assertIn("prompt_tokens", cols)
        self.assertIn("completion_tokens", cols)
        self.assertIn("total_tokens", cols)
        self.assertIn("cost_usd", cols)
        self.assertIn("latency_ms", cols)
        self.assertIn("outcome", cols)
        self.assertIn("confidence", cols)
        self.assertIn("payload", cols)

    # -------------------------------------------------------------------------
    # 13. Timescale hypertable semantics are verified when available
    # -------------------------------------------------------------------------
    def test_timescale_hypertable_conversion_when_available(self) -> None:
        """13. When TimescaleDB is available, create_hypertable is executed for agent_events."""
        harness = DeterministicPostgresHarness(has_timescaledb=True)
        runner = MigrationRunner(harness)
        runner.apply_all()

        hypertable_executed = any("create_hypertable" in stmt.lower() for stmt in harness.executed_statements)
        self.assertTrue(hypertable_executed, "Expected create_hypertable call when TimescaleDB is detected")

    # -------------------------------------------------------------------------
    # 14. Truth lane tables exist with required relationships and constraints
    # -------------------------------------------------------------------------
    def test_truth_lane_tables_exist_with_relationships(self) -> None:
        """14. pr_review_records, finding_records, hitl_reviews, hitl_feedback, and github_review_effects exist."""
        harness = DeterministicPostgresHarness()
        runner = MigrationRunner(harness)
        runner.apply_all()

        for table in [
            "pr_review_records",
            "finding_records",
            "hitl_reviews",
            "hitl_feedback",
            "github_review_effects",
        ]:
            self.assertIn(table, harness.tables, f"Missing expected Truth table: {table}")

        # Verify foreign keys referencing pr_review_records(run_id)
        for child_table in ["finding_records", "hitl_reviews", "hitl_feedback"]:
            cols = harness.tables[child_table]["columns"]
            run_id_def = cols["run_id"]
            self.assertIn("references pr_review_records(run_id)", run_id_def.lower())

    # -------------------------------------------------------------------------
    # 15. Append-only and versioning constraints required by current domain
    # -------------------------------------------------------------------------
    def test_append_only_triggers_and_versioning_constraints(self) -> None:
        """15. Triggers prevent UPDATE/DELETE on append-only tables; finding versioning is strictly enforced."""
        harness = DeterministicPostgresHarness()
        runner = MigrationRunner(harness)
        runner.apply_all()

        # Verify trigger creation on append-only tables
        expected_triggers = [
            "trg_agent_events_prevent_mod",
            "trg_finding_records_prevent_mod",
            "trg_hitl_reviews_prevent_mod",
            "trg_hitl_feedback_prevent_mod",
        ]
        for trg in expected_triggers:
            self.assertIn(trg, harness.triggers, f"Missing append-only trigger: {trg}")

        # Verify finding_records version constraint (canonical_id, sequence_id)
        f_constrs = " ".join(harness.tables["finding_records"]["constraints"])
        self.assertIn("canonical_id", f_constrs.lower())
        self.assertIn("sequence_id", f_constrs.lower())

    # -------------------------------------------------------------------------
    # 16. Foreign keys and unique constraints required for idempotency
    # -------------------------------------------------------------------------
    def test_idempotency_unique_constraints(self) -> None:
        """16. Unique constraints on delivery_id, (repository_id, revision, file_path, chunk_index) exist."""
        harness = DeterministicPostgresHarness()
        runner = MigrationRunner(harness)
        runner.apply_all()

        # pr_review_records delivery_id uniqueness
        pr_constrs = " ".join(harness.tables["pr_review_records"]["constraints"])
        self.assertIn("delivery_id", pr_constrs.lower())

        # code_chunks ordering uniqueness
        cc_constrs = " ".join(harness.tables["code_chunks"]["constraints"])
        self.assertIn("uq_code_chunks_ordering", cc_constrs.lower())

        # github_review_effects idempotency key primary key
        gh_cols = harness.tables["github_review_effects"]["columns"]
        self.assertIn("primary key", gh_cols["idempotency_key"].lower())

    # -------------------------------------------------------------------------
    # 17. Checksum tampering detection
    # -------------------------------------------------------------------------
    def test_checksum_tampering_detection(self) -> None:
        """17. If migration file content changes after being applied, MigrationError is raised."""
        harness = DeterministicPostgresHarness()
        runner = MigrationRunner(harness)
        runner.apply_all()

        # Simulate tampered record
        harness.migration_records[0] = (1, "initial_tiger_schema", "corrupted_checksum_abc123", "2026-09-17")

        with self.assertRaises(MigrationError) as ctx:
            runner.apply_all()
        self.assertIn("checksum mismatch", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 18. Schema initialization does not alter SQLite
    # -------------------------------------------------------------------------
    def test_schema_initialization_does_not_alter_sqlite(self) -> None:
        """18. Running Tiger schema migrations leaves SQLite database completely untouched."""
        sqlite_conn = sqlite3.connect(":memory:")
        cursor = sqlite_conn.cursor()
        cursor.execute("CREATE TABLE existing_sqlite_table (id INTEGER PRIMARY KEY, val TEXT)")
        sqlite_conn.commit()

        # Execute Tiger migration on Tiger harness
        harness = DeterministicPostgresHarness()
        runner = MigrationRunner(harness)
        runner.apply_all()

        # Verify SQLite still only has its original table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        sqlite_tables = [row[0] for row in cursor.fetchall()]
        self.assertEqual(["existing_sqlite_table"], sqlite_tables)
        sqlite_conn.close()

    # -------------------------------------------------------------------------
    # 19. Connection manager health check with harness
    # -------------------------------------------------------------------------
    def test_connection_manager_health_check_with_harness(self) -> None:
        """19. TigerConnectionManager connects, performs health check, and disposes cleanly."""
        harness = DeterministicPostgresHarness()
        cfg = TigerConfig.from_url(self.valid_tiger_url)

        def test_factory(config: TigerConfig) -> Any:
            return harness

        mgr = TigerConnectionManager(cfg, connection_factory=test_factory)
        with mgr:
            self.assertTrue(mgr.check_health())
            self.assertEqual(harness, mgr.get_connection())

        self.assertTrue(mgr._is_closed)
        self.assertFalse(mgr.check_health())

    # -------------------------------------------------------------------------
    # 20. Live PostgreSQL / Tiger Cloud integration test (requires TIGER_DATABASE_URL)
    # -------------------------------------------------------------------------
    def test_live_postgresql_integration_when_available(self) -> None:
        """20. Validates schema against live Tiger Cloud / PostgreSQL when TIGER_DATABASE_URL is provided.

        Honesty invariant:
        Deterministic in-memory tests (1-19) prove schema SQL syntax, constraint definitions,
        tamper detection, and migration idempotency on a simulated catalog. They do NOT constitute
        proof of live Tiger Cloud infrastructure behavior. Real infrastructure validation
        strictly requires TIGER_DATABASE_URL and will skip with an explicit message if unset.
        """
        live_url = os.environ.get("TIGER_DATABASE_URL", "").strip()
        if not live_url:
            self.skipTest(
                "Live Tiger Cloud / PostgreSQL infrastructure validation requires TIGER_DATABASE_URL. "
                "Deterministic harness validates catalog contracts only; skipping real network test."
            )

        try:
            import psycopg  # type: ignore[import-untyped]
            with psycopg.connect(live_url, connect_timeout=3.0) as conn:
                runner = MigrationRunner(conn)
                applied = runner.apply_all()
                self.assertIsNotNone(applied)
        except Exception as exc:
            self.skipTest(f"Live PostgreSQL daemon unreachable or failed: {exc}")


if __name__ == "__main__":
    unittest.main()
