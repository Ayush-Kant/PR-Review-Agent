"""Unit tests for the W1-08B Gate 2 Real Tiger / PostgreSQL validation harness.

Tests offline simulation, mock connection behavior, credential masking, fail-closed handling,
all 13 checks, capabilities distinction, and deterministic cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any
import unittest

from pr_review_agent.adapters.tiger_connection import (
    TigerConfig,
    TigerConnectionManager,
    mask_database_url,
)
from pr_review_agent.adapters.tiger_migrations import MigrationRunner
from scripts.validate_tiger_gate2 import (
    CheckResult,
    TigerGate2Validator,
    ValidationReport,
    parse_tiger_metadata,
)
from tests.test_tiger_data_adapters import (
    SimulationCursor,
    TigerPostgresSimulationConnection,
)


class OfflineTestTigerConnection(TigerPostgresSimulationConnection):
    """In-memory connection simulating PostgreSQL catalog, DDL, and migrations for offline harness testing."""

    def __init__(self, *, simulate_health_failure: bool = False) -> None:
        super().__init__()
        self.simulate_health_failure = simulate_health_failure
        self._is_offline_mock = True
        self._init_extra_schema()

    def _init_extra_schema(self) -> None:
        with self._lock:
            with self.conn:
                self.conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS _gate2_probe (
                        probe_id TEXT PRIMARY KEY
                    );
                    """
                )
                # Pre-seed applied migrations matching disk migrations 001 and 002
                # to test idempotency and checksum matching without live PostgreSQL DDL
                runner = MigrationRunner(self, require_vector=False)
                discovered = runner.discover_migrations()
                for m in discovered:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO schema_migrations (version, name, checksum) VALUES (?, ?, ?);",
                        (m.version, m.name, m.checksum),
                    )

    def execute(self, query: str, params: Any = None) -> Any:
        with self._lock:
            q_strip = query.strip()

            # Health check probe
            if q_strip == "SELECT 1" or q_strip == "SELECT 1;":
                if self.simulate_health_failure:
                    raise sqlite3.OperationalError("Simulated database health failure")
                return SimulationCursor([(1,)])

            # Extension catalog queries
            if "FROM pg_extension" in q_strip:
                return SimulationCursor([("vector",), ("uuid-ossp",), ("pgcrypto",)])
            if "FROM pg_available_extensions" in q_strip:
                return SimulationCursor([("vector",), ("timescaledb",), ("pgcrypto",)])

            return super().execute(query, params)

    def rollback(self) -> None:
        with self._lock:
            if self._in_transaction:
                self.conn.rollback()
                self._in_transaction = False


class TestTigerGate2Validator(unittest.TestCase):
    """Offline validation harness unit tests for Gate 2."""

    def setUp(self) -> None:
        self.client = OfflineTestTigerConnection()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.evidence_path = Path(self.temp_dir.name) / "tiger_gate2_evidence.json"

        config = TigerConfig.from_url("postgresql://tiger_user:secret_pwd@127.0.0.1:5432/pr_review_db?sslmode=require")
        self.manager = TigerConnectionManager(config, connection_factory=lambda cfg: self.client)
        self.manager._is_offline_mock = True

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_url_masking(self) -> None:
        """Verify complete credential masking in PostgreSQL database URLs."""
        masked1 = mask_database_url("postgres://tiger_user:super_secret_pw@db.tiger.cloud:5432/staging_db?sslmode=require")
        self.assertEqual(masked1, "postgres://tiger_user:***@db.tiger.cloud:5432/staging_db?sslmode=require")
        self.assertNotIn("super_secret_pw", masked1)

        masked2 = mask_database_url("postgresql://admin:pass456@127.0.0.1:5432/testdb")
        self.assertEqual(masked2, "postgresql://admin:***@127.0.0.1:5432/testdb")
        self.assertNotIn("pass456", masked2)

    def test_metadata_parsing(self) -> None:
        """Verify non-sensitive metadata extraction from database URLs."""
        meta = parse_tiger_metadata("postgresql://app_user:db_token@tiger-host.internal:5432/review_truth?sslmode=verify-full")
        self.assertEqual(meta["scheme"], "postgresql")
        self.assertEqual(meta["host"], "tiger-host.internal")
        self.assertEqual(meta["port"], 5432)
        self.assertEqual(meta["database"], "review_truth")
        self.assertEqual(meta["sslmode"], "verify-full")
        self.assertTrue(meta["authenticated"])
        self.assertNotIn("db_token", meta["masked_url"])

    def test_no_url_or_manager_behavior(self) -> None:
        """Verify validator fails closed with not_executed when no URL or manager is provided."""
        validator = TigerGate2Validator(tiger_url="", connection_manager=None, evidence_path=self.evidence_path)
        report = validator.validate_all()
        self.assertEqual(report.overall_status, "not_executed")
        self.assertFalse(report.live_tiger_executed)
        self.assertIn("No TIGER_DATABASE_URL provided", report.failure_details or "")

    def test_ssl_mode_disable_rejection(self) -> None:
        """Verify that sslmode=disable fails closed."""
        with self.assertRaises(Exception):
            TigerConfig.from_url("postgresql://u:p@host:5432/db?sslmode=disable")

    def test_offline_validation_success(self) -> None:
        """Run full validation suite with offline simulation client and verify all 13 checks pass."""
        validator = TigerGate2Validator(
            connection_manager=self.manager,
            allow_insecure=True,
            require_staging_isolation=False,
            evidence_path=self.evidence_path,
        )
        report = validator.validate_all()

        self.assertEqual(report.overall_status, "offline_verified")
        self.assertFalse(report.live_tiger_executed)
        self.assertIsNone(report.failure_details)

        # Verify all 13 checks are present and passed
        expected_checks = [
            "ssl_enforcement",
            "health_check",
            "connection_lifecycle",
            "transaction_control",
            "pgvector_capability",
            "migration_checksum_and_idempotency",
            "review_truth_lifecycle",
            "audit_chronology",
            "code_memory_freshness",
            "effect_store_idempotency",
            "stale_sha_rejection",
            "secret_redaction",
            "cleanup",
        ]
        self.assertEqual(len(expected_checks), 13)
        for name in expected_checks:
            self.assertIn(name, report.checks, f"Missing check: {name}")
            self.assertEqual(report.checks[name]["status"], "passed", f"Check {name} failed: {report.checks[name]}")

        # Verify report persistence
        self.assertTrue(self.evidence_path.exists())
        saved = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["overall_status"], "offline_verified")
        self.assertIn("capabilities", saved)
        self.assertIn("modular_production_path", saved)

    def test_capabilities_distinction(self) -> None:
        """Verify that offline simulation distinguishes offline verification from live proof,

        and correctly reports unproven/deferred infrastructure capabilities.
        """
        validator = TigerGate2Validator(
            connection_manager=self.manager,
            allow_insecure=True,
            require_staging_isolation=False,
            evidence_path=self.evidence_path,
        )
        report = validator.validate_all()
        self.assertIsNotNone(report.capabilities)
        caps = report.capabilities or {}

        # Functional capabilities verified offline
        for func_cap in (
            "tiger_postgres_functional_semantics",
            "tiger_ssl_transport",
            "tiger_pgvector_extension",
            "tiger_schema_migrations",
            "tiger_secret_redaction",
        ):
            self.assertIn(func_cap, caps)
            self.assertEqual(caps[func_cap]["status"], "OFFLINE_VERIFIED")
            self.assertEqual(caps[func_cap]["verdict"], "pass")
            # Offline simulation must NEVER claim PROVEN live status
            self.assertNotEqual(caps[func_cap]["status"], "PROVEN")

        # Infrastructure capabilities deferred
        for infra_cap in (
            "tiger_ha_failover",
            "tiger_managed_backups_recovery",
            "tiger_private_networking",
        ):
            self.assertIn(infra_cap, caps)
            self.assertEqual(caps[infra_cap]["status"], "NOT_PROVEN")
            self.assertEqual(caps[infra_cap]["verdict"], "deferred")

    def test_health_check_failure(self) -> None:
        """Verify that health check failure causes validator to report failed status."""
        bad_conn = OfflineTestTigerConnection(simulate_health_failure=True)
        config = TigerConfig.from_url("postgresql://user:pass@127.0.0.1:5432/db?sslmode=require")
        bad_manager = TigerConnectionManager(config, connection_factory=lambda cfg: bad_conn)
        bad_manager._is_offline_mock = True

        validator = TigerGate2Validator(
            connection_manager=bad_manager,
            allow_insecure=True,
            require_staging_isolation=False,
            evidence_path=self.evidence_path,
        )
        report = validator.validate_all()
        self.assertEqual(report.overall_status, "failed")
        self.assertEqual(report.checks["health_check"]["status"], "failed")

    def test_deterministic_cleanup(self) -> None:
        """Verify that cleanup removes harness-created records without touching unrelated records."""
        # Pre-seed unrelated records
        self.client.execute(
            "INSERT INTO pr_review_records (run_id, repository_id, pull_number, head_sha, base_sha, delivery_id, state, policy_version) "
            "VALUES ('unrelated-run-999', 'unrelated-repo', 1, 'sha1', 'sha0', 'unrelated-del-999', 'completed', 'v1');"
        )
        self.client.commit()

        validator = TigerGate2Validator(
            connection_manager=self.manager,
            allow_insecure=True,
            require_staging_isolation=False,
            evidence_path=self.evidence_path,
        )
        report = validator.validate_all()
        self.assertEqual(report.overall_status, "offline_verified")

        # Unrelated record must remain intact
        cur = self.client.execute("SELECT COUNT(*) FROM pr_review_records WHERE run_id = 'unrelated-run-999';")
        count = int(cur.fetchone()[0])
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
