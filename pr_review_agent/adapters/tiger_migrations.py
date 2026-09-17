"""Tiger Cloud / PostgreSQL schema migration runner and capability inspector.

Part of Wave 1 Task 05 (W1-05).
Implements:
- Ordered discovery of versioned SQL migration files
- SHA-256 checksum tracking for tamper detection and idempotency
- Capability inspection for PostgreSQL extensions (pgvector, TimescaleDB, pgvectorscale)
- Safe idempotent migration execution with transaction boundaries
- Extension requirement enforcement (fail-closed on missing required extensions)
- Clean isolation: Does not touch or modify SQLite
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Any

from pr_review_agent.adapters.tiger_connection import (
    TigerConnectionError,
    TigerConnectionManager,
    TigerExtensionError,
    mask_database_url,
)

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


class MigrationError(RuntimeError):
    """Raised when migration discovery, checksum verification, or execution fails."""


@dataclass(frozen=True)
class Migration:
    """Represents a discoverable SQL migration script."""

    version: int
    name: str
    file_path: Path
    sql: str
    checksum: str


@dataclass(frozen=True)
class CapabilityReport:
    """Report of detected PostgreSQL extension capabilities.

    Invariants:
    - has_vector / has_timescaledb / has_vectorscale are True ONLY when the extension is
      actually installed in pg_extension (i.e. usable for DDL/types/functions).
    - vector_available / timescaledb_available / vectorscale_available reflect presence
      in pg_available_extensions (available for installation, but not currently usable).
    """

    has_vector: bool = False
    has_timescaledb: bool = False
    has_vectorscale: bool = False
    vector_available: bool = False
    timescaledb_available: bool = False
    vectorscale_available: bool = False
    installed_extensions: tuple[str, ...] = field(default_factory=tuple)
    available_extensions: tuple[str, ...] = field(default_factory=tuple)


class ExtensionInspector:
    """Inspects PostgreSQL database capabilities and installed extensions."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def inspect(self) -> CapabilityReport:
        """Query pg_extension and pg_available_extensions to determine capability flags."""
        installed: list[str] = []
        available: list[str] = []

        try:
            # Query currently installed extensions
            cur = self._execute_query("SELECT extname FROM pg_extension")
            if cur:
                rows = cur.fetchall() if hasattr(cur, "fetchall") else []
                installed = [row[0].lower() for row in rows]
        except Exception as exc:
            logger.debug("Failed to query pg_extension: %s", exc)

        try:
            # Query available extensions
            cur = self._execute_query("SELECT name FROM pg_available_extensions")
            if cur:
                rows = cur.fetchall() if hasattr(cur, "fetchall") else []
                available = [row[0].lower() for row in rows]
        except Exception as exc:
            logger.debug("Failed to query pg_available_extensions: %s", exc)

        # Distinguish installed (usable) from merely available (not usable until installed)
        has_vector = "vector" in installed
        has_timescaledb = "timescaledb" in installed
        has_vectorscale = "vectorscale" in installed

        vector_available = "vector" in available
        timescaledb_available = "timescaledb" in available
        vectorscale_available = "vectorscale" in available

        return CapabilityReport(
            has_vector=has_vector,
            has_timescaledb=has_timescaledb,
            has_vectorscale=has_vectorscale,
            vector_available=vector_available,
            timescaledb_available=timescaledb_available,
            vectorscale_available=vectorscale_available,
            installed_extensions=tuple(sorted(installed)),
            available_extensions=tuple(sorted(available)),
        )

    def _execute_query(self, query: str) -> Any:
        if hasattr(self.connection, "execute"):
            return self.connection.execute(query)
        elif hasattr(self.connection, "cursor"):
            cur = self.connection.cursor()
            cur.execute(query)
            return cur
        raise TigerConnectionError("Connection object has no execute or cursor method")


class MigrationRunner:
    """Discovers, verifies, and executes SQL schema migrations for Tiger Cloud / Postgres."""

    def __init__(
        self,
        connection: Any,
        migrations_dir: Path | str | None = None,
        *,
        require_vector: bool = False,
    ) -> None:
        self.connection = connection
        self.migrations_dir = Path(migrations_dir) if migrations_dir else MIGRATIONS_DIR
        self.require_vector = require_vector
        self.inspector = ExtensionInspector(connection)

    def discover_migrations(self) -> list[Migration]:
        """Discover and validate all .sql files in the migrations directory in ascending order."""
        if not self.migrations_dir.exists():
            raise MigrationError(f"Migrations directory '{self.migrations_dir}' does not exist")

        sql_files = sorted(self.migrations_dir.glob("*.sql"))
        migrations: list[Migration] = []

        for p in sql_files:
            match = re.match(r"^(\d+)_(.+)\.sql$", p.name)
            if not match:
                logger.warning("Skipping non-conforming migration file: %s", p.name)
                continue

            version = int(match.group(1))
            name = match.group(2)
            sql = p.read_text(encoding="utf-8")
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()

            migrations.append(
                Migration(
                    version=version,
                    name=name,
                    file_path=p,
                    sql=sql,
                    checksum=checksum,
                )
            )

        # Verify strict monotonicity of versions
        versions = [m.version for m in migrations]
        if len(versions) != len(set(versions)):
            raise MigrationError(f"Duplicate migration versions found: {versions}")

        return sorted(migrations, key=lambda m: m.version)

    def init_migration_table(self) -> None:
        """Create the schema_migrations tracking table if it does not already exist."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            checksum VARCHAR(64) NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
        self._execute_sql(create_sql)
        self._commit()

    def get_applied_migrations(self) -> dict[int, str]:
        """Return dict of {version: checksum} for all applied migrations."""
        self.init_migration_table()
        cur = self._execute_sql("SELECT version, checksum FROM schema_migrations ORDER BY version ASC")
        if not cur:
            return {}
        rows = cur.fetchall() if hasattr(cur, "fetchall") else []
        return {int(row[0]): str(row[1]) for row in rows}

    def verify_extensions(self) -> CapabilityReport:
        """Verify installed database extensions before applying migrations.

        Fails closed if require_vector is True and vector is not installed in pg_extension.
        """
        report = self.inspector.inspect()
        if self.require_vector and not report.has_vector:
            if report.vector_available:
                raise TigerExtensionError(
                    "Required PostgreSQL extension 'pgvector' is available in the cluster but NOT installed in pg_extension. "
                    "Migration cannot proceed without vector support actually installed."
                )
            raise TigerExtensionError(
                "Required PostgreSQL extension 'pgvector' is not installed in the target database. "
                "Migration cannot proceed without vector support."
            )
        return report

    def apply_all(self) -> list[Migration]:
        """Apply all pending migrations in order.

        Returns list of newly applied migrations.
        Idempotent: if all migrations are already applied, returns empty list without error.
        """
        self.init_migration_table()
        capabilities = self.verify_extensions()
        logger.info("Tiger Cloud capabilities verified: %s", capabilities)

        applied = self.get_applied_migrations()
        discovered = self.discover_migrations()
        applied_now: list[Migration] = []

        for m in discovered:
            if m.version in applied:
                recorded_checksum = applied[m.version]
                if recorded_checksum != m.checksum:
                    raise MigrationError(
                        f"Migration {m.version} ('{m.name}') checksum mismatch! "
                        f"Recorded: {recorded_checksum}, Current file: {m.checksum}. "
                        f"Migrations are immutable and must not be edited after application."
                    )
                logger.debug("Migration %d ('%s') already applied, skipping.", m.version, m.name)
                continue

            # Fail closed before attempting vector-dependent DDL when pgvector is not installed
            if re.search(r"\bvector\b", m.sql, re.IGNORECASE) and not capabilities.has_vector:
                if capabilities.vector_available:
                    raise TigerExtensionError(
                        f"Migration {m.version} ('{m.name}') contains vector-dependent DDL but 'pgvector' "
                        "is only available, not installed in pg_extension. Automatic extension installation "
                        "is prohibited without Genesis authorization; fail closed."
                    )
                raise TigerExtensionError(
                    f"Migration {m.version} ('{m.name}') contains vector-dependent DDL but 'pgvector' "
                    "is not installed in the target database."
                )

            # Execute migration in a safe transaction
            logger.info("Applying migration %d: %s", m.version, m.name)
            try:
                self._execute_migration(m, capabilities)
                applied_now.append(m)
            except Exception as exc:
                self._rollback()
                safe_err = mask_database_url(str(exc))
                raise MigrationError(f"Failed to apply migration {m.version} ('{m.name}'): {safe_err}") from exc

        return applied_now

    def _execute_migration(self, migration: Migration, capabilities: CapabilityReport) -> None:
        """Execute the SQL script and record completion in schema_migrations."""
        # Execute migration DDL
        self._execute_sql(migration.sql)

        # If TimescaleDB is available, apply hypertable conversion on agent_events
        if capabilities.has_timescaledb:
            try:
                self._execute_sql(
                    "SELECT create_hypertable('agent_events', by_range('timestamp', INTERVAL '1 day'), if_not_exists => TRUE);"
                )
            except Exception as exc:
                logger.warning("Could not convert agent_events to hypertable: %s", exc)

        # Record migration in tracking table
        record_sql = """
        INSERT INTO schema_migrations (version, name, checksum, applied_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """
        # Adapt parameter syntax based on driver / dialect if needed ($1 or %s vs ?)
        self._record_applied(migration.version, migration.name, migration.checksum)
        self._commit()

    def _record_applied(self, version: int, name: str, checksum: str) -> None:
        """Insert migration record into schema_migrations."""
        # Check driver parameter style
        driver_style = getattr(self.connection, "paramstyle", "pyformat")
        if driver_style in ("format", "pyformat"):
            sql = "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)"
        elif driver_style == "numeric":
            sql = "INSERT INTO schema_migrations (version, name, checksum) VALUES ($1, $2, $3)"
        else:
            # qmark or fallback
            sql = "INSERT INTO schema_migrations (version, name, checksum) VALUES (?, ?, ?)"

        try:
            self._execute_sql(sql, (version, name, checksum))
        except Exception:
            # Direct literal insert if parameter substitution fails on custom harness
            escaped_name = name.replace("'", "''")
            literal_sql = f"INSERT INTO schema_migrations (version, name, checksum) VALUES ({version}, '{escaped_name}', '{checksum}')"
            self._execute_sql(literal_sql)

    def _execute_sql(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        if hasattr(self.connection, "execute"):
            if params is not None:
                return self.connection.execute(sql, params)
            return self.connection.execute(sql)
        elif hasattr(self.connection, "cursor"):
            cur = self.connection.cursor()
            if params is not None:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            return cur
        raise TigerConnectionError("Connection object has no execute or cursor method")

    def _commit(self) -> None:
        if hasattr(self.connection, "commit"):
            self.connection.commit()

    def _rollback(self) -> None:
        if hasattr(self.connection, "rollback"):
            self.connection.rollback()
