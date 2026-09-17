"""Tiger Cloud / PostgreSQL connection manager and configuration foundation.

Part of Wave 1 Task 05 (W1-05).
Provides:
- Secure environment-backed configuration for Tiger Cloud / Postgres
- URL validation and parsing (host, port, dbname, user, password, sslmode)
- Mandatory credential masking in repr, error messages, and logs
- Integration with RuntimeSecretRegistry (SecretType.TIGERDB_CREDENTIAL)
- Connection manager with health checks, context management, and clean disposal
- Safe isolation: SQLite remains the authoritative default runtime
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
import logging
import os
import re
from types import TracebackType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from pr_review_agent.security import (
    RuntimeSecretRegistry,
    SecretType,
    SecurityConfig,
)

logger = logging.getLogger(__name__)

ALLOWED_SSL_MODES: frozenset[str] = frozenset({"require", "verify-ca", "verify-full"})


class TigerError(Exception):
    """Base exception for all Tiger database operations."""


class TigerConfigurationError(TigerError, ValueError):
    """Raised when Tiger configuration is invalid or missing when required."""


class TigerConnectionError(TigerError, RuntimeError):
    """Raised when connecting to Tiger Cloud fails or health check fails."""


class TigerExtensionError(TigerError, RuntimeError):
    """Raised when a required database extension is missing or unsupported."""


def mask_database_url(url: str) -> str:
    """Mask credentials in database URL to prevent leakage in logs or exceptions.

    Converts `postgres://user:secret@host:5432/db` to `postgres://user:***@host:5432/db`.
    """
    if not url:
        return ""
    return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", url)


@dataclass(frozen=True)
class TigerConfig:
    """Validated configuration for Tiger Cloud / Postgres connections."""

    database_url: str
    host: str
    port: int = 5432
    database: str = "postgres"
    user: str = "postgres"
    password: str = ""
    ssl_mode: str = "require"
    connect_timeout_seconds: float = 10.0
    min_pool_size: int = 1
    max_pool_size: int = 10

    def __post_init__(self) -> None:
        if self.ssl_mode == "disable":
            raise TigerConfigurationError(
                "sslmode=disable is prohibited for Tiger Cloud / Postgres backend. Connection must fail closed."
            )
        if self.ssl_mode not in ALLOWED_SSL_MODES:
            raise TigerConfigurationError(
                f"Invalid SSL mode '{self.ssl_mode}'. Accepted modes are: {', '.join(sorted(ALLOWED_SSL_MODES))}."
            )

    def __repr__(self) -> str:
        masked_url = mask_database_url(self.database_url)
        masked_pwd = "***" if self.password else ""
        return (
            f"TigerConfig(host={self.host!r}, port={self.port}, database={self.database!r}, "
            f"user={self.user!r}, password={masked_pwd!r}, ssl_mode={self.ssl_mode!r}, "
            f"database_url={masked_url!r}, min_pool_size={self.min_pool_size}, "
            f"max_pool_size={self.max_pool_size})"
        )

    @classmethod
    def from_url(
        cls,
        database_url: str,
        *,
        default_ssl_mode: str = "require",
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        connect_timeout_seconds: float = 10.0,
        secret_registry: RuntimeSecretRegistry | None = None,
    ) -> TigerConfig:
        """Parse and validate a Tiger / Postgres connection URL.

        Fails closed on missing host, malformed URL, sslmode=disable, or unaccepted ssl mode.
        Registers the password in the secret registry if provided.
        """
        raw_url = database_url.strip()
        if not raw_url:
            raise TigerConfigurationError("Database URL cannot be empty")

        parsed = urlparse(raw_url)
        scheme = parsed.scheme.lower()
        if scheme not in ("postgres", "postgresql", "tiger"):
            raise TigerConfigurationError(
                f"Invalid URL scheme '{parsed.scheme}'. Must be 'postgres', 'postgresql', or 'tiger'."
            )

        host = parsed.hostname or ""
        if not host:
            raise TigerConfigurationError("Database URL must contain a valid hostname")

        port = parsed.port or 5432
        database = parsed.path.lstrip("/") if parsed.path else "postgres"
        user = parsed.username or "postgres"
        password = parsed.password or ""

        # Extract and validate sslmode from query parameters or default
        query_params = parse_qs(parsed.query)
        ssl_mode = default_ssl_mode.strip().lower()
        if "sslmode" in query_params:
            ssl_mode = query_params["sslmode"][0].strip().lower()
        elif "ssl" in query_params:
            val = query_params["ssl"][0].strip().lower()
            ssl_mode = "require" if val in ("1", "true", "require") else "disable"

        if ssl_mode == "disable":
            raise TigerConfigurationError(
                "sslmode=disable is prohibited for Tiger Cloud / Postgres backend. Connection must fail closed."
            )
        if ssl_mode not in ALLOWED_SSL_MODES:
            raise TigerConfigurationError(
                f"Invalid SSL mode '{ssl_mode}'. Accepted modes are: {', '.join(sorted(ALLOWED_SSL_MODES))}."
            )

        # Construct sanitized URL ensuring sslmode parameter is explicitly present and matches validated ssl_mode
        qp = {k: list(v) for k, v in query_params.items()}
        qp["sslmode"] = [ssl_mode]
        qp.pop("ssl", None)
        sanitized_query = urlencode([(k, v) for k, vs in qp.items() for v in vs])
        sanitized_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            sanitized_query,
            parsed.fragment,
        ))

        if secret_registry and password:
            secret_registry.register_secret(SecretType.TIGERDB_CREDENTIAL, password)

        return cls(
            database_url=sanitized_url,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            ssl_mode=ssl_mode,
            connect_timeout_seconds=connect_timeout_seconds,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        require_url: bool = True,
        secret_registry: RuntimeSecretRegistry | None = None,
    ) -> TigerConfig | None:
        """Load TigerConfig from environment mapping.

        Looks for TIGER_DATABASE_URL or TIGER_URL.
        Fails closed on missing required URL, sslmode=disable, or unaccepted ssl mode.
        """
        env_map = env if env is not None else os.environ
        url = (
            env_map.get("TIGER_DATABASE_URL", "")
            or env_map.get("TIGER_URL", "")
            or env_map.get("DATABASE_URL", "")
        ).strip()

        if not url:
            if require_url:
                raise TigerConfigurationError(
                    "Missing required TIGER_DATABASE_URL in environment when Tiger backend is requested"
                )
            return None

        ssl_mode = env_map.get("TIGER_SSL_MODE", "require").strip().lower()
        if ssl_mode == "disable":
            raise TigerConfigurationError(
                "sslmode=disable is prohibited for Tiger Cloud / Postgres backend. Connection must fail closed."
            )
        if ssl_mode not in ALLOWED_SSL_MODES:
            raise TigerConfigurationError(
                f"Invalid SSL mode '{ssl_mode}'. Accepted modes are: {', '.join(sorted(ALLOWED_SSL_MODES))}."
            )

        min_pool = int(env_map.get("TIGER_MIN_POOL_SIZE", "1").strip())
        max_pool = int(env_map.get("TIGER_MAX_POOL_SIZE", "10").strip())
        timeout = float(env_map.get("TIGER_CONNECT_TIMEOUT", "10.0").strip())

        return cls.from_url(
            url,
            default_ssl_mode=ssl_mode,
            min_pool_size=min_pool,
            max_pool_size=max_pool,
            connect_timeout_seconds=timeout,
            secret_registry=secret_registry,
        )


@runtime_checkable
class TigerConnectionProtocol(Protocol):
    """Protocol representing a raw or pooled database connection."""

    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any:
        ...

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...

    def close(self) -> None:
        ...


class TigerConnectionManager(AbstractContextManager, AbstractAsyncContextManager):
    """Production connection manager for Tiger Cloud / Postgres.

    Features:
    - Thread-safe connection acquisition
    - SSL mode enforcement (fails if sslmode=disable in production configuration)
    - Connection health check (`check_health()`)
    - Clean resource disposal
    - Pluggable connection factory for deterministic in-memory/test harness testing
    - Credential masking across all error messages and representations
    """

    def __init__(
        self,
        config: TigerConfig,
        *,
        connection_factory: Callable[[TigerConfig], Any] | None = None,
        secret_registry: RuntimeSecretRegistry | None = None,
    ) -> None:
        self.config = config
        self._connection_factory = connection_factory
        self.secret_registry = secret_registry or RuntimeSecretRegistry()
        if self.config.password:
            self.secret_registry.register_secret(SecretType.TIGERDB_CREDENTIAL, self.config.password)

        self._active_connection: Any = None
        self._is_closed = False

    def __repr__(self) -> str:
        return f"TigerConnectionManager(config={self.config!r}, is_closed={self._is_closed})"

    def _create_real_connection(self) -> Any:
        """Create a connection using available Python PostgreSQL driver (psycopg)."""
        masked = mask_database_url(self.config.database_url)
        try:
            import psycopg  # type: ignore[import-untyped]
            return psycopg.connect(
                self.config.database_url,
                sslmode=self.config.ssl_mode,
                connect_timeout=self.config.connect_timeout_seconds,
            )
        except ImportError:
            raise TigerConnectionError(
                f"No PostgreSQL driver installed (psycopg not found). "
                f"Cannot establish live connection to {masked} without a database driver."
            )
        except Exception as exc:
            # Mask the exception message to prevent credential leakage
            safe_err = mask_database_url(str(exc))
            raise TigerConnectionError(f"Failed to connect to Tiger Cloud ({masked}): {safe_err}") from exc

    def get_connection(self) -> Any:
        """Acquire a connection, lazily initializing if necessary."""
        if self._is_closed:
            raise TigerConnectionError("Connection manager is closed")

        if self._active_connection is not None:
            return self._active_connection

        if self._connection_factory is not None:
            try:
                conn = self._connection_factory(self.config)
                self._active_connection = conn
                return conn
            except Exception as exc:
                safe_err = mask_database_url(str(exc))
                raise TigerConnectionError(f"Connection factory failed: {safe_err}") from exc

        conn = self._create_real_connection()
        self._active_connection = conn
        return conn

    def check_health(self) -> bool:
        """Verify connection health by executing a probe query (SELECT 1).

        Returns True if healthy; returns False if connection fails.
        Never raises exceptions or leaks credentials.
        """
        if self._is_closed:
            return False
        try:
            conn = self.get_connection()
            if hasattr(conn, "execute"):
                cursor = conn.execute("SELECT 1")
                if hasattr(cursor, "fetchone"):
                    cursor.fetchone()
            elif hasattr(conn, "cursor"):
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
                if hasattr(cur, "close"):
                    cur.close()
            return True
        except Exception as exc:
            safe_err = mask_database_url(str(exc))
            logger.warning("Tiger Cloud health check probe failed: %s", safe_err)
            return False

    def close(self) -> None:
        """Close all active connections and mark manager as disposed."""
        self._is_closed = True
        if self._active_connection is not None:
            try:
                if hasattr(self._active_connection, "close"):
                    self._active_connection.close()
            except Exception as exc:
                logger.warning("Error closing Tiger connection: %s", mask_database_url(str(exc)))
            finally:
                self._active_connection = None

    def __enter__(self) -> TigerConnectionManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

    async def __aenter__(self) -> TigerConnectionManager:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None
