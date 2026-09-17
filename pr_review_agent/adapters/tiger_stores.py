"""Tiger Cloud / PostgreSQL production data adapters (W1-06).

Implements the three durable state shapes defined in DECISION-d1715a54:
1. TigerCodeMemoryStore: code memory, file indexing, and revision freshness
2. TigerReviewTruthStore: append-only finding lifecycle and review run context
3. TigerAuditSpine: append-only time-ordered event spine and run provenance
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import time
from typing import Any
import uuid

from pr_review_agent.adapters.tiger_connection import (
    TigerConnectionError,
    TigerConnectionManager,
)
from pr_review_agent.observability import (
    AuditRecord,
    RunProvenanceTrace,
    SENSITIVE_KEY_PATTERNS,
    SENSITIVE_VALUE_PATTERNS,
    redact_sensitive_data,
)
from pr_review_agent.orchestration import AuditEvent
from pr_review_agent.policy import (
    ALLOWED_TRANSITIONS,
    CanonicalFinding,
    ReviewTruthRecord,
    TruthState,
)
from pr_review_agent.retrieval import CodeChunk, CodeMemoryStoreProtocol

logger = logging.getLogger(__name__)


class TigerStoreError(RuntimeError):
    """Base error for Tiger Cloud production data adapters."""


class TigerTruthMissingParentError(TigerStoreError):
    """Raised when review truth operations lack registered parent review context."""


class ConflictingFindingError(TigerStoreError):
    """Raised when encountering conflicting canonical finding or review run context."""


class ConcurrentTransitionError(TigerStoreError):
    """Raised when a concurrent transition conflict cannot be resolved."""


class RunNotFoundError(KeyError, TigerStoreError):
    """Raised when reconstruct_run finds no trace of the correlation ID or run ID."""


@dataclass(frozen=True)
class ReviewRunContext:
    """Immutable context representing a pull request review run header."""

    run_id: str
    repository_id: str
    pull_number: int
    head_sha: str
    base_sha: str
    delivery_id: str
    state: str = "in_progress"
    policy_version: str = "v1"


def _execute(conn: Any, query: str, params: Sequence[Any] | Mapping[str, Any] | None = None) -> Any:
    """Helper to execute query across psycopg, sqlite3, or test harnesses."""
    if hasattr(conn, "execute"):
        if params is not None:
            return conn.execute(query, params)
        return conn.execute(query)
    elif hasattr(conn, "cursor"):
        cur = conn.cursor()
        if params is not None:
            cur.execute(query, params)
        else:
            cur.execute(query)
        return cur
    raise TigerConnectionError("Connection object has neither execute nor cursor method")


def _commit(conn: Any) -> None:
    """Helper to commit transaction if supported."""
    if hasattr(conn, "commit"):
        conn.commit()


def _rollback(conn: Any) -> None:
    """Helper to rollback transaction if supported."""
    if hasattr(conn, "rollback"):
        conn.rollback()


def _to_uuid(val: Any) -> str:
    """Convert an opaque domain chunk identifier into a valid RFC 4122 UUID string for Tiger PostgreSQL."""
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, str):
        try:
            return str(uuid.UUID(val))
        except (ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, val))
    return str(uuid.uuid4())


def _parse_timestamp(ts: Any) -> float:
    """Parse a timestamp into a UNIX float across psycopg datetime, ISO string, or numeric."""
    if ts is None:
        return time.time()
    if hasattr(ts, "timestamp"):
        return ts.timestamp()
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            iso_str = ts.replace(" ", "T")
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            try:
                return float(ts)
            except Exception:
                return time.time()
    return time.time()


def _canonical_finding_fingerprint(data: Mapping[str, Any]) -> str:
    """Return a deterministic JSON canonical payload/fingerprint for comparing CanonicalFinding data."""
    raw_lr = data.get("line_range")
    line_range_list = list(raw_lr) if isinstance(raw_lr, (list, tuple)) else None

    material_fields = {
        "canonical_id": data.get("canonical_id"),
        "repository_id": data.get("repository_id"),
        "head_sha": data.get("head_sha"),
        "category": data.get("category"),
        "severity": data.get("severity"),
        "confidence": float(data.get("confidence", 0.0)) if data.get("confidence") is not None else None,
        "summary": data.get("summary"),
        "rationale": data.get("rationale"),
        "file_path": data.get("file_path"),
        "line_range": line_range_list,
        "contributing_candidate_ids": sorted(list(data.get("contributing_candidate_ids") or [])),
        "contributing_specialists": sorted(list(data.get("contributing_specialists") or [])),
        "evidence_refs": sorted(list(data.get("evidence_refs") or [])),
        "remediation": data.get("remediation"),
        "disposition": str(data.get("disposition")),
        "disposition_reason": data.get("disposition_reason"),
        "merge_rationale": data.get("merge_rationale"),
        "policy_version": data.get("policy_version"),
        "delivery_id": data.get("delivery_id"),
        "run_id": data.get("run_id"),
        "raw_severity": data.get("raw_severity"),
        "calibrated_severity": data.get("calibrated_severity"),
        "calibration_rule": data.get("calibration_rule"),
        "calibration_reason": data.get("calibration_reason"),
    }
    return json.dumps(material_fields, sort_keys=True)


def _has_unredacted_secrets(obj: Any) -> bool:
    """Check whether a data structure contains unredacted secrets."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k == "contains_secrets":
                continue
            if SENSITIVE_KEY_PATTERNS.search(str(k)):
                if v != "[REDACTED]":
                    return True
            if _has_unredacted_secrets(v):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_unredacted_secrets(i) for i in obj)
    elif isinstance(obj, str):
        if SENSITIVE_VALUE_PATTERNS.search(obj):
            return True
    return False


# ============================================================================
# 1. TIGER CODE MEMORY STORE
# ============================================================================


class TigerCodeMemoryStore(CodeMemoryStoreProtocol):
    """Tiger Cloud / PostgreSQL-backed repository code memory store.

    Persists into:
    - code_chunks: Chunk contents and unconstrained embeddings (Wave 2)
    - repo_file_index: File-level index versioning and content hashes
    - repository_revisions: Revision-level freshness and atomic snapshot transitions
    """

    def __init__(self, connection_manager: TigerConnectionManager) -> None:
        self.connection_manager = connection_manager

    def index_repository(
        self,
        repository_id: str,
        revision: str,
        chunks: Sequence[CodeChunk] | Mapping[str, str],
        *,
        index_version: str = "v1",
        chunk_line_size: int = 50,
        now: float | None = None,
    ) -> int:
        """Index a snapshot of chunks or files for a revision, ensuring repository/revision isolation."""
        current_time = time.time() if now is None else now
        persisted_now = datetime.fromtimestamp(current_time, tz=timezone.utc)
        conn = self.connection_manager.get_connection()
        indexed_count = 0

        try:
            # Mark prior revisions of this repository as stale atomically
            _execute(
                conn,
                "UPDATE repository_revisions SET is_fresh = FALSE WHERE repository_id = %s AND revision != %s",
                (repository_id, revision),
            )

            # Remove previous chunks for this exact revision if reindexing
            _execute(
                conn,
                "DELETE FROM code_chunks WHERE repository_id = %s AND revision = %s",
                (repository_id, revision),
            )

            if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)):
                # Group chunks by file_path
                files_map: dict[str, list[CodeChunk]] = {}
                for chk in chunks:
                    files_map.setdefault(chk.file_path, []).append(chk)

                for file_path, fchunks in sorted(files_map.items()):
                    combined = "".join(c.content for c in fchunks)
                    file_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                    _execute(
                        conn,
                        """
                        INSERT INTO repo_file_index (
                            repository_id, revision, file_path, content_hash, index_version, is_fresh, indexed_at
                        ) VALUES (%s, %s, %s, %s, %s, TRUE, %s)
                        ON CONFLICT (repository_id, revision, file_path) DO UPDATE SET
                            content_hash = EXCLUDED.content_hash,
                            index_version = EXCLUDED.index_version,
                            is_fresh = TRUE,
                            indexed_at = %s
                        """,
                        (repository_id, revision, file_path, file_hash, index_version, persisted_now, persisted_now),
                    )

                    for idx, chk in enumerate(fchunks):
                        chunk_uuid = _to_uuid(chk.chunk_id)
                        _execute(
                            conn,
                            """
                            INSERT INTO code_chunks (
                                chunk_id, repository_id, revision, file_path,
                                chunk_index, start_line, end_line, content, content_hash,
                                index_version, created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (repository_id, revision, file_path, chunk_index) DO UPDATE SET
                                start_line = EXCLUDED.start_line,
                                end_line = EXCLUDED.end_line,
                                content = EXCLUDED.content,
                                content_hash = EXCLUDED.content_hash,
                                index_version = EXCLUDED.index_version,
                                updated_at = %s
                            """,
                            (
                                chunk_uuid,
                                repository_id,
                                revision,
                                file_path,
                                idx,
                                chk.start_line,
                                chk.end_line,
                                chk.content,
                                chk.content_hash,
                                chk.index_version,
                                persisted_now,
                                persisted_now,
                                persisted_now,
                            ),
                        )
                        indexed_count += 1
            elif isinstance(chunks, Mapping):
                # Process each file into chunks
                for file_path, content in sorted(chunks.items()):
                    file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    # Record / update file index entry
                    _execute(
                        conn,
                        """
                        INSERT INTO repo_file_index (
                            repository_id, revision, file_path, content_hash, index_version, is_fresh, indexed_at
                        ) VALUES (%s, %s, %s, %s, %s, TRUE, %s)
                        ON CONFLICT (repository_id, revision, file_path) DO UPDATE SET
                            content_hash = EXCLUDED.content_hash,
                            index_version = EXCLUDED.index_version,
                            is_fresh = TRUE,
                            indexed_at = %s
                        """,
                        (repository_id, revision, file_path, file_hash, index_version, persisted_now, persisted_now),
                    )

                    lines = content.splitlines(keepends=True)
                    if not lines:
                        continue

                    chunk_idx = 0
                    for i in range(0, len(lines), chunk_line_size):
                        chunk_lines = lines[i : i + chunk_line_size]
                        start_line = i + 1
                        end_line = i + len(chunk_lines)
                        chunk_text = "".join(chunk_lines)
                        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

                        # Derive deterministic UUID from domain coordinates
                        domain_chunk_key = f"{repository_id}:{revision}:{file_path}:{start_line}"
                        chunk_uuid = _to_uuid(domain_chunk_key)

                        _execute(
                            conn,
                            """
                            INSERT INTO code_chunks (
                                chunk_id, repository_id, revision, file_path,
                                chunk_index, start_line, end_line, content, content_hash,
                                index_version, created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (repository_id, revision, file_path, chunk_index) DO UPDATE SET
                                start_line = EXCLUDED.start_line,
                                end_line = EXCLUDED.end_line,
                                content = EXCLUDED.content,
                                content_hash = EXCLUDED.content_hash,
                                index_version = EXCLUDED.index_version,
                                updated_at = %s
                            """,
                            (
                                chunk_uuid,
                                repository_id,
                                revision,
                                file_path,
                                chunk_idx,
                                start_line,
                                end_line,
                                chunk_text,
                                chunk_hash,
                                index_version,
                                persisted_now,
                                persisted_now,
                                persisted_now,
                            ),
                        )
                        chunk_idx += 1
                        indexed_count += 1

            # Record revision-level freshness (supports 0-chunk revisions cleanly)
            _execute(
                conn,
                """
                INSERT INTO repository_revisions (
                    repository_id, revision, is_fresh, indexed_at, chunk_count
                ) VALUES (%s, %s, TRUE, %s, %s)
                ON CONFLICT (repository_id, revision) DO UPDATE SET
                    is_fresh = TRUE,
                    indexed_at = %s,
                    chunk_count = EXCLUDED.chunk_count
                """,
                (repository_id, revision, persisted_now, indexed_count, persisted_now),
            )

            _commit(conn)
            return indexed_count
        except Exception:
            _rollback(conn)
            raise

    def is_fresh(self, repository_id: str, revision: str) -> bool:
        """Return True if the repository revision is known and fresh."""
        conn = self.connection_manager.get_connection()
        cur = _execute(
            conn,
            "SELECT is_fresh FROM repository_revisions WHERE repository_id = %s AND revision = %s",
            (repository_id, revision),
        )
        if not cur:
            return False
        row = cur.fetchone() if hasattr(cur, "fetchone") else None
        return bool(row and row[0])

    def get_chunks(
        self,
        repository_id: str,
        revision: str,
    ) -> list[CodeChunk]:
        """Fetch chunks scoped strictly by repository and revision to preserve tenancy boundaries."""
        return self._get_chunks_for_file(repository_id, revision, file_path=None)

    def _get_chunks_for_file(
        self,
        repository_id: str,
        revision: str,
        file_path: str | None = None,
    ) -> list[CodeChunk]:
        """Internal helper to fetch chunks scoped by repository, revision, and optional file_path."""
        conn = self.connection_manager.get_connection()
        if file_path:
            cur = _execute(
                conn,
                """
                SELECT chunk_id, repository_id, revision, file_path, start_line,
                       end_line, content, content_hash, index_version, created_at
                FROM code_chunks
                WHERE repository_id = %s AND revision = %s AND file_path = %s
                ORDER BY start_line ASC
                """,
                (repository_id, revision, file_path),
            )
        else:
            cur = _execute(
                conn,
                """
                SELECT chunk_id, repository_id, revision, file_path, start_line,
                       end_line, content, content_hash, index_version, created_at
                FROM code_chunks
                WHERE repository_id = %s AND revision = %s
                ORDER BY file_path ASC, start_line ASC
                """,
                (repository_id, revision),
            )

        if not cur:
            return []
        rows = cur.fetchall() if hasattr(cur, "fetchall") else []
        results: list[CodeChunk] = []
        for r in rows:
            created_val = r[9]
            created_ts = _parse_timestamp(created_val)
            # Reconstruct domain chunk identifier
            domain_id = f"{r[1]}:{r[2]}:{r[3]}:{r[4]}"
            results.append(
                CodeChunk(
                    chunk_id=domain_id,
                    repository_id=r[1],
                    revision=r[2],
                    file_path=r[3],
                    start_line=r[4],
                    end_line=r[5],
                    content=r[6],
                    content_hash=r[7],
                    index_version=r[8],
                    created_at=created_ts,
                )
            )
        return results


# ============================================================================
# 2. TIGER REVIEW TRUTH STORE
# ============================================================================


class TigerReviewTruthStore:
    """Tiger Cloud / PostgreSQL-backed Review Truth store tracking finding lifecycles.

    Persists into:
    - pr_review_records: Review-run header (parent context)
    - finding_records: Append-only finding lifecycle records carrying sequence_id
    """

    def __init__(
        self,
        connection_manager: TigerConnectionManager,
        *,
        run_context_resolver: Callable[[str], ReviewRunContext | None] | None = None,
    ) -> None:
        self.connection_manager = connection_manager
        self.run_context_resolver = run_context_resolver

    def register_review_run(
        self,
        run_id: str,
        repository_id: str,
        pull_number: int,
        head_sha: str,
        base_sha: str,
        delivery_id: str,
        state: str = "in_progress",
        policy_version: str = "v1",
    ) -> None:
        """Register a review run header idempotently, failing closed on conflicting provenance."""
        if not all([run_id, repository_id, pull_number, head_sha, base_sha, delivery_id]):
            raise TigerTruthMissingParentError("All review run context fields are required and cannot be empty")

        conn = self.connection_manager.get_connection()

        try:
            # Check existing run_id
            cur = _execute(
                conn,
                "SELECT repository_id, pull_number, head_sha, base_sha, delivery_id FROM pr_review_records WHERE run_id = %s",
                (run_id,),
            )
            existing = cur.fetchone() if cur and hasattr(cur, "fetchone") else None
            if existing:
                if (
                    existing[0] == repository_id
                    and existing[1] == pull_number
                    and existing[2] == head_sha
                    and existing[3] == base_sha
                    and existing[4] == delivery_id
                ):
                    return  # Identical registration is idempotent
                raise ConflictingFindingError(
                    f"Run ID '{run_id}' already registered with conflicting context: "
                    f"existing={existing}, requested={(repository_id, pull_number, head_sha, base_sha, delivery_id)}"
                )

            # Check existing delivery_id
            cur_deliv = _execute(
                conn,
                "SELECT run_id FROM pr_review_records WHERE delivery_id = %s",
                (delivery_id,),
            )
            existing_deliv = cur_deliv.fetchone() if cur_deliv and hasattr(cur_deliv, "fetchone") else None
            if existing_deliv and existing_deliv[0] != run_id:
                raise ConflictingFindingError(
                    f"Delivery ID '{delivery_id}' already associated with a different run ID '{existing_deliv[0]}'"
                )

            _execute(
                conn,
                """
                INSERT INTO pr_review_records (
                    run_id, repository_id, pull_number, head_sha, base_sha, delivery_id, state, policy_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, repository_id, pull_number, head_sha, base_sha, delivery_id, state, policy_version),
            )
            _commit(conn)
        except Exception:
            _rollback(conn)
            raise

    def _resolve_context(
        self,
        finding: CanonicalFinding,
        explicit_run_id: str | None,
        explicit_delivery_id: str | None,
    ) -> ReviewRunContext:
        """Resolve and validate parent review run context, strictly failing closed on uncertainty."""
        # Fail closed if both explicit and finding identifiers are present and differ
        if explicit_run_id and finding.run_id and explicit_run_id != finding.run_id:
            raise ConflictingFindingError(
                f"Explicit run_id '{explicit_run_id}' conflicts with finding.run_id '{finding.run_id}'"
            )
        if explicit_delivery_id and finding.delivery_id and explicit_delivery_id != finding.delivery_id:
            raise ConflictingFindingError(
                f"Explicit delivery_id '{explicit_delivery_id}' conflicts with finding.delivery_id '{finding.delivery_id}'"
            )

        run_id = explicit_run_id or finding.run_id
        delivery_id = explicit_delivery_id or finding.delivery_id

        if not run_id or run_id == "run-unknown":
            if self.run_context_resolver:
                resolved = self.run_context_resolver(finding.canonical_id)
                if resolved:
                    self.register_review_run(
                        run_id=resolved.run_id,
                        repository_id=resolved.repository_id,
                        pull_number=resolved.pull_number,
                        head_sha=resolved.head_sha,
                        base_sha=resolved.base_sha,
                        delivery_id=resolved.delivery_id,
                        state=resolved.state,
                        policy_version=resolved.policy_version,
                    )
                    run_id = resolved.run_id
                    delivery_id = resolved.delivery_id

        if not run_id or run_id == "run-unknown":
            raise TigerTruthMissingParentError(
                f"Cannot record finding '{finding.canonical_id}' without registered review run context"
            )

        conn = self.connection_manager.get_connection()
        cur = _execute(
            conn,
            "SELECT run_id, repository_id, pull_number, head_sha, base_sha, delivery_id, state, policy_version "
            "FROM pr_review_records WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone() if cur and hasattr(cur, "fetchone") else None
        if not row:
            if self.run_context_resolver:
                resolved = self.run_context_resolver(run_id)
                if resolved:
                    self.register_review_run(
                        run_id=resolved.run_id,
                        repository_id=resolved.repository_id,
                        pull_number=resolved.pull_number,
                        head_sha=resolved.head_sha,
                        base_sha=resolved.base_sha,
                        delivery_id=resolved.delivery_id,
                        state=resolved.state,
                        policy_version=resolved.policy_version,
                    )
                    row = (
                        resolved.run_id,
                        resolved.repository_id,
                        resolved.pull_number,
                        resolved.head_sha,
                        resolved.base_sha,
                        resolved.delivery_id,
                        resolved.state,
                        resolved.policy_version,
                    )

        if not row:
            raise TigerTruthMissingParentError(
                f"Review run '{run_id}' is not registered in pr_review_records. Cannot record finding."
            )

        # Validate cross-provenance consistency
        ctx = ReviewRunContext(
            run_id=row[0],
            repository_id=row[1],
            pull_number=row[2],
            head_sha=row[3],
            base_sha=row[4],
            delivery_id=row[5],
            state=row[6],
            policy_version=row[7],
        )

        if finding.repository_id and finding.repository_id != ctx.repository_id:
            raise ConflictingFindingError(
                f"Finding repository '{finding.repository_id}' conflicts with registered run repository '{ctx.repository_id}'"
            )
        if finding.head_sha and finding.head_sha != ctx.head_sha:
            raise ConflictingFindingError(
                f"Finding head_sha '{finding.head_sha}' conflicts with registered run head_sha '{ctx.head_sha}'"
            )
        if run_id and run_id != ctx.run_id and run_id != "run-unknown":
            raise ConflictingFindingError(
                f"Effective run_id '{run_id}' conflicts with registered run context run_id '{ctx.run_id}'"
            )
        if finding.run_id and finding.run_id != ctx.run_id and finding.run_id != "run-unknown":
            raise ConflictingFindingError(
                f"Finding run_id '{finding.run_id}' conflicts with registered run context run_id '{ctx.run_id}'"
            )
        if delivery_id and delivery_id != ctx.delivery_id and delivery_id != "delivery-unknown":
            raise ConflictingFindingError(
                f"Effective delivery_id '{delivery_id}' conflicts with registered run context delivery_id '{ctx.delivery_id}'"
            )
        if finding.delivery_id and finding.delivery_id != ctx.delivery_id and finding.delivery_id != "delivery-unknown":
            raise ConflictingFindingError(
                f"Finding delivery_id '{finding.delivery_id}' conflicts with registered run context delivery_id '{ctx.delivery_id}'"
            )

        return ctx

    def record_initial(
        self,
        finding: CanonicalFinding,
        *,
        delivery_id: str | None = None,
        run_id: str | None = None,
        initial_state: TruthState,
        actor: str | None = None,
        actor_role: str | None = None,
        rationale: str | None = None,
        now: float | None = None,
    ) -> ReviewTruthRecord:
        """Record the initial finding in Review Truth with sequence_id = 1."""
        current_time = time.time() if now is None else now
        persisted_now = datetime.fromtimestamp(current_time, tz=timezone.utc)
        ctx = self._resolve_context(finding, run_id, delivery_id)

        sequence_id = 1
        record_id = f"truth-{finding.canonical_id}-{sequence_id}"
        finding_dict = asdict(finding)
        finding_dict["disposition"] = (
            finding.disposition.value if hasattr(finding.disposition, "value") else str(finding.disposition)
        )
        finding_json = json.dumps(finding_dict, sort_keys=True)
        effective_rationale = rationale or finding.disposition_reason

        conn = self.connection_manager.get_connection()

        try:
            # Check existing sequence 1 record
            cur = _execute(
                conn,
                """
                SELECT record_id, canonical_id, sequence_id, repository_id, head_sha,
                       delivery_id, run_id, state, actor, actor_role, rationale,
                       finding_json, timestamp
                FROM finding_records
                WHERE canonical_id = %s AND sequence_id = 1
                """,
                (finding.canonical_id,),
            )
            existing = cur.fetchone() if cur and hasattr(cur, "fetchone") else None
            if existing:
                # Idempotency check: verify full canonical payload fingerprint and parent provenance
                existing_data = json.loads(existing[11]) if isinstance(existing[11], str) else existing[11]
                if (
                    _canonical_finding_fingerprint(existing_data) == _canonical_finding_fingerprint(finding_dict)
                    and existing[3] == ctx.repository_id
                    and existing[4] == ctx.head_sha
                    and existing[5] == ctx.delivery_id
                    and existing[6] == ctx.run_id
                ):
                    return self._row_to_record(existing)
                raise ConflictingFindingError(
                    f"Conflicting finding data detected for existing canonical ID '{finding.canonical_id}'"
                )

            _execute(
                conn,
                """
                INSERT INTO finding_records (
                    record_id, canonical_id, sequence_id, repository_id, head_sha,
                    delivery_id, run_id, state, actor, actor_role,
                    rationale, finding_json, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record_id,
                    finding.canonical_id,
                    sequence_id,
                    ctx.repository_id,
                    ctx.head_sha,
                    ctx.delivery_id,
                    ctx.run_id,
                    initial_state.value,
                    actor,
                    actor_role,
                    effective_rationale,
                    finding_json,
                    persisted_now,
                ),
            )
            _commit(conn)
        except Exception as exc:
            _rollback(conn)
            # Handle concurrent race: if another worker inserted sequence 1 first
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                cur_retry = _execute(
                    conn,
                    """
                    SELECT record_id, canonical_id, sequence_id, repository_id, head_sha,
                           delivery_id, run_id, state, actor, actor_role, rationale,
                           finding_json, timestamp
                    FROM finding_records
                    WHERE canonical_id = %s AND sequence_id = 1
                    """,
                    (finding.canonical_id,),
                )
                r_row = cur_retry.fetchone() if cur_retry and hasattr(cur_retry, "fetchone") else None
                if r_row:
                    r_data = json.loads(r_row[11]) if isinstance(r_row[11], str) else r_row[11]
                    if (
                        _canonical_finding_fingerprint(r_data) == _canonical_finding_fingerprint(finding_dict)
                        and r_row[3] == ctx.repository_id
                        and r_row[4] == ctx.head_sha
                        and r_row[5] == ctx.delivery_id
                        and r_row[6] == ctx.run_id
                    ):
                        return self._row_to_record(r_row)
                    raise ConflictingFindingError(
                        f"Concurrent conflicting finding detected for '{finding.canonical_id}'"
                    ) from exc
            raise

        return ReviewTruthRecord(
            record_id=record_id,
            canonical_id=finding.canonical_id,
            sequence_id=sequence_id,
            repository_id=ctx.repository_id,
            head_sha=ctx.head_sha,
            delivery_id=ctx.delivery_id,
            run_id=ctx.run_id,
            state=initial_state,
            actor=actor,
            actor_role=actor_role,
            rationale=effective_rationale,
            finding_data=finding_dict,
            timestamp=current_time,
        )

    def record_transition(
        self,
        canonical_id: str,
        new_state: TruthState,
        *,
        actor: str | None = None,
        actor_role: str | None = None,
        rationale: str | None = None,
        updated_finding: CanonicalFinding | None = None,
        now: float | None = None,
    ) -> ReviewTruthRecord:
        """Validate and record a state transition with strict database transaction locking safety."""
        current_time = time.time() if now is None else now
        persisted_now = datetime.fromtimestamp(current_time, tz=timezone.utc)
        conn = self.connection_manager.get_connection()

        # 1. Acquire transaction-level database lock for canonical_id.
        # Must fail-closed on lock acquisition failure and never continue without serialization.
        try:
            _execute(conn, "SELECT pg_advisory_xact_lock(hashtext(%s))", (canonical_id,))
        except Exception as exc:
            _rollback(conn)
            raise ConcurrentTransitionError(
                f"Failed to acquire advisory transaction lock for '{canonical_id}': {exc}"
            ) from exc

        try:
            # 2. Re-read latest committed state after acquiring lock
            cur_latest = _execute(
                conn,
                """
                SELECT record_id, canonical_id, sequence_id, repository_id, head_sha,
                       delivery_id, run_id, state, actor, actor_role,
                       rationale, finding_json, timestamp
                FROM finding_records
                WHERE canonical_id = %s
                ORDER BY sequence_id DESC
                LIMIT 1
                """,
                (canonical_id,),
            )
            latest_row = cur_latest.fetchone() if cur_latest and hasattr(cur_latest, "fetchone") else None
            if latest_row is None:
                raise KeyError(f"Canonical finding {canonical_id} not found in Review Truth")
            latest = self._row_to_record(latest_row)

            # 3. Validate transition after lock acquisition
            current_state = latest.state
            allowed = ALLOWED_TRANSITIONS.get(current_state, frozenset())
            if new_state not in allowed:
                raise ValueError(
                    f"Illegal state transition from {current_state.value} to {new_state.value} for {canonical_id}"
                )

            # Validate updated_finding provenance consistency if provided
            if updated_finding:
                if updated_finding.canonical_id != canonical_id:
                    raise ConflictingFindingError(
                        f"Updated finding canonical_id '{updated_finding.canonical_id}' does not match transition target '{canonical_id}'"
                    )
                if updated_finding.repository_id and updated_finding.repository_id != latest.repository_id:
                    raise ConflictingFindingError(
                        f"Updated finding repository_id '{updated_finding.repository_id}' does not match truth record '{latest.repository_id}'"
                    )
                if updated_finding.head_sha and updated_finding.head_sha != latest.head_sha:
                    raise ConflictingFindingError(
                        f"Updated finding head_sha '{updated_finding.head_sha}' does not match truth record '{latest.head_sha}'"
                    )
                if updated_finding.run_id and updated_finding.run_id != latest.run_id and updated_finding.run_id != "run-unknown":
                    raise ConflictingFindingError(
                        f"Updated finding run_id '{updated_finding.run_id}' does not match truth record '{latest.run_id}'"
                    )
                if updated_finding.delivery_id and updated_finding.delivery_id != latest.delivery_id and updated_finding.delivery_id != "delivery-unknown":
                    raise ConflictingFindingError(
                        f"Updated finding delivery_id '{updated_finding.delivery_id}' does not match truth record '{latest.delivery_id}'"
                    )

            # 4. Allocate next sequence
            seq = latest.sequence_id + 1
            record_id = f"truth-{canonical_id}-{seq}"

            finding_dict = asdict(updated_finding) if updated_finding else dict(latest.finding_data)
            if updated_finding:
                finding_dict["disposition"] = (
                    updated_finding.disposition.value
                    if hasattr(updated_finding.disposition, "value")
                    else str(updated_finding.disposition)
                )
            finding_json = json.dumps(finding_dict, sort_keys=True)

            # 5. Append immutable record and commit
            _execute(
                conn,
                """
                INSERT INTO finding_records (
                    record_id, canonical_id, sequence_id, repository_id, head_sha,
                    delivery_id, run_id, state, actor, actor_role,
                    rationale, finding_json, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record_id,
                    canonical_id,
                    seq,
                    latest.repository_id,
                    latest.head_sha,
                    latest.delivery_id,
                    latest.run_id,
                    new_state.value,
                    actor,
                    actor_role,
                    rationale,
                    finding_json,
                    persisted_now,
                ),
            )
            _commit(conn)
        except Exception as exc:
            _rollback(conn)
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                raise ConcurrentTransitionError(
                    f"Concurrent state transition collision on canonical_id '{canonical_id}', sequence {seq}"
                ) from exc
            raise

        return ReviewTruthRecord(
            record_id=record_id,
            canonical_id=canonical_id,
            sequence_id=seq,
            repository_id=latest.repository_id,
            head_sha=latest.head_sha,
            delivery_id=latest.delivery_id,
            run_id=latest.run_id,
            state=new_state,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
            finding_data=finding_dict,
            timestamp=current_time,
        )

    def get_latest_state(self, canonical_id: str) -> ReviewTruthRecord | None:
        """Fetch the latest lifecycle state record for a canonical finding."""
        conn = self.connection_manager.get_connection()
        cur = _execute(
            conn,
            """
            SELECT record_id, canonical_id, sequence_id, repository_id, head_sha,
                   delivery_id, run_id, state, actor, actor_role,
                   rationale, finding_json, timestamp
            FROM finding_records
            WHERE canonical_id = %s
            ORDER BY sequence_id DESC
            LIMIT 1
            """,
            (canonical_id,),
        )
        if not cur:
            return None
        row = cur.fetchone() if hasattr(cur, "fetchone") else None
        return self._row_to_record(row) if row else None

    def get_history(self, canonical_id: str) -> list[ReviewTruthRecord]:
        """Fetch complete, ordered lifecycle history of a canonical finding."""
        conn = self.connection_manager.get_connection()
        cur = _execute(
            conn,
            """
            SELECT record_id, canonical_id, sequence_id, repository_id, head_sha,
                   delivery_id, run_id, state, actor, actor_role,
                   rationale, finding_json, timestamp
            FROM finding_records
            WHERE canonical_id = %s
            ORDER BY sequence_id ASC
            """,
            (canonical_id,),
        )
        if not cur:
            return []
        rows = cur.fetchall() if hasattr(cur, "fetchall") else []
        return [self._row_to_record(r) for r in rows]

    def list_by_state(self, state: TruthState) -> list[ReviewTruthRecord]:
        """List findings whose latest lifecycle state matches the target state."""
        canonical_ids = self.list_all_canonical()
        results: list[ReviewTruthRecord] = []
        for cid in canonical_ids:
            latest = self.get_latest_state(cid)
            if latest and latest.state == state:
                results.append(latest)
        return results

    def list_all_canonical(self) -> list[str]:
        """List all distinct canonical finding IDs in Review Truth."""
        conn = self.connection_manager.get_connection()
        cur = _execute(conn, "SELECT DISTINCT canonical_id FROM finding_records ORDER BY canonical_id ASC")
        if not cur:
            return []
        rows = cur.fetchall() if hasattr(cur, "fetchall") else []
        return [r[0] for r in rows]

    def _row_to_record(self, row: tuple) -> ReviewTruthRecord:
        f_data = json.loads(row[11]) if isinstance(row[11], str) else row[11]
        ts = row[12]
        ts_float = _parse_timestamp(ts)
        return ReviewTruthRecord(
            record_id=row[0],
            canonical_id=row[1],
            sequence_id=row[2],
            repository_id=row[3],
            head_sha=row[4],
            delivery_id=row[5],
            run_id=row[6],
            state=TruthState(row[7]),
            actor=row[8],
            actor_role=row[9],
            rationale=row[10],
            finding_data=f_data,
            timestamp=ts_float,
        )


# ============================================================================
# 3. TIGER AUDIT SPINE
# ============================================================================


class TigerAuditSpine:
    """Tiger Cloud / PostgreSQL-backed time-ordered append-only event/audit spine.

    Persists into:
    - agent_events: Time-ordered event records with correlation IDs and redaction
    - github_review_effects: Publication effects keyed idempotently
    """

    def __init__(self, connection_manager: TigerConnectionManager) -> None:
        self.connection_manager = connection_manager

    def record_event(
        self,
        event: AuditEvent,
        *,
        repository_id: str | None = None,
        pull_number: int | None = None,
        head_sha: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Persist a time-ordered event with secret redaction. Strictly append-only and non-idempotent."""
        clean_details = redact_sensitive_data(dict(event.details))
        payload_json = json.dumps(clean_details, sort_keys=True)
        persisted_ts = datetime.fromtimestamp(event.timestamp, tz=timezone.utc)

        conn = self.connection_manager.get_connection()
        try:
            cur = _execute(
                conn,
                """
                INSERT INTO agent_events (
                    correlation_id, event_name, step, repository_id,
                    pull_number, head_sha, run_id, payload, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING event_id
                """,
                (
                    event.correlation_id,
                    event.event_name,
                    event.step,
                    repository_id,
                    pull_number,
                    head_sha,
                    run_id,
                    payload_json,
                    persisted_ts,
                ),
            )
            _commit(conn)
            row = cur.fetchone() if cur and hasattr(cur, "fetchone") else None
            return int(row[0]) if row else 1
        except Exception:
            _rollback(conn)
            raise

    def record_events(
        self,
        events: Sequence[AuditEvent],
        *,
        repository_id: str | None = None,
        pull_number: int | None = None,
        head_sha: str | None = None,
        run_id: str | None = None,
    ) -> list[int]:
        """Batch persist a sequence of audit events."""
        return [
            self.record_event(
                e,
                repository_id=repository_id,
                pull_number=pull_number,
                head_sha=head_sha,
                run_id=run_id,
            )
            for e in events
        ]

    def get_events(self, correlation_id: str) -> list[AuditRecord]:
        """Query time-ordered audit events for a given correlation ID."""
        conn = self.connection_manager.get_connection()
        cur = _execute(
            conn,
            """
            SELECT event_id, correlation_id, event_name, step,
                   repository_id, pull_number, head_sha, run_id,
                   timestamp, payload
            FROM agent_events
            WHERE correlation_id = %s
            ORDER BY timestamp ASC, event_id ASC
            """,
            (correlation_id,),
        )
        if not cur:
            return []
        rows = cur.fetchall() if hasattr(cur, "fetchall") else []
        records: list[AuditRecord] = []
        for r in rows:
            ts = r[8]
            ts_float = _parse_timestamp(ts)
            payload = json.loads(r[9]) if isinstance(r[9], str) else r[9]
            records.append(
                AuditRecord(
                    event_id=r[0],
                    correlation_id=r[1],
                    event_name=r[2],
                    step=r[3],
                    repository_id=r[4],
                    pull_number=r[5],
                    head_sha=r[6],
                    run_id=r[7],
                    timestamp=ts_float,
                    details=payload or {},
                )
            )
        return records

    def record_github_effect(
        self,
        idempotency_key: str,
        repository_id: str,
        pull_number: int,
        head_sha: str,
        canonical_id: str,
        status: str,
        *,
        review_id: str | None = None,
        comment_id: str | None = None,
        html_url: str | None = None,
        published_inline: bool = False,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist a GitHub publication side-effect record idempotently."""
        conn = self.connection_manager.get_connection()
        payload_json = json.dumps(payload or {}, sort_keys=True)
        try:
            _execute(
                conn,
                """
                INSERT INTO github_review_effects (
                    idempotency_key, repository_id, pull_number, head_sha,
                    canonical_id, status, review_id, comment_id, html_url,
                    published_inline, reason, payload, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (
                    idempotency_key,
                    repository_id,
                    pull_number,
                    head_sha,
                    canonical_id,
                    status,
                    review_id,
                    comment_id,
                    html_url,
                    published_inline,
                    reason,
                    payload_json,
                ),
            )
            _commit(conn)
        except Exception:
            _rollback(conn)
            raise

    def reconstruct_run(self, correlation_id: str) -> RunProvenanceTrace:
        """Reconstruct provenance trace across Tiger Cloud tables.

        In W1-06, raw webhook deliveries and queue leases reside in Redis/Ingress and are
        not stored in Tiger Cloud. Trace is explicitly marked partial_tiger_only.
        Fails closed with RunNotFoundError if no trace exists in Tiger.
        """
        timeline = self.get_events(correlation_id)
        conn = self.connection_manager.get_connection()

        # Check if review run exists
        cur_rev = _execute(
            conn,
            "SELECT run_id, repository_id, pull_number, head_sha, base_sha, delivery_id "
            "FROM pr_review_records WHERE run_id = %s OR delivery_id = %s",
            (correlation_id, correlation_id),
        )
        rev_row = cur_rev.fetchone() if cur_rev and hasattr(cur_rev, "fetchone") else None

        if not timeline and not rev_row:
            raise RunNotFoundError(f"No run provenance trace found in Tiger Cloud for correlation_id '{correlation_id}'")

        trace = RunProvenanceTrace(correlation_id=correlation_id, timeline=timeline)

        # Populate header fields from review record if present
        if rev_row:
            trace.run_id = rev_row[0]
            trace.repository_id = rev_row[1]
            trace.pull_number = rev_row[2]
            trace.head_sha = rev_row[3]
            trace.delivery_id = rev_row[5]

        # Extract timeline metadata
        for rec in timeline:
            if rec.run_id and not trace.run_id:
                trace.run_id = rec.run_id
            if rec.repository_id and not trace.repository_id:
                trace.repository_id = rec.repository_id
            if rec.pull_number and not trace.pull_number:
                trace.pull_number = rec.pull_number
            if rec.head_sha and not trace.head_sha:
                trace.head_sha = rec.head_sha
            if isinstance(rec.details, dict):
                if not trace.delivery_id and rec.details.get("delivery_id"):
                    trace.delivery_id = str(rec.details["delivery_id"])
                if not trace.coverage_summary and rec.details.get("coverage_summary"):
                    trace.coverage_summary = rec.details["coverage_summary"]
            if "specialist" in rec.event_name or "specialist" in rec.step:
                trace.specialist_steps.append({
                    "step": rec.step,
                    "event_name": rec.event_name,
                    "timestamp": rec.timestamp,
                    "details": rec.details,
                })

        # Query revision freshness
        if trace.repository_id and trace.head_sha:
            cur_fresh = _execute(
                conn,
                "SELECT is_fresh FROM repository_revisions WHERE repository_id = %s AND revision = %s",
                (trace.repository_id, trace.head_sha),
            )
            fresh_row = cur_fresh.fetchone() if cur_fresh and hasattr(cur_fresh, "fetchone") else None
            if fresh_row is not None:
                trace.retrieval_freshness = "fresh" if fresh_row[0] else "stale"

        # Query Review Truth findings
        run_canonical_ids: set[str] = set()
        search_keys = [c for c in (trace.run_id, trace.delivery_id, correlation_id) if c]
        if search_keys:
            placeholders = ",".join(["%s"] * len(search_keys))
            cur_findings = _execute(
                conn,
                f"""
                SELECT canonical_id, sequence_id, state, actor, actor_role,
                       rationale, finding_json, timestamp
                FROM finding_records
                WHERE run_id IN ({placeholders}) OR delivery_id IN ({placeholders})
                ORDER BY sequence_id ASC
                """,
                tuple(search_keys + search_keys),
            )
            rows = cur_findings.fetchall() if cur_findings and hasattr(cur_findings, "fetchall") else []
            seen: set[str] = set()
            for r in rows:
                cid, seq, state, actor, actor_role, rationale, f_json, t_stamp = r
                run_canonical_ids.add(cid)
                if cid not in seen:
                    seen.add(cid)
                    f_data = json.loads(f_json) if isinstance(f_json, str) else f_json
                    trace.findings.append({
                        "canonical_id": cid,
                        "category": f_data.get("category", ""),
                        "severity": f_data.get("severity", ""),
                        "confidence": f_data.get("confidence", 0.0),
                        "summary": f_data.get("summary", ""),
                        "rationale": f_data.get("rationale", ""),
                        "file_path": f_data.get("file_path", ""),
                        "line_range": tuple(f_data.get("line_range", (0, 0))),
                        "evidence_refs": f_data.get("evidence_refs", []),
                        "remediation": f_data.get("remediation"),
                    })
                ts_val = _parse_timestamp(t_stamp)
                trace.policy_transitions.append({
                    "canonical_id": cid,
                    "sequence_id": seq,
                    "state": state,
                    "actor": actor,
                    "actor_role": actor_role,
                    "rationale": rationale,
                    "timestamp": ts_val,
                })

        # Query GitHub effects
        for rec in timeline:
            if isinstance(rec.details, dict):
                if rec.details.get("canonical_id"):
                    run_canonical_ids.add(str(rec.details["canonical_id"]))

        if run_canonical_ids:
            effects_placeholders = ",".join(["%s"] * len(run_canonical_ids))
            cur_effects = _execute(
                conn,
                f"""
                SELECT idempotency_key, canonical_id, status, review_id, comment_id,
                       html_url, published_inline, reason, created_at
                FROM github_review_effects
                WHERE canonical_id IN ({effects_placeholders})
                ORDER BY created_at ASC
                """,
                tuple(run_canonical_ids),
            )
            eff_rows = cur_effects.fetchall() if cur_effects and hasattr(cur_effects, "fetchall") else []
            for e in eff_rows:
                created_ts = _parse_timestamp(e[8])
                trace.github_effects.append({
                    "idempotency_key": e[0],
                    "canonical_id": e[1],
                    "status": e[2],
                    "review_id": e[3],
                    "comment_id": e[4],
                    "html_url": e[5],
                    "published_inline": bool(e[6]),
                    "reason": e[7],
                    "created_at": created_ts,
                    "timestamp": created_ts,
                })

        # Set explicit completeness semantics: W1-06 Tiger reconstruction is partial_tiger_only
        trace.completeness = "partial_tiger_only"
        trace.completeness_reasons = (
            "Webhook deliveries and queue job states reside in Redis/Ingress and are not persisted in Tiger schema in W1-06",
        )
        trace.delivery_status = None
        trace.queue_job_status = None

        # Verify secret redaction
        trace.contains_secrets = _has_unredacted_secrets(asdict(trace))
        return trace
