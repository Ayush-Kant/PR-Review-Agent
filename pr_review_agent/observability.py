"""Event/audit spine, operational telemetry, and repository-user-facing dashboard.

Implements:
- FR-15: Event/audit spine with time-ordered, correlation-ID-bearing events and secret redaction.
- FR-17: Observability and operations exposing repository-user-facing dashboard and machine-readable telemetry.
- AC-11: Traceability of finding provenance from GitHub output back to webhook delivery without secrets.
- AC-15: Repository-user-facing dashboard covering review status, findings, health, queue age, duration, etc.
- NFR-06: Auditability allowing reconstruction of any review decision.
- NFR-08: Structured, correlated, redacted telemetry and operational alerts.
- NFR-10: Usability making held findings and failures understandable without inspecting raw logs.
- NFR-11: WCAG 2.2 AA accessibility for dashboard UI.
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from pr_review_agent.github_output import PublicationStatus
from pr_review_agent.orchestration import AuditEvent
from pr_review_agent.policy import TruthState


# Redaction patterns for secrets, tokens, and credentials (FR-15, FR-18, AC-11)
SENSITIVE_KEY_PATTERNS = re.compile(
    r"(secret|token|password|credential|api[_-]?key|private[_-]?key|authorization)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_PATTERNS = re.compile(
    r"(ghp_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{40,}|Bearer\s+[A-Za-z0-9._\-]{20,})",
    re.IGNORECASE,
)


def redact_sensitive_data(obj: Any) -> Any:
    """Recursively redact secrets, credentials, and tokens from structured audit metadata."""
    if isinstance(obj, Mapping):
        redacted_dict: dict[str, Any] = {}
        for k, v in obj.items():
            if SENSITIVE_KEY_PATTERNS.search(str(k)):
                redacted_dict[k] = "[REDACTED]"
            else:
                redacted_dict[k] = redact_sensitive_data(v)
        return redacted_dict
    elif isinstance(obj, (list, tuple)):
        return [redact_sensitive_data(item) for item in obj]
    elif isinstance(obj, str):
        if SENSITIVE_VALUE_PATTERNS.search(obj) and len(obj) > 20:
            return SENSITIVE_VALUE_PATTERNS.sub("[REDACTED]", obj)
        return obj
    return obj


@dataclass
class AuditRecord:
    """Durable record stored in the audit spine."""

    event_id: int
    correlation_id: str
    event_name: str
    step: str
    repository_id: str | None
    pull_number: int | None
    head_sha: str | None
    run_id: str | None
    timestamp: float
    details: dict[str, Any]


@dataclass
class RunProvenanceTrace:
    """Complete provenance trace of a review run for AC-11 and NFR-06 auditability."""

    correlation_id: str
    run_id: str | None = None
    delivery_id: str | None = None
    repository_id: str | None = None
    pull_number: int | None = None
    head_sha: str | None = None
    delivery_status: str | None = None
    queue_job_status: str | None = None
    attempt_count: int = 0
    retrieval_freshness: str | None = None
    timeline: list[AuditRecord] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    policy_transitions: list[dict[str, Any]] = field(default_factory=list)
    github_effects: list[dict[str, Any]] = field(default_factory=list)
    specialist_steps: list[dict[str, Any]] = field(default_factory=list)
    contains_secrets: bool = False
    coverage_summary: dict[str, Any] | None = None


class AuditSpine:
    """Durable, queryable, append-only event/audit spine in SQLite (FR-15, NFR-06)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    step TEXT NOT NULL,
                    repository_id TEXT,
                    pull_number INTEGER,
                    head_sha TEXT,
                    run_id TEXT,
                    timestamp REAL NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_correlation
                ON audit_events (correlation_id, timestamp)
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_repo_pr
                ON audit_events (repository_id, pull_number)
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_event_name
                ON audit_events (event_name)
                """
            )
            # Enforce append-only invariants via native SQLite triggers
            self.connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_audit_events_prevent_update
                BEFORE UPDATE ON audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'audit_events is append-only: UPDATE operations are prohibited');
                END;
                """
            )
            self.connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_audit_events_prevent_delete
                BEFORE DELETE ON audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'audit_events is append-only: DELETE operations are prohibited');
                END;
                """
            )

    def record_event(
        self,
        event: AuditEvent,
        *,
        repository_id: str | None = None,
        pull_number: int | None = None,
        head_sha: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Persist a time-ordered correlation-bearing event with redaction."""
        clean_details = redact_sensitive_data(dict(event.details))
        details_json = json.dumps(clean_details, sort_keys=True)

        with self._lock:
            with self.connection:
                cursor = self.connection.execute(
                    """
                    INSERT INTO audit_events (
                        correlation_id, event_name, step, repository_id,
                        pull_number, head_sha, run_id, timestamp, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.correlation_id,
                        event.event_name,
                        event.step,
                        repository_id,
                        pull_number,
                        head_sha,
                        run_id,
                        event.timestamp,
                        details_json,
                    ),
                )
                return int(cursor.lastrowid)

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
        event_ids: list[int] = []
        for e in events:
            eid = self.record_event(
                e,
                repository_id=repository_id,
                pull_number=pull_number,
                head_sha=head_sha,
                run_id=run_id,
            )
            event_ids.append(eid)
        return event_ids

    def get_events(self, correlation_id: str) -> list[AuditRecord]:
        """Query time-ordered audit events for a given correlation ID."""
        with self._lock:
            cursor = self.connection.execute(
                """
                SELECT event_id, correlation_id, event_name, step,
                       repository_id, pull_number, head_sha, run_id,
                       timestamp, details_json
                FROM audit_events
                WHERE correlation_id = ?
                ORDER BY timestamp ASC, event_id ASC
                """,
                (correlation_id,),
            )
            records: list[AuditRecord] = []
            for row in cursor.fetchall():
                records.append(
                    AuditRecord(
                        event_id=row[0],
                        correlation_id=row[1],
                        event_name=row[2],
                        step=row[3],
                        repository_id=row[4],
                        pull_number=row[5],
                        head_sha=row[6],
                        run_id=row[7],
                        timestamp=row[8],
                        details=json.loads(row[9]),
                    )
                )
            return records

    def reconstruct_run(self, correlation_id: str) -> RunProvenanceTrace:
        """Reconstruct full provenance trace across webhook, queue, truth, and GitHub effects (AC-11, NFR-06).

        Enforces strict correlation-safe lineage. Does NOT fall back to repository, PR, or SHA
        lookups, which could incorrectly associate records from other runs on the same branch or PR.
        Where an external store (like github_review_effects) lacks a direct correlation_id, lineage
        is bound exclusively via the run's verified canonical findings.
        """
        timeline = self.get_events(correlation_id)
        trace = RunProvenanceTrace(correlation_id=correlation_id, timeline=timeline)

        extracted_job_id: str | None = None
        # Extract metadata from timeline if available
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
                if not trace.run_id and rec.details.get("run_id"):
                    trace.run_id = str(rec.details["run_id"])
                if not extracted_job_id and rec.details.get("job_id"):
                    extracted_job_id = str(rec.details["job_id"])
                if not trace.coverage_summary and rec.details.get("coverage_summary"):
                    trace.coverage_summary = rec.details["coverage_summary"]
                elif not trace.coverage_summary and rec.details.get("coverage"):
                    trace.coverage_summary = rec.details["coverage"]
            if "specialist" in rec.event_name or "specialist" in rec.step:
                trace.specialist_steps.append({
                    "step": rec.step,
                    "event_name": rec.event_name,
                    "timestamp": rec.timestamp,
                    "details": rec.details,
                })

        # Also support composite correlation IDs of the form "delivery_id:run_id"
        if ":" in correlation_id:
            parts = correlation_id.split(":", 1)
            if not trace.delivery_id:
                trace.delivery_id = parts[0]
            if not trace.run_id:
                trace.run_id = parts[1]

        target_delivery = trace.delivery_id or correlation_id

        # 1. Query webhook delivery strictly by delivery_id (no fallback to repository_id or latest)
        try:
            d_row = self.connection.execute(
                """
                SELECT d.delivery_id, d.state, s.snapshot_json
                FROM deliveries d
                LEFT JOIN review_snapshots s ON d.delivery_id = s.delivery_id
                WHERE d.delivery_id = ?
                """,
                (target_delivery,),
            ).fetchone()
            if d_row:
                trace.delivery_id = d_row[0]
                trace.delivery_status = d_row[1]
                if d_row[2]:
                    s_data = json.loads(d_row[2])
                    if not trace.repository_id:
                        trace.repository_id = s_data.get("repository_full_name") or s_data.get("repository_id")
                    if not trace.pull_number:
                        trace.pull_number = s_data.get("pull_request_number")
                    if not trace.head_sha:
                        trace.head_sha = s_data.get("head_sha")
            else:
                delivery_row = self.connection.execute(
                    """
                    SELECT delivery_id, status, repository_id, pull_number, head_sha
                    FROM webhook_deliveries
                    WHERE delivery_id = ?
                    """,
                    (target_delivery,),
                ).fetchone()
                if delivery_row:
                    trace.delivery_id = delivery_row[0]
                    trace.delivery_status = delivery_row[1]
                    if not trace.repository_id:
                        trace.repository_id = delivery_row[2]
                    if not trace.pull_number:
                        trace.pull_number = delivery_row[3]
                    if not trace.head_sha:
                        trace.head_sha = delivery_row[4]
        except sqlite3.OperationalError:
            pass

        # 2. Query queue job strictly by exact job_id or delivery_id (no repository fallback)
        try:
            cols = [col[1] for col in self.connection.execute("PRAGMA table_info(review_jobs)").fetchall()]
            state_col = "state" if "state" in cols else "status"
            pr_col = "pull_request_number" if "pull_request_number" in cols else "pull_number"
            job_candidates = [c for c in (extracted_job_id, trace.delivery_id, correlation_id) if c]
            if job_candidates:
                placeholders = ",".join("?" * len(job_candidates))
                job_row = self.connection.execute(
                    f"""
                    SELECT job_id, {state_col}, attempt_count, repository_id, {pr_col}, head_sha
                    FROM review_jobs
                    WHERE job_id IN ({placeholders}) OR delivery_id IN ({placeholders})
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    job_candidates + job_candidates,
                ).fetchone()
                if job_row:
                    trace.queue_job_status = job_row[1]
                    trace.attempt_count = job_row[2]
                    if not trace.repository_id:
                        trace.repository_id = job_row[3]
                    if not trace.pull_number and job_row[4] is not None:
                        trace.pull_number = job_row[4]
                    if not trace.head_sha:
                        trace.head_sha = job_row[5]
        except sqlite3.OperationalError:
            pass

        # 3. Query retrieval index freshness for exact repository_id and head_sha
        try:
            if trace.repository_id and trace.head_sha:
                r_row = self.connection.execute(
                    """
                    SELECT is_fresh
                    FROM repository_revisions
                    WHERE repository_id = ? AND revision = ?
                    ORDER BY indexed_at DESC LIMIT 1
                    """,
                    (trace.repository_id, trace.head_sha),
                ).fetchone()
                if r_row is not None:
                    trace.retrieval_freshness = "fresh" if r_row[0] == 1 else "stale"
                else:
                    fresh_row = self.connection.execute(
                        """
                        SELECT freshness_state
                        FROM repository_index_metadata
                        WHERE repository_id = ? AND revision = ?
                        ORDER BY indexed_at DESC LIMIT 1
                        """,
                        (trace.repository_id, trace.head_sha),
                    ).fetchone()
                    if fresh_row:
                        trace.retrieval_freshness = fresh_row[0]
        except sqlite3.OperationalError:
            pass

        # 4. Query Review Truth findings strictly by run_id or delivery_id (never repository_id + head_sha)
        run_canonical_ids: set[str] = set()
        try:
            truth_candidates = [c for c in (trace.run_id, trace.delivery_id, correlation_id) if c]
            if ":" in correlation_id:
                for part in correlation_id.split(":"):
                    if part and part not in truth_candidates:
                        truth_candidates.append(part)

            if truth_candidates:
                placeholders = ",".join("?" * len(truth_candidates))
                rows = self.connection.execute(
                    f"""
                    SELECT canonical_id, sequence_id, state, actor, actor_role,
                           rationale, finding_json, timestamp
                    FROM review_truth
                    WHERE run_id IN ({placeholders}) OR delivery_id IN ({placeholders})
                    ORDER BY sequence_id ASC
                    """,
                    truth_candidates + truth_candidates,
                ).fetchall()
                seen_findings: set[str] = set()
                for row in rows:
                    cid, seq, state, actor, actor_role, rationale, f_json, t_stamp = row
                    run_canonical_ids.add(cid)
                    if cid not in seen_findings:
                        seen_findings.add(cid)
                        f_data = json.loads(f_json) if f_json else {}
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
                    trace.policy_transitions.append({
                        "canonical_id": cid,
                        "sequence_id": seq,
                        "state": state,
                        "actor": actor,
                        "actor_role": actor_role,
                        "rationale": rationale,
                        "timestamp": t_stamp,
                    })
        except sqlite3.OperationalError:
            pass

        # 5. Query GitHub effects table strictly by canonical findings tied to this run.
        # Limitation Note: github_review_effects does not persist correlation_id directly.
        # Lineage is preserved by joining through the run's verified canonical finding IDs.
        # If no canonical findings are associated with the run, GitHub effects are omitted
        # rather than guessing by repository_id + pull_number.
        try:
            for rec in timeline:
                if isinstance(rec.details, dict):
                    if rec.details.get("canonical_id"):
                        run_canonical_ids.add(str(rec.details["canonical_id"]))
                    if rec.details.get("finding_id"):
                        run_canonical_ids.add(str(rec.details["finding_id"]))

            if run_canonical_ids:
                placeholders = ",".join("?" * len(run_canonical_ids))
                effect_rows = self.connection.execute(
                    f"""
                    SELECT idempotency_key, canonical_id, status, review_id, comment_id,
                           html_url, published_inline, reason, created_at
                    FROM github_review_effects
                    WHERE canonical_id IN ({placeholders})
                    ORDER BY created_at ASC
                    """,
                    tuple(run_canonical_ids),
                ).fetchall()
                for e in effect_rows:
                    trace.github_effects.append({
                        "idempotency_key": e[0],
                        "canonical_id": e[1],
                        "status": e[2],
                        "review_id": e[3],
                        "comment_id": e[4],
                        "html_url": e[5],
                        "published_inline": bool(e[6]),
                        "reason": e[7],
                        "created_at": e[8],
                        "timestamp": e[8],
                    })
        except sqlite3.OperationalError:
            pass

        # Ensure no secrets leak in reconstructed trace
        def _has_unredacted_secrets(obj: Any) -> bool:
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

        trace.contains_secrets = _has_unredacted_secrets(asdict(trace))
        return trace


@dataclass
class OperationalAlert:
    """Operational alert raised by telemetry evaluation (NFR-08)."""

    alert_id: str
    severity: str  # "critical", "warning", "info"
    category: str  # "queue", "error_rate", "staleness", "publication", "budget"
    message: str
    threshold: str
    current_value: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TelemetrySnapshot:
    """Machine-readable operational telemetry snapshot (FR-17, AC-15, NFR-08)."""

    timestamp: float
    repository_id: str | None
    queue_depth: int
    leased_jobs: int
    dead_letter_count: int
    oldest_queued_age_seconds: float
    total_reviews_completed: int
    average_duration_seconds: float
    total_retries: int
    failure_rate: float
    retrieval_freshness: str
    finding_dispositions: dict[str, int]
    publication_outcomes: dict[str, int]
    active_alerts: list[OperationalAlert] = field(default_factory=list)
    total_cost_usd: float = 0.0


class OperationalTelemetry:
    """Evaluates and serves operational health, queue latency, and telemetry signals."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        cost_ledger: Any | None = None,
        budget_config: Any | None = None,
    ) -> None:
        self.connection = connection
        self.cost_ledger = cost_ledger
        self.budget_config = budget_config

    def _resolve_correlation_repository(
        self,
        corr_id: str,
        ev_list: list[tuple[str, str, float, str, str | None]],
    ) -> str | None:
        """Safely establish repository ownership for an audit correlation using exact durable lineage.

        Collects the complete set of independently available evidence. If evidence conflicts across
        multiple distinct repositories, fails closed by returning None.
        """
        candidate_repos: set[str] = set()

        # 1. Direct repository_id on audit event records
        for _, _, _, _, r_id in ev_list:
            if r_id:
                candidate_repos.add(str(r_id))

        # 2. repository_id in event details_json
        for _, _, _, d_json, _ in ev_list:
            if d_json:
                try:
                    d = json.loads(d_json)
                    if isinstance(d, dict):
                        repo = d.get("repository_id") or d.get("repository") or d.get("repo")
                        if repo:
                            candidate_repos.add(str(repo))
                except Exception:
                    pass

        # Candidate identifiers from correlation_id
        candidate_keys = [corr_id]
        if ":" in corr_id:
            for part in corr_id.split(":"):
                if part and part not in candidate_keys:
                    candidate_keys.append(part)

        # 3. Exact durable lineage via review_jobs
        try:
            placeholders = ",".join("?" * len(candidate_keys))
            cols = [col[1] for col in self.connection.execute("PRAGMA table_info(review_jobs)").fetchall()]
            if "repository_id" in cols:
                job_rows = self.connection.execute(
                    f"""
                    SELECT DISTINCT repository_id FROM review_jobs
                    WHERE job_id IN ({placeholders}) OR delivery_id IN ({placeholders})
                    """,
                    candidate_keys + candidate_keys,
                ).fetchall()
                for j_row in job_rows:
                    if j_row[0]:
                        candidate_repos.add(str(j_row[0]))
        except sqlite3.OperationalError:
            pass

        # 4. Exact durable lineage via webhook_deliveries
        try:
            placeholders = ",".join("?" * len(candidate_keys))
            cols = [col[1] for col in self.connection.execute("PRAGMA table_info(webhook_deliveries)").fetchall()]
            if "repository_id" in cols:
                w_rows = self.connection.execute(
                    f"""
                    SELECT DISTINCT repository_id FROM webhook_deliveries
                    WHERE delivery_id IN ({placeholders})
                    """,
                    candidate_keys,
                ).fetchall()
                for w_row in w_rows:
                    if w_row[0]:
                        candidate_repos.add(str(w_row[0]))
        except sqlite3.OperationalError:
            pass

        # 5. Exact durable lineage via deliveries / review_snapshots
        try:
            placeholders = ",".join("?" * len(candidate_keys))
            s_rows = self.connection.execute(
                f"""
                SELECT s.snapshot_json FROM deliveries d
                JOIN review_snapshots s ON d.delivery_id = s.delivery_id
                WHERE d.delivery_id IN ({placeholders})
                """,
                candidate_keys,
            ).fetchall()
            for s_row in s_rows:
                if s_row[0]:
                    s_data = json.loads(s_row[0])
                    repo = s_data.get("repository_full_name") or s_data.get("repository_id")
                    if repo:
                        candidate_repos.add(str(repo))
        except sqlite3.OperationalError:
            pass

        # 6. Exact durable lineage via review_truth
        try:
            placeholders = ",".join("?" * len(candidate_keys))
            cols = [col[1] for col in self.connection.execute("PRAGMA table_info(review_truth)").fetchall()]
            if "repository_id" in cols:
                t_rows = self.connection.execute(
                    f"""
                    SELECT DISTINCT repository_id FROM review_truth
                    WHERE run_id IN ({placeholders}) OR delivery_id IN ({placeholders})
                    """,
                    candidate_keys + candidate_keys,
                ).fetchall()
                for t_row in t_rows:
                    if t_row[0]:
                        candidate_repos.add(str(t_row[0]))
        except sqlite3.OperationalError:
            pass

        # If exactly one repository is established consistently, return it.
        # If zero (unresolved) or multiple (conflicting evidence), fail closed.
        if len(candidate_repos) == 1:
            return next(iter(candidate_repos))
        return None

    def get_snapshot(
        self,
        repository_id: str | None = None,
        now: float | None = None,
    ) -> TelemetrySnapshot:
        """Compute an un-fabricated telemetry snapshot directly from durable stores."""
        current_time = time.time() if now is None else now

        # 1. Queue telemetry
        queue_depth = 0
        leased_jobs = 0
        dead_letter_count = 0
        oldest_queued_age = 0.0
        total_retries = 0
        failed_jobs = 0
        total_jobs = 0

        try:
            cols = [col[1] for col in self.connection.execute("PRAGMA table_info(review_jobs)").fetchall()]
            state_col = "state" if "state" in cols else "status"
            query = f"SELECT {state_col}, created_at, attempt_count FROM review_jobs"
            params: list[Any] = []
            if repository_id:
                query += " WHERE repository_id = ?"
                params.append(repository_id)

            rows = self.connection.execute(query, params).fetchall()
            total_jobs = len(rows)
            for status, created_at, attempts in rows:
                total_retries += max(0, (attempts or 1) - 1)
                s_lower = str(status).lower()
                if s_lower == "queued":
                    queue_depth += 1
                    age = max(0.0, current_time - float(created_at))
                    if age > oldest_queued_age:
                        oldest_queued_age = age
                elif s_lower in ("leased", "running"):
                    leased_jobs += 1
                elif s_lower == "dead_letter":
                    dead_letter_count += 1
                    failed_jobs += 1
                elif s_lower == "failed":
                    failed_jobs += 1
        except sqlite3.OperationalError:
            pass

        failure_rate = (failed_jobs / total_jobs) if total_jobs > 0 else 0.0

        # 2. Orchestration duration and completed reviews (Defect 2 fix: explicit completion semantics only)
        total_completed = 0
        avg_duration = 0.0
        durations: list[float] = []
        completed_correlations: set[str] = set()

        # Check explicit completed jobs in review_jobs lifecycle
        try:
            cols = [col[1] for col in self.connection.execute("PRAGMA table_info(review_jobs)").fetchall()]
            state_col = "state" if "state" in cols else "status"
            c_query = f"SELECT job_id, delivery_id, created_at, updated_at FROM review_jobs WHERE LOWER({state_col}) = 'completed'"
            c_params: list[Any] = []
            if repository_id:
                c_query += " AND repository_id = ?"
                c_params.append(repository_id)
            for j_id, del_id, c_at, u_at in self.connection.execute(c_query, c_params).fetchall():
                if j_id:
                    completed_correlations.add(str(j_id))
                if del_id:
                    completed_correlations.add(str(del_id))
                dur = max(0.0, float(u_at) - float(c_at))
                durations.append(dur)
        except sqlite3.OperationalError:
            pass

        # Check explicit completion events in audit_events lifecycle
        try:
            terminal_event_names = frozenset({
                "orchestration.completed",
                "review.completed",
                "run.completed",
                "run.finish",
                "review.finished",
                "orchestration.finish",
            })
            a_query = "SELECT correlation_id, event_name, step, timestamp, details_json, repository_id FROM audit_events"
            a_rows = self.connection.execute(a_query).fetchall()

            events_by_corr: dict[str, list[tuple[str, str, float, str, str | None]]] = {}
            for corr_id, ev_name, step, t_stamp, d_json, r_id in a_rows:
                events_by_corr.setdefault(corr_id, []).append((ev_name, step, t_stamp, d_json, r_id))

            for corr_id, ev_list in events_by_corr.items():
                if corr_id in completed_correlations:
                    continue
                if ":" in corr_id:
                    parts = corr_id.split(":")
                    if any(p in completed_correlations for p in parts):
                        continue

                # Scope check: when repository_id is provided, correlation must provably belong to this repository
                if repository_id is not None:
                    resolved_repo = self._resolve_correlation_repository(corr_id, ev_list)
                    if resolved_repo != repository_id:
                        continue

                # Check if correlation contains an explicit terminal completion event
                terminal_event_time: float | None = None
                start_event_time: float | None = None

                for ev_name, step, t_stamp, d_json, _ in ev_list:
                    if start_event_time is None or t_stamp < start_event_time:
                        start_event_time = t_stamp

                    if ev_name in terminal_event_names:
                        if terminal_event_time is None or t_stamp > terminal_event_time:
                            terminal_event_time = t_stamp

                if terminal_event_time is not None and start_event_time is not None:
                    durations.append(max(0.0, terminal_event_time - start_event_time))
                    completed_correlations.add(corr_id)
        except sqlite3.OperationalError:
            pass

        total_completed = len(durations)
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # 3. Finding dispositions in Review Truth (scoped by repository_id)
        dispositions: dict[str, int] = {
            TruthState.APPROVED.value: 0,
            TruthState.AUTO_APPROVED.value: 0,
            TruthState.HELD.value: 0,
            TruthState.SUPERSEDED.value: 0,
            TruthState.REJECTED.value: 0,
            TruthState.DISMISSED.value: 0,
            TruthState.PUBLISHED.value: 0,
        }
        try:
            if repository_id:
                t_query = """
                    SELECT t.state, COUNT(*)
                    FROM review_truth t
                    INNER JOIN (
                        SELECT canonical_id, MAX(sequence_id) as max_seq
                        FROM review_truth
                        WHERE repository_id = ?
                        GROUP BY canonical_id
                    ) latest ON t.canonical_id = latest.canonical_id AND t.sequence_id = latest.max_seq
                    WHERE t.repository_id = ?
                    GROUP BY t.state
                """
                t_params: tuple[Any, ...] = (repository_id, repository_id)
            else:
                t_query = """
                    SELECT t.state, COUNT(*)
                    FROM review_truth t
                    INNER JOIN (
                        SELECT canonical_id, MAX(sequence_id) as max_seq
                        FROM review_truth
                        GROUP BY canonical_id
                    ) latest ON t.canonical_id = latest.canonical_id AND t.sequence_id = latest.max_seq
                    GROUP BY t.state
                """
                t_params = ()

            t_rows = self.connection.execute(t_query, t_params).fetchall()
            for state, count in t_rows:
                if state in dispositions:
                    dispositions[state] = count
        except sqlite3.OperationalError:
            pass

        # 4. GitHub publication outcomes (scoped by repository_id)
        pub_outcomes: dict[str, int] = {
            PublicationStatus.PUBLISHED.value: 0,
            PublicationStatus.ALREADY_PUBLISHED.value: 0,
            PublicationStatus.SUPERSEDED_SHA_MISMATCH.value: 0,
            PublicationStatus.HELD_OR_UNAUTHORIZED.value: 0,
            PublicationStatus.FAILED.value: 0,
            PublicationStatus.AMBIGUOUS.value: 0,
        }
        try:
            if repository_id:
                e_query = """
                    SELECT status, COUNT(*)
                    FROM github_review_effects
                    WHERE repository_id = ?
                    GROUP BY status
                """
                e_params: tuple[Any, ...] = (repository_id,)
            else:
                e_query = """
                    SELECT status, COUNT(*)
                    FROM github_review_effects
                    GROUP BY status
                """
                e_params = ()

            e_rows = self.connection.execute(e_query, e_params).fetchall()
            for status, count in e_rows:
                pub_outcomes[status] = count
        except sqlite3.OperationalError:
            pass

        # 5. Retrieval freshness state (scoped by repository_id)
        freshness_state = "unknown"
        try:
            if repository_id:
                r_query = """
                    SELECT is_fresh
                    FROM repository_revisions
                    WHERE repository_id = ?
                    ORDER BY indexed_at DESC LIMIT 1
                """
                r_row = self.connection.execute(r_query, (repository_id,)).fetchone()
                if r_row is not None:
                    freshness_state = "fresh" if r_row[0] == 1 else "stale"
                else:
                    f_row = self.connection.execute(
                        """
                        SELECT freshness_state
                        FROM repository_index_metadata
                        WHERE repository_id = ?
                        ORDER BY indexed_at DESC LIMIT 1
                        """,
                        (repository_id,),
                    ).fetchone()
                    if f_row:
                        freshness_state = f_row[0]
            else:
                r_row = self.connection.execute(
                    """
                    SELECT is_fresh
                    FROM repository_revisions
                    ORDER BY indexed_at DESC LIMIT 1
                    """
                ).fetchone()
                if r_row is not None:
                    freshness_state = "fresh" if r_row[0] == 1 else "stale"
                else:
                    f_row = self.connection.execute(
                        """
                        SELECT freshness_state
                        FROM repository_index_metadata
                        ORDER BY indexed_at DESC LIMIT 1
                        """
                    ).fetchone()
                    if f_row:
                        freshness_state = f_row[0]
        except sqlite3.OperationalError:
            pass

        # 6. Operational alerts evaluation (NFR-08)
        alerts: list[OperationalAlert] = []
        if oldest_queued_age > 300.0:
            alerts.append(
                OperationalAlert(
                    alert_id="ALERT-QUEUE-AGING",
                    severity="warning",
                    category="queue",
                    message="Queue age threshold exceeded; oldest pending job waiting too long",
                    threshold="300s",
                    current_value=f"{oldest_queued_age:.1f}s",
                    timestamp=current_time,
                )
            )

        if failure_rate > 0.20 and total_jobs >= 3:
            alerts.append(
                OperationalAlert(
                    alert_id="ALERT-ELEVATED-FAILURES",
                    severity="critical",
                    category="error_rate",
                    message="Elevated review failure rate detected",
                    threshold="20%",
                    current_value=f"{failure_rate * 100:.1f}%",
                    timestamp=current_time,
                )
            )

        if dead_letter_count > 0:
            alerts.append(
                OperationalAlert(
                    alert_id="ALERT-DEAD-LETTER-JOBS",
                    severity="critical",
                    category="queue",
                    message="Exhausted retries resulted in dead-letter review jobs",
                    threshold="0",
                    current_value=str(dead_letter_count),
                    timestamp=current_time,
                )
            )

        if pub_outcomes[PublicationStatus.AMBIGUOUS.value] > 0:
            alerts.append(
                OperationalAlert(
                    alert_id="ALERT-AMBIGUOUS-PUBLICATIONS",
                    severity="warning",
                    category="publication",
                    message="Ambiguous GitHub publication states recorded; requires reconciliation",
                    threshold="0",
                    current_value=str(pub_outcomes[PublicationStatus.AMBIGUOUS.value]),
                    timestamp=current_time,
                )
            )

        if freshness_state == "stale":
            alerts.append(
                OperationalAlert(
                    alert_id="ALERT-RETRIEVAL-STALENESS",
                    severity="warning",
                    category="staleness",
                    message="Repository code retrieval index is marked stale",
                    threshold="fresh",
                    current_value="stale",
                    timestamp=current_time,
                )
            )

        # Budget exhaustion & warning alerts (NFR-08, NFR-12)
        total_recorded_cost = 0.0
        if self.cost_ledger is not None:
            total_recorded_cost = (
                self.cost_ledger.get_repository_spend(repository_id)
                if repository_id
                else 0.0
            )
            if (
                self.budget_config is not None
                and self.budget_config.monthly_budget_usd is not None
            ):
                budget_cap = self.budget_config.monthly_budget_usd
                if total_recorded_cost >= budget_cap:
                    alerts.append(
                        OperationalAlert(
                            alert_id="ALERT-BUDGET-EXHAUSTION",
                            severity="critical",
                            category="budget",
                            message="Configured repository budget limit exhausted",
                            threshold=f"${budget_cap:.2f}",
                            current_value=f"${total_recorded_cost:.2f}",
                            timestamp=current_time,
                        )
                    )
                elif (
                    self.budget_config.soft_budget_ratio is not None
                    and total_recorded_cost >= (budget_cap * self.budget_config.soft_budget_ratio)
                ):
                    soft_thresh = budget_cap * self.budget_config.soft_budget_ratio
                    alerts.append(
                        OperationalAlert(
                            alert_id="ALERT-BUDGET-WARNING",
                            severity="warning",
                            category="budget",
                            message="Repository spend reached soft budget warning threshold",
                            threshold=f"${soft_thresh:.2f}",
                            current_value=f"${total_recorded_cost:.2f}",
                            timestamp=current_time,
                        )
                    )

        return TelemetrySnapshot(
            timestamp=current_time,
            repository_id=repository_id,
            queue_depth=queue_depth,
            leased_jobs=leased_jobs,
            dead_letter_count=dead_letter_count,
            oldest_queued_age_seconds=oldest_queued_age,
            total_reviews_completed=total_completed,
            average_duration_seconds=avg_duration,
            total_retries=total_retries,
            failure_rate=failure_rate,
            retrieval_freshness=freshness_state,
            finding_dispositions=dispositions,
            publication_outcomes=pub_outcomes,
            active_alerts=alerts,
            total_cost_usd=round(total_recorded_cost, 6),
        )


class RepositoryDashboard:
    """Repository-user-facing dashboard for developers and maintainers (FR-17, AC-15, NFR-10, NFR-11)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.audit_spine = AuditSpine(connection)
        self.telemetry = OperationalTelemetry(connection)

    def get_dashboard_data(self, repository_id: str | None = None) -> dict[str, Any]:
        """Aggregate machine-readable dashboard data (FR-17, AC-15)."""
        snapshot = self.telemetry.get_snapshot(repository_id)

        # Recent review jobs
        recent_reviews: list[dict[str, Any]] = []
        try:
            cols = [col[1] for col in self.connection.execute("PRAGMA table_info(review_jobs)").fetchall()]
            state_col = "state" if "state" in cols else "status"
            pr_col = "pull_request_number" if "pull_request_number" in cols else "pull_number"
            query = f"""
                SELECT job_id, repository_id, {pr_col}, head_sha, delivery_id, {state_col}, created_at
                FROM review_jobs
                ORDER BY created_at DESC LIMIT 25
            """
            params: list[Any] = []
            if repository_id:
                query = f"""
                    SELECT job_id, repository_id, {pr_col}, head_sha, delivery_id, {state_col}, created_at
                    FROM review_jobs
                    WHERE repository_id = ?
                    ORDER BY created_at DESC LIMIT 25
                """
                params = [repository_id]
            for row in self.connection.execute(query, params).fetchall():
                recent_reviews.append({
                    "job_id": row[0],
                    "repository_id": row[1],
                    "pull_number": row[2],
                    "head_sha": row[3],
                    "delivery_id": row[4],
                    "status": row[5],
                    "created_at": row[6],
                })
        except sqlite3.OperationalError:
            pass

        # Recent findings (scoped by repository_id)
        recent_findings: list[dict[str, Any]] = []
        try:
            if repository_id:
                f_query = """
                    SELECT t.canonical_id, t.repository_id, t.head_sha, t.state, t.finding_json, t.timestamp
                    FROM review_truth t
                    INNER JOIN (
                        SELECT canonical_id, MAX(sequence_id) as max_seq
                        FROM review_truth
                        WHERE repository_id = ?
                        GROUP BY canonical_id
                    ) latest ON t.canonical_id = latest.canonical_id AND t.sequence_id = latest.max_seq
                    WHERE t.repository_id = ?
                    ORDER BY t.timestamp DESC LIMIT 50
                """
                f_params: tuple[Any, ...] = (repository_id, repository_id)
            else:
                f_query = """
                    SELECT t.canonical_id, t.repository_id, t.head_sha, t.state, t.finding_json, t.timestamp
                    FROM review_truth t
                    INNER JOIN (
                        SELECT canonical_id, MAX(sequence_id) as max_seq
                        FROM review_truth
                        GROUP BY canonical_id
                    ) latest ON t.canonical_id = latest.canonical_id AND t.sequence_id = latest.max_seq
                    ORDER BY t.timestamp DESC LIMIT 50
                """
                f_params = ()

            for f in self.connection.execute(f_query, f_params).fetchall():
                cid, repo, sha, state, f_json, t_stamp = f
                f_data = json.loads(f_json) if f_json else {}
                recent_findings.append({
                    "canonical_id": cid,
                    "repository_id": repo,
                    "head_sha": sha,
                    "category": f_data.get("category", ""),
                    "severity": f_data.get("severity", ""),
                    "confidence": f_data.get("confidence", 0.0),
                    "summary": f_data.get("summary", ""),
                    "file_path": f_data.get("file_path", ""),
                    "line_range": tuple(f_data.get("line_range", (0, 0))),
                    "latest_state": state,
                    "created_at": t_stamp,
                })
        except sqlite3.OperationalError:
            pass

        return {
            "telemetry": asdict(snapshot),
            "recent_reviews": recent_reviews,
            "recent_findings": recent_findings,
            "policy_summary": {
                "tenancy": "single-tenant (V1)",
                "source_control": "GitHub Pull Requests only",
                "permissions": "read-only + create review comment",
                "capabilities": "NO merge / NO code modification",
                "hitl_rule": "High/Critical/Security findings require maintainer approval",
            },
        }

    def render_html(self, repository_id: str | None = None) -> str:
        """Render a semantic, WCAG 2.2 AA compliant HTML5 dashboard (NFR-10, NFR-11)."""
        data = self.get_dashboard_data(repository_id)
        telem = data["telemetry"]
        dispositions = telem["finding_dispositions"]
        outcomes = telem["publication_outcomes"]
        alerts = telem["active_alerts"]
        reviews = data["recent_reviews"]
        findings = data["recent_findings"]

        # Alert banner
        alert_rows = ""
        if alerts:
            for a in alerts:
                sev_color = "#e53e3e" if a["severity"] == "critical" else "#dd6b20"
                alert_rows += f"""
                <li role="alert" style="background-color: {sev_color}15; border-left: 4px solid {sev_color}; padding: 0.75rem 1rem; margin-bottom: 0.5rem; border-radius: 4px;">
                    <strong>[{html.escape(a['severity'].upper())}] {html.escape(a['category'].upper())}:</strong>
                    {html.escape(a['message'])} (Threshold: {html.escape(a['threshold'])}, Current: {html.escape(a['current_value'])})
                </li>
                """
        else:
            alert_rows = '<li style="color: #2f855a; padding: 0.5rem 0;">All operational health checks nominal.</li>'

        # Reviews table rows
        review_rows = ""
        if reviews:
            for r in reviews:
                review_rows += f"""
                <tr>
                    <td style="font-family: monospace;">{html.escape(str(r['job_id']))}</td>
                    <td>{html.escape(str(r['repository_id']))} #{r['pull_number']}</td>
                    <td style="font-family: monospace;">{html.escape(str(r['head_sha'])[:8])}</td>
                    <td><span class="badge status-{html.escape(str(r['status']).lower())}">{html.escape(str(r['status']))}</span></td>
                    <td>{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(r['created_at']))} UTC</td>
                </tr>
                """
        else:
            review_rows = '<tr><td colspan="5" style="text-align: center; color: #718096;">No review jobs recorded yet.</td></tr>'

        # Findings table rows
        finding_rows = ""
        if findings:
            for f in findings:
                finding_rows += f"""
                <tr>
                    <td style="font-family: monospace;">{html.escape(str(f['canonical_id']))}</td>
                    <td><strong>{html.escape(str(f['category']).upper())}</strong> / {html.escape(str(f['severity']).upper())}</td>
                    <td>{f['confidence']:.2f}</td>
                    <td>{html.escape(str(f['file_path']))}:{f['line_range'][0]}-{f['line_range'][1]}</td>
                    <td>{html.escape(str(f['summary']))}</td>
                    <td><span class="badge state-{html.escape(str(f['latest_state']).lower())}">{html.escape(str(f['latest_state']))}</span></td>
                </tr>
                """
        else:
            finding_rows = '<tr><td colspan="6" style="text-align: center; color: #718096;">No findings recorded yet.</td></tr>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PR Review Agent — Observability & Review Dashboard</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-color: #c9d1d9;
            --text-muted: #8b949e;
            --heading-color: #f0f6fc;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --focus-outline: #1f6feb;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 1.5rem;
            line-height: 1.5;
        }}
        a:focus, button:focus {{
            outline: 2px solid var(--focus-outline);
            outline-offset: 2px;
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }}
        h1 {{
            color: var(--heading-color);
            font-size: 1.75rem;
            margin: 0 0 0.25rem 0;
        }}
        h2 {{
            color: var(--heading-color);
            font-size: 1.25rem;
            margin-top: 0;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}
        .banner-single-tenant {{
            background-color: #1f242c;
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-blue);
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            margin-bottom: 1.5rem;
        }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
        }}
        .card-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}
        .card-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--heading-color);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
            margin-bottom: 1.5rem;
        }}
        th, td {{
            padding: 0.6rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
            background-color: #12161c;
        }}
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.45rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 12px;
            border: 1px solid transparent;
        }}
        .state-approved, .state-auto_approved, .status-completed, .state-published {{
            background-color: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
            border-color: rgba(63, 185, 80, 0.4);
        }}
        .state-held, .status-queued, .status-leased {{
            background-color: rgba(210, 153, 34, 0.15);
            color: var(--accent-yellow);
            border-color: rgba(210, 153, 34, 0.4);
        }}
        .state-rejected, .state-dismissed, .status-failed, .status-dead_letter {{
            background-color: rgba(248, 81, 73, 0.15);
            color: var(--accent-red);
            border-color: rgba(248, 81, 73, 0.4);
        }}
        .state-superseded {{
            background-color: rgba(139, 148, 158, 0.15);
            color: var(--text-muted);
            border-color: rgba(139, 148, 158, 0.4);
        }}
        footer {{
            border-top: 1px solid var(--border-color);
            padding-top: 1rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
        }}
    </style>
</head>
<body>
    <header role="banner">
        <h1>PR-Review-Agent — Operational Observability</h1>
        <p style="color: var(--text-muted); margin: 0;">Repository-User Dashboard &amp; Telemetry Spine</p>
    </header>

    <main role="main">
        <section class="banner-single-tenant" aria-label="System Constraints and Safety Policy">
            <strong>System Boundaries &amp; Safety Policy:</strong> Single-Tenant V1 &bull; GitHub Pull Requests Only &bull; Read-Only &amp; Review Comment Permissions &bull; <em>Strictly No Merge / No Code-Edit Capabilities</em>.
        </section>

        <section aria-labelledby="health-alerts-heading">
            <h2 id="health-alerts-heading">Operational Health &amp; Active Alerts</h2>
            <ul style="list-style: none; padding: 0; margin-bottom: 1.5rem;">
                {alert_rows}
            </ul>
        </section>

        <section aria-labelledby="telemetry-kpis-heading">
            <h2 id="telemetry-kpis-heading">Key Operational Metrics</h2>
            <div class="grid-cards">
                <div class="card">
                    <div class="card-label">Queue Depth / Pending</div>
                    <div class="card-value">{telem['queue_depth']}</div>
                </div>
                <div class="card">
                    <div class="card-label">Oldest Queue Age</div>
                    <div class="card-value">{telem['oldest_queued_age_seconds']:.1f}s</div>
                </div>
                <div class="card">
                    <div class="card-label">Reviews Completed</div>
                    <div class="card-value">{telem['total_reviews_completed']}</div>
                </div>
                <div class="card">
                    <div class="card-label">Failure / Retry Rate</div>
                    <div class="card-value">{telem['failure_rate'] * 100:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">Retrieval Freshness</div>
                    <div class="card-value" style="font-size: 1.25rem;">{html.escape(str(telem['retrieval_freshness']).upper())}</div>
                </div>
            </div>
        </section>

        <section aria-labelledby="dispositions-heading">
            <h2 id="dispositions-heading">Findings Lifecycle Dispositions</h2>
            <div class="grid-cards">
                <div class="card">
                    <div class="card-label">Published to PR</div>
                    <div class="card-value" style="color: var(--accent-green);">{dispositions.get('PUBLISHED', 0)}</div>
                </div>
                <div class="card">
                    <div class="card-label">Auto-Approved</div>
                    <div class="card-value">{dispositions.get('AUTO_APPROVED', 0)}</div>
                </div>
                <div class="card">
                    <div class="card-label">Held for Maintainer</div>
                    <div class="card-value" style="color: var(--accent-yellow);">{dispositions.get('HELD', 0)}</div>
                </div>
                <div class="card">
                    <div class="card-label">Superseded (Stale SHA)</div>
                    <div class="card-value">{dispositions.get('SUPERSEDED', 0)}</div>
                </div>
                <div class="card">
                    <div class="card-label">Rejected / Dismissed</div>
                    <div class="card-value">{dispositions.get('REJECTED', 0) + dispositions.get('DISMISSED', 0)}</div>
                </div>
            </div>
        </section>

        <section aria-labelledby="recent-reviews-heading">
            <h2 id="recent-reviews-heading">Recent Pull Request Review Jobs</h2>
            <table>
                <caption class="sr-only" style="display:none;">List of recent pull request review jobs</caption>
                <thead>
                    <tr>
                        <th scope="col">Job ID</th>
                        <th scope="col">Repository &amp; PR</th>
                        <th scope="col">Head SHA</th>
                        <th scope="col">Queue Status</th>
                        <th scope="col">Enqueued Time</th>
                    </tr>
                </thead>
                <tbody>
                    {review_rows}
                </tbody>
            </table>
        </section>

        <section aria-labelledby="recent-findings-heading">
            <h2 id="recent-findings-heading">Recent Canonical Findings</h2>
            <table>
                <caption class="sr-only" style="display:none;">List of recent canonical review findings</caption>
                <thead>
                    <tr>
                        <th scope="col">Canonical ID</th>
                        <th scope="col">Category / Severity</th>
                        <th scope="col">Confidence</th>
                        <th scope="col">Location</th>
                        <th scope="col">Summary</th>
                        <th scope="col">Lifecycle State</th>
                    </tr>
                </thead>
                <tbody>
                    {finding_rows}
                </tbody>
            </table>
        </section>
    </main>

    <footer role="contentinfo">
        PR-Review-Agent &bull; Production Observability &bull; WCAG 2.2 AA Compliant &bull; Read-Only Dashboard
    </footer>
</body>
</html>
"""

    def investigate_run(self, correlation_id: str) -> dict[str, Any]:
        """Correlated run investigation report for a single review execution (AC-11, FR-17)."""
        trace = self.audit_spine.reconstruct_run(correlation_id)
        return asdict(trace)
