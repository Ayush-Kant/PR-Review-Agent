"""Durable asynchronous dispatch and LangGraph review lifecycle orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import sqlite3
import time
from typing import Annotated, Any, TypedDict
import uuid

from langgraph.graph import END, START, StateGraph

from pr_review_agent.intake import ReviewSnapshot


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


class SpecialistType(str, Enum):
    SECURITY = "security"
    QUALITY = "quality"
    TESTS = "tests"
    DOCUMENTATION = "documentation"


class SpecialistStatus(str, Enum):
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SpecialistCoverageSummary:
    """Explicit review coverage summary across all evaluated specialists."""

    total_specialists: int
    succeeded_specialists: tuple[str, ...]
    degraded_specialists: tuple[str, ...]
    failed_specialists: tuple[str, ...]
    timeout_specialists: tuple[str, ...]
    skipped_specialists: tuple[str, ...]
    is_full_coverage: bool
    is_degraded: bool
    coverage_ratio: float
    failure_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def timed_out_specialists(self) -> tuple[str, ...]:
        return self.timeout_specialists


ALL_SPECIALISTS: tuple[SpecialistType, ...] = (
    SpecialistType.SECURITY,
    SpecialistType.QUALITY,
    SpecialistType.TESTS,
    SpecialistType.DOCUMENTATION,
)


@dataclass(frozen=True)
class ReviewJob:
    """A durable review job enqueued for processing."""

    job_id: str
    delivery_id: str
    repository_id: str
    pull_request_number: int
    base_sha: str
    head_sha: str
    state: JobState
    attempt_count: int = 0
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    deadline_seconds: float = 60.0
    last_error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    next_run_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CandidateFinding:
    """A candidate defect or improvement finding proposed by a specialist.

    Specialists propose findings; they do not make final publication decisions.
    """

    finding_id: str
    correlation_id: str
    specialist_type: SpecialistType
    category: str
    severity: str
    confidence: float
    summary: str
    rationale: str
    file_path: str | None = None
    line_range: tuple[int, int] | None = None
    evidence_refs: tuple[str, ...] = ()
    remediation: str | None = None


@dataclass(frozen=True)
class SpecialistInput:
    """Concern-specific input provided to an individual specialist."""

    specialist_type: SpecialistType
    correlation_id: str
    instructions: str
    changed_files: tuple[str, ...]
    diff_content: str
    retrieved_evidence: tuple[str, ...]
    head_sha: str


@dataclass(frozen=True)
class SpecialistOutput:
    """The structured output of an individual specialist execution."""

    specialist_type: SpecialistType
    correlation_id: str
    status: SpecialistStatus | str = SpecialistStatus.COMPLETED
    findings: tuple[CandidateFinding, ...] = ()
    error_message: str | None = None
    execution_duration: float = 0.0
    usage: Any | None = None


@dataclass
class AuditEvent:
    """Time-ordered correlation-bearing event in the review audit spine."""

    correlation_id: str
    event_name: str
    step: str
    timestamp: float = field(default_factory=time.time)
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass
class ReviewLifecycleState:
    """State graph execution state for a single review run."""

    run_id: str
    job_id: str
    delivery_id: str
    head_sha: str
    deadline: float
    is_cancelled: bool = False
    step_states: dict[str, str] = field(default_factory=dict)
    specialist_outputs: dict[SpecialistType, SpecialistOutput] = field(default_factory=dict)
    partial_failures: list[str] = field(default_factory=list)
    audit_trail: list[AuditEvent] = field(default_factory=list)
    terminal_status: str | None = None
    aggregation_invoked: bool = False
    is_degraded: bool = False
    degradation_details: dict[str, Any] = field(default_factory=dict)
    coverage_summary: SpecialistCoverageSummary | None = None
    run_cost_summary: Any | None = None


# Helper reducers for LangGraph TypedDict state
def _merge_dict(a: dict, b: dict) -> dict:
    res = dict(a)
    res.update(b)
    return res


def _merge_list(a: list, b: list) -> list:
    return list(a) + list(b)


class _GraphState(TypedDict, total=False):
    run_id: str
    job_id: str
    delivery_id: str
    repository_id: str
    head_sha: str
    correlation_id: str
    changed_files: tuple[str, ...]
    diff_content: str
    retrieved_evidence: tuple[str, ...]
    deadline: float
    is_cancelled: bool
    is_degraded: bool
    degradation_details: Annotated[dict[str, Any], _merge_dict]
    coverage_summary: SpecialistCoverageSummary | None
    run_cost_summary: Any
    step_states: Annotated[dict[str, str], _merge_dict]
    specialist_outputs: Annotated[dict[SpecialistType, SpecialistOutput], _merge_dict]
    partial_failures: Annotated[list[str], _merge_list]
    audit_trail: Annotated[list[AuditEvent], _merge_list]
    terminal_status: str | None
    aggregation_invoked: bool


SpecialistHandler = Callable[[SpecialistInput], SpecialistOutput | asyncio.Future[SpecialistOutput]]


class DurableJobQueue:
    """A durable SQLite-backed review job queue providing Redis/ARQ equivalent semantics."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        budget_config: Any | None = None,
    ) -> None:
        self.connection = connection
        self.budget_config = budget_config
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_jobs (
                    job_id TEXT PRIMARY KEY,
                    delivery_id TEXT NOT NULL,
                    repository_id TEXT NOT NULL,
                    pull_request_number INTEGER NOT NULL,
                    base_sha TEXT NOT NULL,
                    head_sha TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    backoff_base_seconds REAL NOT NULL DEFAULT 1.0,
                    deadline_seconds REAL NOT NULL DEFAULT 60.0,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    next_run_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_jobs_fetch ON review_jobs (state, next_run_at)"
            )

    def enqueue(
        self,
        snapshot: ReviewSnapshot,
        delivery_id: str,
        *,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        deadline_seconds: float = 60.0,
        now: float | None = None,
    ) -> ReviewJob:
        """Enqueue a review job atomically with durable identity correlated to the delivery."""
        current_time = time.time() if now is None else now
        job_id = f"job-{delivery_id}"
        payload = {
            "repository_id": snapshot.repository_id,
            "repository_full_name": snapshot.repository_full_name,
            "pull_request_number": snapshot.pull_request_number,
            "base_sha": snapshot.base_sha,
            "head_sha": snapshot.head_sha,
            "changed_files": list(snapshot.changed_files),
            "policy_version": snapshot.policy_version,
            "prompt_version": snapshot.prompt_version,
            "retrieval_index_version": snapshot.retrieval_index_version,
            "model_configuration": dict(snapshot.model_configuration),
        }
        payload_json = json.dumps(payload, sort_keys=True)

        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO review_jobs (
                    job_id, delivery_id, repository_id, pull_request_number,
                    base_sha, head_sha, state, attempt_count, max_retries,
                    backoff_base_seconds, deadline_seconds, last_error,
                    created_at, updated_at, next_run_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    delivery_id,
                    snapshot.repository_id,
                    snapshot.pull_request_number,
                    snapshot.base_sha,
                    snapshot.head_sha,
                    JobState.QUEUED.value,
                    0,
                    max_retries,
                    backoff_base_seconds,
                    deadline_seconds,
                    current_time,
                    current_time,
                    current_time,
                    payload_json,
                ),
            )
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Failed to persist or fetch job {job_id}")
        return job

    def lease_next_job(
        self,
        now: float | None = None,
        *,
        max_concurrency_per_repo: int | None = None,
    ) -> ReviewJob | None:
        """Atomically lease the next scheduled job using an exclusive write transaction.

        If max_concurrency_per_repo is specified (or configured via budget_config),
        candidate queued jobs are checked against currently running jobs for that repository,
        skipping at-capacity repositories to prevent starvation across repositories (NFR-05).
        """
        effective_cap = (
            max_concurrency_per_repo
            if max_concurrency_per_repo is not None
            else (
                getattr(self.budget_config, "max_concurrent_reviews_per_repo", None)
                if self.budget_config is not None
                else None
            )
        )
        current_time = time.time() if now is None else now
        try:
            self.connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            pass  # Already in transaction
        try:
            if effective_cap is None:
                row = self.connection.execute(
                    """
                    SELECT job_id FROM review_jobs
                    WHERE state = ? AND next_run_at <= ?
                    ORDER BY next_run_at ASC, created_at ASC
                    LIMIT 1
                    """,
                    (JobState.QUEUED.value, current_time),
                ).fetchone()
                if not row:
                    self.connection.commit()
                    return None
                selected_job_id = row[0]
            else:
                candidate_rows = self.connection.execute(
                    """
                    SELECT job_id, repository_id FROM review_jobs
                    WHERE state = ? AND next_run_at <= ?
                    ORDER BY next_run_at ASC, created_at ASC
                    LIMIT 50
                    """,
                    (JobState.QUEUED.value, current_time),
                ).fetchall()
                selected_job_id = None
                for cand_job_id, cand_repo_id in candidate_rows:
                    count_row = self.connection.execute(
                        """
                        SELECT COUNT(*) FROM review_jobs
                        WHERE repository_id = ? AND state = ?
                        """,
                        (cand_repo_id, JobState.RUNNING.value),
                    ).fetchone()
                    running_count = count_row[0] if count_row else 0
                    if running_count < effective_cap:
                        selected_job_id = cand_job_id
                        break
                if selected_job_id is None:
                    self.connection.commit()
                    return None

            self.connection.execute(
                """
                UPDATE review_jobs
                SET state = ?, attempt_count = attempt_count + 1, updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.RUNNING.value, current_time, selected_job_id),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return self.get_job(selected_job_id)

    def mark_completed(self, job_id: str, now: float | None = None) -> ReviewJob:
        """Mark job successfully completed."""
        current_time = time.time() if now is None else now
        with self.connection:
            self.connection.execute(
                "UPDATE review_jobs SET state = ?, updated_at = ? WHERE job_id = ?",
                (JobState.COMPLETED.value, current_time, job_id),
            )
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        return job

    def mark_failed(self, job_id: str, error: str, now: float | None = None) -> ReviewJob:
        """Handle failure: apply exponential backoff retry or transition to dead-letter."""
        current_time = time.time() if now is None else now
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")

        if job.attempt_count >= job.max_retries:
            next_state = JobState.DEAD_LETTER
            next_run = current_time
        else:
            next_state = JobState.QUEUED
            delay = job.backoff_base_seconds * (2 ** (job.attempt_count - 1))
            next_run = current_time + delay

        with self.connection:
            self.connection.execute(
                """
                UPDATE review_jobs
                SET state = ?, last_error = ?, next_run_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (next_state.value, error, next_run, current_time, job_id),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def cancel_job(self, job_id: str, reason: str = "", now: float | None = None) -> ReviewJob:
        """Cancel a job, recording reason."""
        current_time = time.time() if now is None else now
        with self.connection:
            self.connection.execute(
                """
                UPDATE review_jobs
                SET state = ?, last_error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.CANCELLED.value, reason or "Cancelled by operator", current_time, job_id),
            )
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        return job

    def get_job(self, job_id: str) -> ReviewJob | None:
        """Retrieve a review job by its durable ID."""
        row = self.connection.execute(
            """
            SELECT job_id, delivery_id, repository_id, pull_request_number,
                   base_sha, head_sha, state, attempt_count, max_retries,
                   backoff_base_seconds, deadline_seconds, last_error,
                   created_at, updated_at, next_run_at
            FROM review_jobs WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return ReviewJob(
            job_id=row[0],
            delivery_id=row[1],
            repository_id=row[2],
            pull_request_number=row[3],
            base_sha=row[4],
            head_sha=row[5],
            state=JobState(row[6]),
            attempt_count=row[7],
            max_retries=row[8],
            backoff_base_seconds=row[9],
            deadline_seconds=row[10],
            last_error=row[11],
            created_at=row[12],
            updated_at=row[13],
            next_run_at=row[14],
        )


class ReviewOrchestrator:
    """LangGraph review lifecycle orchestrator for parallel specialists and bounded execution."""

    def __init__(
        self,
        specialist_handlers: Mapping[SpecialistType, SpecialistHandler] | None = None,
        default_instructions: Mapping[SpecialistType, str] | None = None,
        *,
        budget_enforcer: Any | None = None,
        cost_ledger: Any | None = None,
        pricing_registry: Any | None = None,
    ) -> None:
        self.specialist_handlers = dict(specialist_handlers or {})
        self.default_instructions = dict(default_instructions or {})
        self.budget_enforcer = budget_enforcer
        self.cost_ledger = cost_ledger
        self.pricing_registry = pricing_registry

    def register_specialist(self, specialist_type: SpecialistType, handler: SpecialistHandler) -> None:
        """Register a handler for a specific review specialist."""
        self.specialist_handlers[specialist_type] = handler

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph StateGraph review lifecycle."""
        graph = StateGraph(_GraphState)

        # Nodes
        graph.add_node("initialize", self._node_initialize)
        graph.add_node("security_specialist", self._make_specialist_node(SpecialistType.SECURITY))
        graph.add_node("quality_specialist", self._make_specialist_node(SpecialistType.QUALITY))
        graph.add_node("tests_specialist", self._make_specialist_node(SpecialistType.TESTS))
        graph.add_node("documentation_specialist", self._make_specialist_node(SpecialistType.DOCUMENTATION))
        graph.add_node("evaluate_terminal", self._node_evaluate_terminal)

        # Edges
        graph.add_edge(START, "initialize")

        def _route_after_init(state: _GraphState) -> str | list[str]:
            term_status = state.get("terminal_status")
            if state.get("is_cancelled") or term_status in (
                JobState.CANCELLED.value,
                JobState.FAILED.value,
                "failed",
                "budget_exhausted",
                "unknown_budget_state",
                "context_cap_exceeded",
            ):
                return END
            return [
                "security_specialist",
                "quality_specialist",
                "tests_specialist",
                "documentation_specialist",
            ]

        graph.add_conditional_edges(
            "initialize",
            _route_after_init,
            [
                END,
                "security_specialist",
                "quality_specialist",
                "tests_specialist",
                "documentation_specialist",
            ],
        )

        graph.add_edge("security_specialist", "evaluate_terminal")
        graph.add_edge("quality_specialist", "evaluate_terminal")
        graph.add_edge("tests_specialist", "evaluate_terminal")
        graph.add_edge("documentation_specialist", "evaluate_terminal")
        graph.add_edge("evaluate_terminal", END)

        return graph.compile()

    async def execute_run(
        self,
        job: ReviewJob,
        snapshot: ReviewSnapshot,
        *,
        diff_content: str = "",
        retrieved_evidence: tuple[str, ...] = (),
        run_deadline_seconds: float | None = None,
        cancellation_token: bool = False,
    ) -> ReviewLifecycleState:
        """Execute the review lifecycle graph across independently scoped specialists using LangGraph."""
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        correlation_id = f"{job.delivery_id}:{run_id}"
        deadline = time.time() + (run_deadline_seconds if run_deadline_seconds is not None else job.deadline_seconds)

        initial_state: _GraphState = {
            "run_id": run_id,
            "job_id": job.job_id,
            "delivery_id": job.delivery_id,
            "repository_id": job.repository_id,
            "head_sha": job.head_sha,
            "correlation_id": correlation_id,
            "changed_files": snapshot.changed_files,
            "diff_content": diff_content,
            "retrieved_evidence": retrieved_evidence,
            "deadline": deadline,
            "is_cancelled": cancellation_token,
            "is_degraded": False,
            "degradation_details": {},
            "coverage_summary": None,
            "run_cost_summary": None,
            "step_states": {},
            "specialist_outputs": {},
            "partial_failures": [],
            "audit_trail": [],
            "terminal_status": None,
            "aggregation_invoked": False,
        }

        app = self._build_graph()
        final_dict: dict = await app.ainvoke(initial_state)

        return ReviewLifecycleState(
            run_id=final_dict["run_id"],
            job_id=final_dict["job_id"],
            delivery_id=final_dict["delivery_id"],
            head_sha=final_dict["head_sha"],
            deadline=final_dict["deadline"],
            is_cancelled=final_dict.get("is_cancelled", False),
            step_states=dict(final_dict.get("step_states", {})),
            specialist_outputs=dict(final_dict.get("specialist_outputs", {})),
            partial_failures=list(final_dict.get("partial_failures", [])),
            audit_trail=list(final_dict.get("audit_trail", [])),
            terminal_status=final_dict.get("terminal_status"),
            aggregation_invoked=final_dict.get("aggregation_invoked", False),
            is_degraded=final_dict.get("is_degraded", False),
            degradation_details=dict(final_dict.get("degradation_details", {})),
            coverage_summary=final_dict.get("coverage_summary"),
            run_cost_summary=final_dict.get("run_cost_summary"),
        )

    async def _node_initialize(self, state: _GraphState) -> dict:
        event = AuditEvent(
            correlation_id=state["correlation_id"],
            event_name="review_run_started",
            step="initialize",
            timestamp=time.time(),
            details={"job_id": state["job_id"], "head_sha": state["head_sha"]},
        )
        if state.get("is_cancelled"):
            cancel_event = AuditEvent(
                correlation_id=state["correlation_id"],
                event_name="review_run_cancelled",
                step="initialize",
                timestamp=time.time(),
                details={"reason": "Cancelled before specialist dispatch"},
            )
            return {
                "step_states": {"initialize": "cancelled"},
                "terminal_status": JobState.CANCELLED.value,
                "audit_trail": [event, cancel_event],
            }

        # Admission & context bounding evaluation (FR-19, AC-13)
        if self.budget_enforcer is not None:
            decision = self.budget_enforcer.check_admission(
                repository_id=state.get("repository_id", ""),
                diff_content=state.get("diff_content", ""),
            )
            if not decision.allowed:
                rej_event = AuditEvent(
                    correlation_id=state["correlation_id"],
                    event_name="review_run_admission_rejected",
                    step="initialize",
                    timestamp=time.time(),
                    details={
                        "status": decision.status,
                        "reason": decision.reason,
                        "repository_id": state.get("repository_id", ""),
                    },
                )
                return {
                    "step_states": {"initialize": decision.status},
                    "terminal_status": JobState.FAILED.value,
                    "audit_trail": [event, rej_event],
                }

            if decision.is_degraded:
                degraded_event = AuditEvent(
                    correlation_id=state["correlation_id"],
                    event_name="context_cap_degraded",
                    step="initialize",
                    timestamp=time.time(),
                    details={
                        "original_diff_bytes": decision.original_diff_bytes,
                        "retained_diff_bytes": decision.retained_diff_bytes,
                        "truncated_diff_bytes": decision.truncated_diff_bytes,
                        "reason": decision.reason,
                    },
                )
                return {
                    "diff_content": decision.degraded_diff_content or "",
                    "is_degraded": True,
                    "degradation_details": {
                        "original_diff_bytes": decision.original_diff_bytes,
                        "retained_diff_bytes": decision.retained_diff_bytes,
                        "truncated_diff_bytes": decision.truncated_diff_bytes,
                        "reason": decision.reason,
                    },
                    "step_states": {"initialize": "completed", "specialists_dispatch": "running"},
                    "audit_trail": [event, degraded_event],
                }

        return {
            "step_states": {"initialize": "completed", "specialists_dispatch": "running"},
            "audit_trail": [event],
        }

    def _make_specialist_node(self, spec_type: SpecialistType):
        async def specialist_node(state: _GraphState) -> dict:
            events = [
                AuditEvent(
                    correlation_id=state["correlation_id"],
                    event_name="specialist_started",
                    step=f"specialist_{spec_type.value}",
                    timestamp=time.time(),
                    details={"specialist_type": spec_type.value},
                )
            ]
            spec_input = SpecialistInput(
                specialist_type=spec_type,
                correlation_id=state["correlation_id"],
                instructions=self.default_instructions.get(spec_type, f"Review for {spec_type.value}"),
                changed_files=state["changed_files"],
                diff_content=state["diff_content"],
                retrieved_evidence=state["retrieved_evidence"],
                head_sha=state["head_sha"],
            )
            output = await self._run_single_specialist(spec_type, spec_input, state["deadline"])

            partial_fails = []
            status_str = str(output.status)
            if status_str in (SpecialistStatus.COMPLETED.value, "completed"):
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="specialist_completed",
                        step=f"specialist_{spec_type.value}",
                        timestamp=time.time(),
                        details={"findings_count": len(output.findings), "duration": output.execution_duration},
                    )
                )
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="specialist_succeeded",
                        step=f"specialist_{spec_type.value}",
                        timestamp=time.time(),
                        details={"findings_count": len(output.findings), "duration": output.execution_duration},
                    )
                )
            elif status_str in (SpecialistStatus.DEGRADED.value, "degraded"):
                partial_fails.append(spec_type.value)
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="specialist_degraded",
                        step=f"specialist_{spec_type.value}",
                        timestamp=time.time(),
                        details={"error": output.error_message, "findings_count": len(output.findings)},
                    )
                )
            elif status_str in (SpecialistStatus.SKIPPED.value, "skipped"):
                partial_fails.append(spec_type.value)
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="specialist_skipped",
                        step=f"specialist_{spec_type.value}",
                        timestamp=time.time(),
                        details={"error": output.error_message},
                    )
                )
            elif status_str in (SpecialistStatus.TIMEOUT.value, "timeout"):
                partial_fails.append(spec_type.value)
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="specialist_timeout",
                        step=f"specialist_{spec_type.value}",
                        timestamp=time.time(),
                        details={"error": output.error_message},
                    )
                )
            else:
                partial_fails.append(spec_type.value)
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="specialist_failed",
                        step=f"specialist_{spec_type.value}",
                        timestamp=time.time(),
                        details={"error": output.error_message},
                    )
                )

            return {
                "specialist_outputs": {spec_type: output},
                "step_states": {f"specialist_{spec_type.value}": status_str},
                "partial_failures": partial_fails,
                "audit_trail": events,
            }

        return specialist_node

    async def _node_evaluate_terminal(self, state: _GraphState) -> dict:
        now = time.time()
        events = []

        is_timeout = now > state["deadline"]
        outputs = state.get("specialist_outputs", {})
        succeeded: list[str] = []
        degraded: list[str] = []
        failed: list[str] = []
        timed_out: list[str] = []
        skipped: list[str] = []
        reasons: dict[str, str] = {}

        for spec_type in ALL_SPECIALISTS:
            out = outputs.get(spec_type)
            if out is None:
                if is_timeout:
                    timed_out.append(spec_type.value)
                    reasons[spec_type.value] = "Specialist timed out exceeding deadline"
                else:
                    skipped.append(spec_type.value)
                    reasons[spec_type.value] = "Specialist omitted or not dispatched"
            elif str(out.status) in (SpecialistStatus.COMPLETED.value, "completed"):
                succeeded.append(spec_type.value)
            elif str(out.status) in (SpecialistStatus.DEGRADED.value, "degraded"):
                degraded.append(spec_type.value)
                if out.error_message:
                    reasons[spec_type.value] = out.error_message
            elif str(out.status) in (SpecialistStatus.TIMEOUT.value, "timeout"):
                timed_out.append(spec_type.value)
                reasons[spec_type.value] = out.error_message or "Specialist timeout"
            elif str(out.status) in (SpecialistStatus.SKIPPED.value, "skipped"):
                skipped.append(spec_type.value)
                reasons[spec_type.value] = out.error_message or "Specialist skipped"
            else:
                failed.append(spec_type.value)
                reasons[spec_type.value] = out.error_message or "Specialist failed"

        completed_count = len(succeeded)
        total_specialists = len(ALL_SPECIALISTS)
        is_full_coverage = (completed_count == total_specialists) and not is_timeout
        is_run_degraded = (not is_full_coverage) or bool(state.get("is_degraded", False))
        coverage_ratio = round(completed_count / total_specialists, 3) if total_specialists > 0 else 0.0

        coverage_summary = SpecialistCoverageSummary(
            total_specialists=total_specialists,
            succeeded_specialists=tuple(succeeded),
            degraded_specialists=tuple(degraded),
            failed_specialists=tuple(failed),
            timeout_specialists=tuple(timed_out),
            skipped_specialists=tuple(skipped),
            is_full_coverage=is_full_coverage,
            is_degraded=is_run_degraded,
            coverage_ratio=coverage_ratio,
            failure_reasons=reasons,
        )

        updated_degradation_details = dict(state.get("degradation_details", {}))
        updated_degradation_details.update({
            "coverage_summary": asdict(coverage_summary),
            "succeeded_specialists": succeeded,
            "failed_specialists": failed,
            "degraded_specialists": degraded,
            "timeout_specialists": timed_out,
            "skipped_specialists": skipped,
            "is_full_coverage": is_full_coverage,
            "coverage_ratio": coverage_ratio,
            "failure_reasons": reasons,
        })

        if is_run_degraded:
            events.append(
                AuditEvent(
                    correlation_id=state["correlation_id"],
                    event_name="specialist_coverage_degraded",
                    step="evaluate_terminal",
                    timestamp=now,
                    details={
                        "total_specialists": total_specialists,
                        "succeeded": succeeded,
                        "degraded": degraded,
                        "failed": failed,
                        "timeout": timed_out,
                        "skipped": skipped,
                        "coverage_ratio": coverage_ratio,
                        "reasons": reasons,
                    },
                )
            )

        if is_timeout:
            events.append(
                AuditEvent(
                    correlation_id=state["correlation_id"],
                    event_name="review_run_timeout",
                    step="evaluate_terminal",
                    timestamp=now,
                    details={"deadline": state["deadline"], "now": now},
                )
            )
            return {
                "step_states": {"specialists_dispatch": "timeout"},
                "terminal_status": "timeout",
                "is_degraded": True,
                "degradation_details": updated_degradation_details,
                "coverage_summary": coverage_summary,
                "aggregation_invoked": False,
                "audit_trail": events,
            }


        # Cost recording and accounting (FR-19)
        run_cost_summary = None
        if self.cost_ledger is not None:
            for spec_type, out in outputs.items():
                u = getattr(out, "usage", None)
                if u is not None:
                    c_usd = u.cost_usd
                    pricing_configured = u.pricing_configured
                    if c_usd is None and self.pricing_registry is not None:
                        calc = self.pricing_registry.calculate_cost(
                            u.provider,
                            u.model,
                            u.prompt_tokens,
                            u.completion_tokens,
                        )
                        if calc is not None:
                            c_usd = calc
                            pricing_configured = True

                    from pr_review_agent.cost_controls import ComponentUsage

                    final_usage = ComponentUsage(
                        component=u.component or f"specialist_{spec_type.value}",
                        provider=u.provider,
                        model=u.model,
                        prompt_tokens=u.prompt_tokens,
                        completion_tokens=u.completion_tokens,
                        total_tokens=u.total_tokens,
                        cost_usd=c_usd,
                        usage_source=u.usage_source,
                        pricing_configured=pricing_configured,
                    )
                    self.cost_ledger.record_usage(
                        run_id=state["run_id"],
                        correlation_id=state["correlation_id"],
                        repository_id=state.get("repository_id", ""),
                        usage=final_usage,
                        timestamp=now,
                    )
            run_cost_summary = self.cost_ledger.get_run_cost_summary(state["run_id"])
            events.append(
                AuditEvent(
                    correlation_id=state["correlation_id"],
                    event_name="cost_recorded",
                    step="evaluate_terminal",
                    timestamp=now,
                    details={
                        "total_prompt_tokens": run_cost_summary.total_prompt_tokens,
                        "total_completion_tokens": run_cost_summary.total_completion_tokens,
                        "total_tokens": run_cost_summary.total_tokens,
                        "total_cost_usd": run_cost_summary.total_cost_usd,
                        "is_cost_complete": run_cost_summary.is_cost_complete,
                        "components_count": len(run_cost_summary.components),
                    },
                )
            )
            if self.budget_enforcer is not None:
                within_limits, limit_reason = self.budget_enforcer.check_run_limits(
                    run_cost_summary.total_tokens,
                    run_cost_summary.total_cost_usd,
                    summary=run_cost_summary,
                )
                if not within_limits:
                    events.append(
                        AuditEvent(
                            correlation_id=state["correlation_id"],
                            event_name="run_budget_limit_exceeded",
                            step="evaluate_terminal",
                            timestamp=now,
                            details={"reason": limit_reason},
                        )
                    )
        elif self.budget_enforcer is not None:
            within_limits, limit_reason = self.budget_enforcer.check_run_limits(
                summary=None,
            )
            if not within_limits:
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="run_budget_limit_exceeded",
                        step="evaluate_terminal",
                        timestamp=now,
                        details={"reason": limit_reason},
                    )
                )

        if completed_count == 0:
            events.append(
                AuditEvent(
                    correlation_id=state["correlation_id"],
                    event_name="review_run_failed",
                    step="evaluate_terminal",
                    timestamp=now,
                    details={"reason": "All specialists failed", "failures": state.get("partial_failures", [])},
                )
            )
            return {
                "step_states": {"specialists_dispatch": "failed"},
                "terminal_status": "failed",
                "is_degraded": True,
                "degradation_details": updated_degradation_details,
                "coverage_summary": coverage_summary,
                "aggregation_invoked": False,
                "run_cost_summary": run_cost_summary,
                "audit_trail": events,
            }

        events.append(
            AuditEvent(
                correlation_id=state["correlation_id"],
                event_name="aggregation_invoked",
                step="aggregation",
                timestamp=now,
                details={
                    "successful_specialists": completed_count,
                    "partial_failures": state.get("partial_failures", []),
                    "is_full_coverage": is_full_coverage,
                },
            )
        )
        return {
            "step_states": {"specialists_dispatch": "completed", "aggregation": "ready"},
            "terminal_status": "completed",
            "is_degraded": is_run_degraded,
            "degradation_details": updated_degradation_details,
            "coverage_summary": coverage_summary,
            "aggregation_invoked": True,
            "run_cost_summary": run_cost_summary,
            "audit_trail": events,
        }

    async def _run_single_specialist(
        self,
        spec_type: SpecialistType,
        spec_input: SpecialistInput,
        deadline: float,
    ) -> SpecialistOutput:
        """Execute an individual specialist with non-blocking thread execution, timeout, and exception safety."""
        start_time = time.time()
        remaining_timeout = max(0.01, deadline - start_time)

        handler = self.specialist_handlers.get(spec_type)
        if handler is None:
            return SpecialistOutput(
                specialist_type=spec_type,
                correlation_id=spec_input.correlation_id,
                status=SpecialistStatus.SKIPPED,
                findings=(),
                error_message=f"No specialist handler registered for {spec_type.value}",
                execution_duration=time.time() - start_time,
            )

        try:
            if asyncio.iscoroutinefunction(handler):
                output = await asyncio.wait_for(handler(spec_input), timeout=remaining_timeout)
            else:
                # Offload synchronous handler to thread to prevent blocking event loop
                result = await asyncio.wait_for(
                    asyncio.to_thread(handler, spec_input),
                    timeout=remaining_timeout,
                )
                if asyncio.iscoroutine(result):
                    output = await asyncio.wait_for(result, timeout=max(0.01, deadline - time.time()))
                else:
                    output = result

            duration = time.time() - start_time
            return SpecialistOutput(
                specialist_type=output.specialist_type,
                correlation_id=output.correlation_id,
                status=output.status,
                findings=output.findings,
                error_message=output.error_message,
                execution_duration=duration,
                usage=getattr(output, "usage", None),
            )
        except asyncio.TimeoutError:
            return SpecialistOutput(
                specialist_type=spec_type,
                correlation_id=spec_input.correlation_id,
                status=SpecialistStatus.TIMEOUT,
                error_message="Specialist execution exceeded run deadline",
                execution_duration=time.time() - start_time,
            )
        except Exception as err:
            return SpecialistOutput(
                specialist_type=spec_type,
                correlation_id=spec_input.correlation_id,
                status=SpecialistStatus.FAILED,
                error_message=str(err),
                execution_duration=time.time() - start_time,
            )


class ReviewWorker:
    """Runtime worker/dispatcher that leases jobs and processes them via ReviewOrchestrator.

    Enforces runtime concurrency caps (NFR-05) by propagating the configured
    CostAndBudgetConfig.max_concurrent_reviews_per_repo into lease_next_job().
    """

    def __init__(
        self,
        queue: DurableJobQueue,
        orchestrator: ReviewOrchestrator,
        *,
        budget_config: Any | None = None,
    ) -> None:
        self.queue = queue
        self.orchestrator = orchestrator
        self.budget_config = budget_config or getattr(queue, "budget_config", None)

    def lease_job(self, now: float | None = None) -> ReviewJob | None:
        """Lease the next eligible job from the queue with configured concurrency cap."""
        cap = (
            getattr(self.budget_config, "max_concurrent_reviews_per_repo", None)
            if self.budget_config is not None
            else None
        )
        return self.queue.lease_next_job(now=now, max_concurrency_per_repo=cap)

    async def run_next_job(
        self,
        snapshot: ReviewSnapshot,
        *,
        diff_content: str = "",
        retrieved_evidence: tuple[str, ...] = (),
        run_deadline_seconds: float | None = None,
        now: float | None = None,
    ) -> tuple[ReviewJob | None, ReviewLifecycleState | None]:
        """Lease the next eligible job, execute the review lifecycle, and record final status."""
        job = self.lease_job(now=now)
        if job is None:
            return None, None
        try:
            state = await self.orchestrator.execute_run(
                job=job,
                snapshot=snapshot,
                diff_content=diff_content,
                retrieved_evidence=retrieved_evidence,
                run_deadline_seconds=run_deadline_seconds,
            )
            if state.terminal_status in (JobState.FAILED.value, "failed"):
                self.queue.mark_failed(job.job_id, "Review run failed during execution", now=now)
            else:
                self.queue.mark_completed(job.job_id, now=now)
            return job, state
        except Exception as exc:
            self.queue.mark_failed(job.job_id, str(exc), now=now)
            raise
