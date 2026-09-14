"""Durable asynchronous dispatch and LangGraph review lifecycle orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
import time
from typing import Annotated, TypedDict
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
    status: str  # 'completed', 'failed', 'timeout'
    findings: tuple[CandidateFinding, ...] = ()
    error_message: str | None = None
    execution_duration: float = 0.0


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
    head_sha: str
    correlation_id: str
    changed_files: tuple[str, ...]
    diff_content: str
    retrieved_evidence: tuple[str, ...]
    deadline: float
    is_cancelled: bool
    step_states: Annotated[dict[str, str], _merge_dict]
    specialist_outputs: Annotated[dict[SpecialistType, SpecialistOutput], _merge_dict]
    partial_failures: Annotated[list[str], _merge_list]
    audit_trail: Annotated[list[AuditEvent], _merge_list]
    terminal_status: str | None
    aggregation_invoked: bool


SpecialistHandler = Callable[[SpecialistInput], SpecialistOutput | asyncio.Future[SpecialistOutput]]


class DurableJobQueue:
    """A durable SQLite-backed review job queue providing Redis/ARQ equivalent semantics."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
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

    def lease_next_job(self, now: float | None = None) -> ReviewJob | None:
        """Atomically lease the next scheduled job using an exclusive write transaction."""
        current_time = time.time() if now is None else now
        try:
            self.connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            pass  # Already in transaction
        try:
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
            job_id = row[0]
            self.connection.execute(
                """
                UPDATE review_jobs
                SET state = ?, attempt_count = attempt_count + 1, updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.RUNNING.value, current_time, job_id),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return self.get_job(job_id)

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
    ) -> None:
        self.specialist_handlers = dict(specialist_handlers or {})
        self.default_instructions = dict(default_instructions or {})

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
            if state.get("is_cancelled") or state.get("terminal_status") == JobState.CANCELLED.value:
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
            "head_sha": job.head_sha,
            "correlation_id": correlation_id,
            "changed_files": snapshot.changed_files,
            "diff_content": diff_content,
            "retrieved_evidence": retrieved_evidence,
            "deadline": deadline,
            "is_cancelled": cancellation_token,
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

        return {
            "step_states": {"initialize": "completed", "specialists_dispatch": "running"},
            "audit_trail": [event],
        }

    def _make_specialist_node(self, spec_type: SpecialistType):
        async def specialist_node(state: _GraphState) -> dict:
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

            events = []
            partial_fails = []
            if output.status == "completed":
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="specialist_completed",
                        step=f"specialist_{spec_type.value}",
                        timestamp=time.time(),
                        details={"findings_count": len(output.findings), "duration": output.execution_duration},
                    )
                )
            else:
                partial_fails.append(spec_type.value)
                events.append(
                    AuditEvent(
                        correlation_id=state["correlation_id"],
                        event_name="specialist_failed" if output.status == "failed" else "specialist_timeout",
                        step=f"specialist_{spec_type.value}",
                        timestamp=time.time(),
                        details={"error": output.error_message},
                    )
                )

            return {
                "specialist_outputs": {spec_type: output},
                "step_states": {f"specialist_{spec_type.value}": output.status},
                "partial_failures": partial_fails,
                "audit_trail": events,
            }

        return specialist_node

    async def _node_evaluate_terminal(self, state: _GraphState) -> dict:
        now = time.time()
        events = []

        if now > state["deadline"]:
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
                "aggregation_invoked": False,
                "audit_trail": events,
            }

        outputs = state.get("specialist_outputs", {})
        completed_count = sum(1 for o in outputs.values() if o.status == "completed")

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
                "aggregation_invoked": False,
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
                },
            )
        )
        return {
            "step_states": {"specialists_dispatch": "completed", "aggregation": "ready"},
            "terminal_status": "completed",
            "aggregation_invoked": True,
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
                status="completed",
                findings=(),
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
            )
        except asyncio.TimeoutError:
            return SpecialistOutput(
                specialist_type=spec_type,
                correlation_id=spec_input.correlation_id,
                status="timeout",
                error_message="Specialist execution exceeded run deadline",
                execution_duration=time.time() - start_time,
            )
        except Exception as err:
            return SpecialistOutput(
                specialist_type=spec_type,
                correlation_id=spec_input.correlation_id,
                status="failed",
                error_message=str(err),
                execution_duration=time.time() - start_time,
            )
