"""Autonomous worker continuously leasing review jobs and executing the full review lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import logging
import sqlite3
import time
from typing import Any

from pr_review_agent.adapters.github import GitHubNetworkClient
from pr_review_agent.adapters.llm import create_specialist_handlers
from pr_review_agent.github_output import (
    GitHubClient,
    GitHubReviewPublisher,
    PublicationResult,
    PublicationStatus,
)
from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.observability import AuditEvent, AuditSpine
from pr_review_agent.orchestration import (
    CandidateFinding,
    DurableJobQueue,
    DurableQueueProtocol,
    JobState,
    ReviewJob,
    ReviewLifecycleState,
    ReviewOrchestrator,
    SpecialistHandler,
    SpecialistStatus,
    SpecialistType,
    WorkflowEngineProtocol,
)
from pr_review_agent.policy import (
    CanonicalFinding,
    FindingAggregator,
    FindingDisposition,
    ReviewPolicyEngine,
    ReviewTruthStore,
    TruthState,
)
from arq.connections import RedisSettings
from arq.constants import default_queue_name

from pr_review_agent.adapters.redis_queue import (
    RedisJobQueue,
    StaleLeaseError,
    review_job_task,
)
from pr_review_agent.service_config import ServiceConfig, load_service_config


logger = logging.getLogger(__name__)


def reconstruct_snapshot(
    job_or_payload: ReviewJob | Mapping[str, Any] | str,
    connection: sqlite3.Connection | None = None,
    queue: DurableQueueProtocol | None = None,
) -> ReviewSnapshot:
    """Reconstruct an immutable ReviewSnapshot from a job or serialized payload."""
    if isinstance(job_or_payload, ReviewJob):
        if hasattr(job_or_payload, "payload_json"):
            data = json.loads(getattr(job_or_payload, "payload_json"))
        elif queue is not None and hasattr(queue, "get_job_payload"):
            data = queue.get_job_payload(job_or_payload.job_id)
        elif connection is not None:
            row = connection.execute(
                "SELECT payload_json FROM review_jobs WHERE job_id = ?",
                (job_or_payload.job_id,),
            ).fetchone()
            data = json.loads(row[0]) if row else {}
        else:

            data = {
                "repository_id": job_or_payload.repository_id,
                "repository_full_name": job_or_payload.repository_id,
                "pull_request_number": job_or_payload.pull_request_number,
                "base_sha": job_or_payload.base_sha,
                "head_sha": job_or_payload.head_sha,
                "changed_files": (),
                "policy_version": "v1",
                "prompt_version": "v1",
                "retrieval_index_version": "v1",
                "model_configuration": {},
            }
    elif isinstance(job_or_payload, str):
        data = json.loads(job_or_payload)
    elif isinstance(job_or_payload, Mapping):
        data = dict(job_or_payload)
    else:
        raise TypeError(f"Cannot reconstruct ReviewSnapshot from {type(job_or_payload).__name__}")

    return ReviewSnapshot(
        repository_id=data["repository_id"],
        repository_full_name=data.get("repository_full_name", data["repository_id"]),
        pull_request_number=int(data["pull_request_number"]),
        base_sha=str(data["base_sha"]),
        head_sha=str(data["head_sha"]),
        changed_files=tuple(data.get("changed_files", ())),
        policy_version=str(data.get("policy_version", "v1")),
        prompt_version=str(data.get("prompt_version", "v1")),
        retrieval_index_version=str(data.get("retrieval_index_version", "v1")),
        model_configuration=dict(data.get("model_configuration", {})),
    )


class AutonomousReviewWorker:
    """Autonomous continuous worker leasing review jobs and driving them to completion."""

    def __init__(
        self,
        config: ServiceConfig,
        connection: sqlite3.Connection | None = None,
        *,
        github_client: GitHubClient | None = None,
        specialist_handlers: Mapping[SpecialistType, SpecialistHandler] | None = None,
        orchestrator: WorkflowEngineProtocol | ReviewOrchestrator | None = None,
        queue: DurableQueueProtocol | DurableJobQueue | None = None,
        audit_spine: AuditSpine | None = None,
        truth_store: ReviewTruthStore | None = None,
        publisher: GitHubReviewPublisher | None = None,
        policy_engine: ReviewPolicyEngine | None = None,
        aggregator: FindingAggregator | None = None,
    ) -> None:
        self.config = config
        self._owns_connection = connection is None
        self.connection = connection or sqlite3.connect(config.database_path, check_same_thread=False)

        self.queue = queue or DurableJobQueue(self.connection)
        self.audit_spine = audit_spine or AuditSpine(self.connection)
        self.truth_store = truth_store or ReviewTruthStore(self.connection)

        self._owns_github_client = github_client is None
        if github_client is not None:
            self.github_client = github_client
        else:
            self.github_client = GitHubNetworkClient(
                config.to_security_config(),
                max_diff_bytes=config.max_diff_bytes,
            )

        self.publisher = publisher or GitHubReviewPublisher(
            self.connection,
            self.truth_store,
            self.github_client,
        )

        if orchestrator is not None:
            self.orchestrator = orchestrator
        else:
            handlers = specialist_handlers
            if handlers is None:
                handlers = create_specialist_handlers(
                    provider=config.model_provider,
                    security_config=config.to_security_config(),
                    model=config.model_name,
                    audit_spine=self.audit_spine,
                )
            default_instructions = {
                SpecialistType.SECURITY: "Review code changes for security vulnerabilities, authentication/authorization bypasses, data exposure, and unsafe cryptographic use.",
                SpecialistType.QUALITY: "Review code changes for correctness bugs, runtime exceptions, logic flaws, resource leaks, and code quality issues.",
                SpecialistType.TESTS: "Review code changes for test coverage gaps, missing regression tests, edge case omissions, and ineffective assertions.",
                SpecialistType.DOCUMENTATION: "Review documentation and docstrings for accuracy against code changes, misleading statements, and missing documentation.",
            }
            self.orchestrator = ReviewOrchestrator(
                specialist_handlers=handlers,
                default_instructions=default_instructions,
            )

        self.policy_engine = policy_engine or ReviewPolicyEngine(policy_version="v1")
        self.aggregator = aggregator or FindingAggregator()
        self.publish_enabled = config.publish_enabled

    async def process_claimed_job(
        self,
        job: ReviewJob,
        lease_token: str | None = None,
        now: float | None = None,
    ) -> ReviewLifecycleState:
        """Process an already-claimed ReviewJob through the full review pipeline with lease token propagation.

        ARQ Integration / Distributed Invariant:
        Retains and propagates lease_token to every terminal queue mutation (mark_completed, mark_failed, cancel_job).
        If another worker has claimed the job (e.g. following lease expiry), terminal mutations
        will raise StaleLeaseError, rejecting stale worker execution without corrupting the queue.
        """
        current_time = time.time() if now is None else now
        effective_token = lease_token if lease_token is not None else getattr(job, "lease_token", None)
        correlation_id = job.delivery_id

        self.audit_spine.record_event(
            AuditEvent(
                correlation_id=correlation_id,
                event_name="worker_job_started",
                step="worker",
                timestamp=current_time,
                details={
                    "job_id": job.job_id,
                    "repository_id": job.repository_id,
                    "pull_request_number": job.pull_request_number,
                    "attempt_count": job.attempt_count,
                    "lease_token": effective_token,
                },
            )
        )

        try:
            snapshot = reconstruct_snapshot(job, connection=self.connection, queue=self.queue)

            # Fetch PR unified diff
            if hasattr(self.github_client, "get_pull_request_diff"):
                diff_content = self.github_client.get_pull_request_diff(
                    snapshot.repository_id,
                    snapshot.pull_request_number,
                )
            else:
                diff_content = ""

            # Execute ReviewOrchestrator with all specialists
            lifecycle_state = await self.orchestrator.execute_run(
                job=job,
                snapshot=snapshot,
                diff_content=diff_content,
            )

            # Record orchestrator audit trail into the audit spine
            for audit_event in lifecycle_state.audit_trail:
                self.audit_spine.record_event(audit_event)

            # Handle terminal failures or cancellations
            if lifecycle_state.terminal_status in (JobState.FAILED.value, "failed"):
                if isinstance(self.queue, RedisJobQueue):
                    await asyncio.to_thread(
                        self.queue.mark_failed,
                        job.job_id,
                        "Review run failed during execution",
                        now=current_time,
                        lease_token=effective_token,
                    )
                else:
                    self.queue.mark_failed(
                        job.job_id,
                        "Review run failed during execution",
                        now=current_time,
                        lease_token=effective_token,
                    )
                self.audit_spine.record_event(
                    AuditEvent(
                        correlation_id=correlation_id,
                        event_name="worker_job_failed",
                        step="worker",
                        timestamp=current_time,
                        details={"job_id": job.job_id, "reason": "Review run failed during execution"},
                    )
                )
                return lifecycle_state

            if lifecycle_state.is_cancelled or lifecycle_state.terminal_status in (JobState.CANCELLED.value, "cancelled"):
                if isinstance(self.queue, RedisJobQueue):
                    await asyncio.to_thread(
                        self.queue.cancel_job,
                        job.job_id,
                        "Review run cancelled during execution",
                        now=current_time,
                        lease_token=effective_token,
                    )
                else:
                    self.queue.cancel_job(
                        job.job_id,
                        "Review run cancelled during execution",
                        now=current_time,
                        lease_token=effective_token,
                    )
                self.audit_spine.record_event(
                    AuditEvent(
                        correlation_id=correlation_id,
                        event_name="worker_job_cancelled",
                        step="worker",
                        timestamp=current_time,
                        details={"job_id": job.job_id},
                    )
                )
                return lifecycle_state

            # Extract candidate findings from specialist outputs (completed or degraded only)
            all_candidate_findings: list[CandidateFinding] = []
            for spec_output in lifecycle_state.specialist_outputs.values():
                if spec_output.status in (
                    SpecialistStatus.COMPLETED.value,
                    SpecialistStatus.DEGRADED.value,
                    "completed",
                    "degraded",
                ):
                    all_candidate_findings.extend(spec_output.findings)

            # Aggregate findings across specialists
            canonical_findings = self.aggregator.aggregate(
                all_candidate_findings,
                repository_id=snapshot.repository_id,
                head_sha=snapshot.head_sha,
            )

            # Apply ReviewPolicyEngine and record into ReviewTruthStore
            for finding in canonical_findings:
                evaluated = self.policy_engine.evaluate(finding, is_fresh=True)
                if evaluated.disposition == FindingDisposition.AUTO_APPROVED:
                    initial_state = TruthState.AUTO_APPROVED
                elif evaluated.disposition == FindingDisposition.HELD:
                    initial_state = TruthState.HELD
                else:
                    initial_state = TruthState.SUPPRESSED

                self.truth_store.record_initial(evaluated, initial_state=initial_state)

            # Publish policy-permitted findings to GitHub (if publishing enabled)
            published_count = 0
            if self.publish_enabled:
                for finding in canonical_findings:
                    latest = self.truth_store.get_latest_state(finding.canonical_id)
                    if latest and latest.state in (TruthState.APPROVED, TruthState.AUTO_APPROVED):
                        pub_res = self.publisher.publish_finding(
                            finding,
                            pull_number=snapshot.pull_request_number,
                            diff_content=diff_content,
                            now=current_time,
                        )
                        if pub_res.status == PublicationStatus.PUBLISHED:
                            published_count += 1
                        self.audit_spine.record_event(
                            AuditEvent(
                                correlation_id=correlation_id,
                                event_name="finding_publication_result",
                                step="publish",
                                timestamp=current_time,
                                details={
                                    "canonical_id": finding.canonical_id,
                                    "status": pub_res.status.value,
                                    "review_id": pub_res.review_id,
                                    "comment_id": pub_res.comment_id,
                                },
                            )
                        )

            # Mark job completed in queue with exact lease token
            if isinstance(self.queue, RedisJobQueue):
                await asyncio.to_thread(
                    self.queue.mark_completed,
                    job.job_id,
                    now=current_time,
                    lease_token=effective_token,
                )
            else:
                self.queue.mark_completed(
                    job.job_id,
                    now=current_time,
                    lease_token=effective_token,
                )

            coverage_data: dict[str, Any] = {}
            if lifecycle_state.coverage_summary:
                cov = lifecycle_state.coverage_summary
                coverage_data = {
                    "is_full_coverage": cov.is_full_coverage,
                    "is_degraded": cov.is_degraded,
                    "coverage_ratio": cov.coverage_ratio,
                    "succeeded": [s.value if hasattr(s, "value") else str(s) for s in cov.succeeded_specialists],
                    "degraded": [s.value if hasattr(s, "value") else str(s) for s in cov.degraded_specialists],
                    "failed": [s.value if hasattr(s, "value") else str(s) for s in cov.failed_specialists],
                    "timed_out": [s.value if hasattr(s, "value") else str(s) for s in cov.timed_out_specialists],
                    "skipped": [s.value if hasattr(s, "value") else str(s) for s in cov.skipped_specialists],
                    "failure_reasons": cov.failure_reasons,
                }

            self.audit_spine.record_event(
                AuditEvent(
                    correlation_id=correlation_id,
                    event_name="worker_job_completed",
                    step="worker",
                    timestamp=current_time,
                    details={
                        "job_id": job.job_id,
                        "canonical_findings_count": len(canonical_findings),
                        "published_count": published_count,
                        "is_degraded": lifecycle_state.is_degraded,
                        "coverage": coverage_data,
                    },
                )
            )

            return lifecycle_state

        except StaleLeaseError:
            logger.warning(
                "Worker lease expired or stolen for job %s (token: %s); mutation rejected",
                job.job_id,
                effective_token,
            )
            self.audit_spine.record_event(
                AuditEvent(
                    correlation_id=correlation_id,
                    event_name="worker_stale_lease_rejected",
                    step="worker",
                    timestamp=current_time,
                    details={"job_id": job.job_id, "lease_token": effective_token},
                )
            )
            raise
        except Exception as exc:
            try:
                if isinstance(self.queue, RedisJobQueue):
                    await asyncio.to_thread(
                        self.queue.mark_failed,
                        job.job_id,
                        str(exc),
                        now=current_time,
                        lease_token=effective_token,
                    )
                else:
                    self.queue.mark_failed(
                        job.job_id,
                        str(exc),
                        now=current_time,
                        lease_token=effective_token,
                    )
            except StaleLeaseError:
                pass  # Stale worker rejection takes precedence
            self.audit_spine.record_event(
                AuditEvent(
                    correlation_id=correlation_id,
                    event_name="worker_job_failed",
                    step="worker",
                    timestamp=current_time,
                    details={"job_id": job.job_id, "error": str(exc)},
                )
            )
            raise

    async def process_one_job(
        self,
        now: float | None = None,
    ) -> tuple[ReviewJob | None, ReviewLifecycleState | None]:
        """Lease one eligible job and process it through the full review pipeline."""
        current_time = time.time() if now is None else now
        if isinstance(self.queue, RedisJobQueue):
            job = await asyncio.to_thread(self.queue.lease_next_job, now=current_time)
        else:
            job = self.queue.lease_next_job(now=current_time)
        if job is None:
            return None, None

        lifecycle_state = await self.process_claimed_job(
            job,
            lease_token=getattr(job, "lease_token", None),
            now=current_time,
        )
        return job, lifecycle_state

    async def run_worker_loop(
        self,
        *,
        poll_interval_seconds: float = 1.0,
        stop_event: asyncio.Event | None = None,
        max_iterations: int | None = None,
    ) -> int:
        """Run continuous review worker loop until stopped or max_iterations reached."""
        iterations = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if max_iterations is not None and iterations >= max_iterations:
                break

            iterations += 1
            try:
                job, _ = await self.process_one_job()
                if job is None:
                    # Queue is empty, pause before next poll
                    await asyncio.sleep(poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing review job in worker loop: %s", e)
                await asyncio.sleep(poll_interval_seconds)

        return iterations

    def close(self) -> None:
        """Cleanly close underlying resources."""
        if self._owns_github_client and hasattr(self.github_client, "close"):
            self.github_client.close()
        if self._owns_connection:
            self.connection.close()


def run_worker(
    config: ServiceConfig | None = None,
    connection: sqlite3.Connection | None = None,
    *,
    poll_interval_seconds: float = 1.0,
    max_iterations: int | None = None,
) -> None:
    """Entrypoint to run the autonomous review worker synchronously."""
    cfg = config or load_service_config(require_live_credentials=True)
    worker = AutonomousReviewWorker(cfg, connection=connection)
    try:
        asyncio.run(worker.run_worker_loop(
            poll_interval_seconds=poll_interval_seconds,
            max_iterations=max_iterations,
        ))
    finally:
        worker.close()


if __name__ == "__main__":
    run_worker()


# ---------------------------------------------------------------------------
# ARQ 0.28.0 Production Worker Bootstrap & Registration
# ---------------------------------------------------------------------------

async def arq_startup(ctx: dict[str, Any]) -> None:
    """Initialize worker dependencies and populate ARQ task context."""
    config: ServiceConfig = ctx.get("config") or load_service_config(require_live_credentials=False)
    ctx["config"] = config

    # Initialize queue if not already injected
    if "queue" not in ctx:
        redis_url = getattr(config, "redis_url", None) or "redis://127.0.0.1:6379/0"
        ctx["queue"] = RedisJobQueue(redis_url=redis_url)

    # Initialize AutonomousReviewWorker if not already injected
    if "worker" not in ctx:
        ctx["worker"] = AutonomousReviewWorker(config, queue=ctx["queue"])


async def arq_shutdown(ctx: dict[str, Any]) -> None:
    """Cleanly close underlying connections and worker resources."""
    worker = ctx.get("worker")
    if worker and hasattr(worker, "close"):
        worker.close()
    queue = ctx.get("queue")
    if queue and hasattr(queue, "close"):
        queue.close()


def get_redis_settings(config: ServiceConfig | None = None) -> RedisSettings:
    """Derive ARQ RedisSettings from application configuration."""
    cfg = config or load_service_config(require_live_credentials=False)
    redis_url = getattr(cfg, "redis_url", None) or "redis://127.0.0.1:6379/0"
    return RedisSettings.from_dsn(redis_url)


class WorkerSettings:
    """ARQ 0.28.0 Worker configuration for distributed PR review task execution.

    Production CLI startup command:
        python -m arq pr_review_agent.worker.WorkerSettings

    Retry Authority & Operational Semantics (Concern 1):
    1. Application Retry Authority:
       ReviewJob.attempt_count, ReviewJob.max_retries, and backoff_base_seconds are the sole
       application retry authority. Generic exception-retry in ARQ is suppressed: application
       failures are caught and governed strictly by ReviewJob lifecycle and RedisJobQueue.mark_failed().
       ARQ's explicit deferral mechanism (Retry(defer=...)) is used exclusively to align ARQ's
       dispatch schedule with the application's exponential backoff.
    2. Pessimistic Worker-Crash Re-Execution:
       ARQ's native pessimistic locking (arq:in-progress:{job_id} TTL) is preserved.
       If a worker process crashes abruptly (SIGKILL, host failure), the in-progress TTL expires
       in Redis and the surviving/restarted ARQ worker picks up the job from arq:queue for re-execution.
    3. Operational Abort / Cancellation (Concern 2):
       allow_abort_jobs = True enables ARQ's background monitoring of abort_jobs_ss (arq:abort).
       Logical JobState.CANCELLED in review:job:{job_id} remains authoritative; ARQ task abort
       serves as an operational optimization to terminate active compute immediately.
    """

    functions = [review_job_task]
    on_startup = arq_startup
    on_shutdown = arq_shutdown
    queue_name = default_queue_name
    redis_settings = get_redis_settings()

    # Concern 2: Enable ARQ operational cancellation listener on abort_jobs_ss
    allow_abort_jobs = True

    # Concern 1: Allow review_job_task to signal controlled deferral via Retry(defer=...)
    # while preventing ARQ from imposing an unintended generic exception retry policy.
    # Set max_tries to a generous ceiling so ARQ transport never truncates application retries
    # or pessimistic crash recovery.
    retry_jobs = True
    max_tries = 10


ReviewWorkerSettings = WorkerSettings
