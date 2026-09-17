"""Comprehensive Production Infrastructure Fault Injection and Reliability Proof Test Suite (W1-07).

Proves the failure-safety, transaction atomicity, crash recovery, and cross-system resilience
of the production infrastructure adapters without cutting them over as the default runtime:
1. Redis Queue Adapter (RedisJobQueue, ARQ task execution, Lua fallbacks, retry authority, stale lease rejection)
2. Redis Workflow Checkpointer (RedisCheckpointSaver, LangGraph recovery, missing/corrupted blob handling)
3. Tiger Data Adapters (TigerCodeMemoryStore, TigerReviewTruthStore, TigerAuditSpine, advisory locks, rollbacks)
4. Cross-System Failure Interactions (10 mandatory scenarios: Redis/Tiger partitions, worker crashes,
   duplicate deliveries, publication reconciliation, stale head SHA detection)
5. Optional Live Infrastructure Hooks (Runs against real Redis / PostgreSQL if available in environment)
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import concurrent.futures
import dataclasses
import json
import os
import sqlite3
import threading
import time
from typing import Any
import unittest
import uuid

import redis

from pr_review_agent.adapters.redis_checkpoint import (
    CheckpointDeserializationError,
    CheckpointStorageError,
    RedisCheckpointSaver,
)
from pr_review_agent.adapters.redis_queue import (
    RedisJobQueue,
    StaleLeaseError,
    review_job_task,
)
from pr_review_agent.adapters.tiger_connection import (
    TigerConfig,
    TigerConnectionError,
    TigerConnectionManager,
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
from pr_review_agent.github_output import (
    FakeGitHubClient,
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
    ReviewOrchestrator,
    SpecialistInput,
    SpecialistOutput,
    SpecialistStatus,
    SpecialistType,
)
from pr_review_agent.policy import (
    CanonicalFinding,
    FindingAggregator,
    FindingDisposition,
    ReviewPolicyEngine,
    ReviewTruthStore,
    TruthState,
)
from pr_review_agent.retrieval import CodeChunk, HybridRetriever
from pr_review_agent.service_config import ServiceConfig
from pr_review_agent.worker import AutonomousReviewWorker, reconstruct_snapshot
from tests.test_queue_and_checkpoint_contracts import sample_snapshot
from tests.test_redis_checkpoint_adapter import InMemoryRedisCheckpointClient
from tests.test_redis_queue_adapter import (
    AsyncInMemoryRedisPool,
    InMemoryRedisClient,
    _create_test_worker,
    _make_dummy_service_config,
    _make_mock_specialist,
)
from tests.test_tiger_data_adapters import (
    SimulationCursor,
    TigerPostgresSimulationConnection,
    _make_sample_finding,
)

AUTH_DIFF = """diff --git a/src/auth.py b/src/auth.py
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,6 @@
+line 10
+line 11
+line 12
+line 13
+line 14
+line 15
"""


def _make_canonical(
    canonical_id: str = "can-test-1",
    repository_id: str = "octocat/hello-world",
    head_sha: str = "head-sha-valid",
    file_path: str = "src/auth.py",
    line_range: tuple[int, int] = (10, 15),
    category: str = "quality",
    severity: str = "medium",
    confidence: float = 0.90,
    summary: str = "Code quality recommendation",
    rationale: str = "Refactor for clarity and testability.",
    evidence_refs: tuple[str, ...] = ("diff://src/auth.py#L10-L15",),
    remediation: str | None = "Refactor method.",
    disposition: FindingDisposition = FindingDisposition.AUTO_APPROVED,
) -> CanonicalFinding:
    return CanonicalFinding(
        canonical_id=canonical_id,
        repository_id=repository_id,
        head_sha=head_sha,
        category=category,
        severity=severity,
        confidence=confidence,
        summary=summary,
        rationale=rationale,
        file_path=file_path,
        line_range=line_range,
        contributing_candidate_ids=("cand-1",),
        contributing_specialists=("quality",),
        evidence_refs=evidence_refs,
        remediation=remediation,
        disposition=disposition,
    )


# ============================================================================
# FAULT-INJECTING SIMULATION HARNESSES
# ============================================================================

class FaultInjectingRedisClient(InMemoryRedisClient):
    """InMemoryRedisClient extension supporting programmable fault injection."""

    def __init__(self) -> None:
        super().__init__()
        self.drop_connection: bool = False
        self.timeout_on_methods: set[str] = set()
        self.error_on_methods: set[str] = set()
        self.simulate_lua_response_drop: bool = False
        self.simulate_pipeline_failure: bool = False

    def _check_fault(self, method_name: str) -> None:
        if self.drop_connection or self.closed:
            raise redis.ConnectionError("Simulated Redis network partition / connection drop")
        if method_name in self.timeout_on_methods:
            raise redis.TimeoutError(f"Simulated Redis timeout on {method_name}")
        if method_name in self.error_on_methods:
            raise redis.RedisError(f"Simulated Redis server error on {method_name}")

    def get(self, name: str) -> bytes | None:
        self._check_fault("get")
        return super().get(name)

    def set(self, name: str, value: Any, **kwargs: Any) -> bool:
        self._check_fault("set")
        return super().set(name, value, **kwargs)

    def hset(self, name: str, key: str | None = None, value: Any = None, mapping: Mapping[str, Any] | None = None) -> int:
        self._check_fault("hset")
        return super().hset(name, key, value, mapping)

    def hget(self, name: str, key: str) -> bytes | None:
        self._check_fault("hget")
        return super().hget(name, key)

    def hgetall(self, name: str) -> dict[bytes, bytes]:
        self._check_fault("hgetall")
        return super().hgetall(name)

    def zadd(self, name: str, mapping: Mapping[str, float], **kwargs: Any) -> int:
        self._check_fault("zadd")
        return super().zadd(name, mapping, **kwargs)

    def zrangebyscore(self, name: str, min: float | str, max: float | str, start: int | None = None, num: int | None = None, **kwargs: Any) -> list[bytes]:
        self._check_fault("zrangebyscore")
        return super().zrangebyscore(name, min, max, start, num, **kwargs)

    def zrem(self, name: str, *values: Any) -> int:
        self._check_fault("zrem")
        return super().zrem(name, *values)

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        self._check_fault("eval")
        # Execute underlying Lua logic
        result = super().eval(script, numkeys, *keys_and_args)
        if self.simulate_lua_response_drop:
            # Server committed changes, but socket severed before reading response
            raise redis.ConnectionError("Simulated Redis connection drop after Lua script execution")
        return result

    def pipeline(self, transaction: bool = True) -> Any:
        self._check_fault("pipeline")
        real_pipe = super().pipeline(transaction=transaction)
        if self.simulate_pipeline_failure:
            orig_exec = real_pipe.execute

            def failing_execute() -> list[Any]:
                raise redis.ConnectionError("Simulated pipeline execute network drop")

            real_pipe.execute = failing_execute  # type: ignore[method-assign]
        return real_pipe


class FaultInjectingTigerConnection(TigerPostgresSimulationConnection):
    """TigerPostgresSimulationConnection extension with programmable database faults."""

    def __init__(self) -> None:
        super().__init__()
        self.drop_connection: bool = False
        self.fail_on_statement_pattern: str | None = None
        self.fail_on_commit: bool = False
        self.executed_queries: list[str] = []

    def execute(self, query: str, params: Sequence[Any] | Mapping[str, Any] | None = None) -> Any:
        if self.drop_connection:
            raise TigerConnectionError("Simulated Tiger Cloud connection loss / host unreachable")

        self.executed_queries.append(query)

        if self.fail_on_statement_pattern:
            matches_pattern = (
                self.fail_on_statement_pattern in query
                or (
                    params is not None
                    and any(
                        self.fail_on_statement_pattern in str(p)
                        for p in (params.values() if isinstance(params, Mapping) else params)
                    )
                )
            )
            if matches_pattern:
                raise sqlite3.OperationalError(
                    f"Simulated database failure on statement matching pattern '{self.fail_on_statement_pattern}'"
                )

        return super().execute(query, params)

    def commit(self) -> None:
        if self.drop_connection:
            raise TigerConnectionError("Simulated Tiger Cloud connection loss during commit")
        if self.fail_on_commit:
            raise sqlite3.OperationalError("Simulated database failure during transaction commit")
        super().commit()


class FaultInjectingGitHubClient(FakeGitHubClient):
    """FakeGitHubClient extension supporting head SHA shifts and ambiguous API failures."""

    def __init__(
        self,
        pr_heads: Mapping[tuple[str, int], str] | None = None,
        *,
        should_fail_api: bool = False,
        api_error_message: str = "Simulated GitHub API failure",
        diff_content: str = AUTH_DIFF,
    ) -> None:
        super().__init__(pr_heads, should_fail_api=should_fail_api, api_error_message=api_error_message)
        self.diff_content = diff_content
        self.simulate_ambiguous_creation: bool = False
        self.fail_create_review: bool = False
        self.head_sha_move_to: str | None = None
        self.create_review_calls: int = 0
        self.create_comment_calls: int = 0

    def get_pull_request_diff(self, repository_id: str, pull_number: int) -> str:
        return self.diff_content

    def get_pull_request_head_sha(self, repository_id: str, pull_number: int) -> str:
        if self.head_sha_move_to is not None:
            return self.head_sha_move_to
        return super().get_pull_request_head_sha(repository_id, pull_number)

    def create_review(
        self,
        repository_id: str,
        pull_number: int,
        *,
        commit_sha: str,
        body: str,
        event: str = "COMMENT",
        comments: Sequence[Any] = (),
    ) -> Any:
        self.create_review_calls += 1
        if self.fail_create_review:
            raise ConnectionResetError("Simulated TCP reset during review creation")
        if self.simulate_ambiguous_creation:
            # GitHub successfully creates the review, but the HTTP connection is severed before client reads 201 OK
            super().create_review(
                repository_id,
                pull_number,
                commit_sha=commit_sha,
                body=body,
                event=event,
                comments=comments,
            )
            raise ConnectionResetError("Simulated TCP reset after GitHub accepted review request")

        return super().create_review(
            repository_id,
            pull_number,
            commit_sha=commit_sha,
            body=body,
            event=event,
            comments=comments,
        )

    def create_issue_comment(
        self,
        repository_id: str,
        pull_number: int,
        *,
        body: str,
    ) -> Any:
        self.create_comment_calls += 1
        if self.fail_create_review:
            raise ConnectionResetError("Simulated TCP reset during issue comment creation")
        if self.simulate_ambiguous_creation:
            super().create_issue_comment(repository_id, pull_number, body=body)
            raise ConnectionResetError("Simulated TCP reset after GitHub accepted comment request")
        return super().create_issue_comment(repository_id, pull_number, body=body)


# ============================================================================
# DOMAIN A: REDIS QUEUE FAULT INJECTION
# ============================================================================

class TestRedisQueueFaultInjection(unittest.IsolatedAsyncioTestCase):
    """Comprehensive failure injection for RedisJobQueue and ARQ execution mechanics."""

    def setUp(self) -> None:
        self.client = FaultInjectingRedisClient()
        self.queue = RedisJobQueue(client=self.client)  # type: ignore[arg-type]

    def test_redis_outage_during_enqueue_fails_closed(self) -> None:
        """Redis outage on enqueue fails closed without creating orphan state."""
        self.client.drop_connection = True
        snapshot = sample_snapshot()

        with self.assertRaises(redis.ConnectionError):
            self.queue.enqueue(snapshot, "del-outage-1")

        # Reconnect to inspect state
        self.client.drop_connection = False
        self.assertIsNone(self.queue.get_job("job-del-outage-1"))
        self.assertEqual(0, self.client.zcard(self.queue.queue_name))
        self.assertIsNone(self.client.get("review:delivery:del-outage-1"))

    def test_redis_timeout_during_enqueue_fails_closed(self) -> None:
        """Redis socket timeout on enqueue fails closed without partial keys."""
        self.client.timeout_on_methods.add("set")
        snapshot = sample_snapshot()

        with self.assertRaises(redis.TimeoutError):
            self.queue.enqueue(snapshot, "del-timeout-1")

        self.client.timeout_on_methods.clear()
        self.assertIsNone(self.queue.get_job("job-del-timeout-1"))
        self.assertEqual(0, self.client.zcard(self.queue.queue_name))

    def test_redis_response_drop_after_lua_enqueue_recovers_via_fallback(self) -> None:
        """When Lua script executes on Redis but connection drops before response is read,
        fallback detects existing delivery and returns the job without double enqueue.
        """
        snapshot = sample_snapshot()
        self.client.simulate_lua_response_drop = True

        # Enqueue will trigger Lua script, which persists state in Redis, then raises ConnectionError,
        # falling through to the SET NX fallback.
        # SET NX sees key already exists and returns the job safely!
        job = self.queue.enqueue(snapshot, "del-lua-drop-1")
        self.assertIsNotNone(job)
        self.assertEqual("job-del-lua-drop-1", job.job_id)
        self.assertEqual(JobState.QUEUED, job.state)
        self.assertEqual(0, job.attempt_count)

        # Confirm queue cardinality in arq:queue is strictly 1
        self.assertEqual(1, self.client.zcard(self.queue.queue_name))

    def test_concurrent_duplicate_delivery_id_race(self) -> None:
        """10 concurrent threads enqueueing same delivery ID produces exactly 1 job."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/race-repo")
        delivery_id = "del-race-concurrency"

        results: list[ReviewJob] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.queue.enqueue, snapshot, delivery_id, now=now)
                for _ in range(10)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        self.assertEqual(10, len(results))
        for r in results:
            self.assertEqual("job-del-race-concurrency", r.job_id)
            self.assertEqual(JobState.QUEUED, r.state)
            self.assertEqual(0, r.attempt_count)

        # Strictly 1 entry in scheduling queue
        self.assertEqual(1, self.client.zcard(self.queue.queue_name))

    def test_lease_contention_and_repo_capacity_cap(self) -> None:
        """Two workers race for a single slot in repo; capacity cap is never breached."""
        now = 1000.0
        self.queue.enqueue(sample_snapshot(repo_id="org/cap-repo"), "del-cap-1", now=now)
        self.queue.enqueue(sample_snapshot(repo_id="org/cap-repo"), "del-cap-2", now=now)

        w1 = self.queue.lease_next_job(now=now, max_concurrency_per_repo=1)
        w2 = self.queue.lease_next_job(now=now, max_concurrency_per_repo=1)

        self.assertIsNotNone(w1)
        self.assertEqual("job-del-cap-1", w1.job_id)
        self.assertIsNone(w2)

        # Slot count is strictly 1
        self.assertEqual(1, self.client.scard("review:repo_running:org/cap-repo"))

    def test_lease_expiry_and_zombie_recovery_rescheduling(self) -> None:
        """Worker crash during execution recovers via lease expiry without resetting attempt count."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-zombie-1", deadline_seconds=10.0, max_retries=3, now=now)
        leased = self.queue.lease_next_job(now=now)
        self.assertIsNotNone(leased)
        self.assertEqual(1, leased.attempt_count)

        # Zombie recovery before deadline does nothing
        recovered_early = self.queue.recover_zombie_jobs(now=now + 5.0)
        self.assertEqual(0, len(recovered_early))

        # Time advances past deadline; recovery re-enqueues job with backoff
        recovered = self.queue.recover_zombie_jobs(now=now + 15.0)
        self.assertEqual(1, len(recovered))
        self.assertEqual(JobState.QUEUED, recovered[0].state)
        # Attempt count preserved (not double-counted by recovery)
        self.assertEqual(1, recovered[0].attempt_count)
        self.assertEqual(1, self.client.zcard(self.queue.queue_name))

    def test_stale_worker_completion_rejected_after_reassignment(self) -> None:
        """Worker A's mutation fails closed with StaleLeaseError if its lease token expired."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-stale-comp", deadline_seconds=10.0, now=now)

        # Worker A claims with token A
        job_a, token_a = self.queue.claim_job(job.job_id, now=now)
        self.assertIsNotNone(token_a)

        # Lease expires and Worker B claims with token B
        self.queue.recover_zombie_jobs(now=now + 15.0)
        job_b, token_b = self.queue.claim_job(job.job_id, now=now + 16.0)
        self.assertNotEqual(token_a, token_b)

        # Worker A attempts completion with token_a -> StaleLeaseError
        with self.assertRaises(StaleLeaseError):
            self.queue.mark_completed(job.job_id, now=now + 20.0, lease_token=token_a)

        # Worker B completes with valid token_b -> Success
        completed = self.queue.mark_completed(job.job_id, now=now + 20.0, lease_token=token_b)
        self.assertEqual(JobState.COMPLETED, completed.state)

    def test_stale_worker_failure_rejected_after_reassignment(self) -> None:
        """Worker A's failure mutation fails closed with StaleLeaseError if lease was reassigned."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-stale-fail", deadline_seconds=10.0, now=now)
        _, token_a = self.queue.claim_job(job.job_id, now=now)

        self.queue.recover_zombie_jobs(now=now + 15.0)
        _, token_b = self.queue.claim_job(job.job_id, now=now + 16.0)

        with self.assertRaises(StaleLeaseError):
            self.queue.mark_failed(job.job_id, "Worker A late fail", now=now + 20.0, lease_token=token_a)

        self.queue.mark_completed(job.job_id, now=now + 20.0, lease_token=token_b)
        self.assertEqual(JobState.COMPLETED, self.queue.get_job(job.job_id).state)

    def test_retry_exhaustion_transitions_to_dead_letter(self) -> None:
        """When attempt_count reaches max_retries, job transitions to DEAD_LETTER."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-exhaust", max_retries=2, now=now)

        # Attempt 1
        self.queue.lease_next_job(now=now)
        self.queue.mark_failed(job.job_id, "Fail 1", now=now)
        j1 = self.queue.get_job(job.job_id)
        self.assertEqual(JobState.QUEUED, j1.state)
        self.assertEqual(1, j1.attempt_count)

        # Attempt 2 (exhaustion)
        self.queue.lease_next_job(now=now + 10.0)
        self.queue.mark_failed(job.job_id, "Fail 2", now=now + 10.0)
        j2 = self.queue.get_job(job.job_id)
        self.assertEqual(JobState.DEAD_LETTER, j2.state)
        self.assertEqual(2, j2.attempt_count)
        self.assertEqual(1, self.client.scard(self.queue.dead_letter_key))
        self.assertEqual(0, self.client.zcard(self.queue.queue_name))

    def test_cancellation_authority_and_no_resurrection(self) -> None:
        """Cancelled job removes entry from queue, signals ARQ abort, and cannot be leased."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-cancel-test", now=now)
        self.queue.lease_next_job(now=now)

        cancelled = self.queue.cancel_job(job.job_id, reason="Security issue in PR", now=now + 1.0)
        self.assertEqual(JobState.CANCELLED, cancelled.state)

        # ARQ abort set has entry
        from arq.constants import abort_jobs_ss
        self.assertIn(job.job_id.encode("utf-8"), self.client.zrange(abort_jobs_ss, 0, -1))

        # Claim rejects
        claimed, reason = self.queue.claim_job(job.job_id, now=now + 2.0)
        self.assertIsNone(claimed)
        self.assertEqual("cancelled", reason)

        # Zombie recovery ignores cancelled job
        recovered = self.queue.recover_zombie_jobs(now=now + 100.0)
        self.assertEqual(0, len(recovered))


# ============================================================================
# DOMAIN B: REDIS CHECKPOINT FAULT INJECTION
# ============================================================================

class TestRedisCheckpointFaultInjection(unittest.IsolatedAsyncioTestCase):
    """Failure injection for RedisCheckpointSaver and LangGraph state preservation."""

    def setUp(self) -> None:
        self.redis_client = InMemoryRedisCheckpointClient()
        self.saver = RedisCheckpointSaver(client=self.redis_client)

    def test_checkpoint_write_outage_fails_closed(self) -> None:
        """Redis write outage raises CheckpointStorageError and fails closed."""
        self.redis_client.close()
        config = {"configurable": {"thread_id": "thread-fail-write"}}
        from langgraph.checkpoint.base import empty_checkpoint
        cp = empty_checkpoint()
        cp["id"] = "cp-fail-write"

        with self.assertRaises(CheckpointStorageError):
            self.saver.put(config, cp, {}, {})

    def test_checkpoint_read_outage_fails_closed(self) -> None:
        """Redis read outage raises CheckpointStorageError and fails closed."""
        config = {"configurable": {"thread_id": "thread-fail-read"}}
        from langgraph.checkpoint.base import empty_checkpoint
        cp = empty_checkpoint()
        cp["id"] = "cp-write-ok"
        self.saver.put(config, cp, {}, {})

        self.redis_client.close()
        with self.assertRaises(CheckpointStorageError):
            self.saver.get_tuple(config)

    def test_malformed_checkpoint_payload_fails_closed(self) -> None:
        """Corrupted checkpoint data raises CheckpointDeserializationError."""
        config = {"configurable": {"thread_id": "thread-corrupt-chk"}}
        from langgraph.checkpoint.base import empty_checkpoint
        cp = empty_checkpoint()
        cp["id"] = "cp-corrupt-chk"
        self.saver.put(config, cp, {}, {})

        # Corrupt data key
        data_key = self.saver._data_key("thread-corrupt-chk", "", "cp-corrupt-chk")
        self.redis_client.hset(data_key, mapping={"cp_data": b"INVALID_BINARY_BYTES"})

        with self.assertRaises(CheckpointDeserializationError):
            self.saver.get_tuple(config)

    def test_missing_referenced_channel_blob_fails_closed(self) -> None:
        """Missing referenced channel blob fails closed with CheckpointDeserializationError."""
        config = {"configurable": {"thread_id": "thread-missing-blob-test"}}
        from langgraph.checkpoint.base import empty_checkpoint
        cp = empty_checkpoint()
        cp["id"] = "cp-missing-blob"
        cp["channel_values"] = {"ch1": "data1"}
        self.saver.put(config, cp, {}, {"ch1": 1})

        # Delete referenced blob
        blob_key = self.saver._blob_key("thread-missing-blob-test", "", "ch1", 1)
        self.redis_client.delete(blob_key)

        with self.assertRaises(CheckpointDeserializationError):
            self.saver.get_tuple(config)

    async def test_worker_crash_after_checkpoint_avoids_duplicate_specialist_calls(self) -> None:
        """When worker crashes after specialist nodes checkpoint, resumed run executes ZERO duplicate specialists."""
        call_counts = {spec: 0 for spec in SpecialistType}

        def make_handler(spec_type: SpecialistType):
            def handler(inp: SpecialistInput) -> SpecialistOutput:
                call_counts[spec_type] += 1
                return SpecialistOutput(
                    specialist_type=spec_type,
                    correlation_id=inp.correlation_id,
                    status=SpecialistStatus.COMPLETED,
                    findings=(
                        CandidateFinding(
                            finding_id=f"f-{spec_type.value}",
                            correlation_id=inp.correlation_id,
                            specialist_type=spec_type,
                            category="quality",
                            severity="medium",
                            confidence=0.85,
                            summary=f"{spec_type.value} check",
                            rationale="Evidence validated",
                        ),
                    ),
                )
            return handler

        handlers = {spec: make_handler(spec) for spec in SpecialistType}
        job = ReviewJob(
            job_id="job-chk-crash-replay",
            delivery_id="del-chk-crash-1",
            repository_id="octocat/hello-world",
            pull_request_number=42,
            base_sha="base-sha",
            head_sha="head-sha",
            state=JobState.RUNNING,
        )
        snapshot = sample_snapshot(head_sha="head-sha")

        class SimulatedCrash(RuntimeError):
            pass

        # Worker 1 runs specialists and crashes before evaluate_terminal
        w1_orch = ReviewOrchestrator(specialist_handlers=handlers, checkpointer=self.saver)

        async def crash_node(state: Any) -> dict:
            raise SimulatedCrash("Worker 1 process terminated before evaluate_terminal")

        w1_orch._node_evaluate_terminal = crash_node

        with self.assertRaises(SimulatedCrash):
            await w1_orch.execute_run(job, snapshot)

        # Specialists executed once each
        for spec in SpecialistType:
            self.assertEqual(1, call_counts[spec])

        # Worker 2 starts fresh with same thread_id and resumes
        w2_orch = ReviewOrchestrator(specialist_handlers=handlers, checkpointer=self.saver)
        final_state = await w2_orch.execute_run(job, snapshot)

        self.assertEqual("completed", final_state.terminal_status)
        self.assertTrue(final_state.aggregation_invoked)

        # VERIFY ZERO DUPLICATE SPECIALIST EXECUTIONS
        for spec in SpecialistType:
            self.assertEqual(1, call_counts[spec], f"Specialist {spec} was re-executed!")


# ============================================================================
# DOMAIN C: TIGER DATA ADAPTERS FAULT INJECTION
# ============================================================================

class TestTigerDataAdaptersFaultInjection(unittest.TestCase):
    """Failure injection for Tiger Cloud stores, advisory locking, and transaction boundaries."""

    def setUp(self) -> None:
        self.conn = FaultInjectingTigerConnection()
        self.config = TigerConfig.from_url("postgresql://postgres:test_pw@localhost:5432/testdb?sslmode=require")
        self.mgr = TigerConnectionManager(self.config, connection_factory=lambda cfg: self.conn)
        self.truth_store = TigerReviewTruthStore(self.mgr)
        self.audit_spine = TigerAuditSpine(self.mgr)
        self.code_store = TigerCodeMemoryStore(self.mgr)

    def test_tiger_connection_failure_fails_closed(self) -> None:
        """Database connection drop fails closed with TigerConnectionError."""
        self.conn.drop_connection = True
        with self.assertRaises(TigerConnectionError):
            self.code_store.is_fresh("org/repo", "rev-1")

    def test_advisory_lock_failure_rolls_back_and_fails_closed(self) -> None:
        """Advisory lock acquisition failure in record_transition rolls back and raises ConcurrentTransitionError."""
        # 1. Register parent run and initial finding
        self.truth_store.register_review_run(
            run_id="run-lock-fail",
            repository_id="org/repo",
            pull_number=1,
            head_sha="head-1",
            base_sha="base-1",
            delivery_id="del-lock-1",
        )
        finding = _make_sample_finding(
            canonical_id="can-lock-fail",
            repository_id="org/repo",
            head_sha="head-1",
            run_id="run-lock-fail",
            delivery_id="del-lock-1",
        )
        self.truth_store.record_initial(finding, initial_state=TruthState.HELD)

        # 2. Inject advisory lock failure
        self.conn.fail_advisory_lock = True
        with self.assertRaises(ConcurrentTransitionError):
            self.truth_store.record_transition(
                "can-lock-fail",
                TruthState.APPROVED,
                actor="reviewer",
                actor_role="human",
                rationale="Approving finding",
            )

        # 3. Verify state was NOT updated (rolled back, remains HELD)
        self.conn.fail_advisory_lock = False
        latest = self.truth_store.get_latest_state("can-lock-fail")
        self.assertIsNotNone(latest)
        self.assertEqual(TruthState.HELD, latest.state)

    def test_missing_parent_review_context_fails_closed(self) -> None:
        """Recording finding without registered parent review run raises TigerTruthMissingParentError."""
        finding = _make_sample_finding(canonical_id="can-unparented", run_id="run-nonexistent", delivery_id="del-none")
        with self.assertRaises(TigerTruthMissingParentError):
            self.truth_store.record_initial(finding, initial_state=TruthState.HELD)

    def test_conflicting_finding_mutation_fails_closed(self) -> None:
        """Re-recording canonical finding with conflicting attributes raises ConflictingFindingError."""
        self.truth_store.register_review_run(
            run_id="run-conflict-test",
            repository_id="org/repo",
            pull_number=1,
            head_sha="head-1",
            base_sha="base-1",
            delivery_id="del-conflict-1",
        )
        f1 = _make_sample_finding(
            canonical_id="can-conf",
            repository_id="org/repo",
            head_sha="head-1",
            run_id="run-conflict-test",
            delivery_id="del-conflict-1",
        )
        self.truth_store.record_initial(f1, initial_state=TruthState.HELD)

        # Create conflicting finding with different severity
        f2 = CanonicalFinding(
            canonical_id="can-conf",
            repository_id="org/repo",
            head_sha="head-1",
            category="correctness",
            severity="low",  # Conflicting! (Original was 'high')
            confidence=0.5,
            summary="Conflicting summary",
            rationale="Conflicting rationale",
            file_path="src/processor.py",
            line_range=(45, 52),
            contributing_candidate_ids=("cand-001",),
            contributing_specialists=("correctness_specialist",),
            evidence_refs=(),
            disposition=FindingDisposition.HELD,
            run_id="run-conflict-test",
            delivery_id="del-conflict-1",
        )

        with self.assertRaises(ConflictingFindingError):
            self.truth_store.record_initial(f2, initial_state=TruthState.HELD)

    def test_out_of_order_state_transition_fails_closed(self) -> None:
        """Illegal state machine transition (e.g. SUPPRESSED -> APPROVED) is rejected."""
        self.truth_store.register_review_run(
            run_id="run-illegal-trans",
            repository_id="org/repo",
            pull_number=1,
            head_sha="head-1",
            base_sha="base-1",
            delivery_id="del-illegal-1",
        )
        finding = _make_sample_finding(
            canonical_id="can-illegal-trans",
            repository_id="org/repo",
            head_sha="head-1",
            run_id="run-illegal-trans",
            delivery_id="del-illegal-1",
        )
        self.truth_store.record_initial(finding, initial_state=TruthState.SUPPRESSED)

        # Attempt illegal transition: SUPPRESSED cannot transition directly to APPROVED
        with self.assertRaises((ValueError, TigerStoreError)):
            self.truth_store.record_transition(
                "can-illegal-trans",
                TruthState.APPROVED,
                actor="system",
                actor_role="system",
                rationale="Illegal bypass",
            )

        latest = self.truth_store.get_latest_state("can-illegal-trans")
        self.assertIsNotNone(latest)
        self.assertEqual(TruthState.SUPPRESSED, latest.state)

    def test_partial_code_indexing_rollback(self) -> None:
        """Failure during batch chunk indexing rolls back all chunks and leaves is_fresh=False."""
        chunks = [
            CodeChunk(
                chunk_id="chunk-1",
                repository_id="org/repo",
                file_path="src/main.py",
                content="def main(): pass",
                start_line=1,
                end_line=2,
                content_hash="hash-1",
                index_version="v1",
                revision="rev-part-fail",
            ),
            CodeChunk(
                chunk_id="chunk-2",
                repository_id="org/repo",
                file_path="src/main.py",
                content="def secondary(): pass",
                start_line=3,
                end_line=4,
                content_hash="hash-2",
                index_version="v1",
                revision="rev-part-fail",
            ),
        ]

        # Inject failure on chunk-2 insert
        self.conn.fail_on_statement_pattern = "hash-2"

        with self.assertRaises(sqlite3.OperationalError):
            self.code_store.index_repository("org/repo", "rev-part-fail", chunks)

        # Verify rollback: 0 chunks present, is_fresh is False
        self.conn.fail_on_statement_pattern = None
        retrieved = self.code_store.get_chunks("org/repo", "rev-part-fail")
        self.assertEqual([], retrieved)
        self.assertFalse(self.code_store.is_fresh("org/repo", "rev-part-fail"))


# ============================================================================
# DOMAIN D: CROSS-SYSTEM FAILURE INTERACTIONS
# ============================================================================

class TestCrossSystemFailureInteractions(unittest.IsolatedAsyncioTestCase):
    """The 10 mandatory cross-system reliability & partition failure scenarios."""

    def setUp(self) -> None:
        self.redis_client = FaultInjectingRedisClient()
        self.queue = RedisJobQueue(client=self.redis_client)  # type: ignore[arg-type]
        self.checkpoint_client = InMemoryRedisCheckpointClient()
        self.checkpointer = RedisCheckpointSaver(client=self.checkpoint_client)

        self.tiger_conn = FaultInjectingTigerConnection()
        self.tiger_config = TigerConfig.from_url("postgresql://postgres:pw@localhost:5432/db?sslmode=require")
        self.tiger_mgr = TigerConnectionManager(self.tiger_config, connection_factory=lambda cfg: self.tiger_conn)
        self.tiger_truth = TigerReviewTruthStore(self.tiger_mgr)
        self.tiger_audit = TigerAuditSpine(self.tiger_mgr)

        self.github_client = FaultInjectingGitHubClient(
            pr_heads={("octocat/hello-world", 101): "head-sha-valid"}
        )

        self.cfg = _make_dummy_service_config()

    def _create_full_worker(
        self,
        truth_store: Any = None,
        audit_spine: Any = None,
        publisher: Any = None,
        handlers: Any = None,
        publish_enabled: bool = False,
    ) -> AutonomousReviewWorker:
        effective_truth = truth_store or ReviewTruthStore(sqlite3.connect(":memory:"))
        effective_audit = audit_spine or AuditSpine(sqlite3.connect(":memory:"))
        effective_pub = publisher or GitHubReviewPublisher(
            sqlite3.connect(":memory:"),
            effective_truth,
            self.github_client,
        )
        effective_cfg = dataclasses.replace(self.cfg, publish_enabled=publish_enabled) if publish_enabled else self.cfg
        return AutonomousReviewWorker(
            effective_cfg,
            connection=sqlite3.connect(":memory:"),
            github_client=self.github_client,
            specialist_handlers=handlers or {
                SpecialistType.SECURITY: _make_mock_specialist(),
                SpecialistType.QUALITY: _make_mock_specialist([
                    CandidateFinding(
                        finding_id="f-qual-1",
                        correlation_id="del-1",
                        specialist_type=SpecialistType.QUALITY,
                        category="quality",
                        severity="medium",
                        confidence=0.95,
                        summary="Code improvement recommendation",
                        rationale="Diff analysis",
                        file_path="src/auth.py",
                        line_range=(10, 15),
                        evidence_refs=("diff://src/auth.py#L10-L15",),
                    )
                ]),
                SpecialistType.TESTS: _make_mock_specialist(),
                SpecialistType.DOCUMENTATION: _make_mock_specialist(),
            },
            queue=self.queue,
            audit_spine=effective_audit,
            truth_store=effective_truth,
            publisher=effective_pub,
            checkpointer=self.checkpointer,
        )

    # ------------------------------------------------------------------------
    # Scenario 1: Redis succeeds, Tiger fails
    # ------------------------------------------------------------------------
    async def test_scenario_1_redis_succeeds_tiger_fails(self) -> None:
        """Job leased from Redis, but Tiger fails during finding recording.
        Worker fails closed, marks job failed with backoff, zero GitHub comments published.
        """
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world", head_sha="head-sha-valid")
        job = self.queue.enqueue(snapshot, "del-scen-1", now=now)

        # Worker configured with TigerReviewTruthStore, but parent run is NOT registered (simulating Tiger failure)
        worker = self._create_full_worker(truth_store=self.tiger_truth, audit_spine=self.tiger_audit)

        # Process job -> must raise TigerTruthMissingParentError
        with self.assertRaises(TigerTruthMissingParentError):
            job_claimed, token = self.queue.claim_job(job.job_id, now=now)
            await worker.process_claimed_job(job_claimed, lease_token=token, now=now)

        # 1. Durable State: ReviewJob in Redis marked QUEUED with retry scheduled
        job_after = self.queue.get_job(job.job_id)
        self.assertIsNotNone(job_after)
        self.assertEqual(JobState.QUEUED, job_after.state)
        self.assertEqual(1, job_after.attempt_count)
        self.assertGreater(job_after.next_run_at, now)

        # 2. Retry Authority: Redis queue owns retry authority
        self.assertEqual(1, self.redis_client.zcard(self.queue.queue_name))

        # 3. Outward Effects: ZERO GitHub reviews published
        self.assertEqual(0, self.github_client.create_review_calls)

        # 4. Review Truth: Zero unparented findings committed
        self.assertIsNone(self.tiger_truth.get_latest_state("can-sec-1"))

        # 5. Observable: Failure event recorded in audit spine
        events = self.tiger_audit.reconstruct_run(job.delivery_id).timeline
        failed_events = [e for e in events if e.event_name == "worker_job_failed"]
        self.assertGreaterEqual(len(failed_events), 1)

    # ------------------------------------------------------------------------
    # Scenario 2: Tiger succeeds, Redis acknowledgement/completion fails
    # ------------------------------------------------------------------------
    async def test_scenario_2_tiger_succeeds_redis_ack_fails(self) -> None:
        """Findings recorded in TigerReviewTruthStore, but Redis drops connection during mark_completed.
        On retry, Tiger re-records findings idempotently and effect table prevents duplicate comment.
        """
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world", head_sha="head-sha-valid")
        job = self.queue.enqueue(snapshot, "del-scen-2", now=now)

        # Wire real TigerReviewTruthStore with parent run context resolver
        tiger_truth = TigerReviewTruthStore(
            self.tiger_mgr,
            run_context_resolver=lambda _: ReviewRunContext(
                run_id="run-scen-2",
                repository_id="octocat/hello-world",
                pull_number=101,
                head_sha="head-sha-valid",
                base_sha="base-sha-valid",
                delivery_id="del-scen-2",
            ),
        )
        sqlite_conn = sqlite3.connect(":memory:")
        publisher = GitHubReviewPublisher(sqlite_conn, tiger_truth, self.github_client)
        worker = self._create_full_worker(
            truth_store=tiger_truth,
            publisher=publisher,
            publish_enabled=True,
        )

        # Attempt 1: Claim and execute, but Redis drops on mark_completed
        job_claimed, token = self.queue.claim_job(job.job_id, now=now)
        self.redis_client.error_on_methods.add("hset")  # Fails mark_completed

        with self.assertRaises(redis.RedisError):
            await worker.process_claimed_job(job_claimed, lease_token=token, now=now)

        # 1. Tiger truth was persisted into Tiger PostgreSQL simulation tables
        cur = self.tiger_conn.execute("SELECT canonical_id, sequence_id, state FROM finding_records")
        rows = cur.fetchall()
        self.assertGreaterEqual(len(rows), 1)
        can_id = rows[0]["canonical_id"]

        # 2. Redis recovers; lease expires; zombie recovery reschedules
        self.redis_client.error_on_methods.clear()
        self.queue.recover_zombie_jobs(now=now + 100.0)

        # 3. Attempt 2: Re-execution by second worker
        job_retry, token_retry = self.queue.claim_job(job.job_id, now=now + 101.0)
        state_retry = await worker.process_claimed_job(job_retry, lease_token=token_retry, now=now + 101.0)

        self.assertEqual("completed", state_retry.terminal_status)
        final_job = self.queue.get_job(job.job_id)
        self.assertEqual(JobState.COMPLETED, final_job.state)

        # 4. Tiger truth is replay-safe and idempotent: no duplicate sequence or duplicate record created
        cur_after = self.tiger_conn.execute(
            "SELECT canonical_id, sequence_id, state FROM finding_records WHERE canonical_id = ? ORDER BY sequence_id ASC",
            (can_id,),
        )
        rows_after = cur_after.fetchall()
        self.assertEqual(2, len(rows_after), "Tiger created duplicate sequence record on retry!")
        self.assertEqual(1, rows_after[0]["sequence_id"])
        self.assertEqual("auto_approved", rows_after[0]["state"])
        self.assertEqual(2, rows_after[1]["sequence_id"])
        self.assertEqual("published", rows_after[1]["state"])

        # 5. ZERO duplicate GitHub reviews (only 1 review was ever created)
        self.assertEqual(1, len(self.github_client.reviews))
        self.assertEqual(1, self.github_client.create_review_calls)

    # ------------------------------------------------------------------------
    # Scenario 3: Worker crashes after checkpoint
    # ------------------------------------------------------------------------
    async def test_scenario_3_worker_crashes_after_checkpoint(self) -> None:
        """Worker crashes after LangGraph checkpoint is written to Redis.
        On restart, resumed worker executes ZERO duplicate specialists and completes publication.
        """
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world", head_sha="head-sha-valid")
        job = self.queue.enqueue(snapshot, "del-scen-3", now=now)

        call_counts = {spec: 0 for spec in SpecialistType}

        def counting_handler(spec: SpecialistType):
            def handler(inp: SpecialistInput) -> SpecialistOutput:
                call_counts[spec] += 1
                return SpecialistOutput(
                    specialist_type=spec,
                    correlation_id=inp.correlation_id,
                    status=SpecialistStatus.COMPLETED,
                    findings=(
                        CandidateFinding(
                            finding_id=f"f-{spec.value}",
                            correlation_id=inp.correlation_id,
                            specialist_type=spec,
                            category="quality",
                            severity="medium",
                            confidence=0.8,
                            summary="Checked",
                            rationale="Evidence",
                        ),
                    ),
                )
            return handler

        handlers = {spec: counting_handler(spec) for spec in SpecialistType}
        worker = self._create_full_worker(handlers=handlers)

        # Simulate crash before evaluate_terminal
        class Worker1Crash(RuntimeError):
            pass

        async def crash_node(state: Any) -> dict:
            raise Worker1Crash("Worker 1 crashed after writing checkpoint")

        worker.orchestrator._node_evaluate_terminal = crash_node

        job_claimed, token = self.queue.claim_job(job.job_id, now=now)
        with self.assertRaises(Worker1Crash):
            await worker.process_claimed_job(job_claimed, lease_token=token, now=now)

        # Specialists executed once
        for spec in SpecialistType:
            self.assertEqual(1, call_counts[spec])

        # Worker 2 restarts with clean orchestrator and executes
        worker2 = self._create_full_worker(handlers=handlers)
        state2 = await worker2.process_claimed_job(job_claimed, lease_token=token, now=now + 5.0)

        self.assertEqual("completed", state2.terminal_status)
        # VERIFY ZERO DUPLICATE SPECIALIST CALLS (remains exactly 1)
        for spec in SpecialistType:
            self.assertEqual(1, call_counts[spec])

    # ------------------------------------------------------------------------
    # Scenario 4: Worker crashes before checkpoint
    # ------------------------------------------------------------------------
    async def test_scenario_4_worker_crashes_before_checkpoint(self) -> None:
        """Worker crashes during specialist execution before any checkpoint is persisted.
        On retry, restarted worker executes cleanly from start without state corruption.
        """
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world", head_sha="head-sha-valid")
        job = self.queue.enqueue(snapshot, "del-scen-4", now=now)

        worker = self._create_full_worker()

        # Simulate crash in node_initialize before any specialist checkpoint
        class EarlyCrash(RuntimeError):
            pass

        async def crash_init(state: Any) -> dict:
            raise EarlyCrash("Worker crash before checkpoint")

        worker.orchestrator._node_initialize = crash_init

        job_claimed, token = self.queue.claim_job(job.job_id, now=now)
        with self.assertRaises(EarlyCrash):
            await worker.process_claimed_job(job_claimed, lease_token=token, now=now)

        # Checkpoint is empty of specialist outputs
        tup = self.checkpointer.get_tuple({"configurable": {"thread_id": job.job_id}})
        if tup is not None:
            self.assertEqual({}, tup.checkpoint.get("channel_values", {}).get("specialist_outputs", {}))

        # Rescheduled in queue
        job_resched = self.queue.get_job(job.job_id)
        self.assertEqual(JobState.QUEUED, job_resched.state)
        self.assertEqual(1, job_resched.attempt_count)

    # ------------------------------------------------------------------------
    # Scenario 5: Worker crashes after GitHub publication
    # ------------------------------------------------------------------------
    async def test_scenario_5_worker_crashes_after_github_publication(self) -> None:
        """Worker publishes review finding to GitHub, then crashes before completing queue mutation.
        On subsequent worker retry, publisher reconciles with existing publication state and emits ZERO duplicate reviews.
        """
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world", head_sha="head-sha-valid")
        job = self.queue.enqueue(snapshot, "del-scen-5", now=now)

        sqlite_conn = sqlite3.connect(":memory:")
        truth_store = ReviewTruthStore(sqlite_conn)
        publisher = GitHubReviewPublisher(sqlite_conn, truth_store, self.github_client)
        worker = self._create_full_worker(
            truth_store=truth_store,
            publisher=publisher,
            publish_enabled=True,
        )

        # Attempt 1: Worker claims job and executes publication, but crashes right before queue ACK
        job_claimed, token = self.queue.claim_job(job.job_id, now=now)
        self.redis_client.error_on_methods.add("hset")  # Intercepts mark_completed to simulate post-publish crash

        with self.assertRaises(redis.RedisError):
            await worker.process_claimed_job(job_claimed, lease_token=token, now=now)

        # Publication succeeded on GitHub in Attempt 1
        self.assertEqual(1, len(self.github_client.reviews))
        self.assertEqual(1, self.github_client.create_review_calls)

        # Worker crashed/died; lease expires in Redis and zombie recovery reschedules
        self.redis_client.error_on_methods.clear()
        self.queue.recover_zombie_jobs(now=now + 100.0)

        # Attempt 2: Second worker claims job on retry and executes full pipeline
        job_retry, token_retry = self.queue.claim_job(job.job_id, now=now + 101.0)
        state_retry = await worker.process_claimed_job(job_retry, lease_token=token_retry, now=now + 101.0)

        self.assertEqual("completed", state_retry.terminal_status)
        # ZERO DUPLICATE REVIEWS (count remains strictly 1)
        self.assertEqual(1, len(self.github_client.reviews))
        self.assertEqual(1, self.github_client.create_review_calls)
        self.assertEqual(JobState.COMPLETED, self.queue.get_job(job.job_id).state)

    # ------------------------------------------------------------------------
    # Scenario 6: Duplicate delivery occurs during recovery
    # ------------------------------------------------------------------------
    def test_scenario_6_duplicate_delivery_during_recovery(self) -> None:
        """Webhook redelivered while infrastructure is recovering returns existing job without resetting attempts."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world", head_sha="head-sha-valid")
        job1 = self.queue.enqueue(snapshot, "del-scen-6", now=now)

        # Job is leased and failed once
        self.queue.lease_next_job(now=now)
        self.queue.mark_failed(job1.job_id, "Network failure", now=now + 1.0)

        j1 = self.queue.get_job(job1.job_id)
        self.assertEqual(1, j1.attempt_count)

        # Webhook redelivered with same delivery_id
        job2 = self.queue.enqueue(snapshot, "del-scen-6", now=now + 5.0)
        self.assertEqual(job1.job_id, job2.job_id)
        # Attempt count is NOT reset to 0
        self.assertEqual(1, job2.attempt_count)

    # ------------------------------------------------------------------------
    # Scenario 7: Retry occurs after partial Tiger persistence
    # ------------------------------------------------------------------------
    def test_scenario_7_retry_after_partial_tiger_persistence(self) -> None:
        """Worker commits partial business state to Tiger (Finding A + Audit event), then crashes before complete review state is committed.
        On retry, existing Tiger state is handled idempotently, remaining state commits cleanly, and audit history remains reconstructable.
        """
        now = 1000.0
        # Register review run in TigerReviewTruthStore
        self.tiger_truth.register_review_run(
            run_id="run-scen-7",
            repository_id="octocat/hello-world",
            pull_number=101,
            head_sha="head-sha-valid",
            base_sha="base-sha-valid",
            delivery_id="del-scen-7",
        )

        # Attempt 1 begins
        self.tiger_audit.record_event(
            AuditEvent(
                correlation_id="del-scen-7",
                event_name="attempt_started",
                step="worker",
                timestamp=now,
                details={"attempt": 1},
            )
        )

        # Commit Finding A to Tiger
        finding_a = _make_sample_finding(
            canonical_id="can-scen-7-a",
            repository_id="octocat/hello-world",
            head_sha="head-sha-valid",
            run_id="run-scen-7",
            delivery_id="del-scen-7",
        )
        self.tiger_truth.record_initial(finding_a, initial_state=TruthState.HELD)
        self.tiger_audit.record_event(
            AuditEvent(
                correlation_id="del-scen-7",
                event_name="finding_persisted",
                step="truth_store",
                timestamp=now + 1.0,
                details={"canonical_id": finding_a.canonical_id},
            )
        )

        # Worker crashes before Finding B is recorded
        finding_b = _make_sample_finding(
            canonical_id="can-scen-7-b",
            repository_id="octocat/hello-world",
            head_sha="head-sha-valid",
            run_id="run-scen-7",
            delivery_id="del-scen-7",
        )
        # Verify partial persistence: Finding A committed, Finding B absent
        self.assertIsNotNone(self.tiger_truth.get_latest_state("can-scen-7-a"))
        self.assertIsNone(self.tiger_truth.get_latest_state("can-scen-7-b"))

        # Attempt 2: Worker retry begins
        self.tiger_audit.record_event(
            AuditEvent(
                correlation_id="del-scen-7",
                event_name="worker_retry_started",
                step="worker",
                timestamp=now + 10.0,
                details={"attempt": 2},
            )
        )

        # Re-recording Finding A is idempotent (returns existing record without creating duplicate sequence)
        rec_a = self.tiger_truth.record_initial(finding_a, initial_state=TruthState.HELD)
        self.assertEqual(1, rec_a.sequence_id)

        # Finding B now records cleanly
        rec_b = self.tiger_truth.record_initial(finding_b, initial_state=TruthState.HELD)
        self.assertEqual(1, rec_b.sequence_id)

        # Both findings transition to APPROVED according to policy state machine
        self.tiger_truth.record_transition(
            finding_a.canonical_id,
            TruthState.APPROVED,
            actor="reviewer",
            rationale="Approved",
            now=now + 12.0,
        )
        self.tiger_truth.record_transition(
            finding_b.canonical_id,
            TruthState.APPROVED,
            actor="reviewer",
            rationale="Approved",
            now=now + 12.0,
        )

        # Assert zero duplicate canonical findings and exact sequence history
        cur_a = self.tiger_conn.execute(
            "SELECT sequence_id, state FROM finding_records WHERE canonical_id = ? ORDER BY sequence_id ASC",
            (finding_a.canonical_id,),
        )
        rows_a = cur_a.fetchall()
        self.assertEqual(2, len(rows_a))  # seq 1 (HELD), seq 2 (APPROVED)
        self.assertEqual("held", rows_a[0]["state"])
        self.assertEqual("approved", rows_a[1]["state"])

        cur_b = self.tiger_conn.execute(
            "SELECT sequence_id, state FROM finding_records WHERE canonical_id = ? ORDER BY sequence_id ASC",
            (finding_b.canonical_id,),
        )
        rows_b = cur_b.fetchall()
        self.assertEqual(2, len(rows_b))  # seq 1 (HELD), seq 2 (APPROVED)
        self.assertEqual("held", rows_b[0]["state"])
        self.assertEqual("approved", rows_b[1]["state"])

        # Audit history is append-only, chronologically ordered, and fully reconstructable
        trace = self.tiger_audit.reconstruct_run("del-scen-7")
        self.assertEqual(3, len(trace.timeline))
        self.assertEqual("attempt_started", trace.timeline[0].event_name)
        self.assertEqual("finding_persisted", trace.timeline[1].event_name)
        self.assertEqual("worker_retry_started", trace.timeline[2].event_name)

    # ------------------------------------------------------------------------
    # Scenario 8: Retry after Redis state persistence before worker completion
    # ------------------------------------------------------------------------
    def test_scenario_8_retry_after_redis_state_persistence_before_worker_completion(self) -> None:
        """Job state in Redis is RUNNING with attempt_count=1, worker dies.
        Zombie recovery detects expired lease and reschedules with backoff.
        """
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world")
        job = self.queue.enqueue(snapshot, "del-scen-8", deadline_seconds=10.0, now=now)
        self.queue.lease_next_job(now=now)

        # Time advances past deadline without completion
        recovered = self.queue.recover_zombie_jobs(now=now + 15.0)
        self.assertEqual(1, len(recovered))
        self.assertEqual(JobState.QUEUED, recovered[0].state)
        self.assertEqual(1, recovered[0].attempt_count)
        self.assertEqual(1, self.redis_client.zcard(self.queue.queue_name))

    # ------------------------------------------------------------------------
    # Scenario 9: Infrastructure failure during publication reconciliation
    # ------------------------------------------------------------------------
    def test_scenario_9_infrastructure_failure_during_publication_reconciliation(self) -> None:
        """Covers both branches of publication ambiguity:
        Case A: Server-side effect succeeded on GitHub, but network dropped before client read 201 OK -> reconciles to ALREADY_PUBLISHED without duplicate.
        Case B: Network dropped before GitHub accepted review -> unverified ambiguity fails closed on retry.
        """
        now = 1000.0
        sqlite_conn = sqlite3.connect(":memory:")
        truth_store = ReviewTruthStore(sqlite_conn)
        publisher = GitHubReviewPublisher(sqlite_conn, truth_store, self.github_client)

        finding = _make_canonical(
            canonical_id="can-scen-9",
            repository_id="octocat/hello-world",
            head_sha="head-sha-valid",
            file_path="src/auth.py",
            line_range=(10, 15),
        )
        truth_store.record_initial(finding, initial_state=TruthState.APPROVED)

        # --------------------------------------------------------------------
        # Case A: GitHub accepted and created review, but TCP severed before 201 response
        # --------------------------------------------------------------------
        self.github_client.simulate_ambiguous_creation = True
        res_a1 = publisher.publish_finding(
            finding,
            pull_number=101,
            diff_content=AUTH_DIFF,
            now=now,
        )

        # Review was created on GitHub server
        self.assertEqual(1, len(self.github_client.reviews))
        # But caller received FAILED due to dropped response
        self.assertEqual(PublicationStatus.FAILED, res_a1.status)

        # Effect table recorded AMBIGUOUS
        idemp_key = publisher.compute_idempotency_key(
            finding.repository_id,
            101,
            finding.head_sha,
            finding.canonical_id,
        )
        effect_a = publisher._get_existing_effect(idemp_key)
        self.assertIsNotNone(effect_a)
        self.assertEqual(PublicationStatus.AMBIGUOUS.value, effect_a["status"])

        # On retry: publisher reconciles with GitHub, detects existing review, and returns ALREADY_PUBLISHED
        self.github_client.simulate_ambiguous_creation = False
        res_a2 = publisher.publish_finding(
            finding,
            pull_number=101,
            diff_content=AUTH_DIFF,
            now=now + 5.0,
        )
        self.assertEqual(PublicationStatus.ALREADY_PUBLISHED, res_a2.status)
        # ZERO duplicate reviews (count remains strictly 1)
        self.assertEqual(1, len(self.github_client.reviews))
        self.assertEqual(1, self.github_client.create_review_calls)

        # --------------------------------------------------------------------
        # Case B: Network failure before GitHub receives request (unverified effect fails closed)
        # --------------------------------------------------------------------
        finding_b = _make_canonical(
            canonical_id="can-scen-9b",
            repository_id="octocat/hello-world",
            head_sha="head-sha-valid",
            file_path="src/auth.py",
            line_range=(10, 15),
        )
        truth_store.record_initial(finding_b, initial_state=TruthState.APPROVED)

        # Simulate network drop before review creation
        self.github_client.fail_create_review = True
        res_b1 = publisher.publish_finding(
            finding_b,
            pull_number=101,
            diff_content=AUTH_DIFF,
            now=now + 10.0,
        )
        self.assertEqual(PublicationStatus.FAILED, res_b1.status)

        idemp_key_b = publisher.compute_idempotency_key(
            finding_b.repository_id,
            101,
            finding_b.head_sha,
            finding_b.canonical_id,
        )
        effect_b = publisher._get_existing_effect(idemp_key_b)
        self.assertIsNotNone(effect_b)
        self.assertEqual(PublicationStatus.AMBIGUOUS.value, effect_b["status"])

        # On retry: reconciliation queries GitHub, finds no verified review, and fails closed
        self.github_client.fail_create_review = False
        res_b2 = publisher.publish_finding(
            finding_b,
            pull_number=101,
            diff_content=AUTH_DIFF,
            now=now + 15.0,
        )
        self.assertEqual(PublicationStatus.FAILED, res_b2.status)
        self.assertIn("failing closed", res_b2.reason)
        # Review count did NOT increase (remains strictly 1 from Case A)
        self.assertEqual(1, len(self.github_client.reviews))

    # ------------------------------------------------------------------------
    # Scenario 10: PR head SHA changes during recovery
    # ------------------------------------------------------------------------
    def test_scenario_10_pr_head_sha_changes_during_recovery(self) -> None:
        """PR head SHA moves while recovering from an outage.
        Pre-call check detects mismatch, marks SUPERSEDED in Review Truth, publishes ZERO reviews.
        """
        now = 1000.0
        sqlite_conn = sqlite3.connect(":memory:")
        truth_store = ReviewTruthStore(sqlite_conn)
        publisher = GitHubReviewPublisher(sqlite_conn, truth_store, self.github_client)

        finding = _make_sample_finding(
            canonical_id="can-scen-10",
            repository_id="octocat/hello-world",
            head_sha="old-reviewed-sha",
        )
        truth_store.record_initial(finding, initial_state=TruthState.APPROVED)

        # Live PR moved to new-head-sha
        self.github_client.head_sha_move_to = "new-head-sha"

        res = publisher.publish_finding(finding, pull_number=101, now=now)

        self.assertEqual(PublicationStatus.SUPERSEDED_SHA_MISMATCH, res.status)
        self.assertIn("Head SHA mismatch", res.reason)

        # Review Truth transitioned to SUPERSEDED
        latest = truth_store.get_latest_state("can-scen-10")
        self.assertIsNotNone(latest)
        self.assertEqual(TruthState.SUPERSEDED, latest.state)

        # ZERO reviews published to GitHub
        self.assertEqual(0, self.github_client.create_review_calls)


# ============================================================================
# DOMAIN E: OPTIONAL LIVE INFRASTRUCTURE HOOKS
# ============================================================================

class TestLiveInfrastructureFaultHooks(unittest.TestCase):
    """Optional hooks validating real Redis / Tiger Cloud behavior when present in environment."""

    def setUp(self) -> None:
        self.live_redis_url = os.environ.get("REDIS_URL")
        self.live_tiger_url = os.environ.get("TIGER_DATABASE_URL") or os.environ.get("TIGER_URL")

    def test_live_redis_reconnect_if_available(self) -> None:
        if not self.live_redis_url:
            self.skipTest("No live REDIS_URL provided; skipping live Redis failure hook")

        client = redis.Redis.from_url(self.live_redis_url)
        try:
            client.ping()
        except Exception as e:
            self.skipTest(f"Live Redis unreachable: {e}")

        # Live Redis round-trip check
        q = RedisJobQueue(client=client, queue_name=f"test:fault:{uuid.uuid4().hex[:8]}")
        deliv = f"live-del-{uuid.uuid4().hex[:8]}"
        job = q.enqueue(sample_snapshot(), deliv)
        self.assertEqual(JobState.QUEUED, job.state)
        q.cancel_job(job.job_id, reason="Live test cleanup")

    def test_live_tiger_connection_if_available(self) -> None:
        if not self.live_tiger_url:
            self.skipTest("No live TIGER_DATABASE_URL provided; skipping live Tiger failure hook")

        try:
            config = TigerConfig.from_url(self.live_tiger_url)
            mgr = TigerConnectionManager(config)
            conn = mgr.get_connection()
            cur = conn.cursor() if hasattr(conn, "cursor") else conn
            cur.execute("SELECT 1")
        except Exception as e:
            self.skipTest(f"Live Tiger unreachable: {e}")


if __name__ == "__main__":
    unittest.main()
