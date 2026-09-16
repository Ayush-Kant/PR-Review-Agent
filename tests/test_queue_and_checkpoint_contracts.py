"""Contract test suites for DurableJobQueue and WorkflowEngine checkpointing.

These tests define the implementation-independent contracts that both the SQLite reference
implementation and future production implementations (Redis/ARQ queue, Redis checkpointer)
must satisfy.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import unittest
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.orchestration import (
    ALL_SPECIALISTS,
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
    WorkflowEngineProtocol,
)


def sample_snapshot(repo_id: str = "repo-contract", head_sha: str = "head-contract-sha") -> ReviewSnapshot:
    return ReviewSnapshot(
        repository_id=repo_id,
        repository_full_name=f"org/{repo_id}",
        pull_request_number=101,
        base_sha="base-contract-sha",
        head_sha=head_sha,
        changed_files=("core.py", "test_core.py"),
        policy_version="policy-v1",
        prompt_version="prompt-v1",
        retrieval_index_version="index-v1",
        model_configuration={"model": "gpt-4o"},
    )


class DurableQueueContractTestSuite:
    """Base reusable contract test suite for any DurableQueueProtocol implementation."""

    def get_queue(self) -> DurableQueueProtocol:
        raise NotImplementedError("Subclasses must implement get_queue()")

    def test_protocol_conformance(self) -> None:
        queue = self.get_queue()
        self.assertIsInstance(queue, DurableQueueProtocol)

    def test_enqueue_and_get_job(self) -> None:
        queue = self.get_queue()
        snapshot = sample_snapshot()
        now = 1000.0

        job = queue.enqueue(
            snapshot,
            delivery_id="delivery-test-1",
            max_retries=3,
            backoff_base_seconds=2.0,
            deadline_seconds=90.0,
            now=now,
        )

        self.assertIsInstance(job, ReviewJob)
        self.assertEqual("job-delivery-test-1", job.job_id)
        self.assertEqual("delivery-test-1", job.delivery_id)
        self.assertEqual(snapshot.repository_id, job.repository_id)
        self.assertEqual(snapshot.pull_request_number, job.pull_request_number)
        self.assertEqual(snapshot.base_sha, job.base_sha)
        self.assertEqual(snapshot.head_sha, job.head_sha)
        self.assertEqual(JobState.QUEUED, job.state)
        self.assertEqual(0, job.attempt_count)
        self.assertEqual(3, job.max_retries)
        self.assertEqual(2.0, job.backoff_base_seconds)
        self.assertEqual(90.0, job.deadline_seconds)

        retrieved = queue.get_job(job.job_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(job.job_id, retrieved.job_id)
        self.assertEqual(JobState.QUEUED, retrieved.state)

    def test_enqueue_idempotency_same_delivery_id(self) -> None:
        queue = self.get_queue()
        snapshot = sample_snapshot()

        job1 = queue.enqueue(snapshot, delivery_id="delivery-idempotent-1")
        job2 = queue.enqueue(snapshot, delivery_id="delivery-idempotent-1")

        self.assertEqual(job1.job_id, job2.job_id)
        self.assertEqual(job1.delivery_id, job2.delivery_id)

    def test_lease_next_job_fifo_and_attempt_increment(self) -> None:
        queue = self.get_queue()
        now = 1000.0
        s1 = sample_snapshot(head_sha="sha-1")
        s2 = sample_snapshot(head_sha="sha-2")

        job1 = queue.enqueue(s1, "delivery-fifo-1", now=now)
        job2 = queue.enqueue(s2, "delivery-fifo-2", now=now + 1.0)

        # First lease should be job1 (earliest created / next_run_at)
        leased1 = queue.lease_next_job(now=now + 2.0)
        self.assertIsNotNone(leased1)
        self.assertEqual(job1.job_id, leased1.job_id)
        self.assertEqual(JobState.RUNNING, leased1.state)
        self.assertEqual(1, leased1.attempt_count)

        # Second lease should be job2
        leased2 = queue.lease_next_job(now=now + 2.0)
        self.assertIsNotNone(leased2)
        self.assertEqual(job2.job_id, leased2.job_id)
        self.assertEqual(JobState.RUNNING, leased2.state)
        self.assertEqual(1, leased2.attempt_count)

    def test_lease_concurrency_cap_per_repo(self) -> None:
        queue = self.get_queue()
        now = 1000.0
        # Enqueue 3 jobs for repo-A, 1 job for repo-B
        queue.enqueue(sample_snapshot(repo_id="repo-A"), "del-A1", now=now)
        queue.enqueue(sample_snapshot(repo_id="repo-A"), "del-A2", now=now + 0.1)
        queue.enqueue(sample_snapshot(repo_id="repo-B"), "del-B1", now=now + 0.2)

        # Lease with cap = 1 for repo-A
        leased1 = queue.lease_next_job(now=now + 1.0, max_concurrency_per_repo=1)
        self.assertIsNotNone(leased1)
        self.assertEqual("repo-A", leased1.repository_id)

        # Next lease must skip repo-A and lease repo-B since repo-A is at capacity
        leased2 = queue.lease_next_job(now=now + 1.0, max_concurrency_per_repo=1)
        self.assertIsNotNone(leased2)
        self.assertEqual("repo-B", leased2.repository_id)

        # Next lease should return None because repo-A is still running at cap 1, and repo-B is also running
        leased3 = queue.lease_next_job(now=now + 1.0, max_concurrency_per_repo=1)
        self.assertIsNone(leased3)

    def test_mark_completed(self) -> None:
        queue = self.get_queue()
        now = 1000.0
        job = queue.enqueue(sample_snapshot(), "del-complete", now=now)
        leased = queue.lease_next_job(now=now)
        self.assertIsNotNone(leased)

        completed = queue.mark_completed(job.job_id, now=now + 10.0)
        self.assertEqual(JobState.COMPLETED, completed.state)

        # Completed job cannot be leased again
        self.assertIsNone(queue.lease_next_job(now=now + 20.0))

    def test_mark_failed_exponential_backoff(self) -> None:
        queue = self.get_queue()
        now = 1000.0
        job = queue.enqueue(
            sample_snapshot(),
            "del-fail-retry",
            max_retries=3,
            backoff_base_seconds=2.0,
            now=now,
        )

        # Attempt 1
        queue.lease_next_job(now=now)
        job_retry1 = queue.mark_failed(job.job_id, error="Transient network blip", now=now)
        self.assertEqual(JobState.QUEUED, job_retry1.state)
        # Expected backoff delay = 2.0 * (2 ** (1 - 1)) = 2.0
        self.assertEqual(now + 2.0, job_retry1.next_run_at)

        # Cannot lease before backoff expires
        self.assertIsNone(queue.lease_next_job(now=now + 1.0))

        # Can lease after backoff expires
        leased_retry1 = queue.lease_next_job(now=now + 2.0)
        self.assertIsNotNone(leased_retry1)
        self.assertEqual(2, leased_retry1.attempt_count)

        # Attempt 2
        job_retry2 = queue.mark_failed(job.job_id, error="Transient error 2", now=now + 2.0)
        # Expected backoff delay = 2.0 * (2 ** (2 - 1)) = 4.0
        self.assertEqual(now + 2.0 + 4.0, job_retry2.next_run_at)

    def test_mark_failed_dead_letter_after_max_retries(self) -> None:
        queue = self.get_queue()
        now = 1000.0
        job = queue.enqueue(
            sample_snapshot(),
            "del-dead-letter",
            max_retries=2,
            backoff_base_seconds=1.0,
            now=now,
        )

        # Attempt 1
        queue.lease_next_job(now=now)
        queue.mark_failed(job.job_id, error="Failure 1", now=now)

        # Attempt 2
        queue.lease_next_job(now=now + 1.0)
        final_job = queue.mark_failed(job.job_id, error="Final terminal failure", now=now + 1.0)

        self.assertEqual(JobState.DEAD_LETTER, final_job.state)
        self.assertEqual("Final terminal failure", final_job.last_error)

        # Dead-lettered job is never leased again
        self.assertIsNone(queue.lease_next_job(now=now + 100.0))

    def test_cancel_job(self) -> None:
        queue = self.get_queue()
        job = queue.enqueue(sample_snapshot(), "del-cancel")
        cancelled = queue.cancel_job(job.job_id, reason="PR closed by author")

        self.assertEqual(JobState.CANCELLED, cancelled.state)
        self.assertEqual("PR closed by author", cancelled.last_error)
        self.assertIsNone(queue.lease_next_job())

    def test_future_scheduled_jobs_not_leased_prematurely(self) -> None:
        queue = self.get_queue()
        now = 1000.0
        job = queue.enqueue(sample_snapshot(), "del-future", now=now)

        # Fail with backoff of 5 seconds
        queue.lease_next_job(now=now)
        queue.mark_failed(job.job_id, error="retry later", now=now)

        self.assertIsNone(queue.lease_next_job(now=now + 0.5))
        self.assertIsNone(queue.lease_next_job(now=now + 0.99))
        self.assertIsNotNone(queue.lease_next_job(now=now + 1.0))

    def test_empty_queue_lease_returns_none(self) -> None:
        queue = self.get_queue()
        self.assertIsNone(queue.lease_next_job())


class SQLiteDurableQueueContractTests(DurableQueueContractTestSuite, unittest.TestCase):
    """Verify SQLite reference implementation satisfies the DurableQueueProtocol contract."""

    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")

    def tearDown(self) -> None:
        self.conn.close()

    def get_queue(self) -> DurableQueueProtocol:
        return DurableJobQueue(self.conn)


class WorkflowCheckpointContractTestSuite(unittest.IsolatedAsyncioTestCase):
    """Verify WorkflowEngine and checkpoint contract behavior."""

    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.queue = DurableJobQueue(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def test_protocol_conformance(self) -> None:
        orchestrator = ReviewOrchestrator()
        self.assertIsInstance(orchestrator, WorkflowEngineProtocol)

    async def test_default_orchestrator_without_checkpointer(self) -> None:
        """Verify default orchestrator runs cleanly without checkpointer for backwards compatibility."""
        orchestrator = ReviewOrchestrator()
        orchestrator.register_specialist(
            SpecialistType.SECURITY,
            lambda inp: SpecialistOutput(
                specialist_type=SpecialistType.SECURITY,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
            ),
        )
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "del-no-checkpointer")

        state = await orchestrator.execute_run(job, snapshot)
        self.assertEqual("completed", state.terminal_status)
        self.assertTrue(state.aggregation_invoked)
        self.assertIsNone(orchestrator.get_workflow_state(job.job_id))

    async def test_orchestrator_with_memory_checkpointer_saves_state(self) -> None:
        """Verify checkpointer seam records execution state in checkpointer at thread_id."""
        checkpointer = MemorySaver()
        orchestrator = ReviewOrchestrator(checkpointer=checkpointer)

        def mock_security(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=SpecialistType.SECURITY,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(
                    CandidateFinding(
                        finding_id="f-sec-chk",
                        correlation_id=inp.correlation_id,
                        specialist_type=SpecialistType.SECURITY,
                        category="security",
                        severity="high",
                        confidence=0.95,
                        summary="SQL injection found",
                        rationale="Unescaped query parameter",
                    ),
                ),
            )

        orchestrator.register_specialist(SpecialistType.SECURITY, mock_security)

        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "del-chk-save")

        state = await orchestrator.execute_run(job, snapshot)
        self.assertEqual("completed", state.terminal_status)

        # Query checkpointer state via thread_id (which is job.job_id)
        saved_state = orchestrator.get_workflow_state(job.job_id)
        self.assertIsNotNone(saved_state)
        values = saved_state.values
        self.assertEqual(job.job_id, values.get("job_id"))
        self.assertEqual(job.head_sha, values.get("head_sha"))
        self.assertEqual("completed", values.get("terminal_status"))
        self.assertTrue(values.get("aggregation_invoked"))

    async def test_orchestrator_checkpoint_thread_isolation(self) -> None:
        """Verify checkpoints for distinct jobs remain strictly isolated."""
        checkpointer = MemorySaver()
        orchestrator = ReviewOrchestrator(checkpointer=checkpointer)

        s1 = sample_snapshot(repo_id="repo-1", head_sha="sha-aaa")
        s2 = sample_snapshot(repo_id="repo-2", head_sha="sha-bbb")

        job1 = self.queue.enqueue(s1, "del-thread-1")
        job2 = self.queue.enqueue(s2, "del-thread-2")

        await orchestrator.execute_run(job1, s1)
        await orchestrator.execute_run(job2, s2)

        state1 = orchestrator.get_workflow_state(job1.job_id)
        state2 = orchestrator.get_workflow_state(job2.job_id)

        self.assertIsNotNone(state1)
        self.assertIsNotNone(state2)
        self.assertEqual("sha-aaa", state1.values.get("head_sha"))
        self.assertEqual("sha-bbb", state2.values.get("head_sha"))
        self.assertNotEqual(state1.values.get("job_id"), state2.values.get("job_id"))

    async def test_checkpoint_security_contract_no_forbidden_secrets(self) -> None:
        """Verify checkpointer state satisfies security contract: no forbidden secrets (Part 4)."""
        from pr_review_agent.orchestration import (
            CHECKPOINT_FORBIDDEN_KEYS,
            validate_checkpoint_state_security,
        )

        checkpointer = MemorySaver()
        orchestrator = ReviewOrchestrator(checkpointer=checkpointer)

        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "del-sec-contract")
        await orchestrator.execute_run(job, snapshot)

        saved_state = orchestrator.get_workflow_state(job.job_id)
        self.assertIsNotNone(saved_state)

        # 1. State must validate clean
        validate_checkpoint_state_security(saved_state.values)

        # 2. Verify all known forbidden keys would be rejected if present
        for forbidden in CHECKPOINT_FORBIDDEN_KEYS:
            dirty_state = dict(saved_state.values)
            dirty_state[forbidden] = "leak-123"
            with self.assertRaises(ValueError):
                validate_checkpoint_state_security(dirty_state)

    async def test_checkpoint_head_sha_immutability_for_publication_safety(self) -> None:
        """Verify checkpoint strictly binds head_sha to protect downstream publication safety."""
        checkpointer = MemorySaver()
        orchestrator = ReviewOrchestrator(checkpointer=checkpointer)

        snapshot = sample_snapshot(head_sha="sha-immutable-verify")
        job = self.queue.enqueue(snapshot, "del-sha-safe")
        await orchestrator.execute_run(job, snapshot)

        saved = orchestrator.get_workflow_state(job.job_id)
        self.assertIsNotNone(saved)
        # Checkpointed head_sha matches snapshot and job, enabling fail-closed publication check
        self.assertEqual("sha-immutable-verify", saved.values.get("head_sha"))


if __name__ == "__main__":
    unittest.main()
