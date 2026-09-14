import asyncio
import os
import sqlite3
import tempfile
import time
import unittest

from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.orchestration import (
    ALL_SPECIALISTS,
    CandidateFinding,
    DurableJobQueue,
    JobState,
    ReviewOrchestrator,
    SpecialistInput,
    SpecialistOutput,
    SpecialistType,
)


def sample_snapshot(head_sha: str = "head-123") -> ReviewSnapshot:
    return ReviewSnapshot(
        repository_id="repo-1",
        repository_full_name="owner/repo",
        pull_request_number=42,
        base_sha="base-000",
        head_sha=head_sha,
        changed_files=("auth.py", "test_auth.py"),
        policy_version="policy-v1",
        prompt_version="prompt-v1",
        retrieval_index_version="index-v1",
        model_configuration={"model": "test-model"},
    )


class DurableJobQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.queue = DurableJobQueue(self.conn)

    def test_enqueue_and_state_transitions(self) -> None:
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-1", max_retries=3, backoff_base_seconds=2.0)
        self.assertEqual("job-delivery-1", job.job_id)
        self.assertEqual(JobState.QUEUED, job.state)
        self.assertEqual(0, job.attempt_count)

        # Lease next job
        leased = self.queue.lease_next_job()
        self.assertIsNotNone(leased)
        self.assertEqual(JobState.RUNNING, leased.state)
        self.assertEqual(1, leased.attempt_count)

        # Mark completed
        completed = self.queue.mark_completed(job.job_id)
        self.assertEqual(JobState.COMPLETED, completed.state)

    def test_retry_and_exponential_backoff(self) -> None:
        snapshot = sample_snapshot()
        now = 1000.0
        job = self.queue.enqueue(snapshot, "delivery-retry", max_retries=3, backoff_base_seconds=2.0, now=now)

        # Attempt 1 fails
        self.queue.lease_next_job(now=now)
        job_after_fail1 = self.queue.mark_failed(job.job_id, error="Transient error 1", now=now)
        self.assertEqual(JobState.QUEUED, job_after_fail1.state)
        # delay = 2.0 * 2^(1 - 1) = 2.0
        self.assertEqual(now + 2.0, job_after_fail1.next_run_at)

        # Attempt 2 fails
        self.queue.lease_next_job(now=now + 2.0)
        job_after_fail2 = self.queue.mark_failed(job.job_id, error="Transient error 2", now=now + 2.0)
        self.assertEqual(JobState.QUEUED, job_after_fail2.state)
        # delay = 2.0 * 2^(2 - 1) = 4.0
        self.assertEqual(now + 2.0 + 4.0, job_after_fail2.next_run_at)

    def test_dead_letter_on_exhausted_retries(self) -> None:
        snapshot = sample_snapshot()
        now = 1000.0
        job = self.queue.enqueue(snapshot, "delivery-exhaust", max_retries=2, backoff_base_seconds=1.0, now=now)

        # Attempt 1
        self.queue.lease_next_job(now=now)
        self.queue.mark_failed(job.job_id, error="err 1", now=now)

        # Attempt 2
        self.queue.lease_next_job(now=now + 1.0)
        final_job = self.queue.mark_failed(job.job_id, error="terminal error", now=now + 1.0)

        self.assertEqual(JobState.DEAD_LETTER, final_job.state)
        self.assertEqual("terminal error", final_job.last_error)
        self.assertIsNone(self.queue.lease_next_job(now=now + 10.0))

    def test_cancel_job(self) -> None:
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-cancel")
        cancelled = self.queue.cancel_job(job.job_id, reason="PR closed")
        self.assertEqual(JobState.CANCELLED, cancelled.state)
        self.assertIsNone(self.queue.lease_next_job())

    def test_duplicate_enqueue_is_idempotent(self) -> None:
        snapshot = sample_snapshot()
        job1 = self.queue.enqueue(snapshot, "delivery-dup")
        job2 = self.queue.enqueue(snapshot, "delivery-dup")
        self.assertEqual(job1.job_id, job2.job_id)

    def test_restart_durability_across_connections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "review_queue.db")
            snapshot = sample_snapshot()

            # Connection 1: Enqueue and close connection
            conn1 = sqlite3.connect(db_path)
            queue1 = DurableJobQueue(conn1)
            job1 = queue1.enqueue(snapshot, "delivery-restart-1")
            self.assertEqual(JobState.QUEUED, job1.state)
            conn1.close()

            # Connection 2: Open fresh connection and verify recovery across restart
            conn2 = sqlite3.connect(db_path)
            queue2 = DurableJobQueue(conn2)
            recovered = queue2.get_job(job1.job_id)
            self.assertIsNotNone(recovered)
            self.assertEqual(JobState.QUEUED, recovered.state)

            # Lease next job using fresh connection
            leased = queue2.lease_next_job()
            self.assertIsNotNone(leased)
            self.assertEqual(job1.job_id, leased.job_id)
            self.assertEqual(JobState.RUNNING, leased.state)
            self.assertEqual(1, leased.attempt_count)
            conn2.close()


class ReviewOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.queue = DurableJobQueue(self.conn)
        self.orchestrator = ReviewOrchestrator()

    async def test_parallel_execution_of_all_four_specialists(self) -> None:
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-parallel")

        executed_specialists: set[SpecialistType] = set()

        def make_handler(spec: SpecialistType):
            def handler(inp: SpecialistInput) -> SpecialistOutput:
                executed_specialists.add(spec)
                finding = CandidateFinding(
                    finding_id=f"f-{spec.value}",
                    correlation_id=inp.correlation_id,
                    specialist_type=spec,
                    category="audit",
                    severity="medium",
                    confidence=0.9,
                    summary=f"Summary from {spec.value}",
                    rationale=f"Rationale from {spec.value}",
                    file_path="auth.py",
                )
                return SpecialistOutput(
                    specialist_type=spec,
                    correlation_id=inp.correlation_id,
                    status="completed",
                    findings=(finding,),
                )
            return handler

        for spec in ALL_SPECIALISTS:
            self.orchestrator.register_specialist(spec, make_handler(spec))

        state = await self.orchestrator.execute_run(job, snapshot)

        self.assertEqual("completed", state.terminal_status)
        self.assertTrue(state.aggregation_invoked)
        self.assertEqual(set(ALL_SPECIALISTS), executed_specialists)
        self.assertEqual(4, len(state.specialist_outputs))
        self.assertEqual([], state.partial_failures)

        # Verify correlation ID is preserved across outputs
        expected_correlation = f"{job.delivery_id}:{state.run_id}"
        for spec, output in state.specialist_outputs.items():
            self.assertEqual(expected_correlation, output.correlation_id)
            self.assertEqual(1, len(output.findings))
            self.assertEqual(expected_correlation, output.findings[0].correlation_id)

        # Verify audit trail contains time-ordered correlated events
        self.assertTrue(any(e.event_name == "review_run_started" for e in state.audit_trail))
        self.assertTrue(any(e.event_name == "aggregation_invoked" for e in state.audit_trail))

    async def test_tolerates_partial_specialist_failure(self) -> None:
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-partial")

        # Security throws exception, Tests times out/fails, Quality & Docs succeed
        def security_handler(inp: SpecialistInput):
            raise ValueError("Security scanner connection error")

        def quality_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=SpecialistType.QUALITY,
                correlation_id=inp.correlation_id,
                status="completed",
                findings=(
                    CandidateFinding(
                        finding_id="f-qual",
                        correlation_id=inp.correlation_id,
                        specialist_type=SpecialistType.QUALITY,
                        category="quality",
                        severity="low",
                        confidence=0.85,
                        summary="Refactoring opportunity",
                        rationale="Function too long",
                    ),
                ),
            )

        self.orchestrator.register_specialist(SpecialistType.SECURITY, security_handler)
        self.orchestrator.register_specialist(SpecialistType.QUALITY, quality_handler)

        state = await self.orchestrator.execute_run(job, snapshot)

        # Run still completes aggregation with partial failure noted
        self.assertEqual("completed", state.terminal_status)
        self.assertTrue(state.aggregation_invoked)
        self.assertIn(SpecialistType.SECURITY.value, state.partial_failures)
        self.assertEqual("failed", state.specialist_outputs[SpecialistType.SECURITY].status)
        self.assertEqual("completed", state.specialist_outputs[SpecialistType.QUALITY].status)

        # Verify audit event for specialist failure was emitted
        audit_names = [e.event_name for e in state.audit_trail]
        self.assertIn("specialist_failed", audit_names)
        self.assertIn("aggregation_invoked", audit_names)

    async def test_run_deadline_and_cancellation(self) -> None:
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-cancel")

        # Test pre-dispatch cancellation
        state = await self.orchestrator.execute_run(job, snapshot, cancellation_token=True)
        self.assertEqual("cancelled", state.terminal_status)
        self.assertFalse(state.aggregation_invoked)

        # Test timeout handling
        async def slow_handler(inp: SpecialistInput) -> SpecialistOutput:
            await asyncio.sleep(0.5)
            return SpecialistOutput(
                specialist_type=inp.specialist_type,
                correlation_id=inp.correlation_id,
                status="completed",
            )

        self.orchestrator.register_specialist(SpecialistType.SECURITY, slow_handler)
        job_timeout = self.queue.enqueue(snapshot, "delivery-timeout", deadline_seconds=0.01)
        state_timeout = await self.orchestrator.execute_run(
            job_timeout, snapshot, run_deadline_seconds=0.01
        )
        self.assertEqual("timeout", state_timeout.terminal_status)

    async def test_all_specialists_failed_results_in_failed_run(self) -> None:
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-all-fail")

        def failing_handler(inp: SpecialistInput):
            raise RuntimeError("Generic specialist failure")

        for spec in ALL_SPECIALISTS:
            self.orchestrator.register_specialist(spec, failing_handler)

        state = await self.orchestrator.execute_run(job, snapshot)
        self.assertEqual("failed", state.terminal_status)
        self.assertFalse(state.aggregation_invoked)
        self.assertEqual(4, len(state.partial_failures))

    async def test_synchronous_blocking_specialist_offloaded_to_thread(self) -> None:
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-sync-blocking")

        def blocking_sync_handler(inp: SpecialistInput) -> SpecialistOutput:
            # Simulate a blocking synchronous computation or blocking call
            time.sleep(0.05)
            return SpecialistOutput(
                specialist_type=SpecialistType.SECURITY,
                correlation_id=inp.correlation_id,
                status="completed",
                findings=(),
            )

        self.orchestrator.register_specialist(SpecialistType.SECURITY, blocking_sync_handler)

        state = await self.orchestrator.execute_run(job, snapshot)
        self.assertEqual("completed", state.terminal_status)
        self.assertEqual("completed", state.specialist_outputs[SpecialistType.SECURITY].status)


if __name__ == "__main__":
    unittest.main()
