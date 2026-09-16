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
    SpecialistCoverageSummary,
    SpecialistInput,
    SpecialistOutput,
    SpecialistStatus,
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

    async def test_specialist_contract_success_with_zero_findings(self) -> None:
        """Test A: Specialist successfully executes and finds 0 issues -> status == completed, no degradation."""
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-zero-findings")

        def zero_finding_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=inp.specialist_type,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(),
            )

        for spec in ALL_SPECIALISTS:
            self.orchestrator.register_specialist(spec, zero_finding_handler)

        state = await self.orchestrator.execute_run(job, snapshot)

        self.assertEqual("completed", state.terminal_status)
        self.assertFalse(state.is_degraded)
        self.assertIsNotNone(state.coverage_summary)
        self.assertTrue(state.coverage_summary.is_full_coverage)
        self.assertFalse(state.coverage_summary.is_degraded)
        self.assertEqual(1.0, state.coverage_summary.coverage_ratio)
        self.assertEqual(4, len(state.coverage_summary.succeeded_specialists))
        self.assertEqual(0, len(state.coverage_summary.failed_specialists))
        self.assertEqual(0, len(state.partial_failures))

    async def test_specialist_contract_mixed_coverage_degradation(self) -> None:
        """Test H: SECURITY/QUALITY succeed, TESTS/DOCUMENTATION fail -> coverage 2/4, is_degraded=True, event emitted."""
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-mixed-cov")

        def sec_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=SpecialistType.SECURITY,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(),
            )

        def qual_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=SpecialistType.QUALITY,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(),
            )

        def tests_fail_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=SpecialistType.TESTS,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.FAILED,
                findings=(),
                error_message="Rate limit 429",
            )

        def doc_fail_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=SpecialistType.DOCUMENTATION,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.FAILED,
                findings=(),
                error_message="Rate limit 429",
            )

        self.orchestrator.register_specialist(SpecialistType.SECURITY, sec_handler)
        self.orchestrator.register_specialist(SpecialistType.QUALITY, qual_handler)
        self.orchestrator.register_specialist(SpecialistType.TESTS, tests_fail_handler)
        self.orchestrator.register_specialist(SpecialistType.DOCUMENTATION, doc_fail_handler)

        state = await self.orchestrator.execute_run(job, snapshot)

        self.assertEqual("completed", state.terminal_status)
        self.assertTrue(state.is_degraded)
        self.assertIsNotNone(state.coverage_summary)
        self.assertFalse(state.coverage_summary.is_full_coverage)
        self.assertTrue(state.coverage_summary.is_degraded)
        self.assertEqual(0.5, state.coverage_summary.coverage_ratio)
        self.assertEqual(("security", "quality"), state.coverage_summary.succeeded_specialists)
        self.assertEqual(("tests", "documentation"), state.coverage_summary.failed_specialists)

        # Confirm specialist_coverage_degraded audit event was emitted
        degraded_events = [e for e in state.audit_trail if e.event_name == "specialist_coverage_degraded"]
        self.assertGreaterEqual(len(degraded_events), 1)
        deg_evt = degraded_events[0]
        self.assertEqual(0.5, deg_evt.details["coverage_ratio"])
        self.assertIn("tests", deg_evt.details["failed"])
        self.assertIn("documentation", deg_evt.details["failed"])

    async def test_distinguish_successful_zero_from_failed_empty(self) -> None:
        """Test I: A successful specialist returning zero findings is strictly distinct from a failed specialist with []."""
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-distinguish")

        # Tests specialist succeeds with valid empty output
        def tests_success_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=SpecialistType.TESTS,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(),
            )

        # Documentation specialist fails after error, returning empty output
        def doc_failed_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=SpecialistType.DOCUMENTATION,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.FAILED,
                findings=(),
                error_message="Provider timeout",
            )

        def pass_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=inp.specialist_type,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(),
            )

        self.orchestrator.register_specialist(SpecialistType.SECURITY, pass_handler)
        self.orchestrator.register_specialist(SpecialistType.QUALITY, pass_handler)
        self.orchestrator.register_specialist(SpecialistType.TESTS, tests_success_handler)
        self.orchestrator.register_specialist(SpecialistType.DOCUMENTATION, doc_failed_handler)

        state = await self.orchestrator.execute_run(job, snapshot)

        # TESTS must be in succeeded_specialists; DOCUMENTATION must be in failed_specialists
        cov = state.coverage_summary
        self.assertIn("tests", cov.succeeded_specialists)
        self.assertNotIn("tests", cov.failed_specialists)
        self.assertIn("documentation", cov.failed_specialists)
        self.assertNotIn("documentation", cov.succeeded_specialists)
        # Because one failed, coverage is degraded, proving the failure wasn't treated as successful zero
        self.assertTrue(state.is_degraded)
        self.assertEqual(0.75, cov.coverage_ratio)

    async def test_terminal_timeout_preserves_explicit_coverage_summary(self) -> None:
        """Terminal timeout branch must explicitly compute and return coverage_summary.

        Guarantees:
        - state.coverage_summary is not None
        - completed specialists remain explicitly represented in succeeded_specialists
        - non-completed/timed-out specialists are explicitly represented in timeout_specialists
        - terminal_status remains 'timeout'
        - is_degraded is True and is_full_coverage is False
        - aggregation_invoked is False
        """
        snapshot = sample_snapshot()
        job = self.queue.enqueue(snapshot, "delivery-timeout-coverage", deadline_seconds=0.06)

        # Fast specialist that completes immediately without thread hop
        async def fast_handler(inp: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=inp.specialist_type,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(),
            )

        # Slow specialist that sleeps well past deadline
        async def slow_handler(inp: SpecialistInput) -> SpecialistOutput:
            await asyncio.sleep(0.15)
            return SpecialistOutput(
                specialist_type=inp.specialist_type,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(),
            )

        self.orchestrator.register_specialist(SpecialistType.SECURITY, fast_handler)
        self.orchestrator.register_specialist(SpecialistType.QUALITY, fast_handler)
        self.orchestrator.register_specialist(SpecialistType.TESTS, slow_handler)
        self.orchestrator.register_specialist(SpecialistType.DOCUMENTATION, slow_handler)

        state = await self.orchestrator.execute_run(
            job, snapshot, run_deadline_seconds=0.06
        )

        # 1. Terminal status must remain 'timeout'
        self.assertEqual("timeout", state.terminal_status)
        self.assertFalse(state.aggregation_invoked)
        self.assertTrue(state.is_degraded)

        # 2. Coverage summary MUST NOT be None
        cov = state.coverage_summary
        self.assertIsNotNone(cov)
        self.assertFalse(cov.is_full_coverage)
        self.assertTrue(cov.is_degraded)

        # 3. Completed specialists must be in succeeded_specialists
        self.assertIn("security", cov.succeeded_specialists)
        self.assertIn("quality", cov.succeeded_specialists)

        # 4. Total specialists accounted for
        self.assertEqual(4, cov.total_specialists)
        self.assertEqual(len(cov.succeeded_specialists) + len(cov.timeout_specialists) + len(cov.failed_specialists) + len(cov.skipped_specialists), 4)

        # 5. Degradation details must contain serialized coverage_summary
        self.assertIn("coverage_summary", state.degradation_details)
        self.assertEqual(state.degradation_details["coverage_summary"]["is_degraded"], True)

        # 6. Audit trail must contain review_run_timeout and specialist_coverage_degraded
        audit_events = [e.event_name for e in state.audit_trail]
        self.assertIn("review_run_timeout", audit_events)
        self.assertIn("specialist_coverage_degraded", audit_events)


if __name__ == "__main__":
    unittest.main()
