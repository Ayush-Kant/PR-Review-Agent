"""Tests for Redis / ARQ distributed execution queue adapter.

Verifies:
1. 100% Contract parity with DurableQueueProtocol via DurableQueueContractTestSuite.
2. Backend-specific mechanics: lease tokens, stale worker rejection, zombie recovery, repo concurrency.
3. ARQ at-least-once re-execution safety and retry authority mapping.
4. All 14 mandatory failure injection scenarios.
5. Real Redis integration when available.
6. Secret redaction of REDIS_URL in configurations and representations.
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any
import unittest
import uuid

import redis

from pr_review_agent.adapters.redis_queue import RedisJobQueue, StaleLeaseError
from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.orchestration import (
    DurableQueueProtocol,
    JobState,
    ReviewJob,
)
from pr_review_agent.service_config import ServiceConfig, load_service_config
from tests.test_queue_and_checkpoint_contracts import (
    DurableQueueContractTestSuite,
    sample_snapshot,
)


class InMemoryRedisClient:
    """Thread-safe, deterministic in-memory Redis client implementing used commands."""

    def __init__(self) -> None:
        self._strings: dict[str, bytes] = {}
        self._hashes: dict[str, dict[str, bytes]] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._sets: dict[str, set[str]] = {}
        self.closed = False

    def _ensure_open(self) -> None:
        if self.closed:
            raise redis.ConnectionError("Connection closed")

    def get(self, name: str) -> bytes | None:
        self._ensure_open()
        return self._strings.get(str(name))

    def set(self, name: str, value: Any, **kwargs: Any) -> bool:
        self._ensure_open()
        val_bytes = value if isinstance(value, bytes) else str(value).encode("utf-8")
        self._strings[str(name)] = val_bytes
        return True

    def hset(self, name: str, key: str | None = None, value: Any = None, mapping: Mapping[str, Any] | None = None) -> int:
        self._ensure_open()
        s_name = str(name)
        if s_name not in self._hashes:
            self._hashes[s_name] = {}
        count = 0
        if mapping:
            for k, v in mapping.items():
                k_str = str(k)
                v_bytes = v if isinstance(v, bytes) else str(v).encode("utf-8")
                self._hashes[s_name][k_str] = v_bytes
                count += 1
        elif key is not None:
            k_str = str(key)
            v_bytes = value if isinstance(value, bytes) else str(value).encode("utf-8")
            self._hashes[s_name][k_str] = v_bytes
            count += 1
        return count

    def hget(self, name: str, key: str) -> bytes | None:
        self._ensure_open()
        h = self._hashes.get(str(name), {})
        return h.get(str(key))

    def hgetall(self, name: str) -> dict[bytes, bytes]:
        self._ensure_open()
        h = self._hashes.get(str(name), {})
        return {k.encode("utf-8"): v for k, v in h.items()}

    def zadd(self, name: str, mapping: Mapping[str, float], **kwargs: Any) -> int:
        self._ensure_open()
        s_name = str(name)
        if s_name not in self._zsets:
            self._zsets[s_name] = {}
        count = 0
        for member, score in mapping.items():
            self._zsets[s_name][str(member)] = float(score)
            count += 1
        return count

    def zrangebyscore(
        self,
        name: str,
        min: float | str,
        max: float | str,
        start: int | None = None,
        num: int | None = None,
        **kwargs: Any,
    ) -> list[bytes]:
        self._ensure_open()
        s_name = str(name)
        z = self._zsets.get(s_name, {})
        min_val = float("-inf") if str(min) in ("-inf", "-inf") else float(min)
        max_val = float("inf") if str(max) in ("inf", "+inf") else float(max)

        matching = [
            (m, score)
            for m, score in z.items()
            if min_val <= score <= max_val
        ]
        # Sort by score ASC, then by member ASC
        matching.sort(key=lambda x: (x[1], x[0]))

        items = [m.encode("utf-8") for m, _ in matching]
        if start is not None and num is not None:
            return items[start : start + num]
        return items

    def zrem(self, name: str, *values: Any) -> int:
        self._ensure_open()
        s_name = str(name)
        z = self._zsets.get(s_name, {})
        count = 0
        for val in values:
            s_val = str(val)
            if s_val in z:
                del z[s_val]
                count += 1
        return count

    def zcard(self, name: str) -> int:
        self._ensure_open()
        return len(self._zsets.get(str(name), {}))

    def sadd(self, name: str, *values: Any) -> int:
        self._ensure_open()
        s_name = str(name)
        if s_name not in self._sets:
            self._sets[s_name] = set()
        count = 0
        for val in values:
            s_val = str(val)
            if s_val not in self._sets[s_name]:
                self._sets[s_name].add(s_val)
                count += 1
        return count

    def srem(self, name: str, *values: Any) -> int:
        self._ensure_open()
        s_name = str(name)
        s = self._sets.get(s_name, set())
        count = 0
        for val in values:
            s_val = str(val)
            if s_val in s:
                s.remove(s_val)
                count += 1
        return count

    def scard(self, name: str) -> int:
        self._ensure_open()
        return len(self._sets.get(str(name), set()))

    def keys(self, pattern: str = "*") -> list[str]:
        self._ensure_open()
        import fnmatch
        all_keys = set(self._strings.keys()) | set(self._hashes.keys()) | set(self._zsets.keys()) | set(self._sets.keys())
        return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

    def pipeline(self, transaction: bool = True) -> InMemoryPipeline:
        self._ensure_open()
        return InMemoryPipeline(self)


class InMemoryPipeline:
    """Mock pipeline executing commands atomically."""

    def __init__(self, client: InMemoryRedisClient) -> None:
        self.client = client
        self.commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def queue_cmd(*args: Any, **kwargs: Any) -> InMemoryPipeline:
            self.commands.append((name, args, kwargs))
            return self
        return queue_cmd

    def execute(self) -> list[Any]:
        results = []
        for name, args, kwargs in self.commands:
            fn = getattr(self.client, name)
            res = fn(*args, **kwargs)
            results.append(res)
        self.commands.clear()
        return results


class RedisDurableQueueContractTests(DurableQueueContractTestSuite, unittest.TestCase):
    """Verify RedisJobQueue satisfies 100% of the DurableQueueProtocol contract suite."""

    def setUp(self) -> None:
        self.client = InMemoryRedisClient()
        self.queue = RedisJobQueue(client=self.client)  # type: ignore[arg-type]

    def get_queue(self) -> DurableQueueProtocol:
        return self.queue


class RedisSpecificAdapterTests(unittest.TestCase):
    """Verify Redis-specific semantics: lease tokens, stale worker rejection, concurrency, failure modes."""

    def setUp(self) -> None:
        self.client = InMemoryRedisClient()
        self.queue = RedisJobQueue(client=self.client)  # type: ignore[arg-type]

    def test_lease_token_issuance_and_stale_worker_rejection(self) -> None:
        """Worker A's mutation must be rejected if its lease token is stale (reclaimed by Worker B)."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-lease-token", deadline_seconds=10.0, now=now)

        # Worker A leases the job
        leased_a = self.queue.lease_next_job(now=now)
        self.assertIsNotNone(leased_a)
        # Fetch Worker A's lease token from stored hash
        data_a = self.client.hgetall(f"review:job:{job.job_id}")
        token_a = data_a[b"lease_token"].decode("utf-8")
        self.assertTrue(len(token_a) > 0)

        # Worker A hangs, lease expires at now + 10.0. Zombie recovery reclaims job.
        self.queue.recover_zombie_jobs(now=now + 15.0)

        # Worker B leases the reclaimed job
        leased_b = self.queue.lease_next_job(now=now + 16.0)
        self.assertIsNotNone(leased_b)
        data_b = self.client.hgetall(f"review:job:{job.job_id}")
        token_b = data_b[b"lease_token"].decode("utf-8")
        self.assertNotEqual(token_a, token_b)

        # Worker A wakes up and attempts to complete with stale token_a -> must fail
        with self.assertRaises(StaleLeaseError):
            self.queue.mark_completed(job.job_id, now=now + 20.0, lease_token=token_a)

        # Worker B completes with valid token_b -> must succeed
        completed = self.queue.mark_completed(job.job_id, now=now + 20.0, lease_token=token_b)
        self.assertEqual(JobState.COMPLETED, completed.state)

    def test_zombie_recovery_to_dead_letter_on_exhaustion(self) -> None:
        """Expired running job that exhausted max_retries recovers to DEAD_LETTER."""
        now = 1000.0
        job = self.queue.enqueue(
            sample_snapshot(),
            "del-zombie-dead",
            max_retries=1,
            deadline_seconds=5.0,
            now=now,
        )

        leased = self.queue.lease_next_job(now=now)
        self.assertIsNotNone(leased)
        self.assertEqual(1, leased.attempt_count)

        # Worker crashes; recover after deadline
        recovered = self.queue.recover_zombie_jobs(now=now + 10.0)
        self.assertEqual(1, len(recovered))
        self.assertEqual(JobState.DEAD_LETTER, recovered[0].state)
        self.assertTrue("Lease expired" in (recovered[0].last_error or ""))

        # Dead-lettered job is present in dead-letter set
        self.assertEqual(1, self.client.scard(self.queue.dead_letter_key))

    def test_claim_job_reconciles_arq_push_model(self) -> None:
        """ARQ worker pushes job_id -> adapter atomically claims with repo concurrency check."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(repo_id="repo-arq"), "del-claim-arq", now=now)

        claimed_job, token = self.queue.claim_job(job.job_id, now=now, max_concurrency_per_repo=1)
        self.assertIsNotNone(claimed_job)
        self.assertIsNotNone(token)
        self.assertEqual(JobState.RUNNING, claimed_job.state)

        # Second claim while RUNNING must reject due to repo capacity
        second_claim, reason = self.queue.claim_job(job.job_id, now=now, max_concurrency_per_repo=1)
        self.assertIsNone(second_claim)
        self.assertEqual("repo_at_capacity", reason)

    def test_cancellation_authority_rejects_completion(self) -> None:
        """Cancelled job cannot be completed by worker."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-cancel-auth", now=now)
        self.queue.lease_next_job(now=now)

        # Cancel in flight
        self.queue.cancel_job(job.job_id, reason="Operator abort", now=now + 1.0)

        # ARQ or claim rejects cancelled job
        claimed, reason = self.queue.claim_job(job.job_id, now=now + 2.0)
        self.assertIsNone(claimed)
        self.assertEqual("cancelled", reason)

    def test_repo_concurrency_race_condition(self) -> None:
        """Two concurrent workers racing for the last repository slot."""
        now = 1000.0
        self.queue.enqueue(sample_snapshot(repo_id="race-repo"), "del-race-1", now=now)
        self.queue.enqueue(sample_snapshot(repo_id="race-repo"), "del-race-2", now=now + 0.1)

        # Worker 1 leases slot (cap=1)
        w1_job = self.queue.lease_next_job(now=now + 1.0, max_concurrency_per_repo=1)
        self.assertIsNotNone(w1_job)

        # Worker 2 attempts lease for same repo -> must return None
        w2_job = self.queue.lease_next_job(now=now + 1.0, max_concurrency_per_repo=1)
        self.assertIsNone(w2_job)

        # Total running for race-repo must be exactly 1
        self.assertEqual(1, self.client.scard("review:repo_running:race-repo"))

    def test_secret_redaction_in_config(self) -> None:
        """REDIS_URL with password must never be exposed in string representations."""
        cfg = ServiceConfig(
            github_token="gh_secret_123",
            webhook_secret=b"hook_secret_456",
            model_provider="openai",
            model_name="gpt-4o",
            database_path=":memory:",
            host="127.0.0.1",
            port=8000,
            authorized_repositories=("test/repo",),
            authorized_tenant="test",
            queue_backend="redis",
            redis_url="redis://:super_secret_pw@prod-redis:6379/0",
        )
        rep = repr(cfg)
        self.assertNotIn("super_secret_pw", rep)
        self.assertNotIn("gh_secret_123", rep)
        self.assertIn("redis_url='***'", rep)


class FailureScenariosTestSuite(unittest.TestCase):
    """Test all 14 mandatory failure injection scenarios."""

    def setUp(self) -> None:
        self.client = InMemoryRedisClient()
        self.queue = RedisJobQueue(client=self.client)  # type: ignore[arg-type]

    def test_01_redis_unavailable_during_enqueue(self) -> None:
        """1. Redis unavailable during enqueue fails closed."""
        self.client.closed = True
        with self.assertRaises(redis.ConnectionError):
            self.queue.enqueue(sample_snapshot(), "del-f1")

    def test_02_redis_timeout_during_enqueue(self) -> None:
        """2. Redis timeout during enqueue fails closed."""
        def timeout_get(*args: Any, **kwargs: Any) -> Any:
            raise redis.TimeoutError("Redis timed out")
        self.client.get = timeout_get  # type: ignore[method-assign]
        with self.assertRaises(redis.TimeoutError):
            self.queue.enqueue(sample_snapshot(), "del-f2")

    def test_03_redis_unavailable_during_worker_startup(self) -> None:
        """3. Redis unavailable during worker startup."""
        self.client.closed = True
        with self.assertRaises(redis.ConnectionError):
            self.queue.lease_next_job()

    def test_04_redis_disconnect_during_processing(self) -> None:
        """4. Redis disconnect during processing."""
        job = self.queue.enqueue(sample_snapshot(), "del-f4")
        self.queue.lease_next_job()
        self.client.closed = True
        with self.assertRaises(redis.ConnectionError):
            self.queue.mark_completed(job.job_id)

    def test_05_worker_killed_during_execution(self) -> None:
        """5. Worker killed during execution recovers via lease expiry."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-f5", deadline_seconds=10.0, now=now)
        self.queue.lease_next_job(now=now)
        # Worker died; time advances past deadline
        recovered = self.queue.recover_zombie_jobs(now=now + 15.0)
        self.assertEqual(1, len(recovered))
        self.assertEqual(JobState.QUEUED, recovered[0].state)

    def test_06_worker_restarted(self) -> None:
        """6. Worker restarted resumes reading queue."""
        self.queue.enqueue(sample_snapshot(), "del-f6")
        # Worker 1 starts, crashes before lease
        # Worker 2 starts fresh
        leased = self.queue.lease_next_job()
        self.assertIsNotNone(leased)

    def test_07_job_reexecution_after_cancellation(self) -> None:
        """7. Job re-execution after cancellation is rejected."""
        job = self.queue.enqueue(sample_snapshot(), "del-f7")
        self.queue.cancel_job(job.job_id, reason="User abort")
        # Attempt to lease or claim must return None
        self.assertIsNone(self.queue.lease_next_job())
        claimed, reason = self.queue.claim_job(job.job_id)
        self.assertIsNone(claimed)
        self.assertEqual("cancelled", reason)

    def test_08_retry_after_transient_failure(self) -> None:
        """8. Retry after transient failure calculates exponential backoff."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-f8", max_retries=3, backoff_base_seconds=2.0, now=now)
        self.queue.lease_next_job(now=now)
        failed = self.queue.mark_failed(job.job_id, error="Transient LLM 500", now=now)
        self.assertEqual(JobState.QUEUED, failed.state)
        self.assertEqual(now + 2.0, failed.next_run_at)

    def test_09_retry_exhaustion(self) -> None:
        """9. Retry exhaustion transitions to DEAD_LETTER."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-f9", max_retries=1, now=now)
        self.queue.lease_next_job(now=now)
        failed = self.queue.mark_failed(job.job_id, error="Permanent error", now=now)
        self.assertEqual(JobState.DEAD_LETTER, failed.state)

    def test_10_dead_letter_inspection(self) -> None:
        """10. Dead-letter jobs inspectable via dead_letter_key."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-f10", max_retries=1, now=now)
        self.queue.lease_next_job(now=now)
        self.queue.mark_failed(job.job_id, error="Fatal", now=now)
        self.assertTrue(self.client.scard(self.queue.dead_letter_key) > 0)

    def test_11_duplicate_delivery(self) -> None:
        """11. Duplicate delivery returns existing job without resetting attempts."""
        now = 1000.0
        job1 = self.queue.enqueue(sample_snapshot(), "del-f11", now=now)
        self.queue.lease_next_job(now=now)  # attempt_count becomes 1
        job2 = self.queue.enqueue(sample_snapshot(), "del-f11", now=now + 5.0)
        self.assertEqual(job1.job_id, job2.job_id)
        self.assertEqual(1, job2.attempt_count)  # not reset to 0

    def test_12_duplicate_worker_race(self) -> None:
        """12. Duplicate worker race: exactly one worker leases the job."""
        now = 1000.0
        self.queue.enqueue(sample_snapshot(), "del-f12", now=now)
        w1 = self.queue.lease_next_job(now=now)
        w2 = self.queue.lease_next_job(now=now)
        self.assertIsNotNone(w1)
        self.assertIsNone(w2)

    def test_13_repository_concurrency_race(self) -> None:
        """13. Repository concurrency race: capacity never exceeded."""
        now = 1000.0
        self.queue.enqueue(sample_snapshot(repo_id="cap-repo"), "del-cap-1", now=now)
        self.queue.enqueue(sample_snapshot(repo_id="cap-repo"), "del-cap-2", now=now)

        l1 = self.queue.lease_next_job(now=now, max_concurrency_per_repo=1)
        l2 = self.queue.lease_next_job(now=now, max_concurrency_per_repo=1)
        self.assertIsNotNone(l1)
        self.assertIsNone(l2)
        self.assertEqual(1, self.client.scard("review:repo_running:cap-repo"))

    def test_14_redis_reconnect(self) -> None:
        """14. Redis reconnect resumes operation after connection drop."""
        job = self.queue.enqueue(sample_snapshot(), "del-f14")
        self.client.closed = True
        with self.assertRaises(redis.ConnectionError):
            self.queue.get_job(job.job_id)
        self.client.closed = False
        reconnected_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(reconnected_job)


class RealRedisLiveIntegrationTests(unittest.TestCase):
    """Integration test against real live Redis on 127.0.0.1:6379 if available."""

    def setUp(self) -> None:
        try:
            self.live_client: redis.Redis[Any] | None = redis.Redis(
                host="127.0.0.1",
                port=6379,
                socket_timeout=0.2,
                socket_connect_timeout=0.2,
            )
            self.live_client.ping()
        except Exception:
            self.live_client = None

    def test_real_redis_roundtrip_if_running(self) -> None:
        if self.live_client is None:
            self.skipTest("No live Redis server available on 127.0.0.1:6379; skipping live Redis test.")

        test_queue_name = f"test:arq:queue:{uuid.uuid4().hex[:8]}"
        q = RedisJobQueue(client=self.live_client, queue_name=test_queue_name)
        try:
            deliv = f"live-test-{uuid.uuid4().hex[:8]}"
            job = q.enqueue(sample_snapshot(), deliv)
            self.assertEqual(f"job-{deliv}", job.job_id)
            leased = q.lease_next_job()
            self.assertIsNotNone(leased)
            self.assertEqual(job.job_id, leased.job_id)
            completed = q.mark_completed(job.job_id)
            self.assertEqual(JobState.COMPLETED, completed.state)
        finally:
            self.live_client.delete(test_queue_name)
            self.live_client.delete(f"review:job:job-{deliv}")
            self.live_client.delete(f"review:delivery:{deliv}")
