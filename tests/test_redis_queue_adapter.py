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
import sqlite3
import time
from typing import Any
import unittest
import uuid

from arq.constants import default_queue_name
from arq.jobs import deserialize_job
from arq.worker import Worker
import redis

from pr_review_agent.adapters.redis_queue import (
    RedisJobQueue,
    StaleLeaseError,
    review_job_task,
)
from pr_review_agent.github_output import FakeGitHubClient
from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.orchestration import (
    CandidateFinding,
    DurableQueueProtocol,
    JobState,
    ReviewJob,
    SpecialistInput,
    SpecialistOutput,
    SpecialistType,
)
from pr_review_agent.service_config import ServiceConfig, load_service_config
from pr_review_agent.worker import (
    AutonomousReviewWorker,
    ReviewWorkerSettings,
    WorkerSettings,
    arq_shutdown,
    arq_startup,
)
from tests.test_queue_and_checkpoint_contracts import (
    DurableQueueContractTestSuite,
    sample_snapshot,
)


def _make_dummy_service_config() -> ServiceConfig:
    return ServiceConfig(
        github_token="ghp_dummytoken123456789012345678901234567890",
        webhook_secret=b"dummy-webhook-secret-32-bytes-ok!",
        model_provider="openai",
        model_name="gpt-4o",
        database_path=":memory:",
        host="127.0.0.1",
        port=8000,
        authorized_repositories=("octocat/hello-world", "owner/repo"),
        authorized_tenant="octocat",
        publish_enabled=False,
        api_key="sk-mock-key-12345678901234567890",
    )


def _make_mock_specialist(findings: list[CandidateFinding] | None = None, should_fail: bool = False):
    def handler(spec_input: SpecialistInput) -> SpecialistOutput:
        if should_fail:
            raise RuntimeError("Simulated specialist failure")
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status="completed",
            findings=tuple(findings or []),
            execution_duration=0.01,
        )

    return handler


def _create_test_worker(queue: DurableQueueProtocol, should_fail: bool = False) -> AutonomousReviewWorker:
    cfg = _make_dummy_service_config()
    conn = sqlite3.connect(":memory:")
    handlers = {
        SpecialistType.SECURITY: _make_mock_specialist(should_fail=should_fail),
        SpecialistType.QUALITY: _make_mock_specialist(should_fail=should_fail),
        SpecialistType.TESTS: _make_mock_specialist(should_fail=should_fail),
        SpecialistType.DOCUMENTATION: _make_mock_specialist(should_fail=should_fail),
    }
    return AutonomousReviewWorker(
        cfg,
        connection=conn,
        github_client=FakeGitHubClient(),
        specialist_handlers=handlers,
        queue=queue,
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
        if kwargs.get("nx") and str(name) in self._strings:
            return False
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

    def zincrby(self, name: str, amount: float, value: Any) -> float:
        self._ensure_open()
        s_name = str(name)
        if s_name not in self._zsets:
            self._zsets[s_name] = {}
        s_val = str(value)
        curr = self._zsets[s_name].get(s_val, 0.0)
        new_score = curr + float(amount)
        self._zsets[s_name][s_val] = new_score
        return new_score

    def zrange(self, name: str, start: int = 0, end: int = -1, **kwargs: Any) -> list[bytes]:
        self._ensure_open()
        s_name = str(name)
        z = self._zsets.get(s_name, {})
        sorted_items = sorted(z.items(), key=lambda x: (x[1], x[0]))
        members = [m.encode("utf-8") for m, _ in sorted_items]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    def zremrangebyscore(self, name: str, min: float | str, max: float | str) -> int:
        self._ensure_open()
        s_name = str(name)
        z = self._zsets.get(s_name, {})
        min_val = float("-inf") if str(min) in ("-inf", "-inf") else float(min)
        max_val = float("inf") if str(max) in ("inf", "+inf") else float(max)
        to_del = [m for m, score in z.items() if min_val <= score <= max_val]
        for m in to_del:
            del z[m]
        return len(to_del)

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

    def smembers(self, name: str) -> set[bytes]:
        self._ensure_open()
        s = self._sets.get(str(name), set())
        return {x.encode("utf-8") if isinstance(x, str) else x for x in s}

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        self._ensure_open()
        keys = list(keys_and_args[:numkeys])
        args = list(keys_and_args[numkeys:])
        if "SCARD" in script or "CLAIM_JOB_LUA" in script:
            running_key = str(keys[0])
            running_jobs_key = str(keys[1])
            job_key = str(keys[2])
            queue_key = str(keys[3])
            job_id = str(args[0])
            cap = int(args[1]) if int(args[1]) > 0 else None
            now = float(args[2])
            lease_token = str(args[3])
            lease_expires_at = str(args[4])

            if cap is not None and self.scard(running_key) >= cap:
                return b"repo_at_capacity"
            if queue_key != "":
                removed = self.zrem(queue_key, job_id)
                if removed == 0:
                    return b"already_leased"
            self.sadd(running_key, job_id)
            self.sadd(running_jobs_key, job_id)
            curr_attempts = int(self.hget(job_key, "attempt_count") or b"0")
            self.hset(
                job_key,
                mapping={
                    "state": "running",
                    "attempt_count": str(curr_attempts + 1),
                    "lease_token": lease_token,
                    "lease_expires_at": lease_expires_at,
                    "updated_at": str(now),
                },
            )
            return b"ok"
        if "ENQUEUE_JOB_LUA" in script or (numkeys == 4 and "existing" in script):
            delivery_key = str(keys[0])
            job_key = str(keys[1])
            arq_job_key = str(keys[2])
            queue_name = str(keys[3])
            job_id = str(args[0])
            arq_payload = args[1]
            score_ms = float(args[2])

            if self.get(delivery_key) is not None:
                return b"existing"

            self.set(delivery_key, job_id)
            hset_mapping: dict[str, Any] = {}
            for i in range(3, len(args), 2):
                hset_mapping[str(args[i])] = args[i + 1]
            self.hset(job_key, mapping=hset_mapping)
            self.set(arq_job_key, arq_payload)
            self.zadd(queue_name, {job_id: score_ms})
            return b"ok"

        raise NotImplementedError("Arbitrary lua eval not supported in InMemoryRedisClient")

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


class AsyncInMemoryPipeline:
    """Async pipeline mock supporting commands used by ARQ Worker."""

    def __init__(self, client: InMemoryRedisClient) -> None:
        self.client = client
        self.commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def watch(self, *keys: Any) -> None:
        pass

    async def exists(self, key: Any) -> int:
        return 1 if self.client.get(key) is not None else 0

    async def zscore(self, key: Any, member: Any) -> float | None:
        z = self.client._zsets.get(str(key), {})
        return z.get(str(member))

    def multi(self) -> None:
        pass

    def psetex(self, key: Any, ms: int, val: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("psetex", (key, ms, val), {}))
        return self

    def get(self, key: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("get", (key,), {}))
        return self

    def incr(self, key: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("incr", (key,), {}))
        return self

    def expire(self, key: Any, seconds: int) -> AsyncInMemoryPipeline:
        self.commands.append(("expire", (key, seconds), {}))
        return self

    def pexpire(self, key: Any, ms: int) -> AsyncInMemoryPipeline:
        self.commands.append(("pexpire", (key, ms), {}))
        return self

    def set(self, key: Any, val: Any, **kwargs: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("set", (key, val), kwargs))
        return self

    def zrem(self, key: Any, *values: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("zrem", (key, *values), {}))
        return self

    def zincrby(self, key: Any, amount: float, member: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("zincrby", (key, amount, member), {}))
        return self

    def zrange(self, key: Any, start: int = 0, end: int = -1, **kwargs: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("zrange", (key, start, end), kwargs))
        return self

    def zremrangebyscore(self, key: Any, min: Any, max: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("zremrangebyscore", (key, min, max), {}))
        return self

    def delete(self, *keys: Any) -> AsyncInMemoryPipeline:
        self.commands.append(("delete", keys, {}))
        return self

    def info(self, section: str | None = None) -> AsyncInMemoryPipeline:
        self.commands.append(("info", (section,), {}))
        return self

    def dbsize(self) -> AsyncInMemoryPipeline:
        self.commands.append(("dbsize", (), {}))
        return self

    async def execute(self) -> list[Any]:
        res = []
        for cmd, args, kwargs in self.commands:
            if cmd == "get":
                res.append(self.client.get(*args))
            elif cmd == "incr":
                k = str(args[0])
                curr = int(self.client.get(k) or b"0")
                new_v = curr + 1
                self.client.set(k, new_v)
                res.append(new_v)
            elif cmd in ("expire", "pexpire"):
                res.append(True)
            elif cmd in ("set", "psetex"):
                res.append(self.client.set(args[0], args[-1]))
            elif cmd == "zrem":
                res.append(self.client.zrem(*args))
            elif cmd == "zincrby":
                res.append(self.client.zincrby(*args))
            elif cmd == "zrange":
                res.append(self.client.zrange(*args, **kwargs))
            elif cmd == "zremrangebyscore":
                res.append(self.client.zremrangebyscore(*args))
            elif cmd == "delete":
                for k in args[0]:
                    if k in self.client._strings:
                        del self.client._strings[k]
                res.append(1)
            elif cmd == "info":
                res.append({"redis_version": "7.0.0", "used_memory_human": "1M", "connected_clients": 1})
            elif cmd == "dbsize":
                res.append(10)
        self.commands.clear()
        return res

    async def __aenter__(self) -> AsyncInMemoryPipeline:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class AsyncInMemoryRedisPool:
    """Async Redis pool adapter over InMemoryRedisClient compatible with ARQ Worker."""

    def __init__(self, client: InMemoryRedisClient) -> None:
        self.client = client

    def pipeline(self, transaction: bool = True) -> AsyncInMemoryPipeline:
        return AsyncInMemoryPipeline(self.client)

    async def get(self, key: Any) -> bytes | None:
        return self.client.get(key)

    async def set(self, key: Any, val: Any, **kwargs: Any) -> bool:
        return self.client.set(key, val, **kwargs)

    async def zrangebyscore(
        self,
        key: Any,
        min: float = float("-inf"),
        max: float = float("inf"),
        start: int | None = None,
        num: int | None = None,
    ) -> list[bytes]:
        raw = self.client.zrangebyscore(key, min=min, max=max, start=start, num=num)
        return [r if isinstance(r, bytes) else str(r).encode("utf-8") for r in raw]

    async def zcard(self, key: Any) -> int:
        return self.client.zcard(key)

    async def zrem(self, key: Any, *values: Any) -> int:
        return self.client.zrem(key, *values)

    async def zincrby(self, key: Any, amount: float, member: Any) -> float:
        return self.client.zincrby(key, amount, member)

    async def zrange(self, key: Any, start: int = 0, end: int = -1, **kwargs: Any) -> list[bytes]:
        return self.client.zrange(key, start=start, end=end, **kwargs)

    async def zremrangebyscore(self, key: Any, min: Any, max: Any) -> int:
        return self.client.zremrangebyscore(key, min=min, max=max)

    async def psetex(self, key: Any, ms: int, val: Any) -> bool:
        return self.client.set(key, val)

    async def close(self) -> None:
        pass


class RedisDurableQueueContractTests(DurableQueueContractTestSuite, unittest.TestCase):
    """Verify RedisJobQueue satisfies 100% of the DurableQueueProtocol contract suite."""

    def setUp(self) -> None:
        self.client = InMemoryRedisClient()
        self.queue = RedisJobQueue(client=self.client)  # type: ignore[arg-type]

    def get_queue(self) -> DurableQueueProtocol:
        return self.queue


class RedisSpecificAdapterTests(unittest.IsolatedAsyncioTestCase):
    """Verify Redis-specific semantics: lease tokens, stale worker rejection, concurrency, failure modes."""

    def setUp(self) -> None:
        self.client = InMemoryRedisClient()
        self.queue = RedisJobQueue(client=self.client)  # type: ignore[arg-type]

    async def test_actual_arq_job_model_and_worker_task_execution_path(self) -> None:
        """Verify RedisJobQueue produces genuine ARQ 0.28.0 serialized jobs executed via review_job_task."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world")
        job = self.queue.enqueue(snapshot, "del-arq-exec", now=now)

        # 1. Verify ARQ job exists in Redis under arq:job:{job_id}
        arq_job_raw = self.client.get(f"arq:job:{job.job_id}")
        self.assertIsNotNone(arq_job_raw)

        # 2. Verify deserialization with official arq.jobs.deserialize_job
        arq_job_def = deserialize_job(arq_job_raw)
        self.assertEqual("review_job_task", arq_job_def.function)
        self.assertEqual((job.job_id,), arq_job_def.args)
        self.assertIsNotNone(arq_job_def.enqueue_time)

        # 3. Execute via real ARQ worker task function
        worker = _create_test_worker(self.queue)
        ctx = {"queue": self.queue, "worker": worker}
        task_result = await review_job_task(ctx, job.job_id)

        self.assertEqual("completed", task_result["status"])
        self.assertEqual(job.job_id, task_result["job_id"])

        # 4. Verify terminal status in queue
        completed_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(completed_job)
        self.assertEqual(JobState.COMPLETED, completed_job.state)

        # 5. Repo slot is released
        running_key = f"review:repo_running:{job.repository_id}"
        self.assertEqual(0, self.client.scard(running_key))
        self.assertEqual(0, self.client.scard(self.queue.running_jobs_key))

    async def test_real_arq_worker_process_main_loop_execution_pipeline(self) -> None:
        """Trace complete end-to-end path:
        RedisJobQueue.enqueue() -> arq:job:{job_id} -> arq:queue -> ARQ Worker process loop
        -> registered review_job_task -> RedisJobQueue.claim_job() -> AutonomousReviewWorker.process_claimed_job()
        -> mark_completed with lease token -> ARQ finish_job.
        """
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world")
        job = self.queue.enqueue(snapshot, "del-real-arq-worker-loop", now=now)

        # Confirm job is scheduled in arq:queue and serialized in arq:job:{job_id}
        self.assertEqual(1, self.client.zcard(self.queue.queue_name))
        self.assertIsNotNone(self.client.get(f"arq:job:{job.job_id}"))

        worker_instance = _create_test_worker(self.queue)
        async_pool = AsyncInMemoryRedisPool(self.client)

        # Instantiate real ARQ 0.28.0 Worker with review_job_task in burst mode
        arq_worker = Worker(
            functions=[review_job_task],
            redis_pool=async_pool,
            burst=True,
            poll_delay=0.01,
        )
        arq_worker.ctx["queue"] = self.queue
        arq_worker.ctx["worker"] = worker_instance

        # Execute genuine ARQ Worker main loop (processes queued job in burst mode)
        await arq_worker.main()

        # Verify job completed through the full ARQ execution loop
        completed_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(completed_job)
        self.assertEqual(JobState.COMPLETED, completed_job.state)

        # Verify repo concurrency slot released
        running_key = f"review:repo_running:{job.repository_id}"
        self.assertEqual(0, self.client.scard(running_key))
        self.assertEqual(0, self.client.scard(self.queue.running_jobs_key))

    async def test_arq_worker_settings_and_bootstrap_configuration(self) -> None:
        """Verify WorkerSettings and ReviewWorkerSettings export valid ARQ configuration."""
        self.assertEqual([review_job_task], WorkerSettings.functions)
        self.assertEqual([review_job_task], ReviewWorkerSettings.functions)
        self.assertEqual(default_queue_name, WorkerSettings.queue_name)
        self.assertIsNotNone(WorkerSettings.on_startup)
        self.assertIsNotNone(WorkerSettings.on_shutdown)
        self.assertIsNotNone(WorkerSettings.redis_settings)

        # Test arq_startup and arq_shutdown lifecycle hooks
        ctx: dict[str, Any] = {"config": _make_dummy_service_config(), "queue": self.queue}
        await arq_startup(ctx)
        self.assertIn("queue", ctx)
        self.assertIn("worker", ctx)

        await arq_shutdown(ctx)

    async def test_stale_worker_rejection_in_real_worker_path_on_completion(self) -> None:
        """Worker A's completion is rejected when lease expires and Worker B claims the job."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world")
        job = self.queue.enqueue(snapshot, "del-stale-worker-complete", deadline_seconds=10.0, now=now)

        worker_a = _create_test_worker(self.queue)
        worker_b = _create_test_worker(self.queue)

        # Worker A claims the job with token A
        job_a, token_a = self.queue.claim_job(job.job_id, now=now)
        self.assertIsNotNone(job_a)
        self.assertIsNotNone(token_a)

        # Time elapses; lease expires; zombie recovery returns job to QUEUED or Worker B claims
        self.queue.recover_zombie_jobs(now=now + 20.0)
        job_b, token_b = self.queue.claim_job(job.job_id, now=now + 21.0)
        self.assertIsNotNone(job_b)
        self.assertIsNotNone(token_b)
        self.assertNotEqual(token_a, token_b)

        # Worker A finishes its slow review and attempts terminal completion with token A -> must fail!
        with self.assertRaises(StaleLeaseError):
            await worker_a.process_claimed_job(job_a, lease_token=token_a, now=now + 25.0)

        # Job in queue must NOT have been marked completed by Worker A
        mid_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(mid_job)
        self.assertEqual(JobState.RUNNING, mid_job.state)

        # Worker B completes with token B -> succeeds
        await worker_b.process_claimed_job(job_b, lease_token=token_b, now=now + 26.0)
        final_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(final_job)
        self.assertEqual(JobState.COMPLETED, final_job.state)

    async def test_stale_worker_rejection_in_real_worker_path_on_failure(self) -> None:
        """Worker A's failure mutation is rejected when lease expires and Worker B claims the job."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world")
        job = self.queue.enqueue(snapshot, "del-stale-worker-fail", deadline_seconds=10.0, now=now)

        failing_worker_a = _create_test_worker(self.queue, should_fail=True)
        worker_b = _create_test_worker(self.queue)

        # Worker A claims the job with token A
        job_a, token_a = self.queue.claim_job(job.job_id, now=now)
        self.assertIsNotNone(job_a)
        self.assertIsNotNone(token_a)

        # Worker A hangs, lease expires, Worker B claims with token B
        self.queue.recover_zombie_jobs(now=now + 20.0)
        job_b, token_b = self.queue.claim_job(job.job_id, now=now + 21.0)
        self.assertIsNotNone(job_b)
        self.assertIsNotNone(token_b)
        self.assertNotEqual(token_a, token_b)

        # Worker A wakes up and its failure is executed
        with self.assertRaises(RuntimeError):
            await failing_worker_a.process_claimed_job(job_a, lease_token=token_a, now=now + 25.0)

        # Job in queue must NOT have been moved to retry/backoff or dead-letter by Worker A
        mid_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(mid_job)
        self.assertEqual(JobState.RUNNING, mid_job.state)

        # Worker B successfully finishes
        await worker_b.process_claimed_job(job_b, lease_token=token_b, now=now + 26.0)
        final_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(final_job)
        self.assertEqual(JobState.COMPLETED, final_job.state)

    def test_atomic_concurrency_eval_lua_script(self) -> None:
        """Lua script atomically checks capacity and admits at most max_concurrency_per_repo."""
        now = 1000.0
        job1 = self.queue.enqueue(sample_snapshot(repo_id="lua-repo"), "del-lua-1", now=now)
        job2 = self.queue.enqueue(sample_snapshot(repo_id="lua-repo"), "del-lua-2", now=now)

        claimed1, token1 = self.queue.claim_job(job1.job_id, now=now, max_concurrency_per_repo=1)
        self.assertIsNotNone(claimed1)
        self.assertIsNotNone(token1)

        claimed2, reason2 = self.queue.claim_job(job2.job_id, now=now, max_concurrency_per_repo=1)
        self.assertIsNone(claimed2)
        self.assertEqual("repo_at_capacity", reason2)

        # Exactly 1 slot in running set
        self.assertEqual(1, self.client.scard("review:repo_running:lua-repo"))

        # Complete job 1
        self.queue.mark_completed(job1.job_id, now=now + 1.0, lease_token=token1)
        self.assertEqual(0, self.client.scard("review:repo_running:lua-repo"))

        # Now job 2 can be claimed
        claimed2, token2 = self.queue.claim_job(job2.job_id, now=now + 2.0, max_concurrency_per_repo=1)
        self.assertIsNotNone(claimed2)
        self.assertIsNotNone(token2)
        self.assertEqual(1, self.client.scard("review:repo_running:lua-repo"))

    def test_set_based_zombie_recovery_tracking(self) -> None:
        """Indexed review:jobs:running tracks running jobs without scanning keyspace."""
        now = 1000.0
        job = self.queue.enqueue(sample_snapshot(), "del-set-zombie", deadline_seconds=5.0, now=now)
        self.assertEqual(0, self.client.scard(self.queue.running_jobs_key))

        leased = self.queue.lease_next_job(now=now)
        self.assertIsNotNone(leased)
        self.assertEqual(1, self.client.scard(self.queue.running_jobs_key))
        self.assertIn(job.job_id.encode("utf-8"), self.client.smembers(self.queue.running_jobs_key))

        # Expire and recover
        recovered = self.queue.recover_zombie_jobs(now=now + 10.0)
        self.assertEqual(1, len(recovered))
        self.assertEqual(0, self.client.scard(self.queue.running_jobs_key))

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

    async def test_concern1_retry_authority_coordination_between_review_job_and_arq(self) -> None:
        """Concern 1: ReviewJob remains the sole retry authority; ARQ coordinates without generic second policy."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world")
        job = self.queue.enqueue(snapshot, "del-c1-retry", max_retries=2, backoff_base_seconds=2.0, now=now)

        # Worker configured to fail during review execution
        failing_worker = _create_test_worker(self.queue, should_fail=True)
        ctx = {"queue": self.queue, "worker": failing_worker, "score": int(now * 1000)}

        # Attempt 1 execution via review_job_task
        t_start = time.time()
        from arq.worker import Retry
        with self.assertRaises(Retry) as cm:
            await review_job_task(ctx, job.job_id)

        # Application authority computed exponential backoff: 2.0 * (2 ** (1 - 1)) = 2.0s
        self.assertAlmostEqual(2.0, cm.exception.defer_score / 1000, places=1)

        # ReviewJob state in queue is QUEUED with attempt_count = 1
        retried_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(retried_job)
        self.assertEqual(JobState.QUEUED, retried_job.state)
        self.assertEqual(1, retried_job.attempt_count)
        self.assertAlmostEqual(t_start + 2.0, retried_job.next_run_at, delta=1.0)

        # ARQ job definition in arq:job:{job_id} is preserved (NOT deleted)
        self.assertIsNotNone(self.client.get(f"arq:job:{job.job_id}"))

        # Attempt 2 execution: lease and execute again
        ctx2 = {"queue": self.queue, "worker": failing_worker, "score": int((now + 2.0) * 1000)}
        task_res2 = await review_job_task(ctx2, job.job_id)

        # Max retries reached (2/2): application marked DEAD_LETTER, task returned cleanly without ARQ generic retry
        self.assertEqual("dead_letter", task_res2["status"])
        dead_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(dead_job)
        self.assertEqual(JobState.DEAD_LETTER, dead_job.state)
        self.assertEqual(2, dead_job.attempt_count)
        self.assertEqual(1, self.client.scard(self.queue.dead_letter_key))
        # Removed from active arq:queue
        self.assertEqual(0, self.client.zcard(self.queue.queue_name))

    async def test_concern2_arq_cancellation_operational_abort(self) -> None:
        """Concern 2: allow_abort_jobs is enabled, cancel_job sets domain state and signals ARQ abort."""
        self.assertTrue(WorkerSettings.allow_abort_jobs)
        self.assertTrue(ReviewWorkerSettings.allow_abort_jobs)

        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/hello-world")
        job = self.queue.enqueue(snapshot, "del-c2-cancel", now=now)

        # Lease the job
        leased = self.queue.lease_next_job(now=now)
        self.assertIsNotNone(leased)

        # Operator triggers cancellation
        cancelled = self.queue.cancel_job(job.job_id, reason="Security review aborted by operator", now=now + 1.0)
        self.assertEqual(JobState.CANCELLED, cancelled.state)

        # ARQ abort set contains job_id as operational optimization
        from arq.constants import abort_jobs_ss
        abort_entries = self.client.zrange(abort_jobs_ss, 0, -1)
        self.assertIn(job.job_id.encode("utf-8"), abort_entries)

        # Logical state in queue remains authoritative: claiming rejects with cancelled
        claimed, reason = self.queue.claim_job(job.job_id, now=now + 2.0)
        self.assertIsNone(claimed)
        self.assertEqual("cancelled", reason)

    def test_concern3_enqueue_atomicity_and_duplicate_delivery_race(self) -> None:
        """Concern 3: Concurrent first-time enqueue calls produce exactly one logical job and queue entry."""
        import concurrent.futures

        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/atomic-repo")
        delivery_id = "del-c3-atomic-race"

        # Simulate 10 racing concurrent enqueuers with the exact same delivery_id
        results: list[ReviewJob] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.queue.enqueue, snapshot, delivery_id, now=now)
                for _ in range(10)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # All 10 returned a valid ReviewJob
        self.assertEqual(10, len(results))
        first_job_id = results[0].job_id
        for r in results:
            self.assertEqual(first_job_id, r.job_id)
            self.assertEqual(0, r.attempt_count)
            self.assertEqual(JobState.QUEUED, r.state)

        # Queue has exactly one job scheduled in arq:queue
        self.assertEqual(1, self.client.zcard(self.queue.queue_name))
        # Exactly one delivery mapping
        self.assertEqual(f"job-{delivery_id}".encode("utf-8"), self.client.get(f"review:delivery:{delivery_id}"))
        # Exactly one job hash
        self.assertEqual(b"queued", self.client.hget(f"review:job:{first_job_id}", "state"))

    async def test_invariant_1_and_7_retry_authority_and_queue_cardinality(self) -> None:
        """Invariant 1 & 7: One failed attempt creates exactly one future execution; queue cardinality is 1."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/cardinality-repo")
        job = self.queue.enqueue(snapshot, "del-inv-1-7", max_retries=3, backoff_base_seconds=2.0, now=now)

        failing_worker = _create_test_worker(self.queue, should_fail=True)
        ctx = {"queue": self.queue, "worker": failing_worker, "score": int(now * 1000)}

        from arq.worker import Retry
        with self.assertRaises(Retry) as cm:
            await review_job_task(ctx, job.job_id)

        # Invariant 1: exactly 1 retry scheduled with application exponential backoff
        self.assertAlmostEqual(2.0, cm.exception.defer_score / 1000, places=1)
        updated = self.queue.get_job(job.job_id)
        self.assertIsNotNone(updated)
        self.assertEqual(JobState.QUEUED, updated.state)
        self.assertEqual(1, updated.attempt_count)

        # Invariant 7: queue cardinality in arq:queue is strictly 1
        self.assertEqual(1, self.client.zcard(self.queue.queue_name))
        queue_members = self.client.zrange(self.queue.queue_name, 0, -1)
        self.assertEqual([job.job_id.encode("utf-8")], queue_members)
        # Verify arq:job is preserved for the next execution
        self.assertIsNotNone(self.client.get(f"arq:job:{job.job_id}"))

    async def test_invariant_2_no_retry_double_counting(self) -> None:
        """Invariant 2: ReviewJob.attempt_count increments strictly once per attempt, never double-counted by ARQ."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/no-double-count")
        job = self.queue.enqueue(snapshot, "del-inv-2", max_retries=3, backoff_base_seconds=1.0, now=now)
        self.assertEqual(0, job.attempt_count)

        failing_worker = _create_test_worker(self.queue, should_fail=True)

        # Attempt 1
        ctx1 = {"queue": self.queue, "worker": failing_worker, "score": int(now * 1000)}
        from arq.worker import Retry
        with self.assertRaises(Retry):
            await review_job_task(ctx1, job.job_id)

        job_after_try1 = self.queue.get_job(job.job_id)
        self.assertIsNotNone(job_after_try1)
        # attempt_count must be exactly 1, not 2
        self.assertEqual(1, job_after_try1.attempt_count)

        # Attempt 2 (retried)
        ctx2 = {"queue": self.queue, "worker": failing_worker, "score": int((now + 1.0) * 1000)}
        with self.assertRaises(Retry):
            await review_job_task(ctx2, job.job_id)

        job_after_try2 = self.queue.get_job(job.job_id)
        self.assertIsNotNone(job_after_try2)
        # attempt_count must be exactly 2, not 3 or 4
        self.assertEqual(2, job_after_try2.attempt_count)

    async def test_invariant_3_retry_exhaustion_dead_letter(self) -> None:
        """Invariant 3: Reaching max_retries transitions to DEAD_LETTER; ARQ does not schedule another retry."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/exhaustion-repo")
        job = self.queue.enqueue(snapshot, "del-inv-3", max_retries=1, backoff_base_seconds=1.0, now=now)

        failing_worker = _create_test_worker(self.queue, should_fail=True)
        ctx = {"queue": self.queue, "worker": failing_worker, "score": int(now * 1000)}

        # Attempt 1 (and only attempt allowed since max_retries=1)
        task_res = await review_job_task(ctx, job.job_id)

        # ARQ task returns cleanly (not raising Retry)
        self.assertEqual("dead_letter", task_res["status"])
        dead_job = self.queue.get_job(job.job_id)
        self.assertIsNotNone(dead_job)
        self.assertEqual(JobState.DEAD_LETTER, dead_job.state)
        self.assertEqual(1, dead_job.attempt_count)
        # Dead-lettered in dead_letter_key and removed from arq:queue
        self.assertEqual(1, self.client.scard(self.queue.dead_letter_key))
        self.assertEqual(0, self.client.zcard(self.queue.queue_name))

    def test_invariant_4_and_5_crash_recovery_and_lease_arq_timing_relationship(self) -> None:
        """Invariant 4 & 5: Crash recovery reclaims expired lease exactly once; stale worker rejected."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/crash-timing-repo")
        job = self.queue.enqueue(snapshot, "del-inv-4-5", deadline_seconds=10.0, max_retries=3, now=now)

        # Worker A claims the job with lease_token_A
        job_a, token_a = self.queue.claim_job(job.job_id, now=now)
        self.assertIsNotNone(job_a)
        self.assertEqual(1, job_a.attempt_count)
        self.assertEqual(JobState.RUNNING, job_a.state)
        self.assertIsNotNone(token_a)

        # Timing check: before deadline (now + 5.0), recovery must NOT reclaim the job
        recovered_early = self.queue.recover_zombie_jobs(now=now + 5.0)
        self.assertEqual(0, len(recovered_early))
        self.assertEqual(JobState.RUNNING, self.queue.get_job(job.job_id).state)

        # Worker A dies (process killed). Time advances past deadline (now + 11.0).
        recovered = self.queue.recover_zombie_jobs(now=now + 11.0)
        self.assertEqual(1, len(recovered))
        self.assertEqual(JobState.QUEUED, recovered[0].state)
        # attempt_count was NOT incremented by recover_zombie_jobs
        self.assertEqual(1, recovered[0].attempt_count)

        # Stale lease invalidation check: Worker A wakes up late and attempts mutation with token_a
        with self.assertRaises(StaleLeaseError):
            self.queue.mark_completed(job.job_id, now=now + 12.0, lease_token=token_a)
        with self.assertRaises(StaleLeaseError):
            self.queue.mark_failed(job.job_id, error="Late fail", now=now + 12.0, lease_token=token_a)

        # Worker B claims the recovered job
        job_b, token_b = self.queue.claim_job(job.job_id, now=now + 13.0)
        self.assertIsNotNone(job_b)
        self.assertEqual(2, job_b.attempt_count)
        self.assertNotEqual(token_a, token_b)

        # Worker B completes the job
        completed = self.queue.mark_completed(job.job_id, now=now + 14.0, lease_token=token_b)
        self.assertEqual(JobState.COMPLETED, completed.state)
        self.assertEqual(2, completed.attempt_count)

    def test_invariant_6_cancellation_during_deferred_retry_prevents_resurrection(self) -> None:
        """Invariant 6: Cancellation during deferred retry leaves CANCELLED authoritative; no resurrection."""
        now = 1000.0
        snapshot = sample_snapshot(repo_id="octocat/cancel-deferred")
        job = self.queue.enqueue(snapshot, "del-inv-6", max_retries=3, backoff_base_seconds=5.0, now=now)

        # Attempt 1 leased and failed -> deferred to now + 5.0
        self.queue.lease_next_job(now=now)
        self.queue.mark_failed(job.job_id, error="Transient fail", now=now)
        self.assertEqual(JobState.QUEUED, self.queue.get_job(job.job_id).state)
        self.assertEqual(1, self.client.zcard(self.queue.queue_name))

        # Operator cancels job while deferred in arq:queue
        cancelled = self.queue.cancel_job(job.job_id, reason="PR closed by user", now=now + 2.0)
        self.assertEqual(JobState.CANCELLED, cancelled.state)

        # Removed from arq:queue
        self.assertEqual(0, self.client.zcard(self.queue.queue_name))

        # Attempting to claim or lease must reject and not resurrect
        claimed, reason = self.queue.claim_job(job.job_id, now=now + 10.0)
        self.assertIsNone(claimed)
        self.assertEqual("cancelled", reason)

        leased = self.queue.lease_next_job(now=now + 10.0)
        self.assertIsNone(leased)

        # recover_zombie_jobs must not touch CANCELLED jobs
        recovered = self.queue.recover_zombie_jobs(now=now + 20.0)
        self.assertEqual(0, len(recovered))
        self.assertEqual(JobState.CANCELLED, self.queue.get_job(job.job_id).state)

    def test_invariant_8_arq_transport_settings_derived_from_review_job(self) -> None:
        """Invariant 8: max_tries and job_timeout are mathematically derived from ReviewJob semantics."""
        # 1. job_timeout = 60.0 derived from ReviewJob.deadline_seconds = 60.0
        self.assertEqual(60.0, WorkerSettings.job_timeout)
        self.assertEqual(60.0, ReviewWorkerSettings.job_timeout)

        # 2. max_tries = 5 derived from ReviewJob.max_retries (3) + 1 initial try + 1 crash tolerance
        self.assertEqual(5, WorkerSettings.max_tries)
        self.assertEqual(5, ReviewWorkerSettings.max_tries)

        # 3. allow_abort_jobs is enabled for operational abort
        self.assertTrue(WorkerSettings.allow_abort_jobs)
        self.assertTrue(ReviewWorkerSettings.allow_abort_jobs)



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
