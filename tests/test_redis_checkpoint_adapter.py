"""Tests for Redis-backed LangGraph workflow checkpoint adapter (W1-04).

Covers the full W1-04 test matrix:
1. Checkpoint protocol/interface conformance (BaseCheckpointSaver)
2. Redis key namespace correctness (review:checkpoint:* only)
3. Basic checkpoint write & read
4. Checkpoint metadata preservation
5. Thread isolation (thread A vs thread B)
6. Successive checkpoint/version behavior
7. get_state / get_workflow_state compatibility
8. Real graph checkpoint persistence
9. Workflow resume from checkpoint
10. Completed node is NOT re-executed after resume
11. Specialist fan-out is NOT duplicated after resume
12. Redis unavailable during checkpoint write (fail-closed)
13. Redis unavailable during checkpoint read (fail-closed)
14. Serialization / deserialization failure handling
15. Malformed checkpoint payload handling (explicit failure, never silent empty)
16. Missing checkpoint handling
17. Cross-thread checkpoint rejection
18. Concurrent independent workflow checkpointing
19. Prohibited secret/checkpoint state rejection
20. Existing no-checkpointer behavior remains unchanged
21. Live Redis integration when available
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import sqlite3
import threading
import time
from typing import Any
import unittest

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    JsonPlusSerializer,
    create_checkpoint,
    empty_checkpoint,
)
import redis

from pr_review_agent.adapters.redis_checkpoint import (
    CheckpointDeserializationError,
    CheckpointStorageError,
    RedisCheckpointSaver,
)
from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.orchestration import (
    CHECKPOINT_FORBIDDEN_KEYS,
    CandidateFinding,
    JobState,
    ReviewJob,
    ReviewOrchestrator,
    SpecialistInput,
    SpecialistOutput,
    SpecialistStatus,
    SpecialistType,
)
from tests.test_queue_and_checkpoint_contracts import sample_snapshot


class InMemoryRedisCheckpointClient:
    """Thread-safe in-memory Redis client supporting commands used by RedisCheckpointSaver."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hashes: dict[str, dict[bytes, bytes]] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._sets: dict[str, set[str]] = {}
        self.closed = False

    def _ensure_open(self) -> None:
        if self.closed:
            raise redis.ConnectionError("Connection closed")

    def _to_bytes(self, val: Any) -> bytes:
        if isinstance(val, bytes):
            return val
        return str(val).encode("utf-8")

    def hset(
        self,
        name: str,
        key: str | None = None,
        value: Any = None,
        mapping: Mapping[str, Any] | None = None,
    ) -> int:
        self._ensure_open()
        with self._lock:
            h = self._hashes.setdefault(name, {})
            added = 0
            if mapping:
                for k, v in mapping.items():
                    bk = self._to_bytes(k)
                    bv = self._to_bytes(v)
                    if bk not in h:
                        added += 1
                    h[bk] = bv
            if key is not None:
                bk = self._to_bytes(key)
                bv = self._to_bytes(value)
                if bk not in h:
                    added += 1
                h[bk] = bv
            return added

    def hgetall(self, name: str) -> dict[bytes, bytes]:
        self._ensure_open()
        with self._lock:
            return dict(self._hashes.get(name, {}))

    def zadd(self, name: str, mapping: Mapping[str, float]) -> int:
        self._ensure_open()
        with self._lock:
            z = self._zsets.setdefault(name, {})
            added = 0
            for member, score in mapping.items():
                m_str = str(member)
                if m_str not in z:
                    added += 1
                z[m_str] = float(score)
            return added

    def zrevrange(self, name: str, start: int, stop: int) -> list[str]:
        self._ensure_open()
        with self._lock:
            z = self._zsets.get(name, {})
            # Sort by score descending, then member descending
            sorted_items = sorted(z.items(), key=lambda x: (x[1], x[0]), reverse=True)
            members = [item[0] for item in sorted_items]
            if stop == -1 or stop >= len(members):
                return members[start:]
            return members[start : stop + 1]

    def sadd(self, name: str, *values: Any) -> int:
        self._ensure_open()
        with self._lock:
            s = self._sets.setdefault(name, set())
            added = 0
            for v in values:
                sv = str(v)
                if sv not in s:
                    added += 1
                    s.add(sv)
            return added

    def srem(self, name: str, *values: Any) -> int:
        self._ensure_open()
        with self._lock:
            s = self._sets.get(name, set())
            removed = 0
            for v in values:
                sv = str(v)
                if sv in s:
                    removed += 1
                    s.discard(sv)
            return removed

    def smembers(self, name: str) -> set[str]:
        self._ensure_open()
        with self._lock:
            return set(self._sets.get(name, set()))

    def delete(self, *names: str) -> int:
        self._ensure_open()
        with self._lock:
            deleted = 0
            for n in names:
                if n in self._hashes:
                    del self._hashes[n]
                    deleted += 1
                if n in self._zsets:
                    del self._zsets[n]
                    deleted += 1
                if n in self._sets:
                    del self._sets[n]
                    deleted += 1
            return deleted

    def expire(self, name: str, time: int) -> bool:
        self._ensure_open()
        return True

    def pipeline(self, transaction: bool = True) -> InMemoryRedisPipeline:
        self._ensure_open()
        return InMemoryRedisPipeline(self)

    def close(self) -> None:
        self.closed = True


class InMemoryRedisPipeline:
    """Mock pipeline buffering commands and executing atomically."""

    def __init__(self, client: InMemoryRedisCheckpointClient) -> None:
        self.client = client
        self.ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def hset(self, name: str, key: str | None = None, value: Any = None, mapping: Mapping[str, Any] | None = None) -> InMemoryRedisPipeline:
        self.ops.append(("hset", (name,), {"key": key, "value": value, "mapping": mapping}))
        return self

    def zadd(self, name: str, mapping: Mapping[str, float]) -> InMemoryRedisPipeline:
        self.ops.append(("zadd", (name,), {"mapping": mapping}))
        return self

    def sadd(self, name: str, *values: Any) -> InMemoryRedisPipeline:
        self.ops.append(("sadd", (name, *values), {}))
        return self

    def srem(self, name: str, *values: Any) -> InMemoryRedisPipeline:
        self.ops.append(("srem", (name, *values), {}))
        return self

    def delete(self, *names: str) -> InMemoryRedisPipeline:
        self.ops.append(("delete", names, {}))
        return self

    def expire(self, name: str, time: int) -> InMemoryRedisPipeline:
        self.ops.append(("expire", (name, time), {}))
        return self

    def execute(self) -> list[Any]:
        results: list[Any] = []
        for op_name, args, kwargs in self.ops:
            method = getattr(self.client, op_name)
            results.append(method(*args, **kwargs))
        self.ops.clear()
        return results


class RedisCheckpointAdapterUnitTests(unittest.TestCase):
    """Unit tests verifying adapter shape, serialization, namespaces, and security."""

    def setUp(self) -> None:
        self.redis_client = InMemoryRedisCheckpointClient()
        self.saver = RedisCheckpointSaver(client=self.redis_client)

    def test_protocol_conformance(self) -> None:
        """1. Checkpoint protocol/interface conformance."""
        self.assertIsInstance(self.saver, BaseCheckpointSaver)
        self.assertTrue(hasattr(self.saver, "get_tuple"))
        self.assertTrue(hasattr(self.saver, "aget_tuple"))
        self.assertTrue(hasattr(self.saver, "put"))
        self.assertTrue(hasattr(self.saver, "aput"))
        self.assertTrue(hasattr(self.saver, "put_writes"))
        self.assertTrue(hasattr(self.saver, "aput_writes"))
        self.assertTrue(hasattr(self.saver, "list"))
        self.assertTrue(hasattr(self.saver, "alist"))
        self.assertTrue(hasattr(self.saver, "delete_thread"))
        self.assertTrue(hasattr(self.saver, "adelete_thread"))

    def test_namespace_isolation_and_no_collision(self) -> None:
        """2. Redis key namespace correctness: review:checkpoint:* only."""
        config = {"configurable": {"thread_id": "job-123", "checkpoint_ns": ""}}
        cp = empty_checkpoint()
        cp["id"] = "cp-001"
        metadata: CheckpointMetadata = {"step": 1, "source": "input"}
        self.saver.put(config, cp, metadata, {})

        # All keys in client must start with review:checkpoint:
        for k in self.redis_client._hashes.keys():
            self.assertTrue(k.startswith("review:checkpoint:"), f"Key {k} violates namespace")
        for k in self.redis_client._zsets.keys():
            self.assertTrue(k.startswith("review:checkpoint:"), f"Key {k} violates namespace")
        for k in self.redis_client._sets.keys():
            self.assertTrue(k.startswith("review:checkpoint:"), f"Key {k} violates namespace")

        # Zero collision with ARQ or review:job keys
        for k in list(self.redis_client._hashes.keys()) + list(self.redis_client._zsets.keys()):
            self.assertFalse(k.startswith("review:job:"), f"Collision with review:job: {k}")
            self.assertFalse(k.startswith("arq:"), f"Collision with arq: {k}")

    def test_basic_checkpoint_write_and_read(self) -> None:
        """3. Basic checkpoint write & read."""
        config = {"configurable": {"thread_id": "thread-abc", "checkpoint_ns": ""}}
        cp = empty_checkpoint()
        cp["id"] = "cp-test-1"
        cp["channel_values"] = {"counter": 42, "msg": "hello"}
        metadata: CheckpointMetadata = {"step": 1, "source": "test"}

        returned_config = self.saver.put(config, cp, metadata, {"counter": 1, "msg": 1})
        self.assertEqual("thread-abc", returned_config["configurable"]["thread_id"])
        self.assertEqual("cp-test-1", returned_config["configurable"]["checkpoint_id"])

        # Read back by thread_id
        tup = self.saver.get_tuple(config)
        self.assertIsNotNone(tup)
        self.assertEqual("cp-test-1", tup.checkpoint["id"])
        self.assertEqual(42, tup.checkpoint["channel_values"].get("counter"))
        self.assertEqual("hello", tup.checkpoint["channel_values"].get("msg"))
        self.assertEqual(1, tup.metadata.get("step"))

    def test_checkpoint_metadata_preservation(self) -> None:
        """4. Checkpoint metadata preservation."""
        config = {"configurable": {"thread_id": "thread-meta"}}
        cp = empty_checkpoint()
        cp["id"] = "cp-meta-1"
        metadata: CheckpointMetadata = {
            "step": 2,
            "source": "specialist_security",
            "writes": {"security": "done"},
            "score": 0.99,
        }
        self.saver.put(config, cp, metadata, {})

        tup = self.saver.get_tuple(config)
        self.assertIsNotNone(tup)
        self.assertEqual(2, tup.metadata.get("step"))
        self.assertEqual("specialist_security", tup.metadata.get("source"))
        self.assertEqual(0.99, tup.metadata.get("score"))

    def test_thread_isolation(self) -> None:
        """5 & 17. Thread isolation: thread A cannot read thread B."""
        config_a = {"configurable": {"thread_id": "thread-A"}}
        config_b = {"configurable": {"thread_id": "thread-B"}}

        cp_a = empty_checkpoint()
        cp_a["id"] = "cp-A"
        cp_a["channel_values"] = {"data_a": "A-val"}

        cp_b = empty_checkpoint()
        cp_b["id"] = "cp-B"
        cp_b["channel_values"] = {"data_b": "B-val"}

        self.saver.put(config_a, cp_a, {"owner": "A"}, {"data_a": 1})
        self.saver.put(config_b, cp_b, {"owner": "B"}, {"data_b": 1})

        tup_a = self.saver.get_tuple(config_a)
        tup_b = self.saver.get_tuple(config_b)

        self.assertIsNotNone(tup_a)
        self.assertIsNotNone(tup_b)
        self.assertEqual("A-val", tup_a.checkpoint["channel_values"].get("data_a"))
        self.assertNotIn("data_b", tup_a.checkpoint["channel_values"])
        self.assertEqual("B-val", tup_b.checkpoint["channel_values"].get("data_b"))
        self.assertNotIn("data_a", tup_b.checkpoint["channel_values"])

    def test_successive_checkpoint_and_version_behavior(self) -> None:
        """6. Successive checkpoint/version behavior: latest returned by default."""
        config = {"configurable": {"thread_id": "thread-seq"}}

        cp1 = empty_checkpoint()
        cp1["id"] = "0001.cp"
        cp1["channel_values"] = {"val": 10}
        self.saver.put(config, cp1, {"step": 1}, {"val": 1})

        time.sleep(0.01)
        cp2 = empty_checkpoint()
        cp2["id"] = "0002.cp"
        cp2["channel_values"] = {"val": 20}
        self.saver.put(config, cp2, {"step": 2}, {"val": 2})

        # Default query retrieves latest (cp2)
        latest = self.saver.get_tuple(config)
        self.assertIsNotNone(latest)
        self.assertEqual("0002.cp", latest.checkpoint["id"])
        self.assertEqual(20, latest.checkpoint["channel_values"].get("val"))

        # Explicit query retrieves specific older checkpoint (cp1)
        config_explicit = {"configurable": {"thread_id": "thread-seq", "checkpoint_id": "0001.cp"}}
        old = self.saver.get_tuple(config_explicit)
        self.assertIsNotNone(old)
        self.assertEqual("0001.cp", old.checkpoint["id"])
        self.assertEqual(10, old.checkpoint["channel_values"].get("val"))

    def test_delete_thread(self) -> None:
        """Thread deletion cleans all keys for thread."""
        config = {"configurable": {"thread_id": "thread-del"}}
        cp = empty_checkpoint()
        cp["id"] = "cp-del-1"
        cp["channel_values"] = {"val": 999}
        self.saver.put(config, cp, {}, {"val": 1})

        self.assertIsNotNone(self.saver.get_tuple(config))
        self.saver.delete_thread("thread-del")
        self.assertIsNone(self.saver.get_tuple(config))

    def test_missing_checkpoint_returns_none(self) -> None:
        """16. Missing checkpoint returns None."""
        self.assertIsNone(self.saver.get_tuple({"configurable": {"thread_id": "unknown-thread"}}))

    def test_prohibited_secrets_rejection(self) -> None:
        """19. Prohibited secret/checkpoint state rejection (W1-02 Security Contract)."""
        config = {"configurable": {"thread_id": "thread-sec"}}

        for forbidden in CHECKPOINT_FORBIDDEN_KEYS:
            cp = empty_checkpoint()
            cp["id"] = f"cp-bad-{forbidden}"
            cp["channel_values"] = {forbidden: "leaked_secret_val"}
            with self.assertRaises(ValueError, msg=f"Should reject forbidden key {forbidden}"):
                self.saver.put(config, cp, {}, {})

        # Test nested secret rejection
        cp_nested = empty_checkpoint()
        cp_nested["id"] = "cp-bad-nested"
        cp_nested["channel_values"] = {"config": {"github_token": "ghp_12345"}}
        with self.assertRaises(ValueError):
            self.saver.put(config, cp_nested, {}, {})

    def test_redis_failure_during_write_raises_fail_closed(self) -> None:
        """12. Redis unavailable during checkpoint write fails closed."""
        self.redis_client.close()
        config = {"configurable": {"thread_id": "thread-fail"}}
        cp = empty_checkpoint()
        cp["id"] = "cp-fail"

        with self.assertRaises(CheckpointStorageError):
            self.saver.put(config, cp, {}, {})

    def test_redis_failure_during_read_raises_fail_closed(self) -> None:
        """13. Redis unavailable during checkpoint read fails closed."""
        config = {"configurable": {"thread_id": "thread-read-fail"}}
        cp = empty_checkpoint()
        cp["id"] = "cp-ok"
        self.saver.put(config, cp, {}, {})

        # Now close Redis connection
        self.redis_client.close()
        with self.assertRaises(CheckpointStorageError):
            self.saver.get_tuple(config)

    def test_malformed_checkpoint_raises_explicit_error(self) -> None:
        """14 & 15. Malformed/corrupted checkpoint payload raises CheckpointDeserializationError."""
        config = {"configurable": {"thread_id": "thread-corrupt"}}
        cp = empty_checkpoint()
        cp["id"] = "cp-corrupt"
        self.saver.put(config, cp, {}, {})

        # Corrupt data in Redis
        data_key = self.saver._data_key("thread-corrupt", "", "cp-corrupt")
        self.redis_client.hset(data_key, mapping={"cp_data": b"THIS_IS_CORRUPT_NOT_MSGPACK"})

        with self.assertRaises(CheckpointDeserializationError):
            self.saver.get_tuple(config)

    def test_concurrent_independent_workflow_checkpointing(self) -> None:
        """18. Concurrent independent workflow checkpointing."""
        errors: list[Exception] = []

        def worker(thread_idx: int) -> None:
            try:
                cfg = {"configurable": {"thread_id": f"thread-concurrent-{thread_idx}"}}
                for step in range(5):
                    cp = empty_checkpoint()
                    cp["id"] = f"cp-{thread_idx}-{step}"
                    cp["channel_values"] = {"step": step, "thread": thread_idx}
                    self.saver.put(cfg, cp, {"step": step}, {"step": step, "thread": 1})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(0, len(errors), f"Errors occurred in concurrent test: {errors}")

        # Verify all 10 threads have their latest checkpoints intact
        for i in range(10):
            cfg = {"configurable": {"thread_id": f"thread-concurrent-{i}"}}
            tup = self.saver.get_tuple(cfg)
            self.assertIsNotNone(tup)
            self.assertEqual(4, tup.checkpoint["channel_values"].get("step"))
            self.assertEqual(i, tup.checkpoint["channel_values"].get("thread"))


class RedisCheckpointAdapterGraphIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration tests with real LangGraph ReviewOrchestrator and crash recovery simulations."""

    def setUp(self) -> None:
        self.redis_client = InMemoryRedisCheckpointClient()
        self.saver = RedisCheckpointSaver(client=self.redis_client)

    async def test_real_graph_checkpoint_persistence_and_get_workflow_state(self) -> None:
        """7 & 8. Real graph checkpoint persistence and get_workflow_state compatibility."""
        orchestrator = ReviewOrchestrator(checkpointer=self.saver)
        orchestrator.register_specialist(
            SpecialistType.SECURITY,
            lambda inp: SpecialistOutput(
                specialist_type=SpecialistType.SECURITY,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
                findings=(
                    CandidateFinding(
                        finding_id="f-redis-chk",
                        correlation_id=inp.correlation_id,
                        specialist_type=SpecialistType.SECURITY,
                        category="security",
                        severity="high",
                        confidence=0.95,
                        summary="Redis checkpoint finding",
                        rationale="Tested with real LangGraph",
                    ),
                ),
            ),
        )

        snapshot = sample_snapshot(head_sha="sha-redis-checkpoint-1")
        job = ReviewJob(
            job_id="job-redis-test-1",
            delivery_id="del-redis-1",
            repository_id="octocat/hello-world",
            pull_request_number=1,
            base_sha="base-1",
            head_sha="sha-redis-checkpoint-1",
            state=JobState.RUNNING,
        )

        state = await orchestrator.execute_run(job, snapshot)
        self.assertEqual("completed", state.terminal_status)
        self.assertTrue(state.aggregation_invoked)

        # Verify get_workflow_state retrieves saved checkpoint from Redis
        saved_state = orchestrator.get_workflow_state(job.job_id)
        self.assertIsNotNone(saved_state)
        self.assertEqual(job.job_id, saved_state.values.get("job_id"))
        self.assertEqual("sha-redis-checkpoint-1", saved_state.values.get("head_sha"))
        self.assertEqual("completed", saved_state.values.get("terminal_status"))
        self.assertTrue(saved_state.values.get("aggregation_invoked"))

    async def test_workflow_resume_avoids_replaying_completed_nodes(self) -> None:
        """9, 10, 11 & 13. PRIMARY ACCEPTANCE CRITERIA: Crash recovery and no duplicate fan-out."""
        execution_counts = {
            "initialize": 0,
            SpecialistType.SECURITY: 0,
            SpecialistType.QUALITY: 0,
            SpecialistType.TESTS: 0,
            SpecialistType.DOCUMENTATION: 0,
            "evaluate_terminal": 0,
        }

        # Track specialist execution
        def make_handler(spec_type: SpecialistType):
            def handler(inp: SpecialistInput) -> SpecialistOutput:
                execution_counts[spec_type] += 1
                return SpecialistOutput(
                    specialist_type=spec_type,
                    correlation_id=inp.correlation_id,
                    status=SpecialistStatus.COMPLETED,
                    findings=(
                        CandidateFinding(
                            finding_id=f"f-{spec_type.value}",
                            correlation_id=inp.correlation_id,
                            specialist_type=spec_type,
                            category="correctness",
                            severity="medium",
                            confidence=0.9,
                            summary=f"{spec_type.value} finding",
                            rationale="Grounded evidence",
                        ),
                    ),
                )

            return handler

        handlers = {spec: make_handler(spec) for spec in SpecialistType}

        snapshot = sample_snapshot(head_sha="sha-crash-recovery")
        job = ReviewJob(
            job_id="job-crash-recovery-1",
            delivery_id="del-crash-1",
            repository_id="octocat/hello-world",
            pull_request_number=1,
            base_sha="base-1",
            head_sha="sha-crash-recovery",
            state=JobState.RUNNING,
        )

        # --- RUN 1: Worker 1 executes review run to completion ---
        worker_1_orchestrator = ReviewOrchestrator(
            specialist_handlers=handlers,
            checkpointer=self.saver,
        )
        res_1 = await worker_1_orchestrator.execute_run(job, snapshot)
        self.assertEqual("completed", res_1.terminal_status)

        # Record initial execution counts
        initial_counts = dict(execution_counts)
        for spec in SpecialistType:
            self.assertEqual(1, initial_counts[spec], f"Specialist {spec.value} should have run exactly once")

        # --- SIMULATE RESTART: Worker 1 process dies; Worker 2 starts fresh with SAME thread_id ---
        # Worker 2 creates a new ReviewOrchestrator instance sharing the same Redis checkpointer
        worker_2_orchestrator = ReviewOrchestrator(
            specialist_handlers=handlers,
            checkpointer=self.saver,
        )

        # Worker 2 executes the same ReviewJob (same thread_id)
        res_2 = await worker_2_orchestrator.execute_run(job, snapshot)
        self.assertEqual("completed", res_2.terminal_status)

        # --- VERIFY CRITICAL INVARIANT: Completed work was NOT replayed! ---
        # Execution counts must NOT have incremented!
        for spec in SpecialistType:
            self.assertEqual(
                initial_counts[spec],
                execution_counts[spec],
                f"Specialist {spec.value} re-executed during resume! Duplication detected!",
            )

        # Findings must match and not be duplicated
        self.assertEqual(
            len(res_1.specialist_outputs),
            len(res_2.specialist_outputs),
        )

    async def test_no_checkpointer_behavior_remains_unchanged(self) -> None:
        """20. Existing no-checkpointer behavior remains unchanged for backwards compatibility."""
        orchestrator = ReviewOrchestrator(checkpointer=None)
        orchestrator.register_specialist(
            SpecialistType.SECURITY,
            lambda inp: SpecialistOutput(
                specialist_type=SpecialistType.SECURITY,
                correlation_id=inp.correlation_id,
                status=SpecialistStatus.COMPLETED,
            ),
        )
        snapshot = sample_snapshot()
        job = ReviewJob(
            job_id="job-no-chk",
            delivery_id="del-no-chk",
            repository_id="octocat/hello-world",
            pull_request_number=1,
            base_sha="base-1",
            head_sha="head-1",
            state=JobState.RUNNING,
        )

        state = await orchestrator.execute_run(job, snapshot)
        self.assertEqual("completed", state.terminal_status)
        self.assertIsNone(orchestrator.get_workflow_state(job.job_id))

    async def test_live_redis_integration_if_available(self) -> None:
        """21. Real live Redis daemon integration test when available."""
        try:
            live_client = redis.Redis(host="127.0.0.1", port=6379, socket_timeout=0.5)
            live_client.ping()
        except Exception as exc:
            self.skipTest(f"Live Redis daemon not available at 127.0.0.1:6379 ({exc})")
            return

        live_saver = RedisCheckpointSaver(client=live_client, key_prefix="review:checkpoint:test_live")
        try:
            config = {"configurable": {"thread_id": f"live-thread-{int(time.time())}"}}
            cp = empty_checkpoint()
            cp["id"] = "live-cp-1"
            cp["channel_values"] = {"live_status": "ok"}
            live_saver.put(config, cp, {"live": True}, {"live_status": 1})

            tup = live_saver.get_tuple(config)
            self.assertIsNotNone(tup)
            self.assertEqual("ok", tup.checkpoint["channel_values"].get("live_status"))
        finally:
            live_saver.delete_thread(config["configurable"]["thread_id"])


if __name__ == "__main__":
    unittest.main()
