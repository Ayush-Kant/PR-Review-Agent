"""Comprehensive tests for production runtime composition behind existing contracts (W1-08A).

Proves:
A. Backend configuration validation (valid/invalid backends, credential requirements)
B. SQLite reference composition (defaults preserved across server and worker)
C. Tiger composition (TigerReviewTruthStore, TigerAuditSpine, TigerCodeMemoryStore, TigerEffectStore)
D. Redis queue composition (RedisJobQueue selected and wired)
E. Redis checkpoint composition (RedisCheckpointSaver selected and wired)
F. Complete Tiger + Redis worker composition (end-to-end review run with production adapters)
G. ARQ startup composition (dependency graph population and clean shutdown)
H. GitHub effect store semantic parity (SQLiteEffectStore vs TigerEffectStore)
I. Webhook delivery cases A-E (failure safety, retry replayability, idempotency)
J. SQLite reference compatibility (zero regression for default local runtime)
K. Selected backend is NOT silently ignored
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import json
import sqlite3
import threading
import time
from typing import Any
import unittest

from starlette.testclient import TestClient

from pr_review_agent.adapters.redis_checkpoint import RedisCheckpointSaver
from tests.test_production_fault_injection import FaultInjectingRedisClient
from tests.test_redis_checkpoint_adapter import InMemoryRedisCheckpointClient
from tests.test_redis_queue_adapter import InMemoryRedisClient
from pr_review_agent.adapters.redis_queue import RedisJobQueue
from pr_review_agent.adapters.tiger_connection import (
    TigerConfig,
    TigerConfigurationError,
    TigerConnectionManager,
)
from pr_review_agent.adapters.tiger_stores import (
    TigerAuditSpine,
    TigerCodeMemoryStore,
    TigerEffectStore,
    TigerReviewTruthStore,
)
from pr_review_agent.github_output import (
    FakeGitHubClient,
    GitHubClient,
    GitHubEffectStoreProtocol,
    GitHubReviewPublisher,
    PublicationResult,
    PublicationStatus,
    SQLiteEffectStore,
)
from pr_review_agent.intake import ReviewSnapshot, WebhookIntake
from pr_review_agent.observability import AuditEvent, AuditSpine
from pr_review_agent.orchestration import (
    CandidateFinding,
    DurableJobQueue,
    DurableQueueProtocol,
    JobState,
    ReviewJob,
    ReviewLifecycleState,
    ReviewOrchestrator,
    SpecialistCoverageSummary,
    SpecialistHandler,
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
    ReviewTruthRecord,
    ReviewTruthStore,
    TruthState,
)
from pr_review_agent.retrieval import CodeChunk, CodeMemoryStore, CodeMemoryStoreProtocol, HybridRetriever
from pr_review_agent.server import create_server_app
from pr_review_agent.service_config import ServiceConfig, load_service_config
from pr_review_agent.worker import AutonomousReviewWorker, arq_shutdown, arq_startup
from tests.test_tiger_data_adapters import TigerPostgresSimulationConnection


def make_test_config(
    database_backend: str = "sqlite",
    queue_backend: str = "sqlite",
    checkpoint_backend: str = "none",
    database_path: str = ":memory:",
    redis_url: str = "",
    tiger_database_url: str = "",
    publish_enabled: bool = False,
    api_key: str = "test-mock-api-key",
) -> ServiceConfig:
    """Create a test ServiceConfig without requiring live environment secrets."""
    return ServiceConfig(
        github_token="ghp_test_secret_token_12345",
        webhook_secret=b"test_webhook_secret_key_67890",
        model_provider="openai",
        model_name="gpt-4o",
        database_path=database_path,
        host="127.0.0.1",
        port=8000,
        authorized_repositories=("test-org/test-repo",),
        authorized_tenant="test-org",
        publish_enabled=publish_enabled,
        api_key=api_key,
        queue_backend=queue_backend,
        redis_url=redis_url,
        checkpoint_backend=checkpoint_backend,
        database_backend=database_backend,
        tiger_database_url=tiger_database_url,
    )


def make_test_snapshot(
    repository_id: str = "test-org/test-repo",
    repository_full_name: str = "test-org/test-repo",
    pull_request_number: int = 1,
    base_sha: str = "base123",
    head_sha: str = "head123",
    changed_files: tuple[str, ...] = ("main.py",),
    policy_version: str = "v1",
    prompt_version: str = "v1",
    retrieval_index_version: str = "v1",
    model_configuration: Mapping[str, str] | None = None,
) -> ReviewSnapshot:
    return ReviewSnapshot(
        repository_id=repository_id,
        repository_full_name=repository_full_name,
        pull_request_number=pull_request_number,
        base_sha=base_sha,
        head_sha=head_sha,
        changed_files=changed_files,
        policy_version=policy_version,
        prompt_version=prompt_version,
        retrieval_index_version=retrieval_index_version,
        model_configuration=model_configuration or {"provider": "openai", "model": "gpt-4o"},
    )


class StubGitHubClient(FakeGitHubClient):
    """Deterministic test double extending FakeGitHubClient."""

    def __init__(self, current_head_sha: str = "sha999") -> None:
        super().__init__(pr_heads={("test-org/test-repo", 1): current_head_sha})

    def get_pull_request_diff(self, repository: str, pull_number: int) -> str:
        return "diff --git a/main.py b/main.py\n@@ -1,3 +1,3 @@\n-old\n+new\n"


class DummySpecialistHandler:
    """Deterministic test specialist returning a candidate finding."""

    def __call__(self, specialist_input: SpecialistInput) -> SpecialistOutput:
        cand = CandidateFinding(
            finding_id="cand-001",
            correlation_id=specialist_input.correlation_id,
            specialist_type=specialist_input.specialist_type,
            category="quality_defect",
            severity="medium",
            confidence=0.9,
            summary="Mutable default argument in function",
            rationale="Using list = [] as default argument can cause state leakage",
            file_path="main.py",
            line_range=(1, 2),
            evidence_refs=("main.py:1",),
            remediation="Use list | None = None",
        )
        return SpecialistOutput(
            specialist_type=specialist_input.specialist_type,
            correlation_id=specialist_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(cand,),
        )


# ============================================================================
# A. BACKEND CONFIGURATION VALIDATION
# ============================================================================

class TestBackendConfigurationValidation(unittest.TestCase):
    """Test environment loading and backend configuration validation."""

    def test_default_backends_are_sqlite_and_none(self) -> None:
        """Verify defaults MUST remain: DATABASE_BACKEND=sqlite, QUEUE_BACKEND=sqlite, CHECKPOINT_BACKEND=none."""
        cfg = make_test_config()
        self.assertEqual(cfg.database_backend, "sqlite")
        self.assertEqual(cfg.queue_backend, "sqlite")
        self.assertEqual(cfg.checkpoint_backend, "none")

    def test_valid_backend_combinations(self) -> None:
        """Verify load_service_config accepts valid backend identifiers."""
        env = {
            "GITHUB_TOKEN": "token",
            "GITHUB_WEBHOOK_SECRET": "secret",
            "GITHUB_REPOSITORY": "org/repo",
            "DATABASE_BACKEND": "tiger",
            "TIGER_DATABASE_URL": "postgresql://usr:pwd@localhost:5432/db?sslmode=require",
            "QUEUE_BACKEND": "redis",
            "REDIS_URL": "redis://localhost:6379/0",
            "CHECKPOINT_BACKEND": "redis",
            "OPENAI_API_KEY": "sk-key",
        }
        cfg = load_service_config(env, require_live_credentials=True)
        self.assertEqual(cfg.database_backend, "tiger")
        self.assertEqual(cfg.queue_backend, "redis")
        self.assertEqual(cfg.checkpoint_backend, "redis")

    def test_invalid_database_backend_rejected(self) -> None:
        """Invalid DATABASE_BACKEND raises ValueError."""
        env = {
            "DATABASE_BACKEND": "mysql",
            "GITHUB_TOKEN": "token",
            "GITHUB_WEBHOOK_SECRET": "secret",
            "GITHUB_REPOSITORY": "org/repo",
        }
        with self.assertRaises(ValueError) as ctx:
            load_service_config(env, require_live_credentials=False)
        self.assertIn("Invalid DATABASE_BACKEND 'mysql'", str(ctx.exception))

    def test_invalid_queue_backend_rejected(self) -> None:
        """Invalid QUEUE_BACKEND raises ValueError."""
        env = {
            "QUEUE_BACKEND": "rabbitmq",
            "GITHUB_TOKEN": "token",
            "GITHUB_WEBHOOK_SECRET": "secret",
            "GITHUB_REPOSITORY": "org/repo",
        }
        with self.assertRaises(ValueError) as ctx:
            load_service_config(env, require_live_credentials=False)
        self.assertIn("Invalid QUEUE_BACKEND 'rabbitmq'", str(ctx.exception))

    def test_invalid_checkpoint_backend_rejected(self) -> None:
        """Invalid CHECKPOINT_BACKEND raises ValueError."""
        env = {
            "CHECKPOINT_BACKEND": "dynamodb",
            "GITHUB_TOKEN": "token",
            "GITHUB_WEBHOOK_SECRET": "secret",
            "GITHUB_REPOSITORY": "org/repo",
        }
        with self.assertRaises(ValueError) as ctx:
            load_service_config(env, require_live_credentials=False)
        self.assertIn("Invalid CHECKPOINT_BACKEND 'dynamodb'", str(ctx.exception))

    def test_live_credentials_fail_closed_on_missing_urls(self) -> None:
        """require_live_credentials=True enforces REDIS_URL and TIGER_DATABASE_URL."""
        env = {
            "GITHUB_TOKEN": "token",
            "GITHUB_WEBHOOK_SECRET": "secret",
            "GITHUB_REPOSITORY": "org/repo",
            "DATABASE_BACKEND": "tiger",
            "OPENAI_API_KEY": "sk-key",
        }
        with self.assertRaises(ValueError) as ctx:
            load_service_config(env, require_live_credentials=True)
        self.assertIn("Missing required TIGER_DATABASE_URL", str(ctx.exception))

        env["TIGER_DATABASE_URL"] = "postgresql://u:p@host:5432/db?sslmode=require"
        env["QUEUE_BACKEND"] = "redis"
        with self.assertRaises(ValueError) as ctx:
            load_service_config(env, require_live_credentials=True)
        self.assertIn("Missing required REDIS_URL", str(ctx.exception))


class TestBackendFailClosedComposition(unittest.TestCase):
    """Verify that selecting production backends with missing configs fails closed without silent fallback."""

    def test_explicit_redis_queue_missing_url_fails_closed(self) -> None:
        """E: QUEUE_BACKEND=redis without REDIS_URL fails closed across server, worker, and arq."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(queue_backend="redis", redis_url="")

        # 1. Server fails closed
        with self.assertRaises(ValueError) as ctx:
            create_server_app(cfg, connection=conn)
        self.assertIn("REDIS_URL is required when QUEUE_BACKEND is 'redis'", str(ctx.exception))

        # 2. Worker fails closed
        with self.assertRaises(ValueError) as ctx:
            AutonomousReviewWorker(cfg, connection=conn)
        self.assertIn("REDIS_URL is required when QUEUE_BACKEND is 'redis'", str(ctx.exception))

        # 3. ARQ startup fails closed
        async def run_arq() -> None:
            await arq_startup({"config": cfg})

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(run_arq())
        self.assertIn("REDIS_URL is required when QUEUE_BACKEND is 'redis'", str(ctx.exception))

        # 4. get_redis_settings fails closed
        from pr_review_agent.worker import get_redis_settings
        with self.assertRaises(ValueError) as ctx:
            get_redis_settings(cfg)
        self.assertIn("REDIS_URL is required when QUEUE_BACKEND is 'redis'", str(ctx.exception))

    def test_explicit_redis_checkpoint_missing_url_fails_closed(self) -> None:
        """F: CHECKPOINT_BACKEND=redis without REDIS_URL fails closed in worker composition."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(checkpoint_backend="redis", redis_url="")

        with self.assertRaises(ValueError) as ctx:
            AutonomousReviewWorker(cfg, connection=conn)
        self.assertIn("REDIS_URL is required when CHECKPOINT_BACKEND is 'redis'", str(ctx.exception))

    def test_explicit_tiger_missing_url_fails_closed(self) -> None:
        """G: DATABASE_BACKEND=tiger without TIGER_DATABASE_URL fails closed across server, worker, and arq."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(database_backend="tiger", tiger_database_url="")

        # 1. Server fails closed
        with self.assertRaises(TigerConfigurationError) as ctx:
            create_server_app(cfg, connection=conn)
        self.assertIn("TIGER_DATABASE_URL is required when DATABASE_BACKEND is 'tiger'", str(ctx.exception))

        # 2. Worker fails closed
        with self.assertRaises(TigerConfigurationError) as ctx:
            AutonomousReviewWorker(cfg, connection=conn)
        self.assertIn("TIGER_DATABASE_URL is required when DATABASE_BACKEND is 'tiger'", str(ctx.exception))

        # 3. ARQ startup fails closed
        async def run_arq() -> None:
            await arq_startup({"config": cfg})

        with self.assertRaises(TigerConfigurationError) as ctx:
            asyncio.run(run_arq())
        self.assertIn("TIGER_DATABASE_URL is required when DATABASE_BACKEND is 'tiger'", str(ctx.exception))


# ============================================================================
# B. SQLITE REFERENCE COMPOSITION
# ============================================================================

class TestSQLiteReferenceComposition(unittest.TestCase):
    """Verify that SQLite composition remains 100% functional as local reference."""

    def test_sqlite_server_composition(self) -> None:
        """Default server app composition wires SQLite adapters."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(database_backend="sqlite", queue_backend="sqlite")
        app = create_server_app(cfg, connection=conn)

        self.assertIsInstance(app.state.job_queue, DurableJobQueue)
        self.assertIsInstance(app.state.audit_spine, AuditSpine)
        self.assertIsInstance(app.state.intake, WebhookIntake)
        self.assertIsNone(app.state.tiger_connection_manager)

    def test_sqlite_worker_composition(self) -> None:
        """Default worker composition wires SQLite adapters and reference retriever."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(database_backend="sqlite", queue_backend="sqlite")
        worker = AutonomousReviewWorker(cfg, connection=conn)

        self.assertIsInstance(worker.queue, DurableJobQueue)
        self.assertIsInstance(worker.audit_spine, AuditSpine)
        self.assertIsInstance(worker.truth_store, ReviewTruthStore)
        self.assertIsInstance(worker.code_memory_store, CodeMemoryStore)
        self.assertIsInstance(worker.effect_store, SQLiteEffectStore)
        self.assertIsInstance(worker.retriever, HybridRetriever)
        self.assertIs(worker.retriever.memory_store, worker.code_memory_store)
        self.assertIsInstance(worker.publisher, GitHubReviewPublisher)
        self.assertIs(worker.publisher.effect_store, worker.effect_store)
        worker.close()


# ============================================================================
# C. TIGER COMPOSITION
# ============================================================================

class TestTigerComposition(unittest.TestCase):
    """Verify that selecting Tiger database wires Tiger adapters across server and worker."""

    def setUp(self) -> None:
        self.sim_conn = TigerPostgresSimulationConnection()
        self.tiger_cfg = TigerConfig.from_url("postgresql://postgres:secret@localhost:5432/testdb?sslmode=require")
        self.tiger_mgr = TigerConnectionManager(
            self.tiger_cfg,
            connection_factory=lambda cfg: self.sim_conn,
        )

    def tearDown(self) -> None:
        self.tiger_mgr.close()

    def test_tiger_server_composition(self) -> None:
        """Server app wires TigerAuditSpine when DATABASE_BACKEND=tiger."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(
            database_backend="tiger",
            tiger_database_url=self.tiger_cfg.database_url,
        )
        app = create_server_app(cfg, connection=conn, tiger_connection_manager=self.tiger_mgr)

        self.assertIsInstance(app.state.audit_spine, TigerAuditSpine)
        self.assertIs(app.state.tiger_connection_manager, self.tiger_mgr)

    def test_tiger_worker_composition(self) -> None:
        """Worker wires TigerReviewTruthStore, TigerAuditSpine, TigerCodeMemoryStore, TigerEffectStore."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(
            database_backend="tiger",
            tiger_database_url=self.tiger_cfg.database_url,
        )
        worker = AutonomousReviewWorker(
            cfg,
            connection=conn,
            tiger_connection_manager=self.tiger_mgr,
        )

        self.assertIsInstance(worker.truth_store, TigerReviewTruthStore)
        self.assertIsInstance(worker.audit_spine, TigerAuditSpine)
        self.assertIsInstance(worker.code_memory_store, TigerCodeMemoryStore)
        self.assertIsInstance(worker.effect_store, TigerEffectStore)
        self.assertIsInstance(worker.retriever, HybridRetriever)
        self.assertIs(worker.retriever.memory_store, worker.code_memory_store)
        self.assertIs(worker.publisher.effect_store, worker.effect_store)
        self.assertIs(worker.publisher.truth_store, worker.truth_store)
        worker.close()

    def test_selected_backend_not_silently_ignored(self) -> None:
        """Worker fails closed if database_backend=tiger but no Tiger configuration or manager is provided."""
        cfg = make_test_config(database_backend="tiger", tiger_database_url="")
        with self.assertRaises(TigerConfigurationError):
            AutonomousReviewWorker(cfg)


# ============================================================================
# D. REDIS QUEUE COMPOSITION
# ============================================================================

class TestRedisQueueComposition(unittest.TestCase):
    """Verify that selecting Redis queue wires RedisJobQueue across server and worker."""

    def test_redis_queue_server_composition(self) -> None:
        """Server app wires RedisJobQueue when QUEUE_BACKEND=redis."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(queue_backend="redis", redis_url="redis://127.0.0.1:6379/0")
        app = create_server_app(cfg, connection=conn)

        self.assertIsInstance(app.state.job_queue, RedisJobQueue)
        self.assertEqual(app.state.job_queue.redis_url, "redis://127.0.0.1:6379/0")

    def test_redis_queue_worker_composition(self) -> None:
        """Worker wires RedisJobQueue when QUEUE_BACKEND=redis."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(queue_backend="redis", redis_url="redis://127.0.0.1:6379/0")
        worker = AutonomousReviewWorker(cfg, connection=conn)

        self.assertIsInstance(worker.queue, RedisJobQueue)
        self.assertEqual(worker.queue.redis_url, "redis://127.0.0.1:6379/0")
        worker.close()


# ============================================================================
# E. REDIS CHECKPOINT COMPOSITION
# ============================================================================

class TestRedisCheckpointComposition(unittest.TestCase):
    """Verify that selecting Redis checkpoint wires RedisCheckpointSaver."""

    def test_redis_checkpoint_worker_composition(self) -> None:
        """Worker wires RedisCheckpointSaver when CHECKPOINT_BACKEND=redis."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cfg = make_test_config(checkpoint_backend="redis", redis_url="redis://127.0.0.1:6379/0")
        worker = AutonomousReviewWorker(cfg, connection=conn)

        self.assertIsInstance(worker.checkpointer, RedisCheckpointSaver)
        self.assertEqual(worker.checkpointer.redis_url, "redis://127.0.0.1:6379/0")
        worker.close()

    def test_owned_redis_resources_closed_and_injected_preserved(self) -> None:
        """I: AutonomousReviewWorker cleans up owned Redis checkpointer and queue, but preserves caller-injected instances."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)

        # 1. Injected resources -> Worker does NOT close them on worker.close()
        in_memory_queue = InMemoryRedisClient()
        injected_queue = RedisJobQueue(client=in_memory_queue)
        in_memory_cp = InMemoryRedisCheckpointClient()
        injected_cp = RedisCheckpointSaver(client=in_memory_cp)

        cfg_injected = make_test_config(
            queue_backend="redis",
            redis_url="redis://127.0.0.1:6379/0",
            checkpoint_backend="redis",
        )
        worker_injected = AutonomousReviewWorker(
            cfg_injected,
            connection=conn,
            queue=injected_queue,
            checkpointer=injected_cp,
        )
        self.assertFalse(worker_injected._owns_queue)
        self.assertFalse(worker_injected._owns_checkpointer)

        worker_injected.close()
        self.assertFalse(in_memory_cp.closed)

        # 2. Owned checkpointer and queue -> Worker closes them on worker.close()
        cfg_owned = make_test_config(
            queue_backend="redis",
            redis_url="redis://127.0.0.1:6379/0",
            checkpoint_backend="redis",
        )
        worker_owned = AutonomousReviewWorker(cfg_owned, connection=conn)
        self.assertTrue(worker_owned._owns_queue)
        self.assertTrue(worker_owned._owns_checkpointer)

        queue_closed = False
        cp_closed = False
        orig_queue_close = worker_owned.queue.close
        orig_cp_close = worker_owned.checkpointer.close

        def close_queue() -> None:
            nonlocal queue_closed
            queue_closed = True
            orig_queue_close()

        def close_cp() -> None:
            nonlocal cp_closed
            cp_closed = True
            orig_cp_close()

        worker_owned.queue.close = close_queue
        worker_owned.checkpointer.close = close_cp

        worker_owned.close()
        self.assertTrue(queue_closed)
        self.assertTrue(cp_closed)
        conn.close()


# ============================================================================
# F. COMPLETE TIGER + REDIS WORKER COMPOSITION
# ============================================================================

class TestCompleteTigerRedisWorkerComposition(unittest.IsolatedAsyncioTestCase):
    """Verify full end-to-end review lifecycle with all production adapters composited."""

    async def asyncSetUp(self) -> None:
        self.sim_conn = TigerPostgresSimulationConnection()
        self.tiger_cfg = TigerConfig.from_url("postgresql://postgres:secret@localhost:5432/testdb?sslmode=require")
        self.tiger_mgr = TigerConnectionManager(
            self.tiger_cfg,
            connection_factory=lambda cfg: self.sim_conn,
        )
        self.in_memory_redis = InMemoryRedisClient()
        self.redis_queue = RedisJobQueue(client=self.in_memory_redis)
        self.in_memory_checkpoint_redis = InMemoryRedisCheckpointClient()
        self.redis_checkpointer = RedisCheckpointSaver(client=self.in_memory_checkpoint_redis)
        self.github_client = StubGitHubClient(current_head_sha="sha999")

        self.cfg = make_test_config(
            database_backend="tiger",
            queue_backend="redis",
            checkpoint_backend="redis",
            tiger_database_url=self.tiger_cfg.database_url,
            redis_url="redis://localhost:6379/0",
            publish_enabled=True,
        )

        handlers = {
            SpecialistType.SECURITY: DummySpecialistHandler(),
            SpecialistType.QUALITY: DummySpecialistHandler(),
            SpecialistType.TESTS: DummySpecialistHandler(),
            SpecialistType.DOCUMENTATION: DummySpecialistHandler(),
        }

        self.worker = AutonomousReviewWorker(
            self.cfg,
            tiger_connection_manager=self.tiger_mgr,
            queue=self.redis_queue,
            checkpointer=self.redis_checkpointer,
            github_client=self.github_client,
            specialist_handlers=handlers,
        )

    async def asyncTearDown(self) -> None:
        self.worker.close()
        self.tiger_mgr.close()

    async def test_complete_tiger_redis_review_lifecycle(self) -> None:
        """Full run: Redis enqueue -> lease -> Tiger audit/truth/effect -> complete."""
        # 1. Enqueue job into Redis
        snapshot = make_test_snapshot(base_sha="base000", head_sha="sha999")
        job = self.redis_queue.enqueue(
            delivery_id="deliv-full-001",
            snapshot=snapshot,
        )
        self.assertEqual(job.state, JobState.QUEUED)

        # 2. Worker leases and processes the job
        processed_job, lifecycle_state = await self.worker.process_one_job()

        self.assertIsNotNone(processed_job)
        self.assertIsNotNone(lifecycle_state)
        self.assertEqual(processed_job.job_id, job.job_id)

        # 3. Verify Redis queue state is COMPLETED
        updated_job = self.redis_queue.get_job(job.job_id)
        self.assertIsNotNone(updated_job)
        self.assertEqual(updated_job.state, JobState.COMPLETED)

        # 4. Verify Tiger ReviewTruthStore recorded review run header and findings
        all_canonical = self.worker.truth_store.list_all_canonical()
        self.assertGreaterEqual(len(all_canonical), 1)
        canonical_id = all_canonical[0]
        truth_records = self.worker.truth_store.get_history(canonical_id)
        self.assertGreaterEqual(len(truth_records), 1)
        self.assertEqual(truth_records[0].repository_id, "test-org/test-repo")

        # 5. Verify Tiger EffectStore recorded publication effect
        idempotency_key = self.worker.publisher.compute_idempotency_key(
            "test-org/test-repo",
            1,
            "sha999",
            canonical_id,
        )
        effect = self.worker.effect_store.get_effect(idempotency_key)
        self.assertIsNotNone(effect)
        self.assertEqual(effect["status"], PublicationStatus.PUBLISHED.value)

        # 6. Verify Tiger AuditSpine recorded events
        trace = self.worker.audit_spine.reconstruct_run(correlation_id="deliv-full-001")
        self.assertGreaterEqual(len(trace.timeline), 2)
        event_names = [e.event_name for e in trace.timeline]
        self.assertIn("worker_job_started", event_names)
        self.assertIn("worker_job_completed", event_names)


# ============================================================================
# G. ARQ STARTUP COMPOSITION
# ============================================================================

class TestARQStartupComposition(unittest.IsolatedAsyncioTestCase):
    """Verify ARQ startup initializes one coherent dependency graph and shutdown cleans up."""

    async def test_arq_startup_and_shutdown_tiger_redis(self) -> None:
        """ARQ startup wires Redis queue, Tiger connection manager, and AutonomousReviewWorker."""
        sim_conn = TigerPostgresSimulationConnection()
        tiger_cfg = TigerConfig.from_url("postgresql://postgres:secret@localhost:5432/testdb?sslmode=require")
        in_memory_redis = InMemoryRedisClient()
        redis_queue = RedisJobQueue(client=in_memory_redis)

        cfg = make_test_config(
            database_backend="tiger",
            queue_backend="redis",
            tiger_database_url=tiger_cfg.database_url,
            redis_url="redis://localhost:6379/0",
        )

        ctx: dict[str, Any] = {
            "config": cfg,
            "queue": redis_queue,
            "tiger_connection_manager": TigerConnectionManager(
                tiger_cfg,
                connection_factory=lambda c: sim_conn,
            ),
        }

        await arq_startup(ctx)

        self.assertIn("worker", ctx)
        worker = ctx["worker"]
        self.assertIsInstance(worker, AutonomousReviewWorker)
        self.assertIsInstance(worker.queue, RedisJobQueue)
        self.assertIsInstance(worker.truth_store, TigerReviewTruthStore)
        self.assertIsInstance(worker.audit_spine, TigerAuditSpine)
        self.assertIsInstance(worker.effect_store, TigerEffectStore)

        await arq_shutdown(ctx)
        self.assertTrue(ctx["tiger_connection_manager"]._is_closed)


# ============================================================================
# H. GITHUB EFFECT STORE SEMANTIC PARITY
# ============================================================================

class TestGitHubEffectStoreSemanticParity(unittest.TestCase):
    """Prove semantic parity between SQLiteEffectStore and TigerEffectStore."""

    def setUp(self) -> None:
        self.sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.sqlite_store = SQLiteEffectStore(self.sqlite_conn)

        self.sim_conn = TigerPostgresSimulationConnection()
        self.tiger_cfg = TigerConfig.from_url("postgresql://postgres:secret@localhost:5432/testdb?sslmode=require")
        self.tiger_mgr = TigerConnectionManager(
            self.tiger_cfg,
            connection_factory=lambda cfg: self.sim_conn,
        )
        self.tiger_store = TigerEffectStore(self.tiger_mgr)

    def tearDown(self) -> None:
        self.sqlite_conn.close()
        self.tiger_mgr.close()

    def test_lookup_nonexistent_returns_none(self) -> None:
        """Both stores return None for nonexistent idempotency keys."""
        self.assertIsNone(self.sqlite_store.get_effect("missing-key"))
        self.assertIsNone(self.tiger_store.get_effect("missing-key"))

    def test_record_and_lookup_pending_effect(self) -> None:
        """Both stores correctly record and retrieve PENDING effects."""
        for store in (self.sqlite_store, self.tiger_store):
            store.record_effect(
                idempotency_key="key-001",
                repository_id="org/repo",
                pull_number=42,
                head_sha="sha111",
                canonical_id="canon-001",
                status="pending",
                payload={"test": "data"},
            )

            retrieved = store.get_effect("key-001")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["idempotency_key"], "key-001")
            self.assertEqual(retrieved["repository_id"], "org/repo")
            self.assertEqual(retrieved["pull_number"], 42)
            self.assertEqual(retrieved["head_sha"], "sha111")
            self.assertEqual(retrieved["canonical_id"], "canon-001")
            self.assertEqual(retrieved["status"], "pending")
            self.assertEqual(retrieved["payload"], {"test": "data"})

    def test_record_published_and_prevent_regression(self) -> None:
        """Both stores permit progression to PUBLISHED and prevent regression to PENDING or FAILED."""
        for store in (self.sqlite_store, self.tiger_store):
            store.record_effect(
                idempotency_key="key-regress",
                repository_id="org/repo",
                pull_number=10,
                head_sha="sha222",
                canonical_id="canon-002",
                status="pending",
            )

            # Advance to published
            store.record_effect(
                idempotency_key="key-regress",
                repository_id="org/repo",
                pull_number=10,
                head_sha="sha222",
                canonical_id="canon-002",
                status="published",
                review_id="rev-900",
                comment_id="com-900",
                html_url="https://github.com/org/repo/pull/10#rev-900",
                published_inline=True,
            )

            curr = store.get_effect("key-regress")
            self.assertIsNotNone(curr)
            self.assertEqual(curr["status"], "published")
            self.assertEqual(curr["review_id"], "rev-900")
            self.assertEqual(curr["comment_id"], "com-900")
            self.assertEqual(curr["html_url"], "https://github.com/org/repo/pull/10#rev-900")
            self.assertTrue(curr["published_inline"])

            # Attempt regression to failed
            store.record_effect(
                idempotency_key="key-regress",
                repository_id="org/repo",
                pull_number=10,
                head_sha="sha222",
                canonical_id="canon-002",
                status="failed",
                reason="Transient failure after publish",
            )

            after = store.get_effect("key-regress")
            self.assertIsNotNone(after)
            # Must remain published! Non-regression invariant.
            self.assertEqual(after["status"], "published")
            self.assertEqual(after["review_id"], "rev-900")

    def test_record_failed_and_ambiguous_transitions(self) -> None:
        """Both stores support FAILED, AMBIGUOUS, and reconciliation."""
        for store in (self.sqlite_store, self.tiger_store):
            # 1. PENDING -> FAILED
            store.record_effect(
                idempotency_key="key-fail",
                repository_id="org/repo",
                pull_number=5,
                head_sha="sha555",
                canonical_id="canon-fail",
                status="pending",
            )
            store.record_effect(
                idempotency_key="key-fail",
                repository_id="org/repo",
                pull_number=5,
                head_sha="sha555",
                canonical_id="canon-fail",
                status="failed",
                reason="Rate limit exceeded",
            )
            failed_effect = store.get_effect("key-fail")
            self.assertIsNotNone(failed_effect)
            self.assertEqual(failed_effect["status"], "failed")
            self.assertEqual(failed_effect["reason"], "Rate limit exceeded")

            # 2. PENDING -> AMBIGUOUS -> PUBLISHED (reconciliation)
            store.record_effect(
                idempotency_key="key-ambig",
                repository_id="org/repo",
                pull_number=6,
                head_sha="sha666",
                canonical_id="canon-ambig",
                status="pending",
            )
            store.record_effect(
                idempotency_key="key-ambig",
                repository_id="org/repo",
                pull_number=6,
                head_sha="sha666",
                canonical_id="canon-ambig",
                status="ambiguous",
                reason="Socket timeout during HTTP request",
            )
            ambig_effect = store.get_effect("key-ambig")
            self.assertIsNotNone(ambig_effect)
            self.assertEqual(ambig_effect["status"], "ambiguous")

            # Reconciliation discovers it was published externally
            store.record_effect(
                idempotency_key="key-ambig",
                repository_id="org/repo",
                pull_number=6,
                head_sha="sha666",
                canonical_id="canon-ambig",
                status="published",
                review_id="rev-reconciled-666",
                html_url="https://github.com/org/repo/pull/6#review-reconciled-666",
            )
            reconciled_effect = store.get_effect("key-ambig")
            self.assertIsNotNone(reconciled_effect)
            self.assertEqual(reconciled_effect["status"], "published")
            self.assertEqual(reconciled_effect["review_id"], "rev-reconciled-666")

    def test_sqlite_effect_store_cannot_regress_published_under_concurrent_write_timing(self) -> None:
        """H: SQLiteEffectStore atomically prevents regression of PUBLISHED even under concurrent multi-threaded writes."""
        store = self.sqlite_store
        key = "race-key-atomic-non-regression"

        # Advance to PUBLISHED first
        store.record_effect(
            idempotency_key=key,
            repository_id="org/repo",
            pull_number=1,
            head_sha="head123",
            canonical_id="cand-1",
            status="published",
            review_id="rev-atomic-1",
            comment_id="com-atomic-1",
            html_url="https://github.com/org/repo/pull/1#rev-1",
            published_inline=True,
        )

        errors: list[Exception] = []

        def worker_thread(target_status: str, reason: str) -> None:
            try:
                for _ in range(25):
                    store.record_effect(
                        idempotency_key=key,
                        repository_id="org/repo",
                        pull_number=1,
                        head_sha="head123",
                        canonical_id="cand-1",
                        status=target_status,
                        reason=reason,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker_thread, args=("failed", "Network crash")),
            threading.Thread(target=worker_thread, args=("pending", "Duplicate start")),
            threading.Thread(target=worker_thread, args=("ambiguous", "Timeout")),
            threading.Thread(target=worker_thread, args=("published", "Retry publish")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

        # Atomic engine invariant: effect MUST remain published
        effect = store.get_effect(key)
        self.assertIsNotNone(effect)
        self.assertEqual(effect["status"], "published")
        self.assertEqual(effect["review_id"], "rev-atomic-1")


# ============================================================================
# I. WEBHOOK DELIVERY AUTHORITY (CASES A - E)
# ============================================================================

class TestWebhookDeliveryAuthority(unittest.TestCase):
    """Verify webhook delivery coordination across failure scenarios (FR-02, NFR-03)."""

    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.secret = b"test_webhook_secret_key_67890"
        self.intake = WebhookIntake(self.conn, self.secret)
        self.in_memory_redis = FaultInjectingRedisClient()
        self.redis_queue = RedisJobQueue(client=self.in_memory_redis)
        self.audit_spine = AuditSpine(self.conn)

        cfg = make_test_config(database_backend="sqlite", queue_backend="redis")
        self.app = create_server_app(
            cfg,
            connection=self.conn,
            job_queue=self.redis_queue,
            audit_spine=self.audit_spine,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.conn.close()

    def _create_signed_payload(
        self,
        delivery_id: str,
        action: str = "opened",
        repository_id: str = "test-org/test-repo",
        pull_request_number: int = 1,
        base_sha: str = "base123",
        head_sha: str = "head123",
    ) -> tuple[bytes, dict[str, str]]:
        import hashlib
        import hmac

        payload = {
            "action": action,
            "delivery_id": delivery_id,
            "repository": {
                "id": 12345,
                "full_name": repository_id,
                "owner": {"login": repository_id.split("/")[0]},
            },
            "pull_request": {
                "number": pull_request_number,
                "base": {"sha": base_sha},
                "head": {"sha": head_sha},
            },
        }
        body = json.dumps(payload).encode("utf-8")
        sig = "sha256=" + hmac.new(self.secret, body, hashlib.sha256).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-GitHub-Delivery": delivery_id,
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": sig,
        }
        return body, headers

    def test_case_a_first_delivery_and_enqueue_success(self) -> None:
        """Case A: First delivery + enqueue success -> HTTP 202, job enqueued, delivery marked enqueued."""
        body, headers = self._create_signed_payload("deliv-case-a")
        res = self.client.post("/webhooks/github", content=body, headers=headers)

        self.assertEqual(res.status_code, 202)
        data = res.json()
        self.assertEqual(data["status"], "accepted")
        self.assertTrue(data["job_id"])

        # Inspect durable queue state
        job = self.redis_queue.get_job_by_delivery("deliv-case-a")
        self.assertIsNotNone(job)
        self.assertEqual(job.job_id, data["job_id"])
        self.assertEqual(job.state, JobState.QUEUED)

        # Inspect intake state
        deliv_state = self.intake.get_delivery_state("deliv-case-a")
        self.assertIsNotNone(deliv_state)
        self.assertEqual(deliv_state, "enqueued")

    def test_case_b_first_delivery_and_enqueue_failure(self) -> None:
        """Case B: First delivery + enqueue failure -> HTTP 503, marked enqueue_failed, no phantom job."""
        # Cause Redis enqueue to fail
        self.in_memory_redis.drop_connection = True

        body, headers = self._create_signed_payload("deliv-case-b")
        res = self.client.post("/webhooks/github", content=body, headers=headers)

        self.assertEqual(res.status_code, 503)
        self.assertIn("Failed to enqueue", res.json()["error"])

        # Verify no phantom job in queue
        self.in_memory_redis.drop_connection = False
        job = self.redis_queue.get_job_by_delivery("deliv-case-b")
        self.assertIsNone(job)

        # Verify intake delivery is marked enqueue_failed (not accepted/duplicate)
        deliv_state = self.intake.get_delivery_state("deliv-case-b")
        self.assertIsNotNone(deliv_state)
        self.assertEqual(deliv_state, "enqueue_failed")

    def test_case_c_retry_after_enqueue_failure(self) -> None:
        """Case C: Retry after enqueue failure -> replaying delivery succeeds and enqueues."""
        # 1. First attempt fails
        self.in_memory_redis.drop_connection = True
        body, headers = self._create_signed_payload("deliv-case-c")
        res1 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res1.status_code, 503)

        # 2. Redis recovers
        self.in_memory_redis.drop_connection = False

        # 3. GitHub retries identical delivery
        res2 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res2.status_code, 202)
        data = res2.json()
        self.assertEqual(data["status"], "accepted")

        # Verify job is now durably enqueued
        job = self.redis_queue.get_job_by_delivery("deliv-case-c")
        self.assertIsNotNone(job)
        self.assertEqual(job.job_id, data["job_id"])

        deliv_state = self.intake.get_delivery_state("deliv-case-c")
        self.assertIsNotNone(deliv_state)
        self.assertEqual(deliv_state, "enqueued")

    def test_case_d_redis_commit_plus_client_response_loss(self) -> None:
        """Case D: First HTTP request creates durable job; client response lost; retry reconciles existing job."""
        body, headers = self._create_signed_payload("deliv-case-d")

        # 1. First webhook request creates the durable job
        res1 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res1.status_code, 202)
        job_id_1 = res1.json()["job_id"]

        # Verify durable queue job was created
        job1 = self.redis_queue.get_job(job_id_1)
        self.assertIsNotNone(job1)

        # Worker leases job and starts attempt 1
        leased_job = self.redis_queue.lease_next_job(now=time.time())
        self.assertIsNotNone(leased_job)
        self.assertEqual(leased_job.job_id, job_id_1)
        self.assertEqual(leased_job.attempt_count, 1)
        self.assertEqual(leased_job.state, JobState.RUNNING)
        original_lease_token = leased_job.lease_token

        # 2. Treat response 1 as lost from client's perspective; second identical webhook arrives
        res2 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2["status"], "duplicate")
        self.assertEqual(data2["job_id"], job_id_1)

        # 3. Invariants: existing job reconciled, same job_id returned, attempt_count unchanged,
        # queue state unchanged, lease token unchanged
        job_after = self.redis_queue.get_job(job_id_1)
        self.assertIsNotNone(job_after)
        self.assertEqual(job_after.job_id, job_id_1)
        self.assertEqual(job_after.attempt_count, 1)
        self.assertEqual(job_after.state, JobState.RUNNING)
        self.assertEqual(job_after.lease_token, original_lease_token)

    def test_case_e_duplicate_delivery_after_successful_enqueue(self) -> None:
        """Case E: Duplicate delivery after success returns existing job without resetting attempts/lease."""
        body, headers = self._create_signed_payload("deliv-case-e")
        res1 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res1.status_code, 202)
        job_id_1 = res1.json()["job_id"]

        # Worker leases job and starts attempt 1
        leased_job = self.redis_queue.lease_next_job(now=time.time())
        self.assertIsNotNone(leased_job)
        self.assertEqual(leased_job.attempt_count, 1)

        # Duplicate delivery arrives from GitHub
        res2 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2["status"], "duplicate")
        self.assertEqual(data2["job_id"], job_id_1)

        # Invariant: attempt_count remains 1, lease_token not corrupted
        job_after = self.redis_queue.get_job(job_id_1)
        self.assertIsNotNone(job_after)
        self.assertEqual(job_after.attempt_count, 1)
        self.assertEqual(job_after.state, JobState.RUNNING)
        self.assertEqual(job_after.lease_token, leased_job.lease_token)

    def test_concurrent_pending_duplicate_cannot_produce_false_duplicate_with_no_job(self) -> None:
        """A: Delivery in local pending state without queue job safely attempts enqueue, never returning false 200."""
        # Pre-seed intake with a pending delivery, but queue has no job
        with self.conn:
            self.conn.execute(
                "INSERT INTO deliveries (delivery_id, state) VALUES (?, 'pending')",
                ("deliv-pending-race",),
            )

        # Case A.1: Redis is unavailable -> returns 503, NEVER false 200 duplicate
        self.in_memory_redis.drop_connection = True
        body, headers = self._create_signed_payload("deliv-pending-race")
        res1 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res1.status_code, 503)

        # Case A.2: Redis is available -> safely attempts deterministic enqueue -> returns 202
        self.in_memory_redis.drop_connection = False
        res2 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res2.status_code, 202)
        job = self.redis_queue.get_job_by_delivery("deliv-pending-race")
        self.assertIsNotNone(job)
        self.assertEqual(job.job_id, res2.json()["job_id"])

    def test_existing_delivery_with_missing_queue_job_replayed_safely(self) -> None:
        """B: Delivery marked enqueued locally but queue job missing -> safely replays enqueue instead of false 200."""
        body, headers = self._create_signed_payload("deliv-missing-job")
        res1 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res1.status_code, 202)
        job_id_1 = res1.json()["job_id"]

        # Evict / delete job from Redis directly to simulate queue loss
        self.in_memory_redis.delete(f"review:job:{job_id_1}")
        self.in_memory_redis.delete("review:delivery:deliv-missing-job")
        self.assertIsNone(self.redis_queue.get_job_by_delivery("deliv-missing-job"))

        # Local intake still has state 'enqueued'
        self.assertEqual(self.intake.get_delivery_state("deliv-missing-job"), "enqueued")

        # Duplicate delivery arrives -> must NOT return false 200 duplicate!
        res2 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res2.status_code, 202)
        replayed_job_id = res2.json()["job_id"]

        # Durable queue job exists again
        job_replayed = self.redis_queue.get_job_by_delivery("deliv-missing-job")
        self.assertIsNotNone(job_replayed)
        self.assertEqual(job_replayed.job_id, replayed_job_id)

    def test_immutable_snapshot_survives_enqueue_failure_and_retry(self) -> None:
        """C: Snapshot created on first intake is immutable and survives enqueue failure and subsequent retry."""
        self.in_memory_redis.drop_connection = True
        body, headers = self._create_signed_payload("deliv-snapshot-immutable")
        res1 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res1.status_code, 503)

        # Snapshot was stored during first intake
        orig_snapshot = self.intake.get_snapshot("deliv-snapshot-immutable")
        self.assertIsNotNone(orig_snapshot)
        self.assertEqual(orig_snapshot.head_sha, "head123")

        # Redis recovers and retry arrives
        self.in_memory_redis.drop_connection = False
        res2 = self.client.post("/webhooks/github", content=body, headers=headers)
        self.assertEqual(res2.status_code, 202)

        # Snapshot in intake is unchanged
        snapshot_after = self.intake.get_snapshot("deliv-snapshot-immutable")
        self.assertEqual(orig_snapshot, snapshot_after)

        # Job in queue has exact snapshot contents
        job = self.redis_queue.get_job_by_delivery("deliv-snapshot-immutable")
        self.assertIsNotNone(job)
        self.assertEqual(job.head_sha, "head123")

    def test_conflicting_retry_snapshot_rejected_fail_closed(self) -> None:
        """D: Retry with identical delivery_id but conflicting head SHA fails closed with HTTP 409."""
        body1, headers1 = self._create_signed_payload("deliv-conflict-test", head_sha="sha_original_123")
        res1 = self.client.post("/webhooks/github", content=body1, headers=headers1)
        self.assertEqual(res1.status_code, 202)

        # Second delivery with SAME delivery_id but DIFFERENT head SHA
        body2, headers2 = self._create_signed_payload("deliv-conflict-test", head_sha="sha_mutated_456")
        res2 = self.client.post("/webhooks/github", content=body2, headers=headers2)
        self.assertEqual(res2.status_code, 409)
        self.assertEqual(res2.json()["status"], "rejected")

        # Verify original snapshot and queue job were NOT corrupted or overwritten
        job = self.redis_queue.get_job_by_delivery("deliv-conflict-test")
        self.assertIsNotNone(job)
        self.assertEqual(job.head_sha, "sha_original_123")

        snapshot = self.intake.get_snapshot("deliv-conflict-test")
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.head_sha, "sha_original_123")


if __name__ == "__main__":
    unittest.main()
