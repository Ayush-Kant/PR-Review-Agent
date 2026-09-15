"""Focused offline tests for live PR validation configuration, server, and worker."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
import pytest
from starlette.testclient import TestClient

from pr_review_agent.github_output import (
    FakeGitHubClient,
    GitHubReviewPublisher,
    PublicationStatus,
    StaleHeadShaError,
)
from pr_review_agent.intake import ReviewSnapshot, WebhookIntake
from pr_review_agent.observability import AuditSpine
from pr_review_agent.orchestration import (
    CandidateFinding,
    DurableJobQueue,
    JobState,
    ReviewJob,
    ReviewOrchestrator,
    SpecialistHandler,
    SpecialistInput,
    SpecialistOutput,
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
from pr_review_agent.security import SecretType, SecurityConfig
from pr_review_agent.server import create_server_app
from pr_review_agent.service_config import ServiceConfig, load_service_config
from pr_review_agent.worker import AutonomousReviewWorker, reconstruct_snapshot


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _sign_payload(secret: bytes, body: bytes) -> str:
    digest = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _make_dummy_service_config(
    *,
    publish_enabled: bool = False,
    model_provider: str = "openai",
    api_key: str = "sk-dummyopenai-12345678901234567890",
) -> ServiceConfig:
    return ServiceConfig(
        github_token="ghp_dummytoken123456789012345678901234567890",
        webhook_secret=b"dummy-webhook-secret-32-bytes-ok!",
        model_provider=model_provider,
        model_name="gpt-4o",
        database_path=":memory:",
        host="127.0.0.1",
        port=8000,
        authorized_repositories=("octocat/hello-world",),
        authorized_tenant="octocat",
        publish_enabled=publish_enabled,
        api_key=api_key,
    )


def _make_mock_specialist(findings: list[CandidateFinding] | None = None) -> SpecialistHandler:
    def handler(spec_input: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status="completed",
            findings=tuple(findings or []),
            execution_duration=0.01,
        )

    return handler


# ---------------------------------------------------------------------------
# 1. Configuration Validation Tests
# ---------------------------------------------------------------------------

def test_config_loading_and_fail_closed() -> None:
    # 1. Missing required variables when require_live_credentials=True raises ValueError
    with pytest.raises(ValueError, match="Missing required GITHUB_TOKEN"):
        load_service_config({}, require_live_credentials=True)

    with pytest.raises(ValueError, match="Missing required GITHUB_WEBHOOK_SECRET"):
        load_service_config(
            {"GITHUB_TOKEN": "token"},
            require_live_credentials=True,
        )

    with pytest.raises(ValueError, match="Missing required GITHUB_REPOSITORY"):
        load_service_config(
            {
                "GITHUB_TOKEN": "token",
                "GITHUB_WEBHOOK_SECRET": "secret",
            },
            require_live_credentials=True,
        )

    with pytest.raises(ValueError, match="Missing required OPENAI_API_KEY"):
        load_service_config(
            {
                "GITHUB_TOKEN": "token",
                "GITHUB_WEBHOOK_SECRET": "secret",
                "GITHUB_REPOSITORY": "owner/repo",
                "MODEL_PROVIDER": "openai",
            },
            require_live_credentials=True,
        )

    # 2. Valid configuration load
    env = {
        "GITHUB_TOKEN": "ghp_validtoken123456789012345678901234567890",
        "GITHUB_WEBHOOK_SECRET": "my-secret",
        "GITHUB_REPOSITORY": "owner/repo,owner/repo2",
        "MODEL_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-mock-key-12345678901234567890",
        "DATABASE_PATH": "test.db",
        "HOST": "0.0.0.0",
        "PORT": "9000",
        "PUBLISH_LIVE_REVIEW": "1",
    }
    cfg = load_service_config(env, require_live_credentials=True)
    assert cfg.github_token == "ghp_validtoken123456789012345678901234567890"
    assert cfg.webhook_secret == b"my-secret"
    assert cfg.authorized_repositories == ("owner/repo", "owner/repo2")
    assert cfg.authorized_tenant == "owner"
    assert cfg.model_provider == "openai"
    assert cfg.model_name == "gpt-4o"
    assert cfg.database_path == "test.db"
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9000
    assert cfg.publish_enabled is True
    assert cfg.api_key == "sk-mock-key-12345678901234567890"


def test_config_secret_masking_in_repr() -> None:
    cfg = _make_dummy_service_config()
    repr_str = repr(cfg)
    assert "ghp_dummytoken" not in repr_str
    assert "dummy-webhook-secret" not in repr_str
    assert "sk-dummyopenai" not in repr_str
    assert "github_token='***'" in repr_str
    assert "webhook_secret=b'***'" in repr_str
    assert "api_key='***'" in repr_str


def test_config_to_security_config_registration() -> None:
    cfg = _make_dummy_service_config()
    sec_cfg = cfg.to_security_config()
    assert sec_cfg.authorized_tenant == "octocat"
    assert "octocat/hello-world" in sec_cfg.authorized_repositories
    # Secret categories must be registered in the registry
    assert SecretType.GITHUB_TOKEN.value in sec_cfg.secret_registry.get_registered_categories()
    assert SecretType.OPENAI_API_KEY.value in sec_cfg.secret_registry.get_registered_categories()
    matches = sec_cfg.secret_registry.scan_exact_matches(cfg.github_token)
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# 2. Server Construction and Routes Tests
# ---------------------------------------------------------------------------

def test_server_routes_and_webhook_ingress() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cfg = _make_dummy_service_config()
    app = create_server_app(cfg, connection=conn)

    client = TestClient(app)

    # 1. GET /healthz returns 200 healthy
    resp_health = client.get("/healthz")
    assert resp_health.status_code == 200
    assert resp_health.json() == {"status": "healthy"}

    # 2. POST /webhooks/github without valid HMAC returns 401 rejected
    bad_resp = client.post(
        "/webhooks/github",
        content=b'{"action": "opened"}',
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "deliv-bad-1",
            "X-Hub-Signature-256": "sha256=invalid",
            "Content-Type": "application/json",
        },
    )
    assert bad_resp.status_code == 401
    assert bad_resp.json()["status"] == "rejected"

    # 3. POST /webhooks/github with valid HMAC enqueues review job and returns 202
    payload = {
        "action": "opened",
        "repository": {
            "id": 12345,
            "full_name": "octocat/hello-world",
        },
        "pull_request": {
            "number": 42,
            "base": {"sha": "basesha00000000000000000000000000000000"},
            "head": {"sha": "headsha11111111111111111111111111111111"},
        },
    }
    raw_body = json.dumps(payload).encode("utf-8")
    sig = _sign_payload(cfg.webhook_secret, raw_body)

    good_resp = client.post(
        "/webhooks/github",
        content=raw_body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "deliv-ok-1",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )
    assert good_resp.status_code == 202
    res_data = good_resp.json()
    assert res_data["status"] == "accepted"
    assert res_data["delivery_id"] == "deliv-ok-1"
    assert res_data["job_id"] == "job-deliv-ok-1"

    # 4. Webhook idempotency / duplicate delivery returns 200 duplicate
    dup_resp = client.post(
        "/webhooks/github",
        content=raw_body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "deliv-ok-1",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )
    assert dup_resp.status_code == 200
    assert dup_resp.json()["status"] == "duplicate"


# ---------------------------------------------------------------------------
# 3. Snapshot Reconstruction Tests
# ---------------------------------------------------------------------------

def test_reconstruct_snapshot_variants() -> None:
    data = {
        "repository_id": "octocat/hello-world",
        "repository_full_name": "octocat/hello-world",
        "pull_request_number": 42,
        "base_sha": "base123",
        "head_sha": "head456",
        "changed_files": ["main.py"],
        "policy_version": "v1",
        "prompt_version": "v1",
        "retrieval_index_version": "v1",
        "model_configuration": {"provider": "openai", "model": "gpt-4o"},
    }
    json_str = json.dumps(data)

    # Reconstruct from dict
    snap1 = reconstruct_snapshot(data)
    assert snap1.repository_id == "octocat/hello-world"
    assert snap1.pull_request_number == 42
    assert snap1.head_sha == "head456"
    assert snap1.changed_files == ("main.py",)

    # Reconstruct from json str
    snap2 = reconstruct_snapshot(json_str)
    assert snap2 == snap1

    # Reconstruct from ReviewJob via connection
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)
    expected_snap = ReviewSnapshot(
        repository_id="octocat/hello-world",
        repository_full_name="octocat/hello-world",
        pull_request_number=42,
        base_sha="base123",
        head_sha="head456",
        changed_files=("main.py",),
        policy_version="v1",
        prompt_version="v1",
        retrieval_index_version="v1",
        model_configuration={"provider": "openai", "model": "gpt-4o"},
    )
    job = queue.enqueue(expected_snap, delivery_id="deliv-recon-1")
    snap3 = reconstruct_snapshot(job, connection=conn)
    assert snap3 == expected_snap

    # Invalid type
    with pytest.raises(TypeError, match="Cannot reconstruct ReviewSnapshot"):
        reconstruct_snapshot(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Autonomous Worker Job Lifecycle Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_worker_job_lifecycle_success() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cfg = _make_dummy_service_config(publish_enabled=True)

    # Setup fake GitHub client with current head SHA
    fake_github = FakeGitHubClient()
    repo_id = "octocat/hello-world"
    pr_num = 42
    head_sha = "headsha11111111111111111111111111111111"
    fake_github.set_head_sha(repo_id, pr_num, head_sha)
    diff_hunk = (
        "diff --git a/main.py b/main.py\n"
        "--- a/main.py\n"
        "+++ b/main.py\n"
        "@@ -10,3 +10,3 @@\n"
        "+line10\n"
        "+line11\n"
        "+line12\n"
    )
    fake_github.get_pull_request_diff = lambda repo, pr: diff_hunk  # type: ignore[assignment]

    # Prepare candidate finding with verified evidence and auto-approvable severity and category
    candidate = CandidateFinding(
        finding_id="find-1",
        correlation_id="deliv-lifecycle-1",
        specialist_type=SpecialistType.QUALITY,
        category="quality",
        severity="low",
        confidence=0.95,
        summary="Potential code smell",
        rationale="Unused local variable",
        file_path="main.py",
        line_range=(10, 12),
        evidence_refs=("ref:main.py:10-12",),
    )

    handlers = {
        SpecialistType.SECURITY: _make_mock_specialist([]),
        SpecialistType.QUALITY: _make_mock_specialist([candidate]),
        SpecialistType.TESTS: _make_mock_specialist([]),
        SpecialistType.DOCUMENTATION: _make_mock_specialist([]),
    }

    worker = AutonomousReviewWorker(
        cfg,
        connection=conn,
        github_client=fake_github,
        specialist_handlers=handlers,
    )

    # Enqueue a job
    snapshot = ReviewSnapshot(
        repository_id=repo_id,
        repository_full_name=repo_id,
        pull_request_number=pr_num,
        base_sha="basesha00000000000000000000000000000000",
        head_sha=head_sha,
        changed_files=("main.py",),
        policy_version="v1",
        prompt_version="v1",
        retrieval_index_version="v1",
        model_configuration={"provider": "openai", "model": "gpt-4o"},
    )
    job = worker.queue.enqueue(snapshot, delivery_id="deliv-lifecycle-1")
    assert job.state == JobState.QUEUED

    # Execute worker on one job
    processed_job, state = await worker.process_one_job()
    assert processed_job is not None
    assert state is not None

    # Job must be marked COMPLETED in the queue
    db_job = worker.queue.get_job(job.job_id)
    assert db_job is not None
    assert db_job.state == JobState.COMPLETED
    assert db_job.last_error is None

    # Finding must be recorded in ReviewTruthStore
    rows = conn.execute("SELECT canonical_id, state FROM review_truth").fetchall()
    assert len(rows) >= 1
    canonical_id, _ = rows[0]
    records = worker.truth_store.get_history(canonical_id)
    assert len(records) >= 1
    # Low severity + high confidence + evidence => AUTO_APPROVED
    assert records[0].state == TruthState.AUTO_APPROVED

    # Finding must be published via FakeGitHubClient
    assert len(fake_github.reviews) == 1
    assert fake_github.reviews[0]["commit_sha"] == head_sha

    # Audit spine has recorded the key events
    trace = worker.audit_spine.get_events("deliv-lifecycle-1")
    event_names = [e.event_name for e in trace]
    assert "worker_job_started" in event_names
    assert "worker_job_completed" in event_names

    # Clean up
    worker.close()


@pytest.mark.anyio
async def test_worker_failure_handling() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cfg = _make_dummy_service_config()

    fake_github = FakeGitHubClient(should_fail_api=True, api_error_message="GitHub unreachable")

    handlers = {
        SpecialistType.SECURITY: _make_mock_specialist([]),
        SpecialistType.QUALITY: _make_mock_specialist([]),
        SpecialistType.TESTS: _make_mock_specialist([]),
        SpecialistType.DOCUMENTATION: _make_mock_specialist([]),
    }

    worker = AutonomousReviewWorker(
        cfg,
        connection=conn,
        github_client=fake_github,
        specialist_handlers=handlers,
    )

    snapshot = ReviewSnapshot(
        repository_id="octocat/hello-world",
        repository_full_name="octocat/hello-world",
        pull_request_number=10,
        base_sha="base",
        head_sha="head",
        changed_files=(),
        policy_version="v1",
        prompt_version="v1",
        retrieval_index_version="v1",
        model_configuration={},
    )
    job = worker.queue.enqueue(snapshot, delivery_id="deliv-fail-1")

    # If get_pull_request_diff fails or specialist fails, process_one_job raises and marks job failed
    # We monkeypatch fake_github.get_pull_request_diff to raise
    fake_github.get_pull_request_diff = lambda repo, pr: (_ for _ in ()).throw(RuntimeError("Diff network error"))  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Diff network error"):
        await worker.process_one_job()

    # Job must be marked failed in queue with backoff or retry state
    db_job = worker.queue.get_job(job.job_id)
    assert db_job is not None
    assert "Diff network error" in (db_job.last_error or "")

    worker.close()


@pytest.mark.anyio
async def test_worker_stale_head_sha_handling() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cfg = _make_dummy_service_config(publish_enabled=True)

    fake_github = FakeGitHubClient()
    repo_id = "octocat/hello-world"
    pr_num = 50
    old_sha = "headsha_old_000000000000000000000000"
    new_sha = "headsha_new_111111111111111111111111"

    # PR head has already moved to new_sha before publishing
    fake_github.set_head_sha(repo_id, pr_num, new_sha)

    candidate = CandidateFinding(
        finding_id="find-stale-1",
        correlation_id="deliv-stale-1",
        specialist_type=SpecialistType.QUALITY,
        category="quality",
        severity="low",
        confidence=0.95,
        summary="Stale test finding",
        rationale="Testing stale head sha detection",
        file_path="foo.py",
        line_range=(1, 2),
        evidence_refs=("ref:foo.py:1-2",),
    )

    handlers = {
        SpecialistType.SECURITY: _make_mock_specialist([]),
        SpecialistType.QUALITY: _make_mock_specialist([candidate]),
        SpecialistType.TESTS: _make_mock_specialist([]),
        SpecialistType.DOCUMENTATION: _make_mock_specialist([]),
    }

    worker = AutonomousReviewWorker(
        cfg,
        connection=conn,
        github_client=fake_github,
        specialist_handlers=handlers,
    )

    snapshot = ReviewSnapshot(
        repository_id=repo_id,
        repository_full_name=repo_id,
        pull_request_number=pr_num,
        base_sha="base",
        head_sha=old_sha,  # Reviewed under old_sha
        changed_files=("foo.py",),
        policy_version="v1",
        prompt_version="v1",
        retrieval_index_version="v1",
        model_configuration={"provider": "openai", "model": "gpt-4o"},
    )
    worker.queue.enqueue(snapshot, delivery_id="deliv-stale-1")

    await worker.process_one_job()

    # Invariant: NO review comments published to GitHub because head SHA was stale
    assert len(fake_github.reviews) == 0

    # Invariant: ReviewTruthStore transitions finding to SUPERSEDED
    rows = conn.execute("SELECT canonical_id, state FROM review_truth").fetchall()
    assert len(rows) >= 1
    canonical_id, _ = rows[0]
    latest = worker.truth_store.get_latest_state(canonical_id)
    assert latest is not None
    assert latest.state == TruthState.SUPERSEDED

    worker.close()


@pytest.mark.anyio
async def test_worker_no_write_github_capability_invariant() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cfg = _make_dummy_service_config(publish_enabled=True)

    fake_github = FakeGitHubClient()
    repo_id = "octocat/hello-world"
    pr_num = 99
    head_sha = "head99"
    fake_github.set_head_sha(repo_id, pr_num, head_sha)

    handlers = {
        SpecialistType.SECURITY: _make_mock_specialist([]),
        SpecialistType.QUALITY: _make_mock_specialist([]),
        SpecialistType.TESTS: _make_mock_specialist([]),
        SpecialistType.DOCUMENTATION: _make_mock_specialist([]),
    }

    worker = AutonomousReviewWorker(
        cfg,
        connection=conn,
        github_client=fake_github,
        specialist_handlers=handlers,
    )

    snapshot = ReviewSnapshot(
        repository_id=repo_id,
        repository_full_name=repo_id,
        pull_request_number=pr_num,
        base_sha="base99",
        head_sha=head_sha,
        changed_files=(),
        policy_version="v1",
        prompt_version="v1",
        retrieval_index_version="v1",
        model_configuration={},
    )
    worker.queue.enqueue(snapshot, delivery_id="deliv-nowrite-1")
    await worker.process_one_job()

    # Safety invariant verification: client has 0 attempted merges or code modifications
    assert fake_github.attempted_merges == 0
    assert fake_github.attempted_code_modifications == 0

    worker.close()


@pytest.mark.anyio
async def test_worker_run_loop_stop_conditions() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cfg = _make_dummy_service_config()
    fake_github = FakeGitHubClient()

    handlers = {
        SpecialistType.SECURITY: _make_mock_specialist([]),
        SpecialistType.QUALITY: _make_mock_specialist([]),
        SpecialistType.TESTS: _make_mock_specialist([]),
        SpecialistType.DOCUMENTATION: _make_mock_specialist([]),
    }

    worker = AutonomousReviewWorker(
        cfg,
        connection=conn,
        github_client=fake_github,
        specialist_handlers=handlers,
    )

    # Test max_iterations limit
    iterations = await worker.run_worker_loop(poll_interval_seconds=0.001, max_iterations=3)
    assert iterations == 3

    # Test stop_event
    stop_event = asyncio.Event()
    stop_event.set()
    iterations2 = await worker.run_worker_loop(poll_interval_seconds=0.001, stop_event=stop_event)
    assert iterations2 == 0

    worker.close()


# ---------------------------------------------------------------------------
# 5. Live Review Harness Multi-Specialist and Safety Tests
# ---------------------------------------------------------------------------

def test_live_pr_review_harness_runs_all_four_specialists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.live_pr_review import run_live_review

    # 1. Safety check: when ENABLE_LIVE_GITHUB_TEST is not set, exits early with 0
    monkeypatch.delenv("ENABLE_LIVE_GITHUB_TEST", raising=False)
    assert run_live_review() == 0

    # 2. When opted-in, verify it executes all four specialists: SECURITY, QUALITY, TESTS, DOCUMENTATION
    monkeypatch.setenv("ENABLE_LIVE_GITHUB_TEST", "1")
    monkeypatch.setenv("PUBLISH_LIVE_REVIEW", "0")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123456789012345678901234567890")
    monkeypatch.setenv("GITHUB_REPOSITORY", "octocat/hello-world")
    monkeypatch.setenv("GITHUB_PR_NUMBER", "42")
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake12345678901234567890")

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "cli_test.db"))

    mock_gh = FakeGitHubClient()
    mock_gh.set_head_sha("octocat/hello-world", 42, "headsha123")
    mock_gh.get_pull_request = lambda repo, pr: {  # type: ignore[assignment]
        "head": {"sha": "headsha123"},
        "base": {"sha": "basesha123"},
        "title": "Test PR",
    }
    mock_gh.get_pull_request_diff = lambda repo, pr: "diff --git a/foo.py b/foo.py\n..."  # type: ignore[assignment]

    executed_specialists: list[SpecialistType] = []

    class MockAdapter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, spec_input: SpecialistInput) -> SpecialistOutput:
            executed_specialists.append(spec_input.specialist_type)
            return SpecialistOutput(
                specialist_type=spec_input.specialist_type,
                correlation_id=spec_input.correlation_id,
                status="completed",
                findings=(),
            )

    monkeypatch.setattr("scripts.live_pr_review.GitHubNetworkClient", lambda sec_config: mock_gh)
    monkeypatch.setattr("scripts.live_pr_review.LLMSpecialistAdapter", MockAdapter)

    ret = run_live_review()
    assert ret == 0
    assert executed_specialists == [
        SpecialistType.SECURITY,
        SpecialistType.QUALITY,
        SpecialistType.TESTS,
        SpecialistType.DOCUMENTATION,
    ]
    # Dry-run safety: verify 0 reviews or comments published
    assert len(mock_gh.reviews) == 0
    assert len(mock_gh.comments) == 0


def test_cross_process_publication_idempotency_via_durable_sqlite(tmp_path: Path) -> None:
    """Offline test demonstrating cross-process persistence and idempotency across separate processes."""
    db_file = str(tmp_path / "durable_harness_test.db")
    diff_content = "diff --git a/src/service.py b/src/service.py\n+ def foo(): pass"

    finding = CanonicalFinding(
        canonical_id="can-durable-idemp-1",
        repository_id="octocat/hello-world",
        head_sha="head123",
        category="quality",
        severity="low",
        confidence=0.92,
        summary="Code quality observation",
        rationale="Detailed observation",
        file_path="src/service.py",
        line_range=(1, 1),
        contributing_candidate_ids=("cand-1",),
        contributing_specialists=("quality",),
        evidence_refs=("diff://src/service.py#L1",),
        remediation="Clean up syntax",
        merge_rationale="Merged 1 finding",
        delivery_id="deliv-1",
        run_id="run-1",
    )

    # Process / Run #1:
    conn1 = sqlite3.connect(db_file)
    truth_store1 = ReviewTruthStore(conn1)
    truth_store1.record_initial(finding, initial_state=TruthState.AUTO_APPROVED)

    mock_gh_1 = FakeGitHubClient()
    mock_gh_1.set_head_sha("octocat/hello-world", 42, "head123")
    publisher1 = GitHubReviewPublisher(conn1, truth_store1, mock_gh_1)

    res1 = publisher1.publish_finding(finding, pull_number=42, diff_content=diff_content)
    assert res1.status == PublicationStatus.PUBLISHED
    assert len(mock_gh_1.reviews) + len(mock_gh_1.comments) == 1
    conn1.commit()
    conn1.close()

    # Process / Run #2: completely separate connection opening the same database file
    conn2 = sqlite3.connect(db_file)
    truth_store2 = ReviewTruthStore(conn2)
    mock_gh_2 = FakeGitHubClient()
    mock_gh_2.set_head_sha("octocat/hello-world", 42, "head123")
    publisher2 = GitHubReviewPublisher(conn2, truth_store2, mock_gh_2)

    # Attempt to republish the same finding
    res2 = publisher2.publish_finding(finding, pull_number=42, diff_content=diff_content)
    assert res2.status == PublicationStatus.ALREADY_PUBLISHED
    # Verify no duplicate network/API publication performed!
    assert len(mock_gh_2.reviews) == 0
    assert len(mock_gh_2.comments) == 0

    # Verify Review Truth state remained PUBLISHED
    latest = truth_store2.get_latest_state(finding.canonical_id)
    assert latest is not None
    assert latest.state == TruthState.PUBLISHED

    conn2.close()


def test_live_harness_uses_configured_database_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify scripts.live_pr_review uses configured DATABASE_PATH rather than :memory:."""
    from scripts.live_pr_review import run_live_review

    test_db = str(tmp_path / "custom_harness.db")
    monkeypatch.setenv("ENABLE_LIVE_GITHUB_TEST", "1")
    monkeypatch.setenv("PUBLISH_LIVE_REVIEW", "0")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123456789012345678901234567890")
    monkeypatch.setenv("GITHUB_REPOSITORY", "octocat/hello-world")
    monkeypatch.setenv("GITHUB_PR_NUMBER", "42")
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake12345678901234567890")
    monkeypatch.setenv("DATABASE_PATH", test_db)

    mock_gh = FakeGitHubClient()
    mock_gh.set_head_sha("octocat/hello-world", 42, "headsha123")
    mock_gh.get_pull_request = lambda repo, pr: {  # type: ignore[assignment]
        "head": {"sha": "headsha123"},
        "base": {"sha": "basesha123"},
        "title": "Test PR",
    }
    mock_gh.get_pull_request_diff = lambda repo, pr: "diff --git a/foo.py b/foo.py\n..."  # type: ignore[assignment]

    class MockAdapter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, spec_input: SpecialistInput) -> SpecialistOutput:
            return SpecialistOutput(
                specialist_type=spec_input.specialist_type,
                correlation_id=spec_input.correlation_id,
                status="completed",
                findings=(),
            )

    monkeypatch.setattr("scripts.live_pr_review.GitHubNetworkClient", lambda sec_config: mock_gh)
    monkeypatch.setattr("scripts.live_pr_review.LLMSpecialistAdapter", MockAdapter)

    ret = run_live_review()
    assert ret == 0
    # Verify the database file was created on disk
    assert os.path.exists(test_db)
    # Verify review_truth table exists in the file
    check_conn = sqlite3.connect(test_db)
    row = check_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='review_truth'").fetchone()
    assert row is not None
    check_conn.close()
