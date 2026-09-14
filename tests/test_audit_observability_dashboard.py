"""Tests for event/audit spine, operational telemetry, and repository dashboard."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import time
import pytest

from pr_review_agent.github_output import FakeGitHubClient, GitHubReviewPublisher, PublicationStatus
from pr_review_agent.intake import ReviewSnapshot, WebhookIntake
from pr_review_agent.observability import (
    AuditRecord,
    AuditSpine,
    OperationalTelemetry,
    RepositoryDashboard,
    redact_sensitive_data,
)
from pr_review_agent.orchestration import AuditEvent, DurableJobQueue, JobState
from pr_review_agent.policy import (
    CanonicalFinding,
    FindingDisposition,
    ReviewTruthStore,
    TruthState,
)
from pr_review_agent.retrieval import CodeMemoryStore


def _sample_snapshot(
    repo: str = "owner/repo",
    pull_request_number: int = 42,
    head_sha: str = "sha-obs-100",
    repository_id: str | None = None,
) -> ReviewSnapshot:
    return ReviewSnapshot(
        repository_id=repository_id or repo,
        repository_full_name=repo,
        pull_request_number=pull_request_number,
        base_sha="sha-base-000",
        head_sha=head_sha,
        changed_files=("src/db.py",),
        policy_version="1.0",
        prompt_version="1.0",
        retrieval_index_version="1.0",
        model_configuration={"provider": "test"},
    )


def _create_sample_canonical(
    canonical_id: str = "can-obs-1",
    head_sha: str = "sha-obs-100",
    repo: str = "owner/repo",
    delivery_id: str = "",
    run_id: str = "",
) -> CanonicalFinding:
    return CanonicalFinding(
        canonical_id=canonical_id,
        repository_id=repo,
        head_sha=head_sha,
        category="security",
        severity="high",
        confidence=0.95,
        summary="SQL Injection vulnerability detected",
        rationale="Raw user parameter passed directly to query without parameterization.",
        file_path="src/db.py",
        line_range=(20, 25),
        contributing_candidate_ids=("cand-sec-1",),
        contributing_specialists=("security",),
        evidence_refs=("diff://src/db.py#L20-L25",),
        remediation="Use parameterized queries with ? placeholders.",
        disposition=FindingDisposition.HELD,
        delivery_id=delivery_id,
        run_id=run_id,
    )


def test_audit_spine_records_and_queries_time_ordered_events() -> None:
    conn = sqlite3.connect(":memory:")
    spine = AuditSpine(conn)

    t0 = 1000.0
    e1 = AuditEvent(correlation_id="corr-1", event_name="webhook.received", step="intake", timestamp=t0, details={"delivery_id": "del-1"})
    e2 = AuditEvent(correlation_id="corr-1", event_name="queue.enqueued", step="queue", timestamp=t0 + 1.0, details={"job_id": "job-1"})
    e3 = AuditEvent(correlation_id="corr-1", event_name="orchestration.completed", step="orchestration", timestamp=t0 + 2.0, details={"status": "done"})

    id1 = spine.record_event(e1, repository_id="owner/repo", pull_number=42, head_sha="sha-1")
    id2 = spine.record_event(e2, repository_id="owner/repo", pull_number=42, head_sha="sha-1")
    id3 = spine.record_event(e3, repository_id="owner/repo", pull_number=42, head_sha="sha-1")

    assert id1 > 0
    assert id2 > id1
    assert id3 > id2

    records = spine.get_events("corr-1")
    assert len(records) == 3
    assert [r.step for r in records] == ["intake", "queue", "orchestration"]
    assert records[0].details["delivery_id"] == "del-1"
    assert records[1].details["job_id"] == "job-1"


def test_audit_spine_redacts_secrets_and_tokens() -> None:
    conn = sqlite3.connect(":memory:")
    spine = AuditSpine(conn)

    event_with_secrets = AuditEvent(
        correlation_id="corr-secret",
        event_name="webhook.validated",
        step="intake",
        timestamp=time.time(),
        details={
            "webhook_secret": "my-super-secret-hmac-key",
            "github_token": "ghp_123456789012345678901234567890123456",
            "authorization_header": "Bearer secret-token-value-here-which-is-long",
            "safe_metadata": "public-repo/pr-123",
            "nested_info": {
                "api_key": "private_api_key_xyz123456789012345",
                "normal_field": 42,
            },
        },
    )

    spine.record_event(event_with_secrets)
    records = spine.get_events("corr-secret")
    assert len(records) == 1
    rec = records[0]

    details = rec.details
    assert details["webhook_secret"] == "[REDACTED]"
    assert details["github_token"] == "[REDACTED]"
    assert details["authorization_header"] == "[REDACTED]"
    assert details["safe_metadata"] == "public-repo/pr-123"
    assert details["nested_info"]["api_key"] == "[REDACTED]"
    assert details["nested_info"]["normal_field"] == 42


def test_audit_spine_reconstructs_full_run_provenance() -> None:
    """AC-11 and NFR-06: Operator can trace full review lifecycle across stores."""
    conn = sqlite3.connect(":memory:")

    # Setup all stores sharing the SQLite database
    secret = b"webhook-test-secret"
    intake = WebhookIntake(conn, secret)
    queue = DurableJobQueue(conn)
    truth_store = ReviewTruthStore(conn)
    spine = AuditSpine(conn)

    corr_id = "corr-lifecycle-100"
    repo = "owner/repo"
    pull = 88
    sha = "sha-audit-123"

    # 1. Intake delivery
    body = json.dumps({
        "action": "opened",
        "repository": {"id": 123, "full_name": repo},
        "pull_request": {
            "number": pull,
            "base": {"sha": "sha-base-000"},
            "head": {"sha": sha},
            "changed_files": 1,
        },
    }).encode()
    import hashlib
    import hmac
    sig = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
    intake.accept(
        headers={
            "x-github-delivery": corr_id,
            "x-github-event": "pull_request",
            "x-hub-signature-256": sig,
        },
        body=body,
        policy_version="1.0",
        prompt_version="1.0",
        retrieval_index_version="1.0",
        model_configuration={"provider": "test"},
    )

    # 2. Queue job
    snapshot = _sample_snapshot(repo=repo, pull_request_number=pull, head_sha=sha)
    job = queue.enqueue(snapshot, corr_id)

    # 3. Audit events for specialist steps
    spine.record_events(
        [
            AuditEvent(corr_id, "specialist.started", "security", time.time(), {"specialist": "security"}),
            AuditEvent(corr_id, "specialist.completed", "security", time.time(), {"candidate_findings": 1}),
            AuditEvent(corr_id, "policy.evaluated", "policy", time.time(), {"disposition": "HELD"}),
        ],
        repository_id=repo,
        pull_number=pull,
        head_sha=sha,
        run_id="run-100",
    )

    # 4. Review Truth finding and transition
    finding = _create_sample_canonical(
        canonical_id="can-audit-trace-1",
        head_sha=sha,
        delivery_id=corr_id,
        run_id="run-100",
    )
    truth_store.record_initial(finding, initial_state=TruthState.HELD, delivery_id=corr_id, run_id="run-100")
    truth_store.record_transition(
        "can-audit-trace-1",
        TruthState.APPROVED,
        actor="maintainer_bob",
        actor_role="maintainer",
        rationale="Verified SQL vulnerability, approving for publication.",
    )

    # 5. Reconstruct trace
    trace = spine.reconstruct_run(corr_id)

    assert trace.correlation_id == corr_id
    assert trace.repository_id == repo
    assert trace.pull_number == pull
    assert trace.head_sha == sha
    assert trace.delivery_status == "accepted"
    assert trace.queue_job_status == JobState.QUEUED.value
    assert len(trace.timeline) == 3
    assert len(trace.specialist_steps) == 2
    assert len(trace.findings) == 1
    assert trace.findings[0]["canonical_id"] == "can-audit-trace-1"
    assert len(trace.policy_transitions) == 2
    assert trace.policy_transitions[1]["state"] == TruthState.APPROVED.value
    assert trace.policy_transitions[1]["actor"] == "maintainer_bob"
    assert trace.contains_secrets is False


def test_operational_telemetry_queue_and_review_metrics() -> None:
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)
    spine = AuditSpine(conn)
    telemetry = OperationalTelemetry(conn)

    t0 = time.time() - 400.0  # enqueued 400s ago

    # Enqueue 3 jobs
    s1 = _sample_snapshot(pull_request_number=1, head_sha="sha-1")
    s2 = _sample_snapshot(pull_request_number=2, head_sha="sha-2")
    s3 = _sample_snapshot(pull_request_number=3, head_sha="sha-3")
    j1 = queue.enqueue(s1, "del-1", now=t0)
    j2 = queue.enqueue(s2, "del-2", now=t0 + 50.0)
    j3 = queue.enqueue(s3, "del-3", now=t0 + 100.0)

    # Lease j2
    queue.lease_next_job(now=t0 + 60.0)

    # Record audit events for j3 completed
    spine.record_event(AuditEvent("del-3", "run.start", "start", timestamp=t0 + 100.0))
    spine.record_event(AuditEvent("del-3", "run.finish", "finish", timestamp=t0 + 125.0))

    snap = telemetry.get_snapshot(now=time.time())

    assert snap.queue_depth == 2  # j1 and j3
    assert snap.leased_jobs == 1   # j2
    assert snap.oldest_queued_age_seconds >= 340.0
    assert snap.total_reviews_completed == 1
    assert snap.average_duration_seconds == 25.0


def test_operational_telemetry_finding_dispositions_and_publication_outcomes() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    telemetry = OperationalTelemetry(conn)

    # Create findings with various states
    f1 = _create_sample_canonical(canonical_id="can-disp-1")
    f2 = _create_sample_canonical(canonical_id="can-disp-2")
    f3 = _create_sample_canonical(canonical_id="can-disp-3")

    truth_store.record_initial(f1, initial_state=TruthState.APPROVED)
    truth_store.record_initial(f2, initial_state=TruthState.HELD)
    truth_store.record_initial(f3, initial_state=TruthState.AUTO_APPROVED)
    truth_store.record_transition(f3.canonical_id, TruthState.PUBLISHED, actor="publisher", actor_role="system", rationale="published")

    snap = telemetry.get_snapshot()

    assert snap.finding_dispositions[TruthState.APPROVED.value] == 1
    assert snap.finding_dispositions[TruthState.HELD.value] == 1
    assert snap.finding_dispositions[TruthState.PUBLISHED.value] == 1
    assert snap.finding_dispositions[TruthState.AUTO_APPROVED.value] == 0


def test_operational_alerts_trigger_on_queue_aging_and_failures() -> None:
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)
    telemetry = OperationalTelemetry(conn)

    t0 = 1000.0
    # Add an old queued job (> 300s)
    s_old = _sample_snapshot(pull_request_number=10, head_sha="sha-old")
    queue.enqueue(s_old, "del-old", now=t0)

    # Evaluate at t0 + 350s
    snap = telemetry.get_snapshot(now=t0 + 350.0)

    alert_ids = [a.alert_id for a in snap.active_alerts]
    assert "ALERT-QUEUE-AGING" in alert_ids


def test_operational_alerts_trigger_on_retrieval_staleness() -> None:
    conn = sqlite3.connect(":memory:")
    code_store = CodeMemoryStore(conn)
    telemetry = OperationalTelemetry(conn)

    # Index repository
    code_store.index_repository("owner/repo", "sha-stale", {"src/main.py": "print('hello')"})
    # Mark stale
    conn.execute("UPDATE repository_revisions SET is_fresh = 0 WHERE repository_id = 'owner/repo'")

    snap = telemetry.get_snapshot()
    assert snap.retrieval_freshness == "stale"
    alert_ids = [a.alert_id for a in snap.active_alerts]
    assert "ALERT-RETRIEVAL-STALENESS" in alert_ids


def test_dashboard_data_structure_and_machine_readable_telemetry() -> None:
    conn = sqlite3.connect(":memory:")
    dashboard = RepositoryDashboard(conn)

    data = dashboard.get_dashboard_data()

    assert "telemetry" in data
    assert "recent_reviews" in data
    assert "recent_findings" in data
    assert "policy_summary" in data
    assert data["policy_summary"]["source_control"] == "GitHub Pull Requests only"
    assert "NO merge" in data["policy_summary"]["capabilities"]


def test_dashboard_html_rendering_and_wcag_accessibility() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    dashboard = RepositoryDashboard(conn)

    f = _create_sample_canonical("can-ui-1")
    truth_store.record_initial(f, initial_state=TruthState.HELD)

    html_content = dashboard.render_html()

    # WCAG 2.2 AA and Semantic Structure Assertions
    assert "<!DOCTYPE html>" in html_content
    assert '<html lang="en">' in html_content
    assert '<header role="banner">' in html_content
    assert '<main role="main">' in html_content
    assert '<footer role="contentinfo">' in html_content
    assert '<th scope="col">' in html_content
    assert "<h1>" in html_content
    assert "<h2" in html_content
    assert "Single-Tenant V1" in html_content
    assert "Strictly No Merge / No Code-Edit Capabilities" in html_content
    assert "can-ui-1" in html_content
    assert "SQL Injection vulnerability detected" in html_content


def test_dashboard_investigation_view_by_correlation_id() -> None:
    conn = sqlite3.connect(":memory:")
    dashboard = RepositoryDashboard(conn)
    spine = dashboard.audit_spine

    spine.record_event(
        AuditEvent("corr-inv-1", "review.started", "dispatch", time.time(), {"meta": "val"}),
        repository_id="owner/repo",
        pull_number=12,
    )

    report = dashboard.investigate_run("corr-inv-1")

    assert report["correlation_id"] == "corr-inv-1"
    assert len(report["timeline"]) == 1
    assert report["timeline"][0]["step"] == "dispatch"


def test_audit_durability_across_sqlite_connection_reopening(tmp_path) -> None:
    db_file = tmp_path / "obs_durability.db"

    # Connection 1: write events
    conn1 = sqlite3.connect(str(db_file))
    spine1 = AuditSpine(conn1)
    spine1.record_event(AuditEvent("corr-durable", "event.one", "step1", timestamp=100.0, details={"k": "v1"}))
    spine1.record_event(AuditEvent("corr-durable", "event.two", "step2", timestamp=101.0, details={"k": "v2"}))
    conn1.close()

    # Connection 2: inspect and verify
    conn2 = sqlite3.connect(str(db_file))
    spine2 = AuditSpine(conn2)
    events = spine2.get_events("corr-durable")

    assert len(events) == 2
    assert events[0].event_name == "event.one"
    assert events[1].event_name == "event.two"
    conn2.close()


def test_telemetry_gracefully_degrades_when_optional_tables_missing() -> None:
    # Completely empty fresh connection with zero tables
    conn = sqlite3.connect(":memory:")
    telemetry = OperationalTelemetry(conn)

    snap = telemetry.get_snapshot()
    assert snap.queue_depth == 0
    assert snap.leased_jobs == 0
    assert snap.dead_letter_count == 0
    assert snap.total_reviews_completed == 0
    assert snap.failure_rate == 0.0
    assert snap.retrieval_freshness == "unknown"


def test_reconstruct_run_isolates_provenance_between_runs_on_same_repo_pr_sha() -> None:
    """Regression test (Defect 1): reconstruct_run must never associate records from another review on same repo/PR/SHA."""
    conn = sqlite3.connect(":memory:")
    secret = b"webhook-secret"
    intake = WebhookIntake(conn, secret)
    queue = DurableJobQueue(conn)
    spine = AuditSpine(conn)
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-shared-100"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    repo = "owner/repo"
    pull = 42
    sha = "sha-shared-100"

    # --- RUN A ---
    corr_a = "del-run-a"
    run_a = "run-id-a"
    body_a = json.dumps({
        "action": "opened",
        "repository": {"id": 12345, "full_name": repo},
        "pull_request": {
            "number": pull,
            "base": {"sha": "sha-base-000"},
            "head": {"sha": sha},
            "changed_files": 1,
        },
    }).encode()
    sig_a = "sha256=" + hmac.new(b"webhook-secret", body_a, hashlib.sha256).hexdigest()
    intake.accept(
        headers={"x-github-delivery": corr_a, "x-github-event": "pull_request", "x-hub-signature-256": sig_a},
        body=body_a,
        policy_version="1.0",
        prompt_version="1.0",
        retrieval_index_version="1.0",
        model_configuration={"provider": "test"},
    )
    snap_a = _sample_snapshot(repo=repo, pull_request_number=pull, head_sha=sha)
    queue.enqueue(snap_a, corr_a)
    spine.record_events(
        [
            AuditEvent(corr_a, "specialist.completed", "security", 1000.0, {"findings": 1}),
            AuditEvent(corr_a, "orchestration.completed", "orchestration", 1010.0, {"status": "done"}),
        ],
        repository_id=repo,
        pull_number=pull,
        head_sha=sha,
        run_id=run_a,
    )
    finding_a = _create_sample_canonical(
        canonical_id="can-finding-a",
        head_sha=sha,
        repo=repo,
        delivery_id=corr_a,
        run_id=run_a,
    )
    truth_store.record_initial(finding_a, initial_state=TruthState.AUTO_APPROVED, delivery_id=corr_a, run_id=run_a)
    diff_a = f"diff --git a/{finding_a.file_path} b/{finding_a.file_path}\n@@ -20,6 +20,6 @@\n+line20\n+line21\n+line22\n+line23\n+line24\n+line25\n"
    pub_res_a = publisher.publish_finding(finding_a, pull_number=pull, diff_content=diff_a)
    assert pub_res_a.status == PublicationStatus.PUBLISHED

    # --- RUN B (Same repo, same PR, same SHA, but distinct delivery, run, finding, and outcome) ---
    corr_b = "del-run-b"
    run_b = "run-id-b"
    body_b = json.dumps({
        "action": "synchronize",
        "repository": {"id": 12345, "full_name": repo},
        "pull_request": {
            "number": pull,
            "base": {"sha": "sha-base-000"},
            "head": {"sha": sha},
            "changed_files": 2,
        },
    }).encode()
    sig_b = "sha256=" + hmac.new(b"webhook-secret", body_b, hashlib.sha256).hexdigest()
    intake.accept(
        headers={"x-github-delivery": corr_b, "x-github-event": "pull_request", "x-hub-signature-256": sig_b},
        body=body_b,
        policy_version="1.0",
        prompt_version="1.0",
        retrieval_index_version="1.0",
        model_configuration={"provider": "test"},
    )
    snap_b = _sample_snapshot(repo=repo, pull_request_number=pull, head_sha=sha)
    queue.enqueue(snap_b, corr_b)
    spine.record_events(
        [
            AuditEvent(corr_b, "specialist.completed", "architecture", 2000.0, {"findings": 2}),
            AuditEvent(corr_b, "orchestration.completed", "orchestration", 2015.0, {"status": "done"}),
        ],
        repository_id=repo,
        pull_number=pull,
        head_sha=sha,
        run_id=run_b,
    )
    finding_b = _create_sample_canonical(
        canonical_id="can-finding-b",
        head_sha=sha,
        repo=repo,
        delivery_id=corr_b,
        run_id=run_b,
    )
    truth_store.record_initial(finding_b, initial_state=TruthState.HELD, delivery_id=corr_b, run_id=run_b)
    pub_res_b = publisher.publish_finding(finding_b, pull_number=pull)
    assert pub_res_b.status == PublicationStatus.HELD_OR_UNAUTHORIZED

    # --- VERIFY RECONSTRUCT RUN A CONTAINS ONLY A AND NEVER B ---
    trace_a = spine.reconstruct_run(corr_a)
    assert trace_a.correlation_id == corr_a
    assert trace_a.delivery_id == corr_a
    assert trace_a.run_id == run_a
    assert len(trace_a.timeline) == 2
    assert all(r.correlation_id == corr_a for r in trace_a.timeline)

    # Findings in trace A must contain only finding A
    finding_ids_a = [f["canonical_id"] for f in trace_a.findings]
    assert "can-finding-a" in finding_ids_a
    assert "can-finding-b" not in finding_ids_a

    # Policy transitions in trace A must contain only finding A
    transition_cids_a = [t["canonical_id"] for t in trace_a.policy_transitions]
    assert "can-finding-a" in transition_cids_a
    assert "can-finding-b" not in transition_cids_a

    # GitHub effects in trace A must contain only finding A's effect
    effect_cids_a = [e["canonical_id"] for e in trace_a.github_effects]
    assert "can-finding-a" in effect_cids_a
    assert "can-finding-b" not in effect_cids_a
    assert trace_a.github_effects[0]["status"] == PublicationStatus.PUBLISHED.value

    # --- VERIFY RECONSTRUCT RUN B CONTAINS ONLY B AND NEVER A ---
    trace_b = spine.reconstruct_run(corr_b)
    assert trace_b.correlation_id == corr_b
    assert trace_b.delivery_id == corr_b
    assert trace_b.run_id == run_b
    assert len(trace_b.timeline) == 2
    assert all(r.correlation_id == corr_b for r in trace_b.timeline)

    finding_ids_b = [f["canonical_id"] for f in trace_b.findings]
    assert "can-finding-b" in finding_ids_b
    assert "can-finding-a" not in finding_ids_b

    transition_cids_b = [t["canonical_id"] for t in trace_b.policy_transitions]
    assert "can-finding-b" in transition_cids_b
    assert "can-finding-a" not in transition_cids_b

    effect_cids_b = [e["canonical_id"] for e in trace_b.github_effects]
    assert "can-finding-b" in effect_cids_b
    assert "can-finding-a" not in effect_cids_b
    assert trace_b.github_effects[0]["status"] == PublicationStatus.HELD_OR_UNAUTHORIZED.value


def test_completed_review_telemetry_ignores_incomplete_runs() -> None:
    """Regression test (Defect 2): total_reviews_completed and average_duration_seconds must count only completed reviews."""
    conn = sqlite3.connect(":memory:")
    spine = AuditSpine(conn)
    telemetry = OperationalTelemetry(conn)

    t0 = 1000.0

    # Incomplete run: only run.start recorded (not completed)
    spine.record_event(
        AuditEvent("corr-incomplete", "run.start", "start", timestamp=t0, details={"step": "start"}),
        repository_id="owner/repo",
    )

    # Complete run: start and terminal completion recorded (duration = 40.0s)
    spine.record_event(
        AuditEvent("corr-complete", "run.start", "start", timestamp=t0, details={"step": "start"}),
        repository_id="owner/repo",
    )
    spine.record_event(
        AuditEvent("corr-complete", "orchestration.completed", "completed", timestamp=t0 + 40.0, details={"status": "completed"}),
        repository_id="owner/repo",
    )

    snap = telemetry.get_snapshot(now=t0 + 50.0)

    # Only corr-complete should be counted as completed
    assert snap.total_reviews_completed == 1
    assert snap.average_duration_seconds == 40.0


def test_dashboard_scoping_prevents_cross_repository_data_leak() -> None:
    """Regression test (Defect 3): scoped dashboard data must not leak jobs/findings from other repositories."""
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)
    truth_store = ReviewTruthStore(conn)
    dashboard = RepositoryDashboard(conn)

    repo_a = "org/repo-alpha"
    repo_b = "org/repo-beta"

    # Enqueue review jobs for repo A and repo B
    snap_a = _sample_snapshot(repo=repo_a, pull_request_number=10, head_sha="sha-a")
    snap_b = _sample_snapshot(repo=repo_b, pull_request_number=20, head_sha="sha-b")
    queue.enqueue(snap_a, "del-a")
    queue.enqueue(snap_b, "del-b")

    # Record findings for repo A and repo B
    fa = _create_sample_canonical(canonical_id="can-alpha-1", head_sha="sha-a", repo=repo_a, delivery_id="del-a")
    fb = _create_sample_canonical(canonical_id="can-beta-1", head_sha="sha-b", repo=repo_b, delivery_id="del-b")
    truth_store.record_initial(fa, initial_state=TruthState.APPROVED, delivery_id="del-a")
    truth_store.record_initial(fb, initial_state=TruthState.HELD, delivery_id="del-b")

    # Scoped dashboard for repo A
    dash_a = dashboard.get_dashboard_data(repository_id=repo_a)
    assert len(dash_a["recent_reviews"]) == 1
    assert dash_a["recent_reviews"][0]["repository_id"] == repo_a
    assert len(dash_a["recent_findings"]) == 1
    assert dash_a["recent_findings"][0]["canonical_id"] == "can-alpha-1"
    assert dash_a["recent_findings"][0]["repository_id"] == repo_a
    assert dash_a["telemetry"]["finding_dispositions"][TruthState.APPROVED.value] == 1
    assert dash_a["telemetry"]["finding_dispositions"][TruthState.HELD.value] == 0

    # Scoped dashboard for repo B
    dash_b = dashboard.get_dashboard_data(repository_id=repo_b)
    assert len(dash_b["recent_reviews"]) == 1
    assert dash_b["recent_reviews"][0]["repository_id"] == repo_b
    assert len(dash_b["recent_findings"]) == 1
    assert dash_b["recent_findings"][0]["canonical_id"] == "can-beta-1"
    assert dash_b["recent_findings"][0]["repository_id"] == repo_b
    assert dash_b["telemetry"]["finding_dispositions"][TruthState.APPROVED.value] == 0
    assert dash_b["telemetry"]["finding_dispositions"][TruthState.HELD.value] == 1


def test_scoped_telemetry_excludes_null_repo_audit_correlation_belonging_to_other_repo() -> None:
    """Regression test (Defect 1): repository-scoped telemetry must not count NULL-repository audit events from other repos."""
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)
    spine = AuditSpine(conn)
    telemetry = OperationalTelemetry(conn)

    repo_a = "org/repo-a"
    repo_b = "org/repo-b"
    t0 = 1000.0

    # Legitimate completed review for Repository A (with repository_id explicitly recorded)
    spine.record_event(
        AuditEvent("corr-a", "run.start", "start", timestamp=t0, details={"step": "start"}),
        repository_id=repo_a,
    )
    spine.record_event(
        AuditEvent("corr-a", "review.completed", "completed", timestamp=t0 + 20.0, details={"status": "completed"}),
        repository_id=repo_a,
    )

    # Correlation for Repository B has repository_id=None on audit events,
    # but exact durable lineage in review_jobs establishes that it belongs to Repository B.
    snap_b = _sample_snapshot(repo=repo_b, pull_request_number=99, head_sha="sha-b-99")
    queue.enqueue(snap_b, "del-b-100")  # durable job recorded with repository_id=repo_b

    spine.record_event(
        AuditEvent("del-b-100", "run.start", "start", timestamp=t0, details={"step": "start"}),
        repository_id=None,
    )
    spine.record_event(
        AuditEvent("del-b-100", "orchestration.completed", "completed", timestamp=t0 + 35.0, details={"status": "completed"}),
        repository_id=None,
    )

    # 1. Scoped snapshot for Repository A: must count ONLY A (not B)
    snap_a = telemetry.get_snapshot(repository_id=repo_a)
    assert snap_a.total_reviews_completed == 1
    assert snap_a.average_duration_seconds == 20.0

    # 2. Scoped snapshot for Repository B: must count B via durable lineage
    snap_b = telemetry.get_snapshot(repository_id=repo_b)
    assert snap_b.total_reviews_completed == 1
    assert snap_b.average_duration_seconds == 35.0

    # 3. Unscoped/global snapshot: counts both completions
    snap_global = telemetry.get_snapshot(repository_id=None)
    assert snap_global.total_reviews_completed == 2
    assert snap_global.average_duration_seconds == (20.0 + 35.0) / 2.0


def test_aggregation_invoked_not_treated_as_terminal_completion() -> None:
    """Regression test (Defect 2): aggregation_invoked is not a terminal completion event."""
    conn = sqlite3.connect(":memory:")
    spine = AuditSpine(conn)
    telemetry = OperationalTelemetry(conn)

    t0 = 1000.0
    spine.record_event(
        AuditEvent("corr-agg-only", "run.start", "start", timestamp=t0, details={"step": "start"}),
        repository_id="owner/repo",
    )
    spine.record_event(
        AuditEvent(
            "corr-agg-only",
            "aggregation_invoked",
            "aggregation",
            timestamp=t0 + 15.0,
            details={"successful_specialists": 2, "partial_failures": []},
        ),
        repository_id="owner/repo",
    )

    snap = telemetry.get_snapshot(now=t0 + 30.0)

    assert snap.total_reviews_completed == 0
    assert snap.average_duration_seconds == 0.0


def test_audit_spine_append_only_triggers_reject_update_and_delete(tmp_path) -> None:
    """Regression test (Item 1): audit_events table is append-only; rejects UPDATE and DELETE."""
    db_file = tmp_path / "append_only.db"

    # Connection 1: write event
    conn1 = sqlite3.connect(str(db_file))
    spine1 = AuditSpine(conn1)
    ev = AuditEvent("corr-append-1", "run.start", "start", timestamp=100.0, details={"status": "init"})
    event_id = spine1.record_event(ev, repository_id="owner/repo", pull_number=1)
    assert event_id > 0

    # UPDATE must be rejected by SQLite trigger
    with pytest.raises(sqlite3.DatabaseError, match="append-only: UPDATE operations are prohibited"):
        conn1.execute("UPDATE audit_events SET event_name = 'tampered' WHERE event_id = ?", (event_id,))

    # DELETE must be rejected by SQLite trigger
    with pytest.raises(sqlite3.DatabaseError, match="append-only: DELETE operations are prohibited"):
        conn1.execute("DELETE FROM audit_events WHERE event_id = ?", (event_id,))

    conn1.close()

    # Connection 2: reopening the DB preserves the original untampered event
    conn2 = sqlite3.connect(str(db_file))
    spine2 = AuditSpine(conn2)
    events = spine2.get_events("corr-append-1")
    assert len(events) == 1
    assert events[0].event_name == "run.start"
    assert events[0].details["status"] == "init"

    # Reopened connection still enforces append-only triggers
    with pytest.raises(sqlite3.DatabaseError, match="append-only: UPDATE operations are prohibited"):
        conn2.execute("UPDATE audit_events SET event_name = 'tampered' WHERE event_id = ?", (event_id,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only: DELETE operations are prohibited"):
        conn2.execute("DELETE FROM audit_events WHERE event_id = ?", (event_id,))
    conn2.close()


def test_scoped_telemetry_fails_closed_on_conflicting_repository_evidence() -> None:
    """Regression test (Item 2): correlation with conflicting repository evidence fails closed in scoped telemetry."""
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)
    spine = AuditSpine(conn)
    telemetry = OperationalTelemetry(conn)

    repo_a = "org/repo-a"
    repo_b = "org/repo-b"
    t0 = 1000.0

    # Correlation corr-conflict has conflicting repository evidence:
    # 1. Audit event explicitly records repository_id=repo_a
    spine.record_event(
        AuditEvent("corr-conflict", "run.start", "start", timestamp=t0, details={"step": "start"}),
        repository_id=repo_a,
    )
    spine.record_event(
        AuditEvent("corr-conflict", "orchestration.completed", "completed", timestamp=t0 + 25.0, details={"status": "completed"}),
        repository_id=repo_a,
    )

    # 2. But queue job for the same correlation explicitly records repository_id=repo_b
    snap_b = _sample_snapshot(repo=repo_b, pull_request_number=10, head_sha="sha-b-10")
    queue.enqueue(snap_b, "corr-conflict")

    # Scoped snapshot for repo A must NOT count it (conflict detected)
    snap_a = telemetry.get_snapshot(repository_id=repo_a)
    assert snap_a.total_reviews_completed == 0
    assert snap_a.average_duration_seconds == 0.0

    # Scoped snapshot for repo B must NOT count it (conflict detected)
    snap_b = telemetry.get_snapshot(repository_id=repo_b)
    assert snap_b.total_reviews_completed == 0
    assert snap_b.average_duration_seconds == 0.0

    # Global/unscoped telemetry still includes it per existing global semantics
    snap_global = telemetry.get_snapshot(repository_id=None)
    assert snap_global.total_reviews_completed == 1
    assert snap_global.average_duration_seconds == 25.0


def test_non_terminal_event_with_arbitrary_done_payload_does_not_count_as_completed() -> None:
    """Regression test (Item 3): non-terminal events with status/state payloads do not count as review completion."""
    conn = sqlite3.connect(":memory:")
    spine = AuditSpine(conn)
    telemetry = OperationalTelemetry(conn)

    t0 = 1000.0
    # Run 1: start + specialist event with arbitrary status="done" payload (not a recognized terminal event)
    spine.record_event(
        AuditEvent("corr-non-terminal", "run.start", "start", timestamp=t0, details={"step": "start"}),
        repository_id="owner/repo",
    )
    spine.record_event(
        AuditEvent(
            "corr-non-terminal",
            "specialist.completed",
            "security",
            timestamp=t0 + 10.0,
            details={"status": "done", "state": "completed", "findings_count": 0},
        ),
        repository_id="owner/repo",
    )

    snap1 = telemetry.get_snapshot(now=t0 + 20.0)
    assert snap1.total_reviews_completed == 0
    assert snap1.average_duration_seconds == 0.0

    # Run 2: legitimate recognized terminal event (run.completed)
    spine.record_event(
        AuditEvent("corr-legit", "run.start", "start", timestamp=t0, details={"step": "start"}),
        repository_id="owner/repo",
    )
    spine.record_event(
        AuditEvent(
            "corr-legit",
            "run.completed",
            "orchestration",
            timestamp=t0 + 30.0,
            details={"status": "done"},
        ),
        repository_id="owner/repo",
    )

    snap2 = telemetry.get_snapshot(now=t0 + 40.0)
    assert snap2.total_reviews_completed == 1
    assert snap2.average_duration_seconds == 30.0



