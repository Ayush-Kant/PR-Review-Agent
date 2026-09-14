"""End-to-End local replay test exercising the complete PR-Review-Agent lifecycle.

Covers:
- Webhook HMAC intake and delivery idempotency (FR-01, FR-02)
- Review snapshot persistence and durable queueing (FR-03, FR-04)
- LangGraph orchestration across independent specialists (FR-05, FR-06)
- Hybrid retrieval and evidence grounding verification (FR-07, FR-08)
- Candidate deduplication and canonical finding creation (FR-09)
- Calibrated confidence, risk policy, and HITL disposition (FR-10, FR-11)
- Current-SHA safe, idempotent GitHub publication via FakeGitHubClient (FR-12, AC-08, AC-09)
- Invariant safety: zero merges and zero code modifications attempted
- Audit spine event correlation and secret redaction (FR-15, AC-11)
- Operational telemetry and repository dashboard HTML rendering (FR-17, AC-15)
- Safety check: stale head SHA detection and suppression (AC-09)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sqlite3
import unittest

from pr_review_agent.cost_controls import (
    BudgetEnforcer,
    CostAndBudgetConfig,
    CostLedger,
    ProviderPricingRegistry,
)
from pr_review_agent.github_output import (
    FakeGitHubClient,
    GitHubReviewPublisher,
    PublicationStatus,
)
from pr_review_agent.intake import WebhookIntake
from pr_review_agent.observability import (
    AuditEvent,
    AuditSpine,
    OperationalTelemetry,
    RepositoryDashboard,
)
from pr_review_agent.orchestration import (
    CandidateFinding,
    DurableJobQueue,
    JobState,
    ReviewOrchestrator,
    ReviewWorker,
    SpecialistInput,
    SpecialistOutput,
    SpecialistType,
)
from pr_review_agent.policy import (
    CanonicalFinding,
    FindingAggregator,
    FindingDisposition,
    MaintainerWorkflow,
    ReviewPolicyEngine,
    ReviewTruthStore,
    TruthState,
)
from pr_review_agent.retrieval import (
    CodeChunk,
    CodeMemoryStore,
    FindingEvidenceValidator,
    HybridRetriever,
)
from pr_review_agent.security import (
    RuntimeSecretRegistry,
    SecretLeakageScanner,
    SecurityConfig,
    SecretType,
)


SAMPLE_UNIFIED_DIFF = """diff --git a/src/auth.py b/src/auth.py
index e69de29..49e29a1 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,12 +10,16 @@ def authenticate_user(username, password_hash):
+    query = f"SELECT * FROM users WHERE username = '{username}'"
+    user = db.execute(query).fetchone()
+    if not user:
+        return None
+    return user
"""


def test_complete_end_to_end_pr_review_replay() -> None:
    """Execute complete end-to-end review lifecycle with FakeGitHubClient."""
    conn = sqlite3.connect(":memory:")
    webhook_secret = b"super-secret-webhook-key-1234"

    # 1. Initialize Subsystems
    intake = WebhookIntake(conn, webhook_secret=webhook_secret)
    queue = DurableJobQueue(conn)
    memory_store = CodeMemoryStore(conn)
    retriever = HybridRetriever(memory_store)
    validator = FindingEvidenceValidator(memory_store)
    aggregator = FindingAggregator()
    policy_engine = ReviewPolicyEngine()
    truth_store = ReviewTruthStore(conn)
    maintainer_workflow = MaintainerWorkflow(truth_store)

    repo_id = "owner/secure-repo"
    pr_num = 42
    head_sha = "sha-head-abc12345"
    base_sha = "sha-base-00000000"

    github_client = FakeGitHubClient(pr_heads={(repo_id, pr_num): head_sha})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)
    audit_spine = AuditSpine(conn)
    telemetry = OperationalTelemetry(conn)
    dashboard = RepositoryDashboard(conn)

    # 2. Populate Code Memory Store with repository files
    memory_store.index_repository(
        repository_id=repo_id,
        revision=head_sha,
        files={
            "src/auth.py": "def authenticate_user(username, password_hash):\n    query = f\"SELECT * FROM users WHERE username = '{username}'\"\n    user = db.execute(query).fetchone()\n    return user\n",
        },
        index_version="v1.0",
    )

    # 3. Construct and Deliver Realistic GitHub Webhook Payload
    raw_payload_dict = {
        "action": "opened",
        "repository": {
            "id": repo_id,
            "full_name": repo_id,
        },
        "pull_request": {
            "number": pr_num,
            "title": "Add authentication lookup function",
            "body": "Implements authenticate_user with user query lookup.",
            "base": {"sha": base_sha},
            "head": {"sha": head_sha},
            "changed_files": 1,
        },
    }
    raw_body = json.dumps(raw_payload_dict).encode("utf-8")
    valid_hmac = "sha256=" + hmac.new(webhook_secret, raw_body, hashlib.sha256).hexdigest()
    delivery_id = "delivery-e2e-real-pr-001"

    headers = {
        "X-GitHub-Delivery": delivery_id,
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": valid_hmac,
    }

    # 4. Stage 1: Webhook Intake (HMAC verified before parsing/scheduling, AC-01, AC-02)
    intake_result = intake.accept(
        headers,
        raw_body,
        policy_version="1.0",
        prompt_version="1.0",
        retrieval_index_version="1.0",
        model_configuration={"provider": "configured", "model": "gpt-4o"},
        changed_files=["src/auth.py"],
    )
    assert intake_result.status == "accepted"
    assert intake_result.delivery_id == delivery_id
    assert intake_result.snapshot is not None
    assert intake_result.snapshot.head_sha == head_sha
    assert intake_result.snapshot.pull_request_number == pr_num

    # 5. Stage 2: Durable Asynchronous Queue (FR-03, FR-04)
    enqueued_job = queue.enqueue(intake_result.snapshot, delivery_id=delivery_id)
    assert enqueued_job.job_id == f"job-{delivery_id}"
    assert enqueued_job.state == JobState.QUEUED

    # 6. Stage 3: Retrieval (Hybrid Lexical + Semantic Search)
    citations = retriever.retrieve(
        repository_id=repo_id,
        revision=head_sha,
        query="authenticate_user SQL query",
        max_k=3,
        max_tokens=500,
    )
    assert len(citations) > 0
    assert "src/auth.py" in citations[0].file_path

    # 7. Stage 4: Orchestrator & Specialist Execution with LangGraph (FR-05, FR-06)
    evidence_ref = f"diff:src/auth.py#L11-L12@{head_sha}"

    def mock_security_specialist(spec_input: SpecialistInput) -> SpecialistOutput:
        finding = CandidateFinding(
            finding_id="finding-sec-sql-injection",
            correlation_id=spec_input.correlation_id,
            specialist_type=SpecialistType.SECURITY,
            category="security",
            severity="critical",
            confidence=0.92,
            summary="SQL injection risk from direct string interpolation in SQL query",
            rationale="User input username is interpolated directly into query string without parameterization.",
            file_path="src/auth.py",
            line_range=(11, 12),
            evidence_refs=(evidence_ref,),
            remediation="Use parameterized queries instead of string formatting.",
        )
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=spec_input.correlation_id,
            status="completed",
            findings=(finding,),
            execution_duration=0.08,
        )

    def mock_quality_specialist(spec_input: SpecialistInput) -> SpecialistOutput:
        finding = CandidateFinding(
            finding_id="finding-qual-sql-injection",
            correlation_id=spec_input.correlation_id,
            specialist_type=SpecialistType.QUALITY,
            category="security",
            severity="critical",
            confidence=0.88,
            summary="Potential SQL query injection in user lookup",
            rationale="Unescaped username parameter used in raw SQL statement.",
            file_path="src/auth.py",
            line_range=(11, 12),
            evidence_refs=(evidence_ref,),
            remediation="Use parameter binding.",
        )
        return SpecialistOutput(
            specialist_type=SpecialistType.QUALITY,
            correlation_id=spec_input.correlation_id,
            status="completed",
            findings=(finding,),
            execution_duration=0.06,
        )

    specialist_handlers = {
        SpecialistType.SECURITY: mock_security_specialist,
        SpecialistType.QUALITY: mock_quality_specialist,
    }

    orchestrator = ReviewOrchestrator(specialist_handlers=specialist_handlers)
    worker = ReviewWorker(queue, orchestrator)

    leased_job, lifecycle_state = asyncio.run(
        worker.run_next_job(
            intake_result.snapshot,
            diff_content=SAMPLE_UNIFIED_DIFF,
            retrieved_evidence=tuple(c.excerpt for c in citations),
        )
    )

    assert leased_job is not None
    assert lifecycle_state is not None
    assert lifecycle_state.terminal_status == "completed"
    assert lifecycle_state.aggregation_invoked is True
    assert len(lifecycle_state.specialist_outputs) == 4

    # Confirm job marked completed in durable queue
    updated_job = queue.get_job(leased_job.job_id)
    assert updated_job is not None
    assert updated_job.state == JobState.COMPLETED

    # 8. Stage 5: Finding Verification & Deduplication (FR-08, FR-09, AC-05, AC-06)
    all_candidates: list[CandidateFinding] = []
    for spec_out in lifecycle_state.specialist_outputs.values():
        all_candidates.extend(spec_out.findings)

    verified_candidates = []
    for cand in all_candidates:
        val_res = validator.validate_finding(
            cand,
            repository_id=repo_id,
            head_sha=head_sha,
            diff_content=SAMPLE_UNIFIED_DIFF,
        )
        assert not val_res.is_suppressed
        verified_candidates.append(cand)

    canonical_findings = aggregator.aggregate(
        verified_candidates,
        repository_id=repo_id,
        head_sha=head_sha,
        run_id=lifecycle_state.run_id,
    )
    assert len(canonical_findings) == 1
    canonical = canonical_findings[0]
    assert canonical.category == "security"
    assert canonical.severity == "critical"
    assert len(canonical.contributing_candidate_ids) == 2

    # 9. Stage 6: Policy Evaluation & HITL Workflow (FR-10, FR-11, AC-07)
    evaluated = policy_engine.evaluate(canonical, is_fresh=True)
    # Critical security finding MUST be held for maintainer approval (AC-07)
    assert evaluated.disposition == FindingDisposition.HELD
    assert "High-impact finding" in evaluated.disposition_reason

    truth_store.record_initial(evaluated, initial_state=TruthState.HELD)
    assert truth_store.get_latest_state(canonical.canonical_id).state == TruthState.HELD

    # Human maintainer reviews evidence and approves finding
    maintainer_workflow.approve(
        canonical.canonical_id,
        actor="Alice Lead Reviewer",
        actor_role="maintainer",
        rationale="SQL injection verified on line 11-12. Approved for inline review publication.",
    )
    assert truth_store.get_latest_state(canonical.canonical_id).state == TruthState.APPROVED

    # 10. Stage 7: GitHub Review Publication via FakeGitHubClient (FR-12, AC-08)
    pub_result = publisher.publish_finding(
        evaluated,
        pull_number=pr_num,
        diff_content=SAMPLE_UNIFIED_DIFF,
    )
    assert pub_result.status == PublicationStatus.PUBLISHED
    assert pub_result.published_inline is True
    assert pub_result.review_id is not None
    assert truth_store.get_latest_state(canonical.canonical_id).state == TruthState.PUBLISHED

    # Verify FakeGitHubClient received the review
    assert len(github_client.reviews) == 1
    review = github_client.reviews[0]
    assert review["repository_id"] == repo_id
    assert review["pull_number"] == pr_num
    assert review["commit_sha"] == head_sha
    assert len(review["comments"]) == 1
    comment = review["comments"][0]
    assert comment["path"] == "src/auth.py"
    assert comment["line"] in (11, 12)
    assert "SQL" in comment["body"] and "injection" in comment["body"]

    # INVARIANT CHECKS: Zero merges, zero code modifications
    assert github_client.attempted_merges == 0
    assert github_client.attempted_code_modifications == 0

    # 11. Stage 8: Audit Spine & Redaction Verification (FR-15, AC-11)
    audit_events = list(lifecycle_state.audit_trail)
    for evt in audit_events:
        audit_spine.record_event(
            evt,
            repository_id=repo_id,
            pull_number=pr_num,
            head_sha=head_sha,
            run_id=lifecycle_state.run_id,
        )

    # Trace provenance from PR to delivery
    trace = audit_spine.reconstruct_run(f"{delivery_id}:{lifecycle_state.run_id}")
    assert trace.contains_secrets is False
    assert len(trace.timeline) > 0

    # 12. Stage 9: Dashboard Visibility & HTML Rendering (FR-17, AC-15)
    dash_data = dashboard.get_dashboard_data(repo_id)
    assert dash_data["telemetry"]["total_reviews_completed"] >= 1
    assert dash_data["telemetry"]["queue_depth"] == 0
    assert len(dash_data["recent_reviews"]) >= 1
    assert len(dash_data["recent_findings"]) >= 1

    html_output = dashboard.render_html(repo_id)
    assert "<!DOCTYPE html>" in html_output
    assert repo_id in html_output
    assert "src/auth.py" in html_output

    # 13. Replay Idempotency Verification (FR-02)
    # Re-send the exact same webhook delivery
    duplicate_intake = intake.accept(
        headers,
        raw_body,
        policy_version="1.0",
        prompt_version="1.0",
        retrieval_index_version="1.0",
        model_configuration={"provider": "configured"},
    )
    assert duplicate_intake.status == "duplicate"
    assert duplicate_intake.delivery_id == delivery_id

    # Re-attempt publishing the same finding
    second_pub = publisher.publish_finding(
        canonical,
        pull_number=pr_num,
        diff_content=SAMPLE_UNIFIED_DIFF,
    )
    assert second_pub.status == PublicationStatus.ALREADY_PUBLISHED
    # Proves no duplicate review created on GitHub
    assert len(github_client.reviews) == 1

    # 14. Safety Check: Stale Head SHA Detection & Suppression (AC-09)
    # Simulate a new commit pushed to the PR before review publication
    new_head_sha = "sha-new-head-99999999"
    github_client.set_head_sha(repo_id, pr_num, new_head_sha)

    stale_candidate = CandidateFinding(
        finding_id="finding-stale-sha",
        correlation_id="corr-stale",
        specialist_type=SpecialistType.TESTS,
        category="tests",
        severity="low",
        confidence=0.85,
        summary="Missing test coverage for edge case",
        rationale="No tests added.",
        file_path="src/auth.py",
        line_range=(10, 15),
    )
    stale_canonical = CanonicalFinding(
        canonical_id="canon-stale-sha",
        repository_id=repo_id,
        head_sha=head_sha,  # Review run was for the OLD head SHA
        category="tests",
        severity="low",
        confidence=0.85,
        summary="Missing test coverage for edge case",
        rationale="No tests added.",
        file_path="src/auth.py",
        line_range=(10, 15),
        contributing_candidate_ids=("finding-stale-sha",),
        contributing_specialists=("tests",),
        evidence_refs=(f"diff:src/auth.py#L10-L15@{head_sha}",),
        disposition=FindingDisposition.AUTO_APPROVED,
    )
    truth_store.record_initial(stale_canonical, initial_state=TruthState.AUTO_APPROVED)

    stale_pub_result = publisher.publish_finding(
        stale_canonical,
        pull_number=pr_num,
        diff_content=SAMPLE_UNIFIED_DIFF,
    )
    assert stale_pub_result.status == PublicationStatus.SUPERSEDED_SHA_MISMATCH
    assert "mismatch" in stale_pub_result.reason.lower()
    # Review count remains strictly 1; stale finding was blocked from publishing (AC-09)
    assert len(github_client.reviews) == 1


def test_safety_check_invalid_hmac_rejected_before_processing() -> None:
    """AC-01/AC-02: Invalid HMAC signature rejected before payload parsing or job scheduling."""
    conn = sqlite3.connect(":memory:")
    intake = WebhookIntake(conn, webhook_secret=b"correct-secret")
    headers = {
        "X-GitHub-Delivery": "del-invalid-sig-999",
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": "sha256=0000000000000000000000000000000000000000000000000000000000000000",
    }
    payload = json.dumps({"action": "opened", "repository": {"id": "repo", "full_name": "repo"}}).encode("utf-8")
    result = intake.accept(
        headers,
        payload,
        policy_version="1.0",
        prompt_version="1.0",
        retrieval_index_version="1.0",
        model_configuration={"provider": "configured"},
    )
    assert result.status == "rejected"
    assert result.snapshot is None


def test_safety_check_ungrounded_finding_suppressed() -> None:
    """AC-05/AC-06: Hallucinated / ungrounded findings are suppressed by evidence validator."""
    conn = sqlite3.connect(":memory:")
    memory_store = CodeMemoryStore(conn)
    validator = FindingEvidenceValidator(memory_store)
    diff = "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1,5 +1,5 @@\n+ pass\n"
    ungrounded_candidate = CandidateFinding(
        finding_id="finding-ungrounded",
        correlation_id="corr-ungrounded",
        specialist_type=SpecialistType.SECURITY,
        category="security",
        severity="critical",
        confidence=0.95,
        summary="Hallucinated buffer overflow",
        rationale="Imaginary C function called.",
        file_path="src/nonexistent.c",
        line_range=(999, 1000),
        evidence_refs=("diff:src/nonexistent.c#L999-L1000@sha-head-abc",),
    )
    val_res = validator.validate_finding(
        ungrounded_candidate,
        repository_id="owner/repo",
        head_sha="sha-head-abc",
        diff_content=diff,
    )
    assert val_res.is_suppressed is True
    assert val_res.status == "suppressed"

