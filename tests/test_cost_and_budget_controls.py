"""Tests for cost and budget controls, provider pricing, concurrency management, and context bounding.

Covers:
- FR-19: Token/model/provider usage and cost accounting at run and component granularity.
- AC-13: Documented degraded, held, or skipped outcomes on budget exhaustion or context/concurrency caps.
- NFR-04: Configurable provisional targets without invented numeric SLOs or fixed defaults.
- NFR-05: Per-repository budgets and concurrency controls preventing cross-repository starvation.
- NFR-12: Auditable explanation of skipped, degraded, or escalated reviews caused by budget policy.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import pytest

from pr_review_agent.cost_controls import (
    AdmissionDecision,
    BudgetEnforcer,
    ComponentUsage,
    CostAndBudgetConfig,
    CostLedger,
    ModelPricingRate,
    ProviderPricingRegistry,
    RepositorySpendSummary,
    RunCostSummary,
    UsageSource,
)
from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.observability import OperationalTelemetry
from pr_review_agent.orchestration import (
    CandidateFinding,
    DurableJobQueue,
    JobState,
    ReviewJob,
    ReviewOrchestrator,
    ReviewWorker,
    SpecialistInput,
    SpecialistOutput,
    SpecialistType,
)
from pr_review_agent.security import SecretLeakageScanner


def _sample_snapshot(
    repo: str = "owner/repo",
    pull_request_number: int = 1,
    head_sha: str = "sha-cost-100",
) -> ReviewSnapshot:
    return ReviewSnapshot(
        repository_id=repo,
        repository_full_name=repo,
        pull_request_number=pull_request_number,
        base_sha="sha-base-000",
        head_sha=head_sha,
        changed_files=("src/module.py",),
        policy_version="1.0",
        prompt_version="1.0",
        retrieval_index_version="1.0",
        model_configuration={"provider": "openai", "model": "gpt-4o"},
    )


# --- 1. No Invented Numeric Defaults (NFR-04, SPEC Decision) ---


def test_config_has_no_invented_numeric_defaults() -> None:
    """SPEC-1 must not invent numeric SLOs, budgets, retention limits, or max PR-size limits."""
    cfg = CostAndBudgetConfig()
    assert cfg.max_tokens_per_run is None
    assert cfg.max_cost_usd_per_run is None
    assert cfg.monthly_budget_usd is None
    assert cfg.max_diff_bytes is None
    assert cfg.max_concurrent_reviews_per_repo is None
    assert cfg.soft_budget_ratio is None
    assert cfg.fail_closed_on_context_cap is False


# --- 2. Provider Pricing Registry (Configurable & Provider-Neutral) ---


def test_pricing_registry_is_configurable_and_reports_none_for_unconfigured_rates() -> None:
    registry = ProviderPricingRegistry()

    # Unconfigured model returns None rather than guessing or fabricating a price
    assert registry.calculate_cost("openai", "gpt-4o", 1000, 500) is None
    assert registry.calculate_cost("groq", "llama-3.3-70b-versatile", 1000, 500) is None

    # Register explicit pricing rate
    registry.register_rate(
        ModelPricingRate(
            provider="openai",
            model="gpt-4o",
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.015,
        )
    )

    # Deterministic calculation: (1000/1000 * 0.005) + (500/1000 * 0.015) = 0.005 + 0.0075 = 0.0125
    cost = registry.calculate_cost("openai", "gpt-4o", 1000, 500)
    assert cost == 0.0125

    # Case-insensitive provider and model matching
    cost_ci = registry.calculate_cost("OpenAI", "GPT-4O", 1000, 500)
    assert cost_ci == 0.0125

    # Missing token count returns None
    assert registry.calculate_cost("openai", "gpt-4o", None, 500) is None
    assert registry.calculate_cost("openai", "gpt-4o", 1000, None) is None


# --- 3. Actual Usage vs Estimation (Component Granularity) ---


def test_component_usage_distinguishes_actual_vs_unavailable() -> None:
    # Actual provider-reported usage
    actual = ComponentUsage(
        component="specialist_security",
        provider="openai",
        model="gpt-4o",
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        cost_usd=0.0105,
        usage_source=UsageSource.PROVIDER_REPORTED,
        pricing_configured=True,
    )
    assert actual.usage_source == UsageSource.PROVIDER_REPORTED
    assert actual.total_tokens == 1500
    assert actual.cost_usd == 0.0105

    # Unavailable usage does not fabricate token counts or cost
    unavail = ComponentUsage(
        component="specialist_quality",
        usage_source=UsageSource.UNAVAILABLE,
    )
    assert unavail.usage_source == UsageSource.UNAVAILABLE
    assert unavail.prompt_tokens is None
    assert unavail.completion_tokens is None
    assert unavail.total_tokens is None
    assert unavail.cost_usd is None


# --- 4. Durable Cost Ledger Persistence and Queries (FR-19, NFR-06) ---


def test_durable_cost_ledger_records_and_queries_component_usage() -> None:
    conn = sqlite3.connect(":memory:")
    ledger = CostLedger(conn)

    t0 = 1000.0
    u_sec = ComponentUsage(
        component="specialist_security",
        provider="openai",
        model="gpt-4o",
        prompt_tokens=800,
        completion_tokens=200,
        total_tokens=1000,
        cost_usd=0.007,
        usage_source=UsageSource.PROVIDER_REPORTED,
        pricing_configured=True,
    )
    u_qual = ComponentUsage(
        component="specialist_quality",
        provider="openai",
        model="gpt-4o",
        prompt_tokens=1200,
        completion_tokens=400,
        total_tokens=1600,
        cost_usd=0.012,
        usage_source=UsageSource.PROVIDER_REPORTED,
        pricing_configured=True,
    )

    ledger.record_usage("run-1", "del-1:run-1", "owner/repo", u_sec, timestamp=t0)
    ledger.record_usage("run-1", "del-1:run-1", "owner/repo", u_qual, timestamp=t0 + 1.0)

    summary = ledger.get_run_cost_summary("run-1")
    assert summary.run_id == "run-1"
    assert summary.total_prompt_tokens == 2000
    assert summary.total_completion_tokens == 600
    assert summary.total_tokens == 2600
    assert summary.total_cost_usd == 0.019
    assert summary.is_cost_complete is True
    assert len(summary.components) == 2

    # Query repository spend
    spend = ledger.get_repository_spend("owner/repo")
    assert spend == 0.019

    # Component breakdown map
    breakdown = ledger.get_component_breakdown("run-1")
    assert "specialist_security" in breakdown
    assert breakdown["specialist_security"].total_tokens == 1000
    assert "specialist_quality" in breakdown
    assert breakdown["specialist_quality"].total_tokens == 1600


# --- 5. Hard Budget Exhaustion Halts Run Without Findings or GitHub (AC-13, NFR-12) ---


def test_hard_budget_exhaustion_halts_execution_without_findings_or_github() -> None:
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)
    ledger = CostLedger(conn)

    # Pre-record spend exceeding monthly budget of $1.00
    prior_usage = ComponentUsage(
        component="prior_run",
        provider="openai",
        model="gpt-4o",
        prompt_tokens=10000,
        completion_tokens=5000,
        total_tokens=15000,
        cost_usd=1.25,
        usage_source=UsageSource.PROVIDER_REPORTED,
        pricing_configured=True,
    )
    ledger.record_usage("prior-run", "corr-0", "owner/repo", prior_usage)

    cfg = CostAndBudgetConfig(monthly_budget_usd=1.00)
    enforcer = BudgetEnforcer(cfg, ledger)

    specialist_called = False

    def mock_security(input_data: SpecialistInput) -> SpecialistOutput:
        nonlocal specialist_called
        specialist_called = True
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=input_data.correlation_id,
            status="completed",
            findings=(
                CandidateFinding(
                    finding_id="f-1",
                    correlation_id=input_data.correlation_id,
                    specialist_type=SpecialistType.SECURITY,
                    category="security",
                    severity="high",
                    confidence=0.9,
                    summary="Test finding",
                    rationale="Test rationale",
                ),
            ),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={SpecialistType.SECURITY: mock_security},
        budget_enforcer=enforcer,
        cost_ledger=ledger,
    )

    snapshot = _sample_snapshot("owner/repo", 1)
    job = queue.enqueue(snapshot, "del-budget-1")

    res = asyncio.run(orchestrator.execute_run(job, snapshot, diff_content="def test(): pass"))

    # Run halted at initialize with failed status
    assert res.terminal_status == JobState.FAILED.value
    assert res.step_states.get("initialize") == "budget_exhausted"
    # Specialist was never called
    assert specialist_called is False
    # No specialist outputs or findings generated
    assert len(res.specialist_outputs) == 0
    # Auditable explanation in event spine
    event_names = [e.event_name for e in res.audit_trail]
    assert "review_run_admission_rejected" in event_names
    rej_event = next(e for e in res.audit_trail if e.event_name == "review_run_admission_rejected")
    assert "meets or exceeds monthly budget cap" in str(rej_event.details.get("reason"))


# --- 6. Soft Budget Threshold Emits Warning (NFR-08, NFR-12) ---


def test_soft_budget_warning_in_operational_telemetry() -> None:
    conn = sqlite3.connect(":memory:")
    ledger = CostLedger(conn)

    # Spend $8.50 out of $10.00 budget (85% > 80% soft ratio)
    usage = ComponentUsage(
        component="test",
        cost_usd=8.50,
        usage_source=UsageSource.PROVIDER_REPORTED,
    )
    ledger.record_usage("run-soft", "corr-soft", "owner/repo", usage)

    cfg = CostAndBudgetConfig(monthly_budget_usd=10.00, soft_budget_ratio=0.80)
    telemetry = OperationalTelemetry(conn, cost_ledger=ledger, budget_config=cfg)

    snap = telemetry.get_snapshot(repository_id="owner/repo")
    alert_ids = [a.alert_id for a in snap.active_alerts]
    assert "ALERT-BUDGET-WARNING" in alert_ids
    assert "ALERT-BUDGET-EXHAUSTION" not in alert_ids
    assert snap.total_cost_usd == 8.50


def test_hard_budget_exhaustion_alert_in_operational_telemetry() -> None:
    conn = sqlite3.connect(":memory:")
    ledger = CostLedger(conn)

    # Spend $10.50 out of $10.00 budget
    usage = ComponentUsage(
        component="test",
        cost_usd=10.50,
        usage_source=UsageSource.PROVIDER_REPORTED,
    )
    ledger.record_usage("run-hard", "corr-hard", "owner/repo", usage)

    cfg = CostAndBudgetConfig(monthly_budget_usd=10.00)
    telemetry = OperationalTelemetry(conn, cost_ledger=ledger, budget_config=cfg)

    snap = telemetry.get_snapshot(repository_id="owner/repo")
    alert_ids = [a.alert_id for a in snap.active_alerts]
    assert "ALERT-BUDGET-EXHAUSTION" in alert_ids
    assert snap.total_cost_usd == 10.50


# --- 7. Concurrency Cap Prevents Repository Starvation (NFR-05) ---


def test_concurrency_cap_prevents_repository_starvation() -> None:
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)

    t0 = 1000.0
    snap_a1 = _sample_snapshot(repo="owner/repo-A", pull_request_number=1)
    snap_a2 = _sample_snapshot(repo="owner/repo-A", pull_request_number=2)
    snap_b1 = _sample_snapshot(repo="owner/repo-B", pull_request_number=1)

    # Enqueue in order: repo-A job 1, repo-A job 2, repo-B job 1
    j_a1 = queue.enqueue(snap_a1, "del-a1", now=t0)
    j_a2 = queue.enqueue(snap_a2, "del-a2", now=t0 + 1.0)
    j_b1 = queue.enqueue(snap_b1, "del-b1", now=t0 + 2.0)

    # Max 1 concurrent review per repository
    # First lease: leases repo-A job 1
    leased_1 = queue.lease_next_job(now=t0 + 3.0, max_concurrency_per_repo=1)
    assert leased_1 is not None
    assert leased_1.job_id == j_a1.job_id
    assert leased_1.repository_id == "owner/repo-A"

    # Second lease: repo-A already has 1 running job. Candidate repo-A job 2 is skipped!
    # Instead, repo-B job 1 is leased, preventing repo-B from starving!
    leased_2 = queue.lease_next_job(now=t0 + 4.0, max_concurrency_per_repo=1)
    assert leased_2 is not None
    assert leased_2.job_id == j_b1.job_id
    assert leased_2.repository_id == "owner/repo-B"

    # Third lease: both repo-A and repo-B have 1 running job. Neither can be leased.
    leased_3 = queue.lease_next_job(now=t0 + 5.0, max_concurrency_per_repo=1)
    assert leased_3 is None

    # When repo-A job 1 completes, repo-A job 2 becomes eligible
    queue.mark_completed(j_a1.job_id, now=t0 + 6.0)
    leased_4 = queue.lease_next_job(now=t0 + 7.0, max_concurrency_per_repo=1)
    assert leased_4 is not None
    assert leased_4.job_id == j_a2.job_id


# --- 8. Concurrency Preserves Retries, Backoff, and Cancellation ---


def test_concurrency_preserves_retries_and_cancellation() -> None:
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)

    t0 = 1000.0
    snap = _sample_snapshot("owner/repo", 1)
    job = queue.enqueue(snap, "del-1", now=t0)

    # Lease with concurrency cap
    leased = queue.lease_next_job(now=t0 + 1.0, max_concurrency_per_repo=2)
    assert leased is not None
    assert leased.state == JobState.RUNNING

    # Fail job -> backoff retry
    failed_job = queue.mark_failed(leased.job_id, error="Transient model failure", now=t0 + 2.0)
    assert failed_job.state == JobState.QUEUED
    assert failed_job.attempt_count == 1
    assert failed_job.next_run_at > t0 + 2.0

    # Cancel job
    cancelled_job = queue.cancel_job(leased.job_id, reason="PR closed by user", now=t0 + 5.0)
    assert cancelled_job.state == JobState.CANCELLED
    assert cancelled_job.last_error == "PR closed by user"


# --- 9. Context Cap Truncation and Auditable Degradation (FR-19, AC-13) ---


def test_context_cap_truncation_records_auditable_degradation() -> None:
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)

    # Configure max_diff_bytes cap of 50 bytes (without inventing a default)
    cfg = CostAndBudgetConfig(max_diff_bytes=50, fail_closed_on_context_cap=False)
    enforcer = BudgetEnforcer(cfg)

    received_diff = ""

    def mock_security(input_data: SpecialistInput) -> SpecialistOutput:
        nonlocal received_diff
        received_diff = input_data.diff_content
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=input_data.correlation_id,
            status="completed",
            findings=(),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={SpecialistType.SECURITY: mock_security},
        budget_enforcer=enforcer,
    )

    full_diff = "diff --git a/test.py b/test.py\n+print('this is an oversized diff that exceeds fifty bytes')"
    snapshot = _sample_snapshot("owner/repo", 1)
    job = queue.enqueue(snapshot, "del-diff-1")

    res = asyncio.run(orchestrator.execute_run(job, snapshot, diff_content=full_diff))

    assert res.terminal_status == "completed"
    assert res.is_degraded is True
    assert res.degradation_details["original_diff_bytes"] == len(full_diff.encode("utf-8"))
    assert res.degradation_details["retained_diff_bytes"] == 50
    assert res.degradation_details["truncated_diff_bytes"] > 0
    # Specialist received bounded prefix diff
    assert len(received_diff.encode("utf-8")) == 50

    # Audit trail contains context_cap_degraded event
    event_names = [e.event_name for e in res.audit_trail]
    assert "context_cap_degraded" in event_names
    deg_event = next(e for e in res.audit_trail if e.event_name == "context_cap_degraded")
    assert deg_event.details["original_diff_bytes"] == len(full_diff.encode("utf-8"))
    assert deg_event.details["retained_diff_bytes"] == 50


def test_context_cap_fail_closed_mode() -> None:
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)

    # fail_closed_on_context_cap enabled
    cfg = CostAndBudgetConfig(max_diff_bytes=50, fail_closed_on_context_cap=True)
    enforcer = BudgetEnforcer(cfg)

    specialist_called = False

    def mock_security(input_data: SpecialistInput) -> SpecialistOutput:
        nonlocal specialist_called
        specialist_called = True
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=input_data.correlation_id,
            status="completed",
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={SpecialistType.SECURITY: mock_security},
        budget_enforcer=enforcer,
    )

    full_diff = "diff --git a/test.py b/test.py\n+print('this is an oversized diff that exceeds fifty bytes')"
    snapshot = _sample_snapshot("owner/repo", 1)
    job = queue.enqueue(snapshot, "del-diff-fc")

    res = asyncio.run(orchestrator.execute_run(job, snapshot, diff_content=full_diff))

    assert res.terminal_status == JobState.FAILED.value
    assert res.step_states.get("initialize") == "context_cap_exceeded"
    assert specialist_called is False


# --- 10. Run Limits and Post-Execution Cost Accounting (FR-19) ---


def test_orchestration_records_component_usage_and_evaluates_run_limits() -> None:
    conn = sqlite3.connect(":memory:")
    queue = DurableJobQueue(conn)
    ledger = CostLedger(conn)
    pricing = ProviderPricingRegistry()
    pricing.register_rate(
        ModelPricingRate(
            provider="openai",
            model="gpt-4o",
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.015,
        )
    )

    cfg = CostAndBudgetConfig(max_tokens_per_run=2000, max_cost_usd_per_run=0.05)
    enforcer = BudgetEnforcer(cfg, ledger)

    def mock_security(input_data: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=input_data.correlation_id,
            status="completed",
            findings=(),
            usage=ComponentUsage(
                component="specialist_security",
                provider="openai",
                model="gpt-4o",
                prompt_tokens=800,
                completion_tokens=200,
                total_tokens=1000,
                cost_usd=None,  # Will be calculated by registry
                usage_source=UsageSource.PROVIDER_REPORTED,
            ),
        )

    def mock_quality(input_data: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=SpecialistType.QUALITY,
            correlation_id=input_data.correlation_id,
            status="completed",
            findings=(),
            usage=ComponentUsage(
                component="specialist_quality",
                provider="openai",
                model="gpt-4o",
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                cost_usd=None,
                usage_source=UsageSource.PROVIDER_REPORTED,
            ),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={
            SpecialistType.SECURITY: mock_security,
            SpecialistType.QUALITY: mock_quality,
        },
        budget_enforcer=enforcer,
        cost_ledger=ledger,
        pricing_registry=pricing,
    )

    snapshot = _sample_snapshot("owner/repo", 1)
    job = queue.enqueue(snapshot, "del-cost-1")

    res = asyncio.run(orchestrator.execute_run(job, snapshot))

    assert res.terminal_status == "completed"
    assert res.run_cost_summary is not None
    # 1000 + 1500 = 2500 tokens (exceeds 2000 cap)
    assert res.run_cost_summary.total_tokens == 2500
    # Pricing calculated: security (0.004 + 0.003 = 0.007) + quality (0.005 + 0.0075 = 0.0125) = 0.0195
    assert res.run_cost_summary.total_cost_usd == 0.0195

    # Audit events contain cost_recorded and run_budget_limit_exceeded
    event_names = [e.event_name for e in res.audit_trail]
    assert "cost_recorded" in event_names
    assert "run_budget_limit_exceeded" in event_names


# --- 11. Security Boundaries: No Secrets in Cost Records or Events ---


def test_no_secrets_in_cost_ledger_or_budget_events() -> None:
    conn = sqlite3.connect(":memory:")
    ledger = CostLedger(conn)
    scanner = SecretLeakageScanner()

    # Record usage
    usage = ComponentUsage(
        component="specialist_security",
        provider="openai",
        model="gpt-4o",
        prompt_tokens=500,
        completion_tokens=100,
        total_tokens=600,
        cost_usd=0.004,
        usage_source=UsageSource.PROVIDER_REPORTED,
    )
    ledger.record_usage("run-sec-check", "corr-sec", "owner/repo", usage)

    # Read from DB and scan for secrets
    rows = conn.execute("SELECT * FROM component_usage_records WHERE run_id = 'run-sec-check'").fetchall()
    serialized = str(rows)
    scan_res = scanner.scan_text(serialized, "usage_records")
    assert not scan_res.has_secret
    assert len(scan_res.matches) == 0


# --- 12. Runtime Concurrency Wiring (Worker/Dispatcher Propagation) ---


def test_runtime_worker_and_queue_propagate_concurrency_cap() -> None:
    """Configured concurrency cap must be propagated through normal runtime path to lease_next_job()."""
    conn = sqlite3.connect(":memory:")
    cfg = CostAndBudgetConfig(max_concurrent_reviews_per_repo=1)
    queue = DurableJobQueue(conn, budget_config=cfg)

    def dummy_handler(input_data: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=input_data.specialist_type,
            correlation_id=input_data.correlation_id,
            status="completed",
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={SpecialistType.SECURITY: dummy_handler}
    )
    worker = ReviewWorker(queue, orchestrator, budget_config=cfg)

    # Queue 2 jobs for repo-a, 1 job for repo-b
    snap_a1 = _sample_snapshot("org/repo-a", 1, "sha-a1")
    snap_a2 = _sample_snapshot("org/repo-a", 2, "sha-a2")
    snap_b1 = _sample_snapshot("org/repo-b", 1, "sha-b1")

    t0 = time.time()
    j_a1 = queue.enqueue(snap_a1, "del-a1", now=t0)
    j_a2 = queue.enqueue(snap_a2, "del-a2", now=t0 + 1.0)
    j_b1 = queue.enqueue(snap_b1, "del-b1", now=t0 + 2.0)

    # 1. Normal runtime call via ReviewWorker (no manual max_concurrency_per_repo argument)
    leased_1 = worker.lease_job(now=t0 + 5.0)
    assert leased_1 is not None
    assert leased_1.job_id == j_a1.job_id
    assert leased_1.repository_id == "org/repo-a"

    # 2. Second normal runtime call: repo-a has 1 running job, so cap=1 forces skipping repo-a to lease repo-b
    leased_2 = worker.lease_job(now=t0 + 6.0)
    assert leased_2 is not None
    assert leased_2.job_id == j_b1.job_id
    assert leased_2.repository_id == "org/repo-b"

    # 3. Third normal runtime call: both repos at capacity (1 running each) -> returns None
    leased_3 = worker.lease_job(now=t0 + 7.0)
    assert leased_3 is None

    # 4. DurableJobQueue itself propagates budget_config without manual args
    leased_queue_direct = queue.lease_next_job(now=t0 + 8.0)
    assert leased_queue_direct is None

    # 5. Complete job a1, then worker.run_next_job() executes next job for repo-a
    queue.mark_completed(j_a1.job_id)
    executed_job, lifecycle = asyncio.run(
        worker.run_next_job(snap_a2, diff_content="", now=t0 + 10.0)
    )
    assert executed_job is not None
    assert executed_job.job_id == j_a2.job_id
    assert lifecycle is not None
    assert lifecycle.terminal_status == "completed"

    # Verify job state transitioned to completed in DB
    updated_a2 = queue.get_job(j_a2.job_id)
    assert updated_a2 is not None
    assert updated_a2.state == JobState.COMPLETED


# --- 13. Incomplete/Unknown Cost Handling (Never Treated as Zero Spend) ---


def test_complete_spend_normal_budget_decisions() -> None:
    """When all prior usage records have known cost, budget admission behaves normally."""
    conn = sqlite3.connect(":memory:")
    ledger = CostLedger(conn)
    cfg = CostAndBudgetConfig(monthly_budget_usd=10.0)
    enforcer = BudgetEnforcer(cfg, ledger)

    # Initially 0 spend
    summary_empty = ledger.get_repository_spend_summary("owner/repo")
    assert summary_empty.known_cost_usd == 0.0
    assert not summary_empty.has_unknown_cost
    assert summary_empty.unknown_cost_records_count == 0

    dec_init = enforcer.check_admission("owner/repo", "diff")
    assert dec_init.allowed is True
    assert dec_init.status == "admitted"

    # Record complete, known usage of $6.00
    ledger.record_usage(
        run_id="run-1",
        correlation_id="corr-1",
        repository_id="owner/repo",
        usage=ComponentUsage(
            component="specialist_security",
            cost_usd=6.0,
            usage_source=UsageSource.PROVIDER_REPORTED,
        ),
    )

    # Still under $10 cap
    dec_under = enforcer.check_admission("owner/repo", "diff")
    assert dec_under.allowed is True
    assert dec_under.status == "admitted"

    # Record another $5.00 -> total $11.00 >= $10.00
    ledger.record_usage(
        run_id="run-2",
        correlation_id="corr-2",
        repository_id="owner/repo",
        usage=ComponentUsage(
            component="specialist_quality",
            cost_usd=5.0,
            usage_source=UsageSource.PROVIDER_REPORTED,
        ),
    )

    summary_over = ledger.get_repository_spend_summary("owner/repo")
    assert summary_over.known_cost_usd == 11.0
    assert not summary_over.has_unknown_cost

    dec_over = enforcer.check_admission("owner/repo", "diff")
    assert dec_over.allowed is False
    assert dec_over.status == "budget_exhausted"


def test_unknown_or_unavailable_prior_spend_fails_closed_under_hard_budget() -> None:
    """Prior unavailable/unknown usage must never be treated as zero spend to admit runs under hard budget."""
    conn = sqlite3.connect(":memory:")
    ledger = CostLedger(conn)
    cfg = CostAndBudgetConfig(monthly_budget_usd=50.0)
    enforcer = BudgetEnforcer(cfg, ledger)

    # Record a usage where cost is UNAVAILABLE / None
    ledger.record_usage(
        run_id="run-unavail",
        correlation_id="corr-unavail",
        repository_id="owner/repo",
        usage=ComponentUsage(
            component="specialist_security",
            cost_usd=None,
            usage_source=UsageSource.UNAVAILABLE,
        ),
    )

    summary = ledger.get_repository_spend_summary("owner/repo")
    assert summary.has_unknown_cost is True
    assert summary.unknown_cost_records_count == 1
    # Known cost is 0.0, but because has_unknown_cost is True, it must NOT be admitted
    assert summary.known_cost_usd == 0.0

    # BudgetEnforcer must fail closed with unknown_budget_state
    decision = enforcer.check_admission("owner/repo", "diff content")
    assert decision.allowed is False
    assert decision.status == "unknown_budget_state"
    assert "unknown/unavailable cost" in (decision.reason or "")

    # Execute run via ReviewOrchestrator to verify admission rejection and routing to END
    def dummy_handler(input_data: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=input_data.specialist_type,
            correlation_id=input_data.correlation_id,
            status="completed",
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={SpecialistType.SECURITY: dummy_handler},
        budget_enforcer=enforcer,
        cost_ledger=ledger,
    )
    queue = DurableJobQueue(conn)
    snap = _sample_snapshot("owner/repo", 1)
    job = queue.enqueue(snap, "del-unavail-test")

    res = asyncio.run(orchestrator.execute_run(job, snap, diff_content="diff"))
    assert res.terminal_status == JobState.FAILED.value
    assert res.step_states.get("initialize") == "unknown_budget_state"
    # Specialists were never dispatched
    assert len(res.specialist_outputs) == 0

    event_names = [e.event_name for e in res.audit_trail]
    assert "review_run_admission_rejected" in event_names
    rej_event = next(e for e in res.audit_trail if e.event_name == "review_run_admission_rejected")
    assert rej_event.details.get("status") == "unknown_budget_state"


def test_run_limits_do_not_falsely_satisfy_missing_or_incomplete_usage() -> None:
    """Missing or unavailable token/cost usage must not be treated as zero to satisfy run limits."""
    cfg = CostAndBudgetConfig(max_tokens_per_run=5000, max_cost_usd_per_run=1.00)
    enforcer = BudgetEnforcer(cfg)

    # 1. Component with unavailable tokens
    summary_unavail_tokens = RunCostSummary(
        run_id="run-incomplete-tokens",
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_tokens=0,
        total_cost_usd=0.0,
        is_cost_complete=True,
        has_unavailable_tokens=True,
    )
    ok_tok, reason_tok = enforcer.check_run_limits(summary=summary_unavail_tokens)
    assert ok_tok is False
    assert reason_tok is not None
    assert "Cannot verify max_tokens_per_run" in reason_tok

    # 2. Component with incomplete cost
    summary_incomplete_cost = RunCostSummary(
        run_id="run-incomplete-cost",
        total_prompt_tokens=100,
        total_completion_tokens=50,
        total_tokens=150,
        total_cost_usd=0.0,  # Missing cost must not be treated as $0.00
        is_cost_complete=False,
        has_unavailable_tokens=False,
    )
    ok_cost, reason_cost = enforcer.check_run_limits(summary=summary_incomplete_cost)
    assert ok_cost is False
    assert reason_cost is not None
    assert "Cannot verify max_cost_usd_per_run" in reason_cost

    # 3. Verify in ReviewOrchestrator: specialist reporting UNAVAILABLE usage triggers limit verification event
    conn = sqlite3.connect(":memory:")
    ledger = CostLedger(conn)
    queue = DurableJobQueue(conn)
    orchestrator_enforcer = BudgetEnforcer(cfg, ledger)

    def mock_unavail_specialist(input_data: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=input_data.correlation_id,
            status="completed",
            usage=ComponentUsage(
                component="specialist_security",
                provider=None,
                model=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                cost_usd=None,
                usage_source=UsageSource.UNAVAILABLE,
            ),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={SpecialistType.SECURITY: mock_unavail_specialist},
        budget_enforcer=orchestrator_enforcer,
        cost_ledger=ledger,
    )

    snap = _sample_snapshot("owner/repo", 1)
    job = queue.enqueue(snap, "del-incomplete-run")
    res = asyncio.run(orchestrator.execute_run(job, snap))

    assert res.run_cost_summary is not None
    assert res.run_cost_summary.has_unavailable_tokens is True
    assert res.run_cost_summary.is_cost_complete is False

    event_names = [e.event_name for e in res.audit_trail]
    assert "run_budget_limit_exceeded" in event_names
    exceeded_event = next(e for e in res.audit_trail if e.event_name == "run_budget_limit_exceeded")
    assert "Cannot verify" in str(exceeded_event.details.get("reason"))

