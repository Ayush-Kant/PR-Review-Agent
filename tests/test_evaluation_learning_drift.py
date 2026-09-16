"""Tests for golden-PR evaluation, regression promotion gates, feedback capture, drift detection, and rollback controls.

Covers:
- FR-16: Versioned golden pull-request dataset, development and holdout evaluation, regression promotion gates.
- FR-20: Feedback capture linked to findings and policy versions, material drift detection, explicit promotion, rollback controls.
- AC-12: Prevention of production promotion when holdout regression gate fails, identifying version and failed metric.
- NFR-04: Configurable provisional targets without invented numeric SLOs or fixed defaults.
- NFR-07: Strict repository tenancy isolation for feedback, golden cases, and drift evaluations.
- NFR-09: Versioned policy transitions and golden-set regression gates before promotion.
- FR-18: Secret leakage scanning over golden datasets, feedback, and drift reports.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import time
import pytest

from pr_review_agent.evaluation import (
    DatasetSplit,
    DriftConfig,
    DriftDetector,
    DriftReport,
    EvaluationCaseResult,
    EvaluationMetrics,
    EvaluationRunner,
    FeedbackDisposition,
    FeedbackLedger,
    FeedbackRecord,
    GoldenFinding,
    GoldenPRCase,
    GoldenPRDataset,
    PolicyPromotionManager,
    PolicyPromotionRecord,
    PromotionGateEvaluator,
    PromotionGateResult,
    RegressionGateConfig,
    SEVERITY_TIERS,
    compute_severity_distance,
    load_golden_dataset,
)
from pr_review_agent.orchestration import CandidateFinding, SpecialistType
from pr_review_agent.security import SecretLeakageScanner


def _make_sample_candidate(
    category: str = "correctness",
    severity: str = "medium",
    file_path: str = "src/main.py",
    line_range: tuple[int, int] = (10, 20),
    confidence: float = 0.85,
    summary: str = "Test candidate",
) -> CandidateFinding:
    return CandidateFinding(
        finding_id=f"cand-{time.time()}",
        correlation_id="corr-test",
        specialist_type=SpecialistType.QUALITY,
        category=category,
        severity=severity,
        confidence=confidence,
        summary=summary,
        rationale="Rationale",
        file_path=file_path,
        line_range=line_range,
    )


# --- 1. No Invented Numeric Defaults (NFR-04, SPEC Decision) ---


def test_configs_have_no_invented_numeric_defaults() -> None:
    """SPEC-1 must not invent numeric thresholds, SLOs, or drift tolerances."""
    reg_cfg = RegressionGateConfig()
    assert reg_cfg.min_precision is None
    assert reg_cfg.min_recall is None
    assert reg_cfg.min_f1 is None
    assert reg_cfg.min_critical_finding_recall is None
    assert reg_cfg.max_regression_cost_usd is None
    assert reg_cfg.max_regression_duration_seconds is None

    drift_cfg = DriftConfig()
    assert drift_cfg.max_rejection_rate_drift is None
    assert drift_cfg.max_dispute_rate is None
    assert drift_cfg.high_confidence_threshold is None
    assert drift_cfg.max_calibration_gap is None
    assert drift_cfg.max_model_behavior_drift is None
    assert drift_cfg.max_cost_drift_ratio is None
    assert drift_cfg.max_staleness_ratio is None


# --- 2. Golden Dataset Structure & Split Filtering (FR-16) ---


def test_golden_dataset_structure_and_split_filtering() -> None:
    f1 = GoldenFinding("f1", "security", "critical", "src/auth.py", (10, 20), is_required=True)
    f2 = GoldenFinding("f2", "correctness", "medium", "src/util.py", (50, 60), is_required=False)

    case_dev = GoldenPRCase(
        case_id="case-dev-1",
        repository_id="owner/repo",
        title="Dev PR",
        expected_findings=(f1,),
        split=DatasetSplit.DEVELOPMENT,
    )
    case_holdout = GoldenPRCase(
        case_id="case-holdout-1",
        repository_id="owner/repo",
        title="Holdout PR",
        expected_findings=(f2,),
        split=DatasetSplit.HOLDOUT,
    )

    dataset = GoldenPRDataset(
        dataset_id="golden-v1",
        version="1.0",
        cases=(case_dev, case_holdout),
    )

    dev_cases = dataset.get_cases(split=DatasetSplit.DEVELOPMENT)
    assert len(dev_cases) == 1
    assert dev_cases[0].case_id == "case-dev-1"

    holdout_cases = dataset.get_cases(split=DatasetSplit.HOLDOUT)
    assert len(holdout_cases) == 1
    assert holdout_cases[0].case_id == "case-holdout-1"

    all_cases = dataset.get_cases()
    assert len(all_cases) == 2


# --- 3. Deterministic 1:1 Finding Matching & Line Tolerance (FR-16) ---


def test_deterministic_finding_matching_with_tolerances() -> None:
    runner = EvaluationRunner()

    expected = GoldenFinding(
        finding_id="exp-1",
        category="security",
        severity="critical",
        file_path="src/login.py",
        line_range=(100, 110),
        line_tolerance=5,
    )
    case = GoldenPRCase(
        case_id="case-1",
        repository_id="owner/repo",
        expected_findings=(expected,),
        split=DatasetSplit.DEVELOPMENT,
    )

    # 1. Matching candidate within line tolerance (104, 114 is within tolerance bounds of 100-5 to 110+5)
    cand_match = _make_sample_candidate(
        category="security",
        severity="critical",
        file_path="src/login.py",
        line_range=(104, 114),
    )
    res_match = runner.evaluate_case(case, [cand_match])
    assert res_match.true_positives == 1
    assert res_match.false_positives == 0
    assert res_match.false_negatives == 0
    assert res_match.critical_detected == 1

    # 2. Line outside tolerance (120, 130 does not overlap 95-115)
    cand_out = _make_sample_candidate(
        category="security",
        severity="critical",
        file_path="src/login.py",
        line_range=(120, 130),
    )
    res_out = runner.evaluate_case(case, [cand_out])
    assert res_out.true_positives == 0
    assert res_out.false_positives == 1
    assert res_out.false_negatives == 1
    assert res_out.critical_detected == 0

    # 3. Category mismatch
    cand_wrong_cat = _make_sample_candidate(
        category="documentation",
        severity="critical",
        file_path="src/login.py",
        line_range=(100, 110),
    )
    res_wrong_cat = runner.evaluate_case(case, [cand_wrong_cat])
    assert res_wrong_cat.true_positives == 0
    assert res_wrong_cat.false_negatives == 1

    # 4. Severity mismatch
    cand_wrong_sev = _make_sample_candidate(
        category="security",
        severity="low",
        file_path="src/login.py",
        line_range=(100, 110),
    )
    res_wrong_sev = runner.evaluate_case(case, [cand_wrong_sev])
    assert res_wrong_sev.true_positives == 0

    # 5. Duplicate candidate cannot claim the same expected finding twice (1:1 matching)
    cand_dup = _make_sample_candidate(
        category="security",
        severity="critical",
        file_path="src/login.py",
        line_range=(100, 110),
    )
    res_dup = runner.evaluate_case(case, [cand_match, cand_dup])
    assert res_dup.true_positives == 1
    assert res_dup.false_positives == 1
    assert res_dup.false_negatives == 0


# --- 4. Precision, Recall, F1, and Critical Finding Metrics (FR-16) ---


def test_evaluation_metrics_precision_recall_f1_critical() -> None:
    runner = EvaluationRunner()

    f_req1 = GoldenFinding("f1", "security", "critical", "a.py", (1, 5), is_required=True)
    f_req2 = GoldenFinding("f2", "correctness", "medium", "b.py", (1, 5), is_required=True)
    f_opt = GoldenFinding("f3", "documentation", "low", "c.py", (1, 5), is_required=False)

    case = GoldenPRCase(
        case_id="case-metrics",
        repository_id="owner/repo",
        expected_findings=(f_req1, f_req2, f_opt),
        split=DatasetSplit.DEVELOPMENT,
    )
    dataset = GoldenPRDataset("ds", "1.0", (case,))

    # Candidate detects f_req1 (TP), misses f_req2 (FN), and generates an hallucinated finding (FP)
    c1 = _make_sample_candidate(category="security", severity="critical", file_path="a.py", line_range=(1, 5))
    c2 = _make_sample_candidate(category="security", severity="low", file_path="extra.py", line_range=(1, 5))

    metrics = runner.evaluate_dataset(
        dataset,
        {"case-metrics": [c1, c2]},
        split=DatasetSplit.DEVELOPMENT,
        case_durations={"case-metrics": 2.5},
        case_costs={"case-metrics": 0.015},
    )

    # 1 TP (c1), 1 FP (c2), 1 FN (f_req2 is required and missed). Note: f_opt is not required, so omission != FN.
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5  # 1 / (1 + 1)
    assert metrics.recall == 0.5     # 1 / (1 + 1 required)
    assert metrics.f1_score == 0.5   # 2 * 0.5 * 0.5 / 1.0 = 0.5
    assert metrics.critical_findings_expected == 1
    assert metrics.critical_findings_detected == 1
    assert metrics.critical_finding_recall == 1.0
    assert metrics.mean_duration_seconds == 2.5
    assert metrics.total_cost_usd == 0.015
    assert metrics.is_cost_complete is True


# --- 5. Holdout Regression Gate & Identification of Failed Metric (AC-12, NFR-09) ---


def test_holdout_regression_gate_passes_when_all_thresholds_met() -> None:
    evaluator = PromotionGateEvaluator()
    config = RegressionGateConfig(
        min_precision=0.80,
        min_recall=0.75,
        min_f1=0.75,
        min_critical_finding_recall=1.0,
        max_regression_cost_usd=0.50,
        max_regression_duration_seconds=5.0,
    )

    metrics_pass = EvaluationMetrics(
        split=DatasetSplit.HOLDOUT,
        total_cases=10,
        expected_required_count=20,
        candidate_findings_count=22,
        true_positives=18,
        false_positives=4,
        false_negatives=2,
        precision=0.8182,
        recall=0.90,
        f1_score=0.8571,
        critical_findings_expected=5,
        critical_findings_detected=5,
        critical_finding_recall=1.0,
        mean_duration_seconds=3.2,
        total_cost_usd=0.25,
        is_cost_complete=True,
    )

    result = evaluator.evaluate_gate("policy-v2.0", metrics_pass, config)
    assert result.passed is True
    assert len(result.failed_gates) == 0
    assert result.candidate_version == "policy-v2.0"
    assert result.dataset_split == DatasetSplit.HOLDOUT


def test_holdout_regression_gate_fails_and_identifies_failed_metric() -> None:
    """AC-12: Promotion is prevented when holdout regression gate fails; result identifies version and failed metric."""
    evaluator = PromotionGateEvaluator()
    config = RegressionGateConfig(
        min_precision=0.90,
        min_critical_finding_recall=1.0,
    )

    metrics_fail = EvaluationMetrics(
        split=DatasetSplit.HOLDOUT,
        total_cases=5,
        expected_required_count=10,
        candidate_findings_count=12,
        true_positives=8,
        false_positives=4,
        false_negatives=2,
        precision=0.6667,  # < 0.90
        recall=0.80,
        f1_score=0.7273,
        critical_findings_expected=2,
        critical_findings_detected=1,  # Missed a critical finding => 0.5 < 1.0
        critical_finding_recall=0.5,
        mean_duration_seconds=2.0,
        total_cost_usd=0.10,
        is_cost_complete=True,
    )

    result = evaluator.evaluate_gate("candidate-prompt-v3", metrics_fail, config)
    assert result.passed is False
    assert result.candidate_version == "candidate-prompt-v3"
    assert "min_precision" in result.failed_gates
    assert "min_critical_finding_recall" in result.failed_gates
    assert any("precision" in r for r in result.reasons)
    assert any("critical finding recall" in r for r in result.reasons)


def test_development_split_cannot_satisfy_holdout_promotion_gate() -> None:
    """Development split evaluation cannot promote to production."""
    evaluator = PromotionGateEvaluator()
    config = RegressionGateConfig(min_precision=0.5)

    metrics_dev = EvaluationMetrics(
        split=DatasetSplit.DEVELOPMENT,
        total_cases=5,
        expected_required_count=5,
        candidate_findings_count=5,
        true_positives=5,
        false_positives=0,
        false_negatives=0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        critical_findings_expected=1,
        critical_findings_detected=1,
        critical_finding_recall=1.0,
    )

    result = evaluator.evaluate_gate("policy-v2", metrics_dev, config)
    assert result.passed is False
    assert "dataset_split" in result.failed_gates
    assert any("requires 'holdout' split" in r for r in result.reasons)


# --- 6. Incomplete Cost/Latency Fails Closed Under Configured Gate ---


def test_incomplete_cost_or_duration_fails_closed_under_configured_gate() -> None:
    evaluator = PromotionGateEvaluator()
    config = RegressionGateConfig(max_regression_cost_usd=1.00, max_regression_duration_seconds=5.0)

    # Incomplete cost
    metrics_incomplete_cost = EvaluationMetrics(
        split=DatasetSplit.HOLDOUT,
        total_cases=2,
        expected_required_count=2,
        candidate_findings_count=2,
        true_positives=2,
        false_positives=0,
        false_negatives=0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        critical_findings_expected=0,
        critical_findings_detected=0,
        critical_finding_recall=1.0,
        mean_duration_seconds=2.0,
        total_cost_usd=None,
        is_cost_complete=False,
    )

    res_cost = evaluator.evaluate_gate("cand-1", metrics_incomplete_cost, config)
    assert res_cost.passed is False
    assert "max_regression_cost_usd" in res_cost.failed_gates
    assert any("Cannot verify max_regression_cost_usd" in r for r in res_cost.reasons)

    # Missing duration
    metrics_unavail_duration = EvaluationMetrics(
        split=DatasetSplit.HOLDOUT,
        total_cases=2,
        expected_required_count=2,
        candidate_findings_count=2,
        true_positives=2,
        false_positives=0,
        false_negatives=0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        critical_findings_expected=0,
        critical_findings_detected=0,
        critical_finding_recall=1.0,
        mean_duration_seconds=None,
        total_cost_usd=0.10,
        is_cost_complete=True,
        has_unavailable_latency=True,
    )

    res_dur = evaluator.evaluate_gate("cand-1", metrics_unavail_duration, config)
    assert res_dur.passed is False
    assert "max_regression_duration_seconds" in res_dur.failed_gates
    assert any("Cannot verify max_regression_duration_seconds" in r for r in res_dur.reasons)


# --- 7. Feedback Ledger: Repository Isolation & Append-Only (FR-20, NFR-07) ---


def test_feedback_ledger_records_dispositions_with_tenant_isolation() -> None:
    conn = sqlite3.connect(":memory:")
    ledger = FeedbackLedger(conn)

    t0 = time.time()
    fb_repo1 = FeedbackRecord(
        feedback_id="fb-1",
        repository_id="org/repo-1",
        canonical_id="canon-1",
        run_id="run-1",
        policy_version="1.0",
        disposition=FeedbackDisposition.APPROVED,
        actor="maintainer-alice",
        actor_role="maintainer",
        rationale="Accurate finding and fix",
        confidence_at_review=0.92,
        timestamp=t0,
    )
    fb_repo2 = FeedbackRecord(
        feedback_id="fb-2",
        repository_id="org/repo-2",
        canonical_id="canon-2",
        run_id="run-2",
        policy_version="1.0",
        disposition=FeedbackDisposition.DISPUTED,
        actor="maintainer-bob",
        actor_role="maintainer",
        rationale="False positive in test directory",
        confidence_at_review=0.88,
        timestamp=t0 + 10.0,
    )

    ledger.record_feedback(fb_repo1)
    ledger.record_feedback(fb_repo2)

    # Repository isolation query (NFR-07)
    records_repo1 = ledger.get_feedback("org/repo-1")
    assert len(records_repo1) == 1
    assert records_repo1[0].feedback_id == "fb-1"
    assert records_repo1[0].disposition == FeedbackDisposition.APPROVED

    records_repo2 = ledger.get_feedback("org/repo-2")
    assert len(records_repo2) == 1
    assert records_repo2[0].feedback_id == "fb-2"
    assert records_repo2[0].disposition == FeedbackDisposition.DISPUTED

    # Query without repository ID is rejected
    with pytest.raises(ValueError, match="Repository ID is required"):
        ledger.get_feedback("")

    # Enforce append-only semantics via SQLite trigger
    with pytest.raises(sqlite3.DatabaseError, match="reviewer_feedback is append-only"):
        conn.execute("UPDATE reviewer_feedback SET disposition = 'rejected' WHERE feedback_id = 'fb-1'")

    with pytest.raises(sqlite3.DatabaseError, match="reviewer_feedback is append-only"):
        conn.execute("DELETE FROM reviewer_feedback WHERE feedback_id = 'fb-1'")


# --- 8. No Automatic Promotion from Raw Feedback (SPEC Non-Goals) ---


def test_raw_feedback_cannot_auto_promote_or_mutate_policy() -> None:
    """SPEC non-goal: No automatic learning, retraining, or promotion directly from reviewer feedback."""
    conn = sqlite3.connect(":memory:")
    ledger = FeedbackLedger(conn)
    manager = PolicyPromotionManager(conn, initial_active_version="1.0")

    # Record 100 positive approvals in feedback ledger
    for i in range(100):
        ledger.record_feedback(
            FeedbackRecord(
                feedback_id=f"fb-bulk-{i}",
                repository_id="owner/repo",
                canonical_id=f"canon-{i}",
                run_id=f"run-{i}",
                policy_version="1.0",
                disposition=FeedbackDisposition.APPROVED,
                actor="alice",
                actor_role="maintainer",
                timestamp=time.time(),
            )
        )

    # Active version remains completely unchanged; feedback ledger cannot mutate policy
    assert manager.active_policy_version == "1.0"
    assert len(manager.get_history()) == 0


# --- 9. Promotion Requires Both Passing Holdout & Explicit Human Approval (FR-20, AC-12) ---


def test_promotion_requires_both_passing_holdout_and_human_approval() -> None:
    conn = sqlite3.connect(":memory:")
    manager = PolicyPromotionManager(conn, initial_active_version="1.0")

    # 1. Gate result that passed on holdout
    passing_gate = PromotionGateResult(
        passed=True,
        candidate_version="policy-2.0",
        baseline_version="1.0",
        dataset_split=DatasetSplit.HOLDOUT,
        metrics=EvaluationMetrics(
            split=DatasetSplit.HOLDOUT,
            total_cases=10,
            expected_required_count=10,
            candidate_findings_count=10,
            true_positives=10,
            false_positives=0,
            false_negatives=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            critical_findings_expected=1,
            critical_findings_detected=1,
            critical_finding_recall=1.0,
        ),
        failed_gates=(),
        reasons=(),
    )

    # 2. Gate result that failed
    failing_gate = PromotionGateResult(
        passed=False,
        candidate_version="policy-bad",
        baseline_version="1.0",
        dataset_split=DatasetSplit.HOLDOUT,
        metrics=passing_gate.metrics,
        failed_gates=("min_precision",),
        reasons=("Precision too low",),
    )

    # Attempt promotion with missing human approver => rejected
    with pytest.raises(ValueError, match="Explicit human approval is mandatory"):
        manager.promote_policy_version("policy-2.0", passing_gate, human_approver="", reason="automated")

    # Attempt promotion with failing holdout gate => rejected
    with pytest.raises(ValueError, match="holdout regression gate failed"):
        manager.promote_policy_version("policy-bad", failing_gate, human_approver="Ayush Kant", reason="Try promote")

    assert manager.active_policy_version == "1.0"

    # Successful promotion with BOTH passing holdout gate and human approval
    promo_record = manager.promote_policy_version(
        "policy-2.0",
        passing_gate,
        human_approver="Ayush Kant",
        reason="Holdout evaluation passed with 100% precision/recall on golden test set",
    )
    assert promo_record.action == "promoted"
    assert promo_record.policy_version == "policy-2.0"
    assert promo_record.human_actor == "Ayush Kant"
    assert manager.active_policy_version == "policy-2.0"


# --- 10. Explicit Auditable Policy Rollback (FR-20) ---


def test_explicit_auditable_policy_rollback() -> None:
    conn = sqlite3.connect(":memory:")
    manager = PolicyPromotionManager(conn, initial_active_version="1.0")

    # First promote 1.0 so it is recorded as a successfully promoted version
    gate_1 = PromotionGateResult(
        passed=True,
        candidate_version="1.0",
        baseline_version=None,
        dataset_split=DatasetSplit.HOLDOUT,
        metrics=EvaluationMetrics(
            split=DatasetSplit.HOLDOUT, total_cases=1, expected_required_count=1,
            candidate_findings_count=1, true_positives=1, false_positives=0, false_negatives=0,
            precision=1.0, recall=1.0, f1_score=1.0, critical_findings_expected=0,
            critical_findings_detected=0, critical_finding_recall=1.0,
        ),
        failed_gates=(), reasons=(),
    )
    manager.promote_policy_version("1.0", gate_1, human_approver="Alice", reason="Initial base promotion")

    # Promote 2.0
    gate_2 = PromotionGateResult(
        passed=True,
        candidate_version="2.0",
        baseline_version="1.0",
        dataset_split=DatasetSplit.HOLDOUT,
        metrics=gate_1.metrics,
        failed_gates=(), reasons=(),
    )
    manager.promote_policy_version("2.0", gate_2, human_approver="Alice", reason="Version 2.0 launch")
    assert manager.active_policy_version == "2.0"

    # Rollback to 1.0 without human actor is rejected
    with pytest.raises(ValueError, match="Human actor is required"):
        manager.rollback_policy_version("1.0", human_actor="", reason="Rollback")

    # Explicit rollback to previously promoted 1.0
    rb_record = manager.rollback_policy_version(
        "1.0",
        human_actor="Bob Maintainer",
        reason="Unexpected elevated disputes in production repository",
    )
    assert rb_record.action == "rolled_back"
    assert rb_record.policy_version == "1.0"
    assert rb_record.human_actor == "Bob Maintainer"
    assert manager.active_policy_version == "1.0"

    # Verify audit history
    history = manager.get_history()
    assert len(history) == 3
    assert history[0].action == "promoted"
    assert history[1].action == "promoted"
    assert history[2].action == "rolled_back"


def test_rollback_safety_rejects_never_promoted_version() -> None:
    """Issue 2: Attempting rollback to a never-promoted version must be rejected and active state unchanged."""
    conn = sqlite3.connect(":memory:")
    manager = PolicyPromotionManager(conn, initial_active_version="1.0")

    gate = PromotionGateResult(
        passed=True,
        candidate_version="2.0",
        baseline_version="1.0",
        dataset_split=DatasetSplit.HOLDOUT,
        metrics=EvaluationMetrics(
            split=DatasetSplit.HOLDOUT, total_cases=1, expected_required_count=1,
            candidate_findings_count=1, true_positives=1, false_positives=0, false_negatives=0,
            precision=1.0, recall=1.0, f1_score=1.0, critical_findings_expected=0,
            critical_findings_detected=0, critical_finding_recall=1.0,
        ),
        failed_gates=(), reasons=(),
    )
    manager.promote_policy_version("2.0", gate, human_approver="Alice", reason="Promote 2.0")
    assert manager.active_policy_version == "2.0"

    # Attempt rollback to a version that was never promoted ("9.9")
    with pytest.raises(ValueError, match="never successfully promoted"):
        manager.rollback_policy_version("9.9", human_actor="Bob", reason="Testing invalid rollback")

    # Active state must remain unchanged
    assert manager.active_policy_version == "2.0"


def test_durable_active_policy_version_recovered_by_new_manager() -> None:
    """Issue 3: New PolicyPromotionManager recovers active version from durable SQLite history."""
    conn = sqlite3.connect(":memory:")
    manager1 = PolicyPromotionManager(conn, initial_active_version="1.0")
    assert manager1.active_policy_version == "1.0"

    gate = PromotionGateResult(
        passed=True,
        candidate_version="2.0",
        baseline_version="1.0",
        dataset_split=DatasetSplit.HOLDOUT,
        metrics=EvaluationMetrics(
            split=DatasetSplit.HOLDOUT, total_cases=1, expected_required_count=1,
            candidate_findings_count=1, true_positives=1, false_positives=0, false_negatives=0,
            precision=1.0, recall=1.0, f1_score=1.0, critical_findings_expected=0,
            critical_findings_detected=0, critical_finding_recall=1.0,
        ),
        failed_gates=(), reasons=(),
    )
    manager1.promote_policy_version("2.0", gate, human_approver="Alice", reason="Promote 2.0")
    assert manager1.active_policy_version == "2.0"

    # Create a NEW manager instance sharing the same SQLite database with initial_active_version="1.0"
    manager2 = PolicyPromotionManager(conn, initial_active_version="1.0")
    # Must recover "2.0" from durable promotion history
    assert manager2.active_policy_version == "2.0"

    # Promote 3.0 via manager2
    gate_3 = PromotionGateResult(
        passed=True, candidate_version="3.0", baseline_version="2.0", dataset_split=DatasetSplit.HOLDOUT,
        metrics=gate.metrics, failed_gates=(), reasons=(),
    )
    manager2.promote_policy_version("3.0", gate_3, human_approver="Charlie", reason="Promote 3.0")
    assert manager2.active_policy_version == "3.0"

    # Rollback to 2.0 via manager2
    manager2.rollback_policy_version("2.0", human_actor="Dave", reason="Rollback to 2.0")
    assert manager2.active_policy_version == "2.0"

    # Create manager3, should recover "2.0"
    manager3 = PolicyPromotionManager(conn, initial_active_version="1.0")
    assert manager3.active_policy_version == "2.0"


# --- 11. Drift Detection (Quality, Calibration, and Cost Invariants) (FR-20) ---


def test_drift_detector_evaluates_quality_and_dispute_drift() -> None:
    cfg = DriftConfig(max_rejection_rate_drift=0.15, max_dispute_rate=0.20)
    detector = DriftDetector(cfg)

    # Baseline: 10 reviews, 1 rejection (10% rejection rate)
    baseline = [
        FeedbackRecord("b1", "repo", "c1", "r1", "1.0", disposition=FeedbackDisposition.APPROVED),
        FeedbackRecord("b2", "repo", "c2", "r2", "1.0", disposition=FeedbackDisposition.APPROVED),
        FeedbackRecord("b3", "repo", "c3", "r3", "1.0", disposition=FeedbackDisposition.APPROVED),
        FeedbackRecord("b4", "repo", "c4", "r4", "1.0", disposition=FeedbackDisposition.APPROVED),
        FeedbackRecord("b5", "repo", "c5", "r5", "1.0", disposition=FeedbackDisposition.REJECTED),
    ]

    # Current: 5 reviews, 3 rejections (60% rejection rate) => drift = +50% > 15% threshold
    current = [
        FeedbackRecord("c1", "repo", "c1", "r1", "1.0", disposition=FeedbackDisposition.REJECTED),
        FeedbackRecord("c2", "repo", "c2", "r2", "1.0", disposition=FeedbackDisposition.REJECTED),
        FeedbackRecord("c3", "repo", "c3", "r3", "1.0", disposition=FeedbackDisposition.REJECTED),
        FeedbackRecord("c4", "repo", "c4", "r4", "1.0", disposition=FeedbackDisposition.APPROVED),
        FeedbackRecord("c5", "repo", "c5", "r5", "1.0", disposition=FeedbackDisposition.DISPUTED),
    ]

    report = detector.evaluate_drift("repo", baseline, current)
    assert report.is_drift_detected is True

    q_signal = next(s for s in report.signals if s.signal_type == "quality_drift")
    assert q_signal.exceeded is True
    assert q_signal.baseline_value == 0.2
    assert q_signal.current_value == 0.6

    disp_signal = next(s for s in report.signals if s.signal_type == "dispute_drift")
    assert disp_signal.current_value == 0.2  # 1/5 = 20%
    assert disp_signal.exceeded is False    # 20% <= 20% threshold


def test_calibration_drift_evaluated_only_when_high_confidence_threshold_configured() -> None:
    """Contract correction 1: Calibration drift is only evaluated when high_confidence_threshold is explicitly configured."""
    feedback = [
        FeedbackRecord("f1", "repo", "c1", "r1", "1.0", disposition=FeedbackDisposition.REJECTED, confidence_at_review=0.95),
        FeedbackRecord("f2", "repo", "c2", "r2", "1.0", disposition=FeedbackDisposition.REJECTED, confidence_at_review=0.90),
    ]

    # 1. Config with max_calibration_gap set, but high_confidence_threshold is None => SKIPPED
    cfg_unconfigured = DriftConfig(high_confidence_threshold=None, max_calibration_gap=0.10)
    detector_unconfigured = DriftDetector(cfg_unconfigured)
    report_skip = detector_unconfigured.evaluate_drift("repo", [], feedback)
    sig_names = [s.signal_type for s in report_skip.signals]
    assert "calibration_drift" not in sig_names

    # 2. Config with explicit high_confidence_threshold => EVALUATED
    cfg_configured = DriftConfig(high_confidence_threshold=0.85, max_calibration_gap=0.10)
    detector_configured = DriftDetector(cfg_configured)
    report_eval = detector_configured.evaluate_drift("repo", [], feedback)
    cal_signal = next(s for s in report_eval.signals if s.signal_type == "calibration_drift")
    assert cal_signal.is_evaluable is True
    assert cal_signal.current_value == 1.0  # 2/2 high-confidence findings were rejected
    assert cal_signal.exceeded is True


def test_cost_drift_safe_handling_for_incomplete_or_zero_baseline() -> None:
    """Contract correction 2: Explicit safe handling for unavailable, incomplete, or zero baseline cost."""
    cfg = DriftConfig(max_cost_drift_ratio=1.5)  # Alert if cost grows by >50% (ratio > 1.5)
    detector = DriftDetector(cfg)

    # 1. Baseline cost is unavailable/None => cannot claim zero drift, must be unevaluable/non-passing
    rep_none = detector.evaluate_drift("repo", [], [], baseline_cost_usd=None, current_cost_usd=0.50)
    sig_none = next(s for s in rep_none.signals if s.signal_type == "cost_drift")
    assert sig_none.is_evaluable is False
    assert sig_none.exceeded is True
    assert "unavailable or incomplete" in sig_none.details

    # 2. Incomplete cost accounting flag
    rep_incomp = detector.evaluate_drift("repo", [], [], baseline_cost_usd=0.20, current_cost_usd=0.30, is_cost_complete=False)
    sig_incomp = next(s for s in rep_incomp.signals if s.signal_type == "cost_drift")
    assert sig_incomp.is_evaluable is False
    assert sig_incomp.exceeded is True

    # 3. Baseline cost is 0.0 => cannot divide by zero
    rep_zero = detector.evaluate_drift("repo", [], [], baseline_cost_usd=0.0, current_cost_usd=0.20)
    sig_zero = next(s for s in rep_zero.signals if s.signal_type == "cost_drift")
    assert sig_zero.is_evaluable is False
    assert sig_zero.exceeded is True
    assert "division by zero" in sig_zero.details

    # 4. Normal valid evaluation: baseline $0.10, current $0.20 => 2.0x ratio > 1.5x threshold => exceeded
    rep_valid = detector.evaluate_drift("repo", [], [], baseline_cost_usd=0.10, current_cost_usd=0.20)
    sig_valid = next(s for s in rep_valid.signals if s.signal_type == "cost_drift")
    assert sig_valid.is_evaluable is True
    assert sig_valid.exceeded is True
    assert sig_valid.current_value == 0.20
    assert sig_valid.baseline_value == 0.10


# --- 12. Secret Leakage Protection (FR-18) ---


def test_secret_leakage_protection_in_evaluation_and_feedback() -> None:
    conn = sqlite3.connect(":memory:")
    ledger = FeedbackLedger(conn)

    # Attempt to record feedback containing raw GitHub PAT secret in rationale
    secret_pat = "ghp_" + "A" * 36
    leaky_feedback = FeedbackRecord(
        feedback_id="fb-leak",
        repository_id="owner/repo",
        canonical_id="canon-1",
        run_id="run-1",
        policy_version="1.0",
        actor="hacker",
        rationale=f"Secret token used was {secret_pat}",
    )

    with pytest.raises(ValueError, match="Secret detected in feedback rationale"):
        ledger.record_feedback(leaky_feedback)

    # Verify no records entered database
    assert len(ledger.get_feedback("owner/repo")) == 0


def test_drift_detector_evaluates_model_behavior_drift() -> None:
    """Issue 1: Model behavior drift is evaluated when configured and skipped when unconfigured/insufficient."""
    # 1. Unconfigured -> skipped
    unconfigured_cfg = DriftConfig()
    detector_unconf = DriftDetector(unconfigured_cfg)
    rep_unconf = detector_unconf.evaluate_drift("repo", [], [])
    assert not any(s.signal_type == "model_behavior_drift" for s in rep_unconf.signals)

    # 2. Configured, but insufficient feedback in baseline or current -> is_evaluable=False, exceeded=False
    configured_cfg = DriftConfig(max_model_behavior_drift=0.25)
    detector_conf = DriftDetector(configured_cfg)
    rep_insufficient = detector_conf.evaluate_drift("repo", [], [])
    sig_insuf = next(s for s in rep_insufficient.signals if s.signal_type == "model_behavior_drift")
    assert sig_insuf.is_evaluable is False
    assert sig_insuf.exceeded is False

    # 3. Configured with sufficient feedback -> evaluated
    # Baseline: all gpt-4o (100%)
    baseline = [
        FeedbackRecord(
            f"b{i}", "repo", f"c{i}", f"r{i}", "1.0",
            model_configuration={"model": "gpt-4o"},
            disposition=FeedbackDisposition.APPROVED,
        )
        for i in range(10)
    ]
    # Current: 2 gpt-4o, 8 claude-3.5-sonnet (80% claude shift > 25% threshold)
    current = [
        FeedbackRecord(
            f"c{i}", "repo", f"c{i}", f"r{i}", "1.0",
            model_configuration={"model": "claude-3.5-sonnet" if i < 8 else "gpt-4o"},
            disposition=FeedbackDisposition.APPROVED,
        )
        for i in range(10)
    ]

    report = detector_conf.evaluate_drift("repo", baseline, current)
    sig_eval = next(s for s in report.signals if s.signal_type == "model_behavior_drift")
    assert sig_eval.is_evaluable is True
    assert sig_eval.exceeded is True
    assert sig_eval.current_value == 0.8


def test_secret_scanning_rejects_secret_in_feedback_model_configuration() -> None:
    """Issue 4: Secret in feedback model_configuration is rejected without leaking secret in message."""
    conn = sqlite3.connect(":memory:")
    ledger = FeedbackLedger(conn)
    secret_pat = "ghp_" + "B" * 36

    fb_leaky_config = FeedbackRecord(
        feedback_id="fb-cfg-leak",
        repository_id="owner/repo",
        canonical_id="canon-1",
        run_id="run-1",
        policy_version="1.0",
        model_configuration={"api_key": secret_pat, "model": "custom"},
        actor="dev",
        rationale="Valid rationale",
    )

    with pytest.raises(ValueError, match="Secret detected in feedback model_configuration") as exc_info:
        ledger.record_feedback(fb_leaky_config)

    # Invariant: Exception message must NOT leak raw secret
    assert secret_pat not in str(exc_info.value)
    assert "sensitive token found" in str(exc_info.value)
    assert len(ledger.get_feedback("owner/repo")) == 0


def test_secret_scanning_rejects_secret_in_golden_pr_fields() -> None:
    """Issue 4: Secret in golden PR diff_content or free-form fields is rejected without leaking secrets."""
    secret_pat = "ghp_" + "C" * 36

    # 1. diff_content with secret
    with pytest.raises(ValueError, match="Secret detected in golden PR case 'case-leak' diff_content") as exc1:
        GoldenPRCase(
            case_id="case-leak",
            repository_id="owner/repo",
            diff_content=f"--- a/file.py\n+++ b/file.py\n+TOKEN = '{secret_pat}'",
        )
    assert secret_pat not in str(exc1.value)

    # 2. title with secret
    with pytest.raises(ValueError, match="title"):
        GoldenPRCase(
            case_id="case-title-leak",
            repository_id="owner/repo",
            title=f"Fix bug with token {secret_pat}",
        )

    # 3. metadata with secret
    with pytest.raises(ValueError, match="metadata"):
        GoldenPRCase(
            case_id="case-meta-leak",
            repository_id="owner/repo",
            metadata={"secret_token": secret_pat},
        )

    # 4. expected finding summary with secret
    with pytest.raises(ValueError, match="Secret detected in golden finding 'f-leak' summary"):
        GoldenPRCase(
            case_id="case-find-leak",
            repository_id="owner/repo",
            expected_findings=(
                GoldenFinding(
                    finding_id="f-leak",
                    category="security",
                    severity="critical",
                    file_path="src/app.py",
                    line_range=(1, 5),
                    summary=f"Found leaked token {secret_pat}",
                ),
            ),
        )


# --- Category Family Normalization, Severity Distance, and Golden Benchmark Tests ---


def test_evaluator_category_family_normalization() -> None:
    """EvaluationRunner._is_finding_match matches across aliases within the same category family."""
    runner = EvaluationRunner()

    # 1. 'code_smell' and 'maintainability' match expected 'quality_defect'
    exp_quality = GoldenFinding(
        finding_id="exp-q",
        category="quality_defect",
        severity="medium",
        file_path="src/main.py",
        line_range=(10, 20),
    )
    cand_smell = _make_sample_candidate(category="code_smell", severity="medium")
    cand_maint = _make_sample_candidate(category="maintainability", severity="medium")
    cand_sec = _make_sample_candidate(category="security", severity="medium")

    assert runner._is_finding_match(exp_quality, cand_smell) is True
    assert runner._is_finding_match(exp_quality, cand_maint) is True
    assert runner._is_finding_match(exp_quality, cand_sec) is False

    # 2. 'docstring' matches expected 'documentation'
    exp_doc = GoldenFinding(
        finding_id="exp-d",
        category="documentation",
        severity="low",
        file_path="src/main.py",
        line_range=(10, 20),
    )
    cand_docstring = _make_sample_candidate(category="docstring", severity="low")
    assert runner._is_finding_match(exp_doc, cand_docstring) is True


def test_severity_distance_computation() -> None:
    """compute_severity_distance computes deterministic distance between ordered severity tiers."""
    assert compute_severity_distance("medium", "medium") == 0
    assert compute_severity_distance("high", "medium") == 1
    assert compute_severity_distance("medium", "high") == 1
    assert compute_severity_distance("critical", "info") == 4
    assert compute_severity_distance("low", "critical") == 3


def test_evaluation_case_and_dataset_severity_metrics() -> None:
    """EvaluationRunner tracks exact match rate, mean distance, over/under rates."""
    runner = EvaluationRunner()

    case = GoldenPRCase(
        case_id="case-sev-metrics",
        repository_id="owner/repo",
        expected_findings=(
            GoldenFinding(
                finding_id="exp-1",
                category="correctness",
                severity="medium",
                file_path="src/app.py",
                line_range=(10, 15),
                severity_tolerance=1,
            ),
            GoldenFinding(
                finding_id="exp-2",
                category="quality",
                severity="medium",
                file_path="src/app.py",
                line_range=(20, 25),
                severity_tolerance=1,
            ),
            GoldenFinding(
                finding_id="exp-3",
                category="documentation",
                severity="low",
                file_path="src/app.py",
                line_range=(30, 35),
                severity_tolerance=1,
            ),
        ),
    )

    # Candidate 1: Exact match (medium vs medium) -> dist=0
    # Candidate 2: Over-severity by 1 tier (high vs medium) -> dist=1, over=1
    # Candidate 3: Over-severity by 1 tier (medium vs low) -> dist=1, over=1
    c1 = CandidateFinding(
        finding_id="c-1",
        correlation_id="corr-1",
        specialist_type=SpecialistType.QUALITY,
        category="correctness",
        severity="medium",
        confidence=0.9,
        summary="C1",
        rationale="R1",
        file_path="src/app.py",
        line_range=(10, 15),
    )
    c2 = CandidateFinding(
        finding_id="c-2",
        correlation_id="corr-1",
        specialist_type=SpecialistType.QUALITY,
        category="quality",
        severity="high",
        confidence=0.9,
        summary="C2",
        rationale="R2",
        file_path="src/app.py",
        line_range=(20, 25),
    )
    c3 = CandidateFinding(
        finding_id="c-3",
        correlation_id="corr-1",
        specialist_type=SpecialistType.DOCUMENTATION,
        category="documentation",
        severity="medium",
        confidence=0.9,
        summary="C3",
        rationale="R3",
        file_path="src/app.py",
        line_range=(30, 35),
    )

    res = runner.evaluate_case(case, [c1, c2, c3])
    assert res.true_positives == 3
    assert res.false_positives == 0
    assert res.false_negatives == 0
    assert res.exact_severity_matches == 1
    assert res.one_tier_deviations == 2
    assert res.major_deviations == 0
    assert res.severity_over_count == 2
    assert res.severity_under_count == 0
    assert res.total_severity_distance == 2

    # In dataset evaluation
    dataset = GoldenPRDataset(
        dataset_id="ds-test",
        version="1.0.0",
        cases=(case,),
    )
    metrics = runner.evaluate_dataset(dataset, {case.case_id: [c1, c2, c3]})
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.severity_exact_match_rate == round(1 / 3, 4)
    assert metrics.mean_severity_distance == round(2 / 3, 4)
    assert metrics.over_severity_rate == round(2 / 3, 4)
    assert metrics.under_severity_rate == 0.0
    assert metrics.one_tier_deviation_rate == round(2 / 3, 4)


def test_promotion_gate_severity_metrics_enforcement() -> None:
    """PromotionGateEvaluator blocks candidates violating severity exact match, distance, or over/under thresholds."""
    evaluator = PromotionGateEvaluator()

    base_metrics = EvaluationMetrics(
        split=DatasetSplit.HOLDOUT,
        total_cases=1,
        expected_required_count=2,
        candidate_findings_count=2,
        true_positives=2,
        false_positives=0,
        false_negatives=0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        critical_findings_expected=0,
        critical_findings_detected=0,
        critical_finding_recall=1.0,
        severity_exact_match_rate=0.75,
        mean_severity_distance=0.35,
        over_severity_rate=0.25,
        under_severity_rate=0.0,
    )

    # 1. Satisfied thresholds -> PASS
    config_pass = RegressionGateConfig(
        min_f1=0.80,
        min_severity_exact_match_rate=0.70,
        max_mean_severity_distance=0.50,
        max_over_severity_rate=0.30,
    )
    res_pass = evaluator.evaluate_gate("v2.0", base_metrics, config_pass)
    assert res_pass.passed is True
    assert len(res_pass.failed_gates) == 0

    # 2. Strict exact match threshold -> FAIL
    config_fail_exact = RegressionGateConfig(min_severity_exact_match_rate=0.90)
    res_fail_exact = evaluator.evaluate_gate("v2.0", base_metrics, config_fail_exact)
    assert res_fail_exact.passed is False
    assert "min_severity_exact_match_rate" in res_fail_exact.failed_gates

    # 3. Strict max distance threshold -> FAIL
    config_fail_dist = RegressionGateConfig(max_mean_severity_distance=0.20)
    res_fail_dist = evaluator.evaluate_gate("v2.0", base_metrics, config_fail_dist)
    assert res_fail_dist.passed is False
    assert "max_mean_severity_distance" in res_fail_dist.failed_gates

    # 4. Strict max over-severity threshold -> FAIL
    config_fail_over = RegressionGateConfig(max_over_severity_rate=0.10)
    res_fail_over = evaluator.evaluate_gate("v2.0", base_metrics, config_fail_over)
    assert res_fail_over.passed is False
    assert "max_over_severity_rate" in res_fail_over.failed_gates


def test_load_golden_dataset_schema_and_security_validation() -> None:
    """load_golden_dataset parses golden_prs_v1.json and ensures zero secrets."""
    dataset = load_golden_dataset("data/golden_prs_v1.json")
    assert dataset.dataset_id == "golden_prs_v1"
    assert dataset.version == "1.0.0"
    assert len(dataset.cases) >= 5

    dev_cases = dataset.get_cases(DatasetSplit.DEVELOPMENT)
    assert len(dev_cases) == 5

    archetypes = {c.metadata.get("archetype") for c in dev_cases}
    assert "empty_input_runtime_crash" in archetypes
    assert "insecure_prng_auth_token" in archetypes
    assert "mutable_default_argument" in archetypes
    assert "docstring_contradicts_implementation" in archetypes
    assert "test_without_assertion" in archetypes

    # Ensure zero secrets in file
    scanner = SecretLeakageScanner()
    content = Path("data/golden_prs_v1.json").read_text(encoding="utf-8")
    scan_res = scanner.scan_text(content)
    assert scan_res.has_secret is False
    assert len(scan_res.matches) == 0


def test_offline_golden_benchmark_script_execution() -> None:
    """scripts/evaluate_golden.py runs offline benchmark and passes on both dev and holdout splits."""
    from scripts.evaluate_golden import run_evaluation

    # Development split run
    dev_status = run_evaluation(
        dataset_path="data/golden_prs_v1.json",
        split_str="development",
        run_gate=False,
    )
    assert dev_status == 0

    # Holdout split run with regression gate
    holdout_status = run_evaluation(
        dataset_path="data/golden_prs_v1.json",
        split_str="holdout",
        run_gate=True,
    )
    assert holdout_status == 0
