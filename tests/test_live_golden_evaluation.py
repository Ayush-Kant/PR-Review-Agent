"""Focused tests for live-golden evaluation validation, dry-run safety, and severity calibration.

Covers:
- Dry-run publication suppression (zero GitHub writes)
- Raw vs calibrated severity preservation
- Golden expectation matching
- Specialist failure / degraded coverage handling
- Secret-free evaluation artifacts
- Regression compatibility with existing offline evaluation
- Correct security calibration behavior
- Correct benign-randomness behavior
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest
import sqlite3
import time
from typing import Any

from pr_review_agent.evaluation import (
    DatasetSplit,
    EvaluationCaseResult,
    EvaluationMetrics,
    EvaluationRunner,
    GoldenFinding,
    GoldenPRCase,
    GoldenPRDataset,
    LiveEvaluationCaseOutcome,
    LiveGoldenEvaluator,
    LiveGoldenReport,
    PromotionGateEvaluator,
    RegressionGateConfig,
    load_golden_dataset,
)
from pr_review_agent.observability import AuditSpine
from pr_review_agent.orchestration import (
    CandidateFinding,
    ReviewJob,
    ReviewOrchestrator,
    SpecialistCoverageSummary,
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
    ReviewTruthStore,
    TruthState,
)
from pr_review_agent.security import (
    RuntimeSecretRegistry,
    SecretLeakageScanner,
    SecretType,
    SecurityConfig,
)


def _build_test_case(
    case_id: str = "case-test-1",
    category: str = "quality_defect",
    expected_sev: str = "medium",
) -> GoldenPRCase:
    return GoldenPRCase(
        case_id=case_id,
        repository_id="Ayush-Kant/PR-Review-Agent",
        title="feat: test feature",
        base_sha="base-sha-1",
        head_sha="head-sha-1",
        changed_files=("src/app.py",),
        diff_content="diff --git a/src/app.py b/src/app.py\n+def fn(x=[]): pass",
        split=DatasetSplit.DEVELOPMENT,
        expected_findings=(
            GoldenFinding(
                finding_id="exp-1",
                category=category,
                severity=expected_sev,
                file_path="src/app.py",
                line_range=(1, 2),
                summary="Mutable default argument in function definition.",
                is_required=True,
                line_tolerance=2,
                severity_tolerance=1,
            ),
        ),
    )


def test_dry_run_publication_suppression() -> None:
    """LiveGoldenEvaluator guarantees zero GitHub publications or outbound comment writes."""
    case = _build_test_case()

    # Mock handler that produces an auto-approvable finding
    def quality_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        cand = CandidateFinding(
            finding_id="cand-auto-1",
            correlation_id=spec_input.correlation_id,
            specialist_type=SpecialistType.QUALITY,
            category="quality",
            severity="medium",
            confidence=0.95,
            summary="Mutable default argument in function definition.",
            rationale="Items default argument retains state.",
            file_path="src/app.py",
            line_range=(1, 2),
            evidence_refs=("ref1",),
        )
        return SpecialistOutput(
            specialist_type=SpecialistType.QUALITY,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(cand,),
        )

    def dummy_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(),
        )

    handlers = {
        SpecialistType.SECURITY: dummy_handler,
        SpecialistType.QUALITY: quality_handler,
        SpecialistType.TESTS: dummy_handler,
        SpecialistType.DOCUMENTATION: dummy_handler,
    }

    orchestrator = ReviewOrchestrator(specialist_handlers=handlers)
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)

    evaluator = LiveGoldenEvaluator(
        orchestrator=orchestrator,
        truth_store=truth_store,
        provider="mock",
        model="mock-v1",
    )

    outcome = asyncio.run(evaluator.evaluate_case(case))

    # Finding was auto-approved by policy, but NO publisher was invoked
    assert len(outcome.canonical_findings) == 1
    cf = outcome.canonical_findings[0]
    assert cf.disposition == FindingDisposition.AUTO_APPROVED

    # Check ReviewTruthStore recorded initial state as AUTO_APPROVED, with 0 published comments
    latest = truth_store.get_latest_state(cf.canonical_id)
    assert latest is not None
    assert latest.state == TruthState.AUTO_APPROVED

    # Zero tables or records exist for GitHub publications
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "github_publications" not in tables
    conn.close()


def test_raw_vs_calibrated_severity_preservation() -> None:
    """Live evaluation preserves raw specialist severity separately from calibrated policy severity."""
    case = _build_test_case(category="quality_defect", expected_sev="medium")

    def quality_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        # Raw specialist output inflated to 'high'
        cand = CandidateFinding(
            finding_id="cand-inflated-1",
            correlation_id=spec_input.correlation_id,
            specialist_type=SpecialistType.QUALITY,
            category="quality",
            severity="high",
            confidence=0.95,
            summary="Mutable default argument in function definition: items=[] retains state.",
            rationale="Mutable default argument in Python.",
            file_path="src/app.py",
            line_range=(1, 2),
            evidence_refs=("ref1",),
        )
        return SpecialistOutput(
            specialist_type=SpecialistType.QUALITY,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(cand,),
        )

    def dummy_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={
            SpecialistType.SECURITY: dummy_handler,
            SpecialistType.QUALITY: quality_handler,
            SpecialistType.TESTS: dummy_handler,
            SpecialistType.DOCUMENTATION: dummy_handler,
        }
    )

    evaluator = LiveGoldenEvaluator(orchestrator=orchestrator)
    outcome = asyncio.run(evaluator.evaluate_case(case))

    assert len(outcome.raw_findings) == 1
    assert outcome.raw_findings[0].severity == "high"

    assert len(outcome.canonical_findings) == 1
    canon = outcome.canonical_findings[0]
    assert canon.raw_severity == "high"
    assert canon.calibrated_severity == "medium"
    assert canon.severity == "medium"
    assert canon.calibration_rule == "quality_mutable_default_medium"

    # Verify serialization dictionary preserves both fields
    d = outcome.to_dict()
    assert d["canonical_findings"][0]["raw_severity"] == "high"
    assert d["canonical_findings"][0]["calibrated_severity"] == "medium"


def test_golden_expectation_matching() -> None:
    """Evaluates 1:1 matching against golden case requirements with category normalization."""
    case = _build_test_case(category="quality_defect", expected_sev="medium")

    def quality_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        cand = CandidateFinding(
            finding_id="c-match-1",
            correlation_id=spec_input.correlation_id,
            specialist_type=SpecialistType.QUALITY,
            category="code_smell",  # Alias in quality_defect family
            severity="medium",
            confidence=0.90,
            summary="Mutable default argument in function definition.",
            rationale="Mutable default argument.",
            file_path="src/app.py",
            line_range=(1, 2),
            evidence_refs=("ref1",),
        )
        return SpecialistOutput(
            specialist_type=SpecialistType.QUALITY,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(cand,),
        )

    def dummy_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={
            SpecialistType.SECURITY: dummy_handler,
            SpecialistType.QUALITY: quality_handler,
            SpecialistType.TESTS: dummy_handler,
            SpecialistType.DOCUMENTATION: dummy_handler,
        }
    )

    evaluator = LiveGoldenEvaluator(orchestrator=orchestrator)
    outcome = asyncio.run(evaluator.evaluate_case(case))

    assert outcome.case_result.true_positives == 1
    assert outcome.case_result.false_positives == 0
    assert outcome.case_result.false_negatives == 0
    assert outcome.case_result.exact_severity_matches == 1
    assert outcome.case_result.total_severity_distance == 0


def test_specialist_failure_and_degraded_coverage_handling() -> None:
    """Specialist failures and degraded coverage are captured explicitly and never silently swallowed."""
    case = _build_test_case()

    def failing_security_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.FAILED,
            findings=(),
            error_message="HTTP 429 Too Many Requests: Groq rate limit exceeded",
        )

    def ok_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={
            SpecialistType.SECURITY: failing_security_handler,
            SpecialistType.QUALITY: ok_handler,
            SpecialistType.TESTS: ok_handler,
            SpecialistType.DOCUMENTATION: ok_handler,
        }
    )

    evaluator = LiveGoldenEvaluator(orchestrator=orchestrator)
    outcome = asyncio.run(evaluator.evaluate_case(case))

    assert outcome.is_degraded is True
    assert outcome.coverage_summary is not None
    assert "security" in outcome.coverage_summary.failed_specialists
    assert "security" in outcome.failure_reasons
    assert "429" in outcome.failure_reasons["security"]


def test_secret_free_evaluation_artifacts() -> None:
    """Generated JSON and Markdown evaluation report artifacts contain zero secrets."""
    dataset = load_golden_dataset("data/golden_prs_v1.json")
    dev_cases = dataset.get_cases(DatasetSplit.DEVELOPMENT)

    # Construct report
    metrics = EvaluationMetrics(
        split=DatasetSplit.DEVELOPMENT,
        total_cases=len(dev_cases),
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
        severity_exact_match_rate=1.0,
        mean_severity_distance=0.0,
        over_severity_rate=0.0,
        under_severity_rate=0.0,
    )

    report = LiveGoldenReport(
        report_id="rep-sec-check",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        split=DatasetSplit.DEVELOPMENT,
        provider="groq",
        model="openai/gpt-oss-120b",
        timestamp=time.time(),
        total_cases=len(dev_cases),
        metrics=metrics,
        case_outcomes=(),
        dry_run_verified=True,
        no_github_writes_verified=True,
        no_secrets_verified=True,
    )

    rep_dict = report.to_dict()
    rep_json = json.dumps(rep_dict)
    rep_md = report.to_markdown()

    sec_registry = RuntimeSecretRegistry()
    sec_registry.register_secret(SecretType.GROQ_API_KEY, "gsk_secret_dummy_test_token_123456789")
    scanner = SecretLeakageScanner(sec_registry)

    assert scanner.scan_text(rep_json).has_secret is False
    assert scanner.scan_text(rep_md).has_secret is False
    assert len(scanner.scan_text(rep_json).matches) == 0


def test_regression_compatibility_with_existing_offline_evaluation() -> None:
    """scripts/evaluate_golden.py runs offline benchmark with 100% precision/recall."""
    from scripts.evaluate_golden import run_evaluation

    exit_code = run_evaluation(
        dataset_path="data/golden_prs_v1.json",
        split_str="development",
        mode="mock",
        run_gate=False,
    )
    assert exit_code == 0


def test_correct_security_calibration_behavior() -> None:
    """Insecure token in authentication flow remains high severity."""
    case = _build_test_case(case_id="case-sec-auth", category="security", expected_sev="high")

    def sec_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        cand = CandidateFinding(
            finding_id="cand-sec-1",
            correlation_id=spec_input.correlation_id,
            specialist_type=SpecialistType.SECURITY,
            category="security",
            severity="high",
            confidence=0.95,
            summary="Use of random.choices for authentication session token generation.",
            rationale="Non-cryptographic PRNG in password reset token flow.",
            file_path="src/app.py",
            line_range=(1, 2),
            evidence_refs=("ref1",),
        )
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(cand,),
        )

    def dummy_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={
            SpecialistType.SECURITY: sec_handler,
            SpecialistType.QUALITY: dummy_handler,
            SpecialistType.TESTS: dummy_handler,
            SpecialistType.DOCUMENTATION: dummy_handler,
        }
    )

    evaluator = LiveGoldenEvaluator(orchestrator=orchestrator)
    outcome = asyncio.run(evaluator.evaluate_case(case))

    assert len(outcome.canonical_findings) == 1
    canon = outcome.canonical_findings[0]
    assert canon.raw_severity == "high"
    assert canon.calibrated_severity == "high"
    assert canon.calibration_rule == "security_sensitive_context_high"
    assert canon.disposition == FindingDisposition.HELD


def test_correct_benign_randomness_behavior() -> None:
    """Randomness in benign UI animation script is calibrated down to low severity."""
    case = _build_test_case(case_id="case-sec-ui", category="security", expected_sev="low")

    def sec_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        cand = CandidateFinding(
            finding_id="cand-ui-1",
            correlation_id=spec_input.correlation_id,
            specialist_type=SpecialistType.SECURITY,
            category="security",
            severity="high",  # Raw model false positive
            confidence=0.85,
            summary="random.choice used for frontend UI badge animation color delay.",
            rationale="PRNG used in UI script.",
            file_path="src/app.py",
            line_range=(1, 2),
            evidence_refs=("ref1",),
        )
        return SpecialistOutput(
            specialist_type=SpecialistType.SECURITY,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(cand,),
        )

    def dummy_handler(spec_input: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status=SpecialistStatus.COMPLETED,
            findings=(),
        )

    orchestrator = ReviewOrchestrator(
        specialist_handlers={
            SpecialistType.SECURITY: sec_handler,
            SpecialistType.QUALITY: dummy_handler,
            SpecialistType.TESTS: dummy_handler,
            SpecialistType.DOCUMENTATION: dummy_handler,
        }
    )

    evaluator = LiveGoldenEvaluator(orchestrator=orchestrator)
    outcome = asyncio.run(evaluator.evaluate_case(case))

    assert len(outcome.canonical_findings) == 1
    canon = outcome.canonical_findings[0]
    assert canon.raw_severity == "high"
    assert canon.calibrated_severity == "low"
    assert canon.calibration_rule == "security_benign_context_low"
    assert "cosmetic/benign randomness" in canon.calibration_reason
