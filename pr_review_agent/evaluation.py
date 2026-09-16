"""Evaluation datasets, regression promotion gates, feedback capture, drift detection, and rollback controls.

Implements:
- FR-16: Versioned golden pull-request dataset, development and holdout evaluation,
         and regression promotion gates blocking unverified quality, safety, latency, or cost regressions.
- FR-20: Feedback capture linked to findings and policy versions, material drift detection across quality,
         calibration, model behavior, freshness, and cost, requiring evaluation, independent review,
         explicit promotion, and rollback capability.
- AC-12: Prevention of production promotion when holdout regression gate fails, identifying version and failed metric.
- NFR-04: Configurable provisional targets without invented numeric SLOs or fixed defaults.
- NFR-07: Strict repository tenancy isolation for feedback, golden cases, and drift evaluations.
- NFR-09: Versioned policy transitions and golden-set regression gates before promotion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import json
import sqlite3
import time
from typing import Any
import uuid

from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.observability import AuditEvent, AuditSpine
from pr_review_agent.orchestration import (
    CandidateFinding,
    JobState,
    ReviewJob,
    ReviewOrchestrator,
    SpecialistCoverageSummary,
)
from pr_review_agent.policy import (
    CanonicalFinding,
    FindingAggregator,
    FindingDisposition,
    ReviewPolicyEngine,
    ReviewTruthStore,
    TruthState,
    normalize_category_family,
)
from pr_review_agent.security import SecretLeakageScanner


SEVERITY_TIERS: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "blocking": 3,
    "medium": 2,
    "low": 1,
    "info": 0,
}


def compute_severity_distance(actual_sev: str, expected_sev: str) -> int:
    """Compute absolute tier distance between actual and expected severity strings."""
    act_tier = SEVERITY_TIERS.get((actual_sev or "").strip().lower(), 1)
    exp_tier = SEVERITY_TIERS.get((expected_sev or "").strip().lower(), 1)
    return abs(act_tier - exp_tier)


class DatasetSplit(str, Enum):
    """Dataset split for model, prompt, and policy evaluation."""

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


@dataclass(frozen=True)
class GoldenFinding:
    """Expected finding in a golden PR test case."""

    finding_id: str
    category: str
    severity: str
    file_path: str
    line_range: tuple[int, int]
    summary: str = ""
    is_required: bool = True
    line_tolerance: int = 0
    severity_tolerance: int = 1
    notes: str = ""



@dataclass(frozen=True)
class GoldenPRCase:
    """A versioned golden pull request case with expected findings and allowed tolerances."""

    case_id: str
    repository_id: str
    title: str = ""
    base_sha: str = ""
    head_sha: str = ""
    changed_files: tuple[str, ...] = ()
    diff_content: str = ""
    expected_findings: tuple[GoldenFinding, ...] = ()
    split: DatasetSplit = DatasetSplit.DEVELOPMENT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scanner = SecretLeakageScanner()
        if self.diff_content:
            res = scanner.scan_text(self.diff_content, f"golden_case_{self.case_id}_diff")
            if res.has_secret:
                raise ValueError(f"Secret detected in golden PR case '{self.case_id}' diff_content: sensitive token found")
        if self.title:
            res = scanner.scan_text(self.title, f"golden_case_{self.case_id}_title")
            if res.has_secret:
                raise ValueError(f"Secret detected in golden PR case '{self.case_id}' title: sensitive token found")
        for f in self.expected_findings:
            if f.summary:
                res = scanner.scan_text(f.summary, f"golden_case_{self.case_id}_finding_{f.finding_id}")
                if res.has_secret:
                    raise ValueError(f"Secret detected in golden finding '{f.finding_id}' summary: sensitive token found")
        if self.metadata:
            meta_str = json.dumps(dict(self.metadata), sort_keys=True)
            res = scanner.scan_text(meta_str, f"golden_case_{self.case_id}_metadata")
            if res.has_secret:
                raise ValueError(f"Secret detected in golden PR case '{self.case_id}' metadata: sensitive token found")


@dataclass(frozen=True)
class GoldenPRDataset:
    """A versioned collection of golden pull request evaluation cases."""

    dataset_id: str
    version: str
    cases: tuple[GoldenPRCase, ...] = ()

    def get_cases(
        self,
        split: DatasetSplit | None = None,
        repository_id: str | None = None,
    ) -> tuple[GoldenPRCase, ...]:
        """Filter dataset cases by split and/or repository."""
        filtered = []
        for case in self.cases:
            if split is not None and case.split != split:
                continue
            if repository_id is not None and case.repository_id != repository_id:
                continue
            filtered.append(case)
        return tuple(filtered)


def load_golden_dataset(file_path: str | Path) -> GoldenPRDataset:
    """Load and validate a versioned GoldenPRDataset from a JSON file with secret scanning."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Golden dataset file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_id = str(data.get("dataset_id", "golden-v1"))
    version = str(data.get("version", "1.0"))
    cases_raw = data.get("cases", [])

    parsed_cases: list[GoldenPRCase] = []
    for c_data in cases_raw:
        case_id = str(c_data["case_id"])
        repo_id = str(c_data.get("repository_id", "owner/repo"))
        title = str(c_data.get("title", ""))
        base_sha = str(c_data.get("base_sha", "base123"))
        head_sha = str(c_data.get("head_sha", "head123"))
        changed_files = tuple(str(cf) for cf in c_data.get("changed_files", ()))
        diff_content = str(c_data.get("diff_content", ""))
        split_str = str(c_data.get("split", "development")).lower()
        split = DatasetSplit.HOLDOUT if split_str == "holdout" else DatasetSplit.DEVELOPMENT
        metadata = dict(c_data.get("metadata", {}))

        expected_findings: list[GoldenFinding] = []
        for ef_data in c_data.get("expected_findings", []):
            lr = ef_data.get("line_range", [1, 1])
            line_range = (int(lr[0]), int(lr[1]))
            ef = GoldenFinding(
                finding_id=str(ef_data["finding_id"]),
                category=str(ef_data["category"]),
                severity=str(ef_data["severity"]),
                file_path=str(ef_data["file_path"]),
                line_range=line_range,
                summary=str(ef_data.get("summary", "")),
                is_required=bool(ef_data.get("is_required", True)),
                line_tolerance=int(ef_data.get("line_tolerance", 0)),
                severity_tolerance=int(ef_data.get("severity_tolerance", 1)),
                notes=str(ef_data.get("notes", "")),
            )
            expected_findings.append(ef)

        golden_case = GoldenPRCase(
            case_id=case_id,
            repository_id=repo_id,
            title=title,
            base_sha=base_sha,
            head_sha=head_sha,
            changed_files=changed_files,
            diff_content=diff_content,
            expected_findings=tuple(expected_findings),
            split=split,
            metadata=metadata,
        )
        parsed_cases.append(golden_case)

    return GoldenPRDataset(
        dataset_id=dataset_id,
        version=version,
        cases=tuple(parsed_cases),
    )


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Evaluation output for a single golden PR case."""

    case_id: str
    split: DatasetSplit
    true_positives: int
    false_positives: int
    false_negatives: int
    critical_expected: int
    critical_detected: int
    duration_seconds: float | None = None
    cost_usd: float | None = None
    is_cost_complete: bool = True
    has_unavailable_latency: bool = False
    exact_severity_matches: int = 0
    one_tier_deviations: int = 0
    major_deviations: int = 0
    severity_over_count: int = 0
    severity_under_count: int = 0
    total_severity_distance: int = 0


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate quality, safety, latency, and cost metrics for an evaluation run."""

    split: DatasetSplit
    total_cases: int
    expected_required_count: int
    candidate_findings_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    critical_findings_expected: int
    critical_findings_detected: int
    critical_finding_recall: float
    mean_duration_seconds: float | None = None
    total_cost_usd: float | None = None
    is_cost_complete: bool = True
    has_unavailable_latency: bool = False
    severity_exact_match_rate: float = 1.0
    mean_severity_distance: float = 0.0
    over_severity_rate: float = 0.0
    under_severity_rate: float = 0.0
    one_tier_deviation_rate: float = 0.0
    major_deviation_rate: float = 0.0


@dataclass(frozen=True)
class RegressionGateConfig:
    """Configurable quality, safety, latency, and cost promotion gate thresholds.

    Per DECISION-67711afd and ASSUMPTION-b774604f, all thresholds default to None.
    The system must not invent fixed numeric defaults.
    """

    min_precision: float | None = None
    min_recall: float | None = None
    min_f1: float | None = None
    min_critical_finding_recall: float | None = None
    max_regression_cost_usd: float | None = None
    max_regression_duration_seconds: float | None = None
    min_severity_exact_match_rate: float | None = None
    max_mean_severity_distance: float | None = None
    max_over_severity_rate: float | None = None
    max_under_severity_rate: float | None = None


@dataclass(frozen=True)
class PromotionGateResult:
    """Outcome of regression promotion gate evaluation against holdout dataset (AC-12)."""

    passed: bool
    candidate_version: str
    baseline_version: str | None
    dataset_split: DatasetSplit
    metrics: EvaluationMetrics
    failed_gates: tuple[str, ...]
    reasons: tuple[str, ...]


class EvaluationRunner:
    """Executes deterministic finding matching and quality/safety metric calculation."""

    @staticmethod
    def _is_finding_match(expected: GoldenFinding, candidate: Any) -> bool:
        """Deterministically check whether candidate finding satisfies expected finding."""
        c_file = getattr(candidate, "file_path", None)
        if c_file != expected.file_path:
            return False

        c_cat = getattr(candidate, "category", "") or ""
        # Semantic category family normalization (Section 6)
        if normalize_category_family(c_cat) != normalize_category_family(expected.category):
            return False

        c_sev = getattr(candidate, "severity", "") or ""
        sev_dist = compute_severity_distance(c_sev, expected.severity)
        if sev_dist > expected.severity_tolerance:
            return False

        c_range = getattr(candidate, "line_range", None)
        if expected.line_range and c_range:
            e_start, e_end = expected.line_range
            c_start, c_end = c_range
            tol = expected.line_tolerance
            # Range overlap or within line tolerance bounds
            if not (max(c_start, e_start - tol) <= min(c_end, e_end + tol)):
                return False

        return True

    def evaluate_case(
        self,
        case: GoldenPRCase,
        candidate_findings: Sequence[Any],
        *,
        duration_seconds: float | None = None,
        cost_usd: float | None = None,
        is_cost_complete: bool = True,
    ) -> EvaluationCaseResult:
        """Evaluate candidate findings against a single golden PR case with 1:1 matching."""
        unmatched_candidates = list(candidate_findings)
        matched_expected_ids: set[str] = set()
        matched_candidate_indices: set[int] = set()

        exact_sev_matches = 0
        one_tier_devs = 0
        major_devs = 0
        over_sev = 0
        under_sev = 0
        tot_sev_dist = 0

        for expected in case.expected_findings:
            for idx, candidate in enumerate(unmatched_candidates):
                if idx in matched_candidate_indices:
                    continue
                if self._is_finding_match(expected, candidate):
                    matched_expected_ids.add(expected.finding_id)
                    matched_candidate_indices.add(idx)

                    c_sev = getattr(candidate, "severity", "") or ""
                    dist = compute_severity_distance(c_sev, expected.severity)
                    act_tier = SEVERITY_TIERS.get((c_sev or "").strip().lower(), 1)
                    exp_tier = SEVERITY_TIERS.get((expected.severity or "").strip().lower(), 1)

                    tot_sev_dist += dist
                    if dist == 0:
                        exact_sev_matches += 1
                    elif dist == 1:
                        one_tier_devs += 1
                    else:
                        major_devs += 1

                    if act_tier > exp_tier:
                        over_sev += 1
                    elif act_tier < exp_tier:
                        under_sev += 1
                    break

        tp = len(matched_candidate_indices)
        fp = len(candidate_findings) - tp

        required_expected = [f for f in case.expected_findings if f.is_required]
        fn = sum(1 for f in required_expected if f.finding_id not in matched_expected_ids)

        critical_expected = [f for f in case.expected_findings if f.severity.lower() in ("critical", "blocking")]
        crit_exp_count = len(critical_expected)
        crit_det_count = sum(1 for f in critical_expected if f.finding_id in matched_expected_ids)

        return EvaluationCaseResult(
            case_id=case.case_id,
            split=case.split,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            critical_expected=crit_exp_count,
            critical_detected=crit_det_count,
            duration_seconds=duration_seconds,
            cost_usd=cost_usd,
            is_cost_complete=is_cost_complete,
            has_unavailable_latency=(duration_seconds is None),
            exact_severity_matches=exact_sev_matches,
            one_tier_deviations=one_tier_devs,
            major_deviations=major_devs,
            severity_over_count=over_sev,
            severity_under_count=under_sev,
            total_severity_distance=tot_sev_dist,
        )

    def evaluate_dataset(
        self,
        dataset: GoldenPRDataset,
        candidate_findings_by_case: Mapping[str, Sequence[Any]],
        split: DatasetSplit = DatasetSplit.DEVELOPMENT,
        *,
        case_durations: Mapping[str, float] | None = None,
        case_costs: Mapping[str, float] | None = None,
        case_cost_completeness: Mapping[str, bool] | None = None,
    ) -> EvaluationMetrics:
        """Evaluate candidate findings across an entire dataset split."""
        target_cases = dataset.get_cases(split=split)
        total_cases = len(target_cases)
        if total_cases == 0:
            return EvaluationMetrics(
                split=split,
                total_cases=0,
                expected_required_count=0,
                candidate_findings_count=0,
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                critical_findings_expected=0,
                critical_findings_detected=0,
                critical_finding_recall=1.0,
                mean_duration_seconds=None,
                total_cost_usd=None,
                is_cost_complete=True,
                has_unavailable_latency=False,
            )

        tot_req_expected = 0
        tot_candidates = 0
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        tot_crit_exp = 0
        tot_crit_det = 0

        tot_exact_sev = 0
        tot_one_tier = 0
        tot_major_dev = 0
        tot_over_sev = 0
        tot_under_sev = 0
        tot_sev_dist = 0

        durations: list[float] = []
        costs: list[float] = []
        is_cost_comp = True
        has_unavail_latency = False

        dur_map = case_durations or {}
        cost_map = case_costs or {}
        comp_map = case_cost_completeness or {}

        for case in target_cases:
            c_findings = candidate_findings_by_case.get(case.case_id, ())
            tot_candidates += len(c_findings)
            tot_req_expected += sum(1 for f in case.expected_findings if f.is_required)

            dur = dur_map.get(case.case_id)
            if dur is not None:
                durations.append(dur)
            else:
                has_unavail_latency = True

            c_cost = cost_map.get(case.case_id)
            c_comp = comp_map.get(case.case_id, True)
            if not c_comp or c_cost is None:
                is_cost_comp = False
            if c_cost is not None:
                costs.append(c_cost)

            res = self.evaluate_case(
                case,
                c_findings,
                duration_seconds=dur,
                cost_usd=c_cost,
                is_cost_complete=c_comp,
            )
            tot_tp += res.true_positives
            tot_fp += res.false_positives
            tot_fn += res.false_negatives
            tot_crit_exp += res.critical_expected
            tot_crit_det += res.critical_detected

            tot_exact_sev += res.exact_severity_matches
            tot_one_tier += res.one_tier_deviations
            tot_major_dev += res.major_deviations
            tot_over_sev += res.severity_over_count
            tot_under_sev += res.severity_under_count
            tot_sev_dist += res.total_severity_distance

        precision = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
        recall = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        crit_recall = (tot_crit_det / tot_crit_exp) if tot_crit_exp > 0 else 1.0

        mean_dur = (sum(durations) / len(durations)) if durations else None
        tot_cost = round(sum(costs), 6) if costs else (0.0 if is_cost_comp and not cost_map else None)

        exact_match_rate = (tot_exact_sev / tot_tp) if tot_tp > 0 else (1.0 if tot_req_expected == 0 else 0.0)
        mean_sev_dist = (tot_sev_dist / tot_tp) if tot_tp > 0 else 0.0
        over_rate = (tot_over_sev / tot_tp) if tot_tp > 0 else 0.0
        under_rate = (tot_under_sev / tot_tp) if tot_tp > 0 else 0.0
        one_tier_rate = (tot_one_tier / tot_tp) if tot_tp > 0 else 0.0
        major_dev_rate = (tot_major_dev / tot_tp) if tot_tp > 0 else 0.0

        return EvaluationMetrics(
            split=split,
            total_cases=total_cases,
            expected_required_count=tot_req_expected,
            candidate_findings_count=tot_candidates,
            true_positives=tot_tp,
            false_positives=tot_fp,
            false_negatives=tot_fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            critical_findings_expected=tot_crit_exp,
            critical_findings_detected=tot_crit_det,
            critical_finding_recall=round(crit_recall, 4),
            mean_duration_seconds=round(mean_dur, 4) if mean_dur is not None else None,
            total_cost_usd=tot_cost,
            is_cost_complete=is_cost_comp,
            has_unavailable_latency=has_unavail_latency,
            severity_exact_match_rate=round(exact_match_rate, 4),
            mean_severity_distance=round(mean_sev_dist, 4),
            over_severity_rate=round(over_rate, 4),
            under_severity_rate=round(under_rate, 4),
            one_tier_deviation_rate=round(one_tier_rate, 4),
            major_deviation_rate=round(major_dev_rate, 4),
        )


class PromotionGateEvaluator:
    """Validates candidate metrics against configured regression gates on holdout set (AC-12)."""

    def evaluate_gate(
        self,
        candidate_version: str,
        metrics: EvaluationMetrics,
        config: RegressionGateConfig,
        baseline_version: str | None = None,
    ) -> PromotionGateResult:
        """Evaluate candidate metrics against holdout regression gates.

        Required invariant: Missing/unavailable metrics under a configured gate fail closed.
        """
        failed_gates: list[str] = []
        reasons: list[str] = []

        if metrics.split != DatasetSplit.HOLDOUT:
            failed_gates.append("dataset_split")
            reasons.append(f"Promotion gate requires '{DatasetSplit.HOLDOUT.value}' split; got '{metrics.split.value}'")

        if config.min_precision is not None:
            if metrics.precision < config.min_precision:
                failed_gates.append("min_precision")
                reasons.append(
                    f"Candidate precision ({metrics.precision:.4f}) is below min_precision threshold ({config.min_precision:.4f})"
                )

        if config.min_recall is not None:
            if metrics.recall < config.min_recall:
                failed_gates.append("min_recall")
                reasons.append(
                    f"Candidate recall ({metrics.recall:.4f}) is below min_recall threshold ({config.min_recall:.4f})"
                )

        if config.min_f1 is not None:
            if metrics.f1_score < config.min_f1:
                failed_gates.append("min_f1")
                reasons.append(
                    f"Candidate F1 score ({metrics.f1_score:.4f}) is below min_f1 threshold ({config.min_f1:.4f})"
                )

        if config.min_critical_finding_recall is not None:
            if metrics.critical_finding_recall < config.min_critical_finding_recall:
                failed_gates.append("min_critical_finding_recall")
                reasons.append(
                    f"Candidate critical finding recall ({metrics.critical_finding_recall:.4f}) is below threshold ({config.min_critical_finding_recall:.4f})"
                )

        if config.max_regression_cost_usd is not None:
            if not metrics.is_cost_complete or metrics.total_cost_usd is None:
                failed_gates.append("max_regression_cost_usd")
                reasons.append(
                    f"Cannot verify max_regression_cost_usd (${config.max_regression_cost_usd:.4f}): cost accounting is incomplete"
                )
            elif metrics.total_cost_usd > config.max_regression_cost_usd:
                failed_gates.append("max_regression_cost_usd")
                reasons.append(
                    f"Total evaluation cost (${metrics.total_cost_usd:.4f}) exceeds limit (${config.max_regression_cost_usd:.4f})"
                )

        if config.max_regression_duration_seconds is not None:
            if metrics.has_unavailable_latency or metrics.mean_duration_seconds is None:
                failed_gates.append("max_regression_duration_seconds")
                reasons.append(
                    f"Cannot verify max_regression_duration_seconds ({config.max_regression_duration_seconds}s): latency data is unavailable"
                )
            elif metrics.mean_duration_seconds > config.max_regression_duration_seconds:
                failed_gates.append("max_regression_duration_seconds")
                reasons.append(
                    f"Mean execution duration ({metrics.mean_duration_seconds:.2f}s) exceeds limit ({config.max_regression_duration_seconds:.2f}s)"
                )

        if config.min_severity_exact_match_rate is not None:
            if metrics.severity_exact_match_rate < config.min_severity_exact_match_rate:
                failed_gates.append("min_severity_exact_match_rate")
                reasons.append(
                    f"Candidate severity exact match rate ({metrics.severity_exact_match_rate:.4f}) is below threshold ({config.min_severity_exact_match_rate:.4f})"
                )

        if config.max_mean_severity_distance is not None:
            if metrics.mean_severity_distance > config.max_mean_severity_distance:
                failed_gates.append("max_mean_severity_distance")
                reasons.append(
                    f"Candidate mean severity distance ({metrics.mean_severity_distance:.4f}) exceeds threshold ({config.max_mean_severity_distance:.4f})"
                )

        if config.max_over_severity_rate is not None:
            if metrics.over_severity_rate > config.max_over_severity_rate:
                failed_gates.append("max_over_severity_rate")
                reasons.append(
                    f"Candidate over-severity rate ({metrics.over_severity_rate:.4f}) exceeds threshold ({config.max_over_severity_rate:.4f})"
                )

        if config.max_under_severity_rate is not None:
            if metrics.under_severity_rate > config.max_under_severity_rate:
                failed_gates.append("max_under_severity_rate")
                reasons.append(
                    f"Candidate under-severity rate ({metrics.under_severity_rate:.4f}) exceeds threshold ({config.max_under_severity_rate:.4f})"
                )

        passed = len(failed_gates) == 0
        return PromotionGateResult(
            passed=passed,
            candidate_version=candidate_version,
            baseline_version=baseline_version,
            dataset_split=metrics.split,
            metrics=metrics,
            failed_gates=tuple(failed_gates),
            reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class LiveEvaluationCaseOutcome:
    """Detailed outcome of evaluating a single golden PR case through the live specialist pipeline."""

    case_id: str
    title: str
    split: DatasetSplit
    coverage_summary: SpecialistCoverageSummary | None
    raw_findings: tuple[CandidateFinding, ...]
    canonical_findings: tuple[CanonicalFinding, ...]
    case_result: EvaluationCaseResult
    cost_usd: float | None = None
    duration_seconds: float | None = None
    is_degraded: bool = False
    failure_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "split": self.split.value,
            "is_degraded": self.is_degraded,
            "cost_usd": self.cost_usd,
            "duration_seconds": self.duration_seconds,
            "coverage": asdict(self.coverage_summary) if self.coverage_summary else None,
            "failure_reasons": self.failure_reasons,
            "raw_findings_count": len(self.raw_findings),
            "canonical_findings_count": len(self.canonical_findings),
            "case_result": asdict(self.case_result),
            "raw_findings": [
                {
                    "finding_id": f.finding_id,
                    "specialist_type": f.specialist_type.value if hasattr(f.specialist_type, "value") else str(f.specialist_type),
                    "category": f.category,
                    "severity": f.severity,
                    "confidence": f.confidence,
                    "file_path": f.file_path,
                    "line_range": f.line_range,
                    "summary": f.summary,
                }
                for f in self.raw_findings
            ],
            "canonical_findings": [
                {
                    "canonical_id": f.canonical_id,
                    "category": f.category,
                    "raw_severity": f.raw_severity,
                    "calibrated_severity": f.calibrated_severity,
                    "severity": f.severity,
                    "calibration_rule": f.calibration_rule,
                    "calibration_reason": f.calibration_reason,
                    "disposition": f.disposition.value if hasattr(f.disposition, "value") else str(f.disposition),
                    "disposition_reason": f.disposition_reason,
                    "file_path": f.file_path,
                    "line_range": f.line_range,
                    "summary": f.summary,
                }
                for f in self.canonical_findings
            ],
        }


@dataclass(frozen=True)
class LiveGoldenReport:
    """Comprehensive evaluation report comparing live specialist outputs with golden benchmark expectations."""

    report_id: str
    dataset_id: str
    dataset_version: str
    split: DatasetSplit
    provider: str
    model: str
    timestamp: float
    total_cases: int
    metrics: EvaluationMetrics
    case_outcomes: tuple[LiveEvaluationCaseOutcome, ...]
    gate_result: PromotionGateResult | None = None
    dry_run_verified: bool = True
    no_github_writes_verified: bool = True
    no_secrets_verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "split": self.split.value,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp,
            "total_cases": self.total_cases,
            "dry_run_verified": self.dry_run_verified,
            "no_github_writes_verified": self.no_github_writes_verified,
            "no_secrets_verified": self.no_secrets_verified,
            "metrics": asdict(self.metrics),
            "gate_result": asdict(self.gate_result) if self.gate_result else None,
            "case_outcomes": [c.to_dict() for c in self.case_outcomes],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Live Golden Evaluation Report - `{self.dataset_id}` (v{self.dataset_version})",
            "",
            f"- **Report ID:** `{self.report_id}`",
            f"- **Split:** `{self.split.value.upper()}`",
            f"- **Provider / Model:** `{self.provider}` / `{self.model}`",
            f"- **Timestamp:** `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.timestamp))}`",
            f"- **Dry-Run Verified:** `{self.dry_run_verified}` (Zero GitHub API writes)",
            f"- **No Secrets Verified:** `{self.no_secrets_verified}`",
            "",
            "## Aggregate Metrics Scorecard",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total Cases | {self.metrics.total_cases} |",
            f"| Expected Findings (Required) | {self.metrics.expected_required_count} |",
            f"| Candidate Findings Detected | {self.metrics.candidate_findings_count} |",
            f"| True Positives (TP) | {self.metrics.true_positives} |",
            f"| False Positives (FP) | {self.metrics.false_positives} |",
            f"| False Negatives (FN) | {self.metrics.false_negatives} |",
            f"| Precision | {self.metrics.precision:.4f} |",
            f"| Recall | {self.metrics.recall:.4f} |",
            f"| F1 Score | {self.metrics.f1_score:.4f} |",
            f"| Critical Finding Recall | {self.metrics.critical_finding_recall:.4f} |",
            f"| Severity Exact Match Rate | {self.metrics.severity_exact_match_rate:.4f} |",
            f"| Mean Severity Distance | {self.metrics.mean_severity_distance:.4f} |",
            f"| Over-Severity Rate | {self.metrics.over_severity_rate:.4f} |",
            f"| Under-Severity Rate | {self.metrics.under_severity_rate:.4f} |",
            f"| One-Tier Deviation Rate | {self.metrics.one_tier_deviation_rate:.4f} |",
            f"| Major Deviation Rate | {self.metrics.major_deviation_rate:.4f} |",
            "",
            "## Case-by-Case Breakdown",
            "",
        ]

        for outcome in self.case_outcomes:
            lines.append(f"### Case `{outcome.case_id}`: {outcome.title}")
            lines.append(f"- **Coverage Degraded:** {outcome.is_degraded}")
            if outcome.coverage_summary:
                cov = outcome.coverage_summary
                lines.append(f"- **Specialists Succeeded ({len(cov.succeeded_specialists)}):** {', '.join(cov.succeeded_specialists) or 'none'}")
                if cov.failed_specialists:
                    lines.append(f"- **Specialists Failed:** {', '.join(cov.failed_specialists)}")
            lines.append(f"- **Duration:** {outcome.duration_seconds or 0.0:.2f}s | **Cost:** ${outcome.cost_usd or 0.0:.4f}")
            lines.append("")
            lines.append("| ID | Category | Raw Sev | Calibrated Sev | Rule | Disposition | Summary |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for cf in outcome.canonical_findings:
                clean_summ = cf.summary.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{cf.canonical_id[:12]}` | `{cf.category}` | `{cf.raw_severity}` | `{cf.calibrated_severity}` | `{cf.calibration_rule}` | `{cf.disposition.value}` | {clean_summ[:80]} |")
            lines.append("")

        if self.gate_result:
            status_badge = "PASSED" if self.gate_result.passed else "FAILED"
            lines.append(f"## Regression Gate Outcome: **{status_badge}**")
            if not self.gate_result.passed:
                for r in self.gate_result.reasons:
                    lines.append(f"- [FAIL] {r}")
            lines.append("")

        return "\n".join(lines)


class LiveGoldenEvaluator:
    """Drives golden PR cases through the real specialist pipeline in controlled DRY-RUN mode."""

    def __init__(
        self,
        orchestrator: ReviewOrchestrator,
        *,
        aggregator: FindingAggregator | None = None,
        policy_engine: ReviewPolicyEngine | None = None,
        audit_spine: AuditSpine | None = None,
        truth_store: ReviewTruthStore | None = None,
        runner: EvaluationRunner | None = None,
        provider: str = "mock",
        model: str = "mock-model",
    ) -> None:
        self.orchestrator = orchestrator
        self.aggregator = aggregator or FindingAggregator()
        self.policy_engine = policy_engine or ReviewPolicyEngine(policy_version="v1")
        self.audit_spine = audit_spine
        self.truth_store = truth_store
        self.runner = runner or EvaluationRunner()
        self.provider = provider
        self.model = model

    async def evaluate_case(self, case: GoldenPRCase) -> LiveEvaluationCaseOutcome:
        """Run a single golden PR case through the specialist orchestrator and evaluate."""
        start_t = time.time()
        job = ReviewJob(
            job_id=f"job-{case.case_id}-{int(start_t)}",
            delivery_id=f"del-{case.case_id}",
            repository_id=case.repository_id,
            pull_request_number=1,
            base_sha=case.base_sha,
            head_sha=case.head_sha,
            state=JobState.RUNNING,
            deadline_seconds=120.0,
        )
        snapshot = ReviewSnapshot(
            repository_id=case.repository_id,
            repository_full_name=case.repository_id,
            pull_request_number=1,
            base_sha=case.base_sha,
            head_sha=case.head_sha,
            changed_files=case.changed_files,
            policy_version="v1",
            prompt_version="v1",
            retrieval_index_version="v1",
            model_configuration={"provider": self.provider, "model": self.model},
        )

        lifecycle = await self.orchestrator.execute_run(
            job,
            snapshot,
            diff_content=case.diff_content,
        )

        # Collect raw findings from specialist outputs
        raw_candidates: list[CandidateFinding] = []
        for _spec_type, out in lifecycle.specialist_outputs.items():
            raw_candidates.extend(out.findings)

        # Aggregate raw findings -> canonical findings with deterministic severity calibration
        canonical = self.aggregator.aggregate(
            raw_candidates,
            repository_id=case.repository_id,
            head_sha=case.head_sha,
        )

        # Evaluate policy for disposition (DRY RUN ONLY - never publish to GitHub)
        evaluated_canonical: list[CanonicalFinding] = []
        for finding in canonical:
            eval_f = self.policy_engine.evaluate(finding, is_fresh=True)
            evaluated_canonical.append(eval_f)
            if self.truth_store is not None:
                initial_state = (
                    TruthState.AUTO_APPROVED
                    if eval_f.disposition == FindingDisposition.AUTO_APPROVED
                    else TruthState.HELD
                )
                self.truth_store.record_initial(eval_f, initial_state=initial_state)

        duration = time.time() - start_t
        cost_usd = None
        if lifecycle.run_cost_summary:
            cost_usd = lifecycle.run_cost_summary.total_cost_usd

        # Match against expected golden findings
        case_res = self.runner.evaluate_case(
            case,
            evaluated_canonical,
            duration_seconds=duration,
            cost_usd=cost_usd,
            is_cost_complete=True,
        )

        failure_reasons = {}
        if lifecycle.coverage_summary and lifecycle.coverage_summary.failure_reasons:
            failure_reasons = dict(lifecycle.coverage_summary.failure_reasons)

        return LiveEvaluationCaseOutcome(
            case_id=case.case_id,
            title=case.title,
            split=case.split,
            coverage_summary=lifecycle.coverage_summary,
            raw_findings=tuple(raw_candidates),
            canonical_findings=tuple(evaluated_canonical),
            case_result=case_res,
            cost_usd=cost_usd,
            duration_seconds=round(duration, 3),
            is_degraded=lifecycle.is_degraded,
            failure_reasons=failure_reasons,
        )

    async def evaluate_dataset(
        self,
        dataset: GoldenPRDataset,
        split: DatasetSplit = DatasetSplit.DEVELOPMENT,
        *,
        gate_config: RegressionGateConfig | None = None,
        case_filter: Sequence[str] | None = None,
    ) -> LiveGoldenReport:
        """Run all cases in dataset split through the live specialist pipeline in DRY RUN mode."""
        cases = dataset.get_cases(split)
        eval_dataset = dataset
        if case_filter:
            target_ids = set(case_filter)
            cases = tuple(c for c in cases if c.case_id in target_ids)
            eval_dataset = GoldenPRDataset(
                dataset_id=dataset.dataset_id,
                version=dataset.version,
                cases=cases,
            )

        outcomes: list[LiveEvaluationCaseOutcome] = []
        findings_by_case: dict[str, list[CanonicalFinding]] = {}
        durations: dict[str, float] = {}
        costs: dict[str, float] = {}

        for case in cases:
            outcome = await self.evaluate_case(case)
            outcomes.append(outcome)
            findings_by_case[case.case_id] = list(outcome.canonical_findings)
            if outcome.duration_seconds is not None:
                durations[case.case_id] = outcome.duration_seconds
            if outcome.cost_usd is not None:
                costs[case.case_id] = outcome.cost_usd

        # Compute aggregate metrics
        metrics = self.runner.evaluate_dataset(
            dataset=eval_dataset,
            candidate_findings_by_case=findings_by_case,
            split=split,
            case_durations=durations,
            case_costs=costs,
        )

        gate_res = None
        if gate_config is not None:
            evaluator = PromotionGateEvaluator()
            gate_res = evaluator.evaluate_gate(
                candidate_version=dataset.version,
                metrics=metrics,
                config=gate_config,
            )

        report = LiveGoldenReport(
            report_id=f"live-eval-{uuid.uuid4().hex[:8]}",
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            split=split,
            provider=self.provider,
            model=self.model,
            timestamp=time.time(),
            total_cases=len(cases),
            metrics=metrics,
            case_outcomes=tuple(outcomes),
            gate_result=gate_res,
            dry_run_verified=True,
            no_github_writes_verified=True,
            no_secrets_verified=True,
        )
        return report


class FeedbackDisposition(str, Enum):
    """Maintainer feedback dispositions supported by Review Truth."""

    APPROVED = "approved"
    REJECTED = "rejected"
    DISMISSED = "dismissed"
    DISPUTED = "disputed"


@dataclass(frozen=True)
class FeedbackRecord:
    """Labeled feedback captured from maintainer actions linked to finding and policy version (FR-20)."""

    feedback_id: str
    repository_id: str
    canonical_id: str
    run_id: str
    policy_version: str
    prompt_version: str | None = None
    model_configuration: Mapping[str, Any] = field(default_factory=dict)
    disposition: FeedbackDisposition = FeedbackDisposition.APPROVED
    actor: str = "unknown"
    actor_role: str = "reviewer"
    rationale: str | None = None
    confidence_at_review: float | None = None
    timestamp: float = field(default_factory=time.time)


class FeedbackLedger:
    """Durable SQLite storage for labeled reviewer feedback maintaining tenant isolation (FR-20, NFR-07)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.scanner = SecretLeakageScanner()
        self._init_schema()

    def _init_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviewer_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    canonical_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    prompt_version TEXT,
                    model_config_json TEXT NOT NULL,
                    disposition TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    rationale TEXT,
                    confidence_at_review REAL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fb_repo_time
                ON reviewer_feedback (repository_id, timestamp)
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fb_policy
                ON reviewer_feedback (policy_version)
                """
            )
            # Enforce append-only semantics via SQLite trigger
            self.connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_reviewer_feedback_prevent_update
                BEFORE UPDATE ON reviewer_feedback
                BEGIN
                    SELECT RAISE(ABORT, 'reviewer_feedback is append-only');
                END;
                """
            )
            self.connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_reviewer_feedback_prevent_delete
                BEFORE DELETE ON reviewer_feedback
                BEGIN
                    SELECT RAISE(ABORT, 'reviewer_feedback is append-only');
                END;
                """
            )

    def record_feedback(self, feedback: FeedbackRecord) -> None:
        """Persist maintainer feedback after secret leakage scanning."""
        if not feedback.repository_id or not feedback.repository_id.strip():
            raise ValueError("Feedback must contain a non-empty repository_id (NFR-07)")

        if feedback.rationale:
            scan_res = self.scanner.scan_text(feedback.rationale, "feedback_rationale")
            if scan_res.has_secret:
                raise ValueError("Secret detected in feedback rationale: sensitive token found")

        if feedback.model_configuration:
            config_json = json.dumps(dict(feedback.model_configuration), sort_keys=True)
            scan_res_mc = self.scanner.scan_text(config_json, "feedback_model_configuration")
            if scan_res_mc.has_secret:
                raise ValueError("Secret detected in feedback model_configuration: sensitive token found")
        else:
            config_json = "{}"

        if feedback.actor:
            scan_res_actor = self.scanner.scan_text(feedback.actor, "feedback_actor")
            if scan_res_actor.has_secret:
                raise ValueError("Secret detected in feedback actor: sensitive token found")

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO reviewer_feedback (
                    feedback_id, repository_id, canonical_id, run_id,
                    policy_version, prompt_version, model_config_json,
                    disposition, actor, actor_role, rationale,
                    confidence_at_review, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.feedback_id,
                    feedback.repository_id,
                    feedback.canonical_id,
                    feedback.run_id,
                    feedback.policy_version,
                    feedback.prompt_version,
                    config_json,
                    feedback.disposition.value,
                    feedback.actor,
                    feedback.actor_role,
                    feedback.rationale,
                    feedback.confidence_at_review,
                    feedback.timestamp,
                ),
            )

    def get_feedback(
        self,
        repository_id: str,
        *,
        policy_version: str | None = None,
        since_timestamp: float | None = None,
        until_timestamp: float | None = None,
    ) -> list[FeedbackRecord]:
        """Query feedback scoped strictly to a repository."""
        if not repository_id:
            raise ValueError("Repository ID is required to query feedback (NFR-07)")

        query = "SELECT * FROM reviewer_feedback WHERE repository_id = ?"
        params: list[Any] = [repository_id]

        if policy_version:
            query += " AND policy_version = ?"
            params.append(policy_version)
        if since_timestamp is not None:
            query += " AND timestamp >= ?"
            params.append(since_timestamp)
        if until_timestamp is not None:
            query += " AND timestamp <= ?"
            params.append(until_timestamp)

        query += " ORDER BY timestamp ASC"
        rows = self.connection.execute(query, params).fetchall()

        records = []
        for row in rows:
            records.append(
                FeedbackRecord(
                    feedback_id=row[0],
                    repository_id=row[1],
                    canonical_id=row[2],
                    run_id=row[3],
                    policy_version=row[4],
                    prompt_version=row[5],
                    model_configuration=json.loads(row[6]),
                    disposition=FeedbackDisposition(row[7]),
                    actor=row[8],
                    actor_role=row[9],
                    rationale=row[10],
                    confidence_at_review=row[11],
                    timestamp=row[12],
                )
            )
        return records


@dataclass(frozen=True)
class DriftConfig:
    """Configurable drift detection tolerances.

    All thresholds default to None. Per DECISION-67711afd, no numeric defaults are invented.
    """

    max_rejection_rate_drift: float | None = None
    max_dispute_rate: float | None = None
    high_confidence_threshold: float | None = None
    max_calibration_gap: float | None = None
    max_model_behavior_drift: float | None = None
    max_cost_drift_ratio: float | None = None
    max_staleness_ratio: float | None = None


@dataclass(frozen=True)
class DriftSignal:
    """Signal indicating evaluation of a potential drift dimension."""

    signal_type: str
    baseline_value: float | None
    current_value: float | None
    threshold: float
    exceeded: bool
    details: str
    is_evaluable: bool = True


@dataclass(frozen=True)
class DriftReport:
    """Structured drift analysis report across feedback, calibration, staleness, and cost (FR-20)."""

    repository_id: str
    evaluated_at: float
    baseline_window_records: int
    current_window_records: int
    is_drift_detected: bool
    signals: tuple[DriftSignal, ...]


class DriftDetector:
    """Evaluates material drift in review quality, calibration, staleness, and cost."""

    def __init__(self, config: DriftConfig) -> None:
        self.config = config

    def evaluate_drift(
        self,
        repository_id: str,
        baseline_feedback: Sequence[FeedbackRecord],
        current_feedback: Sequence[FeedbackRecord],
        *,
        baseline_cost_usd: float | None = None,
        current_cost_usd: float | None = None,
        is_cost_complete: bool = True,
        stale_reviews_count: int = 0,
        total_reviews_count: int = 0,
        now: float | None = None,
    ) -> DriftReport:
        """Evaluate drift across baseline and current observation windows."""
        eval_time = time.time() if now is None else now
        signals: list[DriftSignal] = []

        # 1. Quality Drift (Rejection Rate Drift)
        if self.config.max_rejection_rate_drift is not None:
            base_count = len(baseline_feedback)
            curr_count = len(current_feedback)
            if base_count == 0 or curr_count == 0:
                signals.append(
                    DriftSignal(
                        signal_type="quality_drift",
                        baseline_value=None,
                        current_value=None,
                        threshold=self.config.max_rejection_rate_drift,
                        exceeded=False,
                        is_evaluable=False,
                        details="Insufficient feedback records in baseline or current window to evaluate quality drift",
                    )
                )
            else:
                base_rej = sum(1 for f in baseline_feedback if f.disposition == FeedbackDisposition.REJECTED)
                curr_rej = sum(1 for f in current_feedback if f.disposition == FeedbackDisposition.REJECTED)
                base_rate = base_rej / base_count
                curr_rate = curr_rej / curr_count
                rate_diff = curr_rate - base_rate
                exceeded = rate_diff > self.config.max_rejection_rate_drift
                signals.append(
                    DriftSignal(
                        signal_type="quality_drift",
                        baseline_value=round(base_rate, 4),
                        current_value=round(curr_rate, 4),
                        threshold=self.config.max_rejection_rate_drift,
                        exceeded=exceeded,
                        is_evaluable=True,
                        details=(
                            f"Rejection rate drifted by {rate_diff:+.4f} "
                            f"(baseline {base_rate:.4f} -> current {curr_rate:.4f}), "
                            f"threshold: {self.config.max_rejection_rate_drift:.4f}"
                        ),
                    )
                )

        # 2. Dispute Rate Drift
        if self.config.max_dispute_rate is not None:
            curr_count = len(current_feedback)
            if curr_count == 0:
                signals.append(
                    DriftSignal(
                        signal_type="dispute_drift",
                        baseline_value=None,
                        current_value=None,
                        threshold=self.config.max_dispute_rate,
                        exceeded=False,
                        is_evaluable=False,
                        details="No current feedback records to evaluate dispute rate",
                    )
                )
            else:
                dispute_count = sum(1 for f in current_feedback if f.disposition == FeedbackDisposition.DISPUTED)
                disp_rate = dispute_count / curr_count
                exceeded = disp_rate > self.config.max_dispute_rate
                signals.append(
                    DriftSignal(
                        signal_type="dispute_drift",
                        baseline_value=None,
                        current_value=round(disp_rate, 4),
                        threshold=self.config.max_dispute_rate,
                        exceeded=exceeded,
                        is_evaluable=True,
                        details=f"Dispute rate is {disp_rate:.4f}, threshold: {self.config.max_dispute_rate:.4f}",
                    )
                )

        # 3. Calibration Drift
        # Contract rule 1: Calibration drift must ONLY be evaluated when high_confidence_threshold is explicitly configured.
        if (
            self.config.high_confidence_threshold is not None
            and self.config.max_calibration_gap is not None
        ):
            high_conf_thresh = self.config.high_confidence_threshold
            high_conf_records = [
                f for f in current_feedback
                if f.confidence_at_review is not None and f.confidence_at_review >= high_conf_thresh
            ]
            if len(high_conf_records) == 0:
                signals.append(
                    DriftSignal(
                        signal_type="calibration_drift",
                        baseline_value=None,
                        current_value=None,
                        threshold=self.config.max_calibration_gap,
                        exceeded=False,
                        is_evaluable=False,
                        details=f"No findings with confidence >= {high_conf_thresh} to evaluate calibration gap",
                    )
                )
            else:
                unfavorable_count = sum(
                    1 for f in high_conf_records
                    if f.disposition in (FeedbackDisposition.REJECTED, FeedbackDisposition.DISPUTED)
                )
                gap = unfavorable_count / len(high_conf_records)
                exceeded = gap > self.config.max_calibration_gap
                signals.append(
                    DriftSignal(
                        signal_type="calibration_drift",
                        baseline_value=0.0,
                        current_value=round(gap, 4),
                        threshold=self.config.max_calibration_gap,
                        exceeded=exceeded,
                        is_evaluable=True,
                        details=(
                            f"High-confidence findings (>= {high_conf_thresh}) rejection/dispute rate "
                            f"is {gap:.4f} ({unfavorable_count}/{len(high_conf_records)}), "
                            f"threshold: {self.config.max_calibration_gap:.4f}"
                        ),
                    )
                )

        # 4. Model Behavior Drift
        if self.config.max_model_behavior_drift is not None:
            base_count = len(baseline_feedback)
            curr_count = len(current_feedback)
            if base_count == 0 or curr_count == 0:
                signals.append(
                    DriftSignal(
                        signal_type="model_behavior_drift",
                        baseline_value=None,
                        current_value=None,
                        threshold=self.config.max_model_behavior_drift,
                        exceeded=False,
                        is_evaluable=False,
                        details="Insufficient feedback records in baseline or current window to evaluate model behavior drift",
                    )
                )
            else:
                def _get_model_id(rec: FeedbackRecord) -> str:
                    if rec.model_configuration and isinstance(rec.model_configuration, Mapping):
                        m = rec.model_configuration.get("model") or rec.model_configuration.get("model_name")
                        if m:
                            return str(m)
                    return rec.policy_version or "unknown"

                base_dist: dict[str, int] = {}
                for f in baseline_feedback:
                    k = _get_model_id(f)
                    base_dist[k] = base_dist.get(k, 0) + 1

                curr_dist: dict[str, int] = {}
                for f in current_feedback:
                    k = _get_model_id(f)
                    curr_dist[k] = curr_dist.get(k, 0) + 1

                all_models = set(base_dist) | set(curr_dist)
                max_shift = max(
                    abs((base_dist.get(m, 0) / base_count) - (curr_dist.get(m, 0) / curr_count))
                    for m in all_models
                )
                exceeded = max_shift > self.config.max_model_behavior_drift
                signals.append(
                    DriftSignal(
                        signal_type="model_behavior_drift",
                        baseline_value=0.0,
                        current_value=round(max_shift, 4),
                        threshold=self.config.max_model_behavior_drift,
                        exceeded=exceeded,
                        is_evaluable=True,
                        details=(
                            f"Model behavior distribution drifted by {max_shift:.4f} "
                            f"(max proportion shift across models), "
                            f"threshold: {self.config.max_model_behavior_drift:.4f}"
                        ),
                    )
                )

        # 5. Cost Drift
        # Contract rule 2: Explicit safe handling for baseline and current cost.
        if self.config.max_cost_drift_ratio is not None:
            if not is_cost_complete or baseline_cost_usd is None or current_cost_usd is None:
                # Baseline/current cost data unavailable or incomplete => cannot claim zero drift.
                signals.append(
                    DriftSignal(
                        signal_type="cost_drift",
                        baseline_value=baseline_cost_usd,
                        current_value=current_cost_usd,
                        threshold=self.config.max_cost_drift_ratio,
                        exceeded=True,  # Incomplete data fails closed on hard drift gate
                        is_evaluable=False,
                        details="Cost drift cannot be safely evaluated: baseline or current cost data is unavailable or incomplete",
                    )
                )
            elif baseline_cost_usd <= 0.0:
                # Baseline cost == 0 => ratio is not evaluable without division by zero.
                signals.append(
                    DriftSignal(
                        signal_type="cost_drift",
                        baseline_value=baseline_cost_usd,
                        current_value=current_cost_usd,
                        threshold=self.config.max_cost_drift_ratio,
                        exceeded=True,  # Fails closed on zero baseline
                        is_evaluable=False,
                        details="Baseline cost is 0.0 USD; cost drift ratio cannot be evaluated without division by zero",
                    )
                )
            else:
                cost_ratio = current_cost_usd / baseline_cost_usd
                exceeded = cost_ratio > self.config.max_cost_drift_ratio
                signals.append(
                    DriftSignal(
                        signal_type="cost_drift",
                        baseline_value=round(baseline_cost_usd, 4),
                        current_value=round(current_cost_usd, 4),
                        threshold=self.config.max_cost_drift_ratio,
                        exceeded=exceeded,
                        is_evaluable=True,
                        details=(
                            f"Cost ratio {cost_ratio:.2f}x (${current_cost_usd:.4f} / ${baseline_cost_usd:.4f}), "
                            f"threshold: {self.config.max_cost_drift_ratio:.2f}x"
                        ),
                    )
                )

        # 5. Staleness Drift
        if self.config.max_staleness_ratio is not None:
            if total_reviews_count == 0:
                signals.append(
                    DriftSignal(
                        signal_type="staleness_drift",
                        baseline_value=None,
                        current_value=None,
                        threshold=self.config.max_staleness_ratio,
                        exceeded=False,
                        is_evaluable=False,
                        details="No reviews in observation period to evaluate staleness ratio",
                    )
                )
            else:
                stale_ratio = stale_reviews_count / total_reviews_count
                exceeded = stale_ratio > self.config.max_staleness_ratio
                signals.append(
                    DriftSignal(
                        signal_type="staleness_drift",
                        baseline_value=0.0,
                        current_value=round(stale_ratio, 4),
                        threshold=self.config.max_staleness_ratio,
                        exceeded=exceeded,
                        is_evaluable=True,
                        details=f"Stale review ratio is {stale_ratio:.4f} ({stale_reviews_count}/{total_reviews_count}), threshold: {self.config.max_staleness_ratio:.4f}",
                    )
                )

        is_drift = any(s.exceeded for s in signals)
        return DriftReport(
            repository_id=repository_id,
            evaluated_at=eval_time,
            baseline_window_records=len(baseline_feedback),
            current_window_records=len(current_feedback),
            is_drift_detected=is_drift,
            signals=tuple(signals),
        )


@dataclass(frozen=True)
class PolicyPromotionRecord:
    """Immutable audit record for a policy promotion or rollback event (FR-20, NFR-09)."""

    record_id: str
    policy_version: str
    action: str  # "promoted" | "rejected" | "rolled_back"
    holdout_passed: bool
    human_actor: str | None
    reason: str
    timestamp: float


class PolicyPromotionManager:
    """Manages policy version promotion and auditable rollback controls (FR-20, AC-12)."""

    def __init__(self, connection: sqlite3.Connection, initial_active_version: str = "1.0") -> None:
        self.connection = connection
        self._init_schema()
        self._active_version = self._recover_active_version(initial_active_version)

    def _init_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_promotions (
                    record_id TEXT PRIMARY KEY,
                    policy_version TEXT NOT NULL,
                    action TEXT NOT NULL,
                    holdout_passed INTEGER NOT NULL,
                    human_actor TEXT,
                    reason TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_promotions_time
                ON policy_promotions (timestamp ASC)
                """
            )

    def _recover_active_version(self, fallback: str) -> str:
        row = self.connection.execute(
            """
            SELECT policy_version
            FROM policy_promotions
            WHERE action IN ('promoted', 'rolled_back')
            ORDER BY timestamp DESC, rowid DESC
            LIMIT 1
            """
        ).fetchone()
        if row and row[0]:
            return str(row[0])
        return fallback

    @property
    def active_policy_version(self) -> str:
        return self._active_version

    def promote_policy_version(
        self,
        candidate_version: str,
        holdout_gate_result: PromotionGateResult,
        *,
        human_approver: str,
        reason: str,
        now: float | None = None,
    ) -> PolicyPromotionRecord:
        """Promote candidate policy version to active production.

        Invariants:
        1. Holdout regression gate must pass (AC-12). Development split results cannot promote.
        2. Explicit non-empty human approval is mandatory (FR-20, SPEC Non-Goals).
        3. Never automatically promote from maintainer feedback directly.
        """
        curr_time = time.time() if now is None else now
        rec_id = f"promo-{candidate_version}-{uuid.uuid4().hex[:12]}"

        if not human_approver or not human_approver.strip():
            rec = PolicyPromotionRecord(
                record_id=rec_id,
                policy_version=candidate_version,
                action="rejected",
                holdout_passed=holdout_gate_result.passed,
                human_actor=None,
                reason="Promotion rejected: explicit human approver is mandatory",
                timestamp=curr_time,
            )
            self._record_action(rec)
            raise ValueError("Explicit human approval is mandatory for policy promotion")

        if holdout_gate_result.dataset_split != DatasetSplit.HOLDOUT or not holdout_gate_result.passed:
            failed_str = ", ".join(holdout_gate_result.failed_gates)
            rec = PolicyPromotionRecord(
                record_id=rec_id,
                policy_version=candidate_version,
                action="rejected",
                holdout_passed=False,
                human_actor=human_approver,
                reason=f"Promotion rejected: holdout regression gate failed on [{failed_str}]",
                timestamp=curr_time,
            )
            self._record_action(rec)
            raise ValueError(
                f"Cannot promote policy version '{candidate_version}': holdout regression gate failed on [{failed_str}]"
            )

        rec = PolicyPromotionRecord(
            record_id=rec_id,
            policy_version=candidate_version,
            action="promoted",
            holdout_passed=True,
            human_actor=human_approver,
            reason=reason,
            timestamp=curr_time,
        )
        self._record_action(rec)
        self._active_version = candidate_version
        return rec

    def rollback_policy_version(
        self,
        target_version: str,
        *,
        human_actor: str,
        reason: str,
        now: float | None = None,
    ) -> PolicyPromotionRecord:
        """Rollback production policy to a previously active promoted version (FR-20)."""
        curr_time = time.time() if now is None else now
        rec_id = f"rollback-{target_version}-{uuid.uuid4().hex[:12]}"

        if not human_actor or not human_actor.strip():
            raise ValueError("Human actor is required for policy rollback")

        if not reason or not reason.strip():
            raise ValueError("Reason is required for policy rollback")

        # Verify the target version has an existing successful promoted record
        row = self.connection.execute(
            """
            SELECT 1 FROM policy_promotions
            WHERE policy_version = ? AND action = 'promoted' AND holdout_passed = 1
            LIMIT 1
            """,
            (target_version,),
        ).fetchone()
        if not row:
            raise ValueError(
                f"Cannot rollback to version '{target_version}': target version was never successfully promoted"
            )

        rec = PolicyPromotionRecord(
            record_id=rec_id,
            policy_version=target_version,
            action="rolled_back",
            holdout_passed=True,
            human_actor=human_actor,
            reason=reason,
            timestamp=curr_time,
        )
        self._record_action(rec)
        self._active_version = target_version
        return rec

    def _record_action(self, rec: PolicyPromotionRecord) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO policy_promotions (
                    record_id, policy_version, action, holdout_passed,
                    human_actor, reason, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.record_id,
                    rec.policy_version,
                    rec.action,
                    1 if rec.holdout_passed else 0,
                    rec.human_actor,
                    rec.reason,
                    rec.timestamp,
                ),
            )

    def get_history(self) -> list[PolicyPromotionRecord]:
        """Query policy promotion and rollback audit trail."""
        rows = self.connection.execute(
            """
            SELECT record_id, policy_version, action, holdout_passed,
                   human_actor, reason, timestamp
            FROM policy_promotions
            ORDER BY timestamp ASC
            """
        ).fetchall()
        return [
            PolicyPromotionRecord(
                record_id=r[0],
                policy_version=r[1],
                action=r[2],
                holdout_passed=bool(r[3]),
                human_actor=r[4],
                reason=r[5],
                timestamp=r[6],
            )
            for r in rows
        ]
