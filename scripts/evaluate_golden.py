#!/usr/bin/env python3
"""Golden PR benchmark evaluation runner (supporting both offline mock and live LLM modes).

Evaluates the PR Reviewer finding generation, severity calibration, and specialist coverage against
the versioned golden benchmark dataset (data/golden_prs_v1.json).

Usage:
    # Offline mock evaluation (fast, deterministic, zero network credentials)
    python scripts/evaluate_golden.py
    python scripts/evaluate_golden.py --split holdout --gate

    # Live controlled DRY-RUN evaluation (executes real LLM specialists, zero GitHub writes)
    python scripts/evaluate_golden.py --mode live --provider groq
    python scripts/evaluate_golden_live.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pr_review_agent.adapters.llm import create_specialist_handlers
from pr_review_agent.cost_controls import ProviderPricingRegistry
from pr_review_agent.evaluation import (
    DatasetSplit,
    EvaluationRunner,
    GoldenPRCase,
    GoldenPRDataset,
    LiveGoldenEvaluator,
    LiveGoldenReport,
    PromotionGateEvaluator,
    RegressionGateConfig,
    load_golden_dataset,
)
from pr_review_agent.observability import AuditSpine
from pr_review_agent.orchestration import (
    CandidateFinding,
    ReviewOrchestrator,
    SpecialistType,
)
from pr_review_agent.policy import (
    CanonicalFinding,
    FindingAggregator,
    ReviewPolicyEngine,
    ReviewTruthStore,
    SeverityCalibrator,
)
from pr_review_agent.security import (
    RuntimeSecretRegistry,
    SecretLeakageScanner,
    SecurityConfig,
)


def resolve_runtime_env_keys() -> None:
    """Ensure runtime API keys from Windows user environment or registry are visible in os.environ."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
            for var_name in ("GROQ_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN"):
                if not os.environ.get(var_name):
                    try:
                        val, _ = winreg.QueryValueEx(key, var_name)
                        if val:
                            os.environ[var_name] = str(val).strip()
                    except FileNotFoundError:
                        pass
        except Exception:
            pass


def _specialist_for_category(category: str) -> SpecialistType:
    cat = category.lower()
    if "sec" in cat:
        return SpecialistType.SECURITY
    if "test" in cat:
        return SpecialistType.TESTS
    if "doc" in cat:
        return SpecialistType.DOCUMENTATION
    return SpecialistType.QUALITY


def generate_mock_findings_for_case(case: GoldenPRCase) -> list[CanonicalFinding]:
    """Generate simulated specialist raw findings and run them through deterministic calibration."""
    candidates: list[CandidateFinding] = []

    for i, exp in enumerate(case.expected_findings):
        # Simulate realistic model outputs, including pre-calibration severity inflation
        simulated_raw_sev = exp.severity
        notes = (exp.notes or "").lower()
        summary = exp.summary

        # Simulate raw model tending to over-inflate quality and doc defects to 'high'
        if "mutable default" in notes or "mutable default" in summary.lower():
            simulated_raw_sev = "high"
        elif "contradict" in notes or "contradict" in summary.lower():
            simulated_raw_sev = "high"
        elif "missing test" in notes or "smoke test" in summary.lower():
            simulated_raw_sev = "medium"

        cand = CandidateFinding(
            finding_id=f"sim-{case.case_id}-{i}",
            correlation_id=case.case_id,
            specialist_type=_specialist_for_category(exp.category),
            category=exp.category,
            severity=simulated_raw_sev,
            confidence=0.92,
            summary=exp.summary,
            rationale=f"Simulated rationale for {exp.category} finding.",
            file_path=exp.file_path,
            line_range=exp.line_range,
            evidence_refs=(f"diff-ref-{exp.file_path}:{exp.line_range}",),
            remediation=f"Remediate {exp.category} finding per repository guidelines.",
        )
        candidates.append(cand)

    # Run raw findings through FindingAggregator to test real aggregation & calibration
    aggregator = FindingAggregator()
    canonical = aggregator.aggregate(
        candidates,
        repository_id=case.repository_id,
        head_sha=case.head_sha,
    )
    return canonical


def run_evaluation(
    dataset_path: str | Path = "data/golden_prs_v1.json",
    split_str: str = "development",
    run_gate: bool = False,
    mode: str = "mock",
    provider: str = "groq",
    model_name: str | None = None,
    output_report: str | None = None,
    output_markdown: str | None = None,
    case_filter: list[str] | None = None,
) -> int:
    """Run golden evaluation (mock or live) and output metrics scorecard."""
    resolve_runtime_env_keys()

    print("=" * 75)
    print(f"REVIEWER QUALITY CALIBRATION & GOLDEN BENCHMARK ({mode.upper()} MODE)")
    print("=" * 75)

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        print(f"Error: Golden dataset not found at {dataset_file}", file=sys.stderr)
        return 1

    dataset = load_golden_dataset(dataset_file)
    target_split = DatasetSplit.HOLDOUT if split_str.lower() == "holdout" else DatasetSplit.DEVELOPMENT
    cases = dataset.get_cases(target_split)
    if case_filter:
        target_ids = set(case_filter)
        cases = tuple(c for c in cases if c.case_id in target_ids)

    print(f"Dataset:       {dataset.dataset_id} (version {dataset.version})")
    print(f"Split:         {target_split.value.upper()} ({len(cases)} cases evaluated)")
    print(f"Execution:     {'LIVE DRY-RUN (Real LLM Specialist Pipeline)' if mode == 'live' else 'OFFLINE MOCK (Deterministic Simulation)'}")
    print("-" * 75)

    if mode == "live":
        clean_provider = provider.strip().lower()
        active_model = model_name or ("openai/gpt-oss-120b" if clean_provider == "groq" else "gpt-4o-mini")
        print(f"Provider:      {clean_provider.upper()} (model: {active_model})")

        key_var = "GROQ_API_KEY" if clean_provider == "groq" else "OPENAI_API_KEY"
        if not os.environ.get(key_var):
            print(f"\n[!] ERROR: Live evaluation requires {key_var} in environment or Windows user profile.", file=sys.stderr)
            print("Set the key or run with '--mode mock' for offline evaluation.", file=sys.stderr)
            return 1

        db_path = "golden_eval_review.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        audit_spine = AuditSpine(conn)
        truth_store = ReviewTruthStore(conn)

        sec_registry = RuntimeSecretRegistry()
        sec_config = SecurityConfig(
            authorized_tenant="Ayush-Kant",
            authorized_repositories=["Ayush-Kant/PR-Review-Agent"],
            secret_registry=sec_registry,
            env_provider=os.environ.get,
        )

        handlers = create_specialist_handlers(
            provider=clean_provider,
            security_config=sec_config,
            model=active_model,
            audit_spine=audit_spine,
            max_retries=5,
            retry_backoff_seconds=2.0,
            max_retry_backoff_seconds=15.0,
        )

        default_instructions = {
            SpecialistType.SECURITY: (
                "Review code changes for security vulnerabilities, authentication/authorization bypasses, data exposure, and unsafe cryptographic use.\n"
                "Security Review Guidelines:\n"
                "- Generic use of random, non-cryptographic hashing, or debug logging is NOT automatically a security vulnerability.\n"
                "- A security finding requires evidence that the behavior is security-sensitive in context.\n"
                "- For randomness specifically, classify it as a security issue only when the code path is used for something security-sensitive such as: "
                "authentication secrets, password reset tokens, session identifiers, CSRF tokens, cryptographic material, authorization/security tokens, or other explicitly security-sensitive values.\n"
                "- If the code clearly uses randomness for a benign identifier, demo/test value, display value, sampling, non-security ID, etc., do not emit a high/medium security vulnerability finding merely because the API is non-cryptographic.\n"
                "- When the context is ambiguous, prefer omission or a lower-severity informational/quality observation rather than an unsupported security claim.\n"
                "- Never suppress a genuine security issue when surrounding code/evidence establishes security-sensitive use."
            ),
            SpecialistType.QUALITY: (
                "Review code changes for correctness bugs, runtime exceptions, logic flaws, resource leaks, and code quality issues.\n"
                "Severity Guidelines:\n"
                "- Genuine runtime crashes, unhandled exceptions, and data corruption may be HIGH severity.\n"
                "- Code smells, maintainability concerns, and mutable default arguments should normally be MEDIUM or LOW severity."
            ),
            SpecialistType.TESTS: (
                "Review code changes for test coverage gaps, missing regression tests, edge case omissions, and ineffective assertions.\n"
                "Severity Guidelines:\n"
                "- Ineffective assertions or missing smoke test assertions should normally be LOW or MEDIUM severity."
            ),
            SpecialistType.DOCUMENTATION: (
                "Review documentation and docstrings for accuracy against code changes, misleading statements, and missing documentation.\n"
                "Severity Guidelines:\n"
                "- Documentation discrepancies are normally LOW severity, or MEDIUM if they materially contradict runtime behavior or public APIs. They must not be HIGH."
            ),
        }

        pricing = ProviderPricingRegistry()
        orchestrator = ReviewOrchestrator(
            specialist_handlers=handlers,
            default_instructions=default_instructions,
            pricing_registry=pricing,
        )

        live_evaluator = LiveGoldenEvaluator(
            orchestrator=orchestrator,
            audit_spine=audit_spine,
            truth_store=truth_store,
            provider=clean_provider,
            model=active_model,
        )

        gate_cfg = None
        if run_gate:
            gate_cfg = RegressionGateConfig(
                min_precision=0.80,
                min_recall=0.80,
                min_f1=0.80,
                min_critical_finding_recall=1.0,
                min_severity_exact_match_rate=0.70,
                max_mean_severity_distance=0.50,
                max_over_severity_rate=0.25,
            )

        print("[*] Executing live specialist evaluation across golden cases...")
        report: LiveGoldenReport = asyncio.run(
            live_evaluator.evaluate_dataset(
                dataset=dataset,
                split=target_split,
                gate_config=gate_cfg,
                case_filter=case_filter,
            )
        )
        metrics = report.metrics

        # Print case breakdown
        print(f"\n{'Case ID':<38} {'Coverage':<10} {'Exp':<5} {'Act':<5} {'Raw->Calibrated Severity'}")
        print("-" * 75)
        for outcome in report.case_outcomes:
            cov_str = "Full" if not outcome.is_degraded else "Degraded"
            sev_transitions = ", ".join(f"{f.raw_severity}->{f.calibrated_severity}" for f in outcome.canonical_findings) or "none"
            case_obj = next((c for c in cases if c.case_id == outcome.case_id), None)
            exp_count = len(case_obj.expected_findings) if case_obj else 0
            print(f"{outcome.case_id:<38} {cov_str:<10} {exp_count:<5} {len(outcome.canonical_findings):<5} {sev_transitions}")

        # Scan and persist artifacts
        scanner = SecretLeakageScanner(sec_registry)
        rep_json = str(report.to_dict())
        rep_md = report.to_markdown()

        scan_json = scanner.scan_text(rep_json)
        scan_md = scanner.scan_text(rep_md)
        if scan_json.has_secret or scan_md.has_secret:
            print("\n[!] CRITICAL SECURITY ALERT: Secret detected in evaluation report artifact!", file=sys.stderr)
            return 1

        out_json_path = Path(output_report or "artifacts/live_golden_report.json")
        out_md_path = Path(output_markdown or "artifacts/live_golden_report.md")
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

        import json as json_lib
        out_json_path.write_text(json_lib.dumps(report.to_dict(), indent=2), encoding="utf-8")
        out_md_path.write_text(rep_md, encoding="utf-8")
        print(f"\n[+] Artifacts saved:\n    - JSON:     {out_json_path}\n    - Markdown: {out_md_path}")
        print("    - Security: Verified 0 leaked secrets / credentials in artifacts.")

    else:
        # Mock / Offline mode
        candidate_findings_by_case: dict[str, list[CanonicalFinding]] = {}
        for case in cases:
            findings = generate_mock_findings_for_case(case)
            candidate_findings_by_case[case.case_id] = findings

        runner = EvaluationRunner()
        metrics = runner.evaluate_dataset(
            dataset=dataset,
            candidate_findings_by_case=candidate_findings_by_case,
            split=target_split,
        )

        print(f"{'Case ID':<42} {'Exp':<5} {'Act':<5} {'Calibrated Rules'}")
        print("-" * 75)
        for case in cases:
            findings = candidate_findings_by_case.get(case.case_id, [])
            rules = ", ".join(f.calibration_rule for f in findings if f.calibration_rule) or "none"
            print(f"{case.case_id:<42} {len(case.expected_findings):<5} {len(findings):<5} {rules}")

    print("=" * 75)
    print("EVALUATION METRICS SCORECARD")
    print("=" * 75)
    print(f"Total Cases:                   {metrics.total_cases}")
    print(f"Expected Findings (Required):  {metrics.expected_required_count}")
    print(f"Candidate Findings Detected:   {metrics.candidate_findings_count}")
    print(f"True Positives (TP):           {metrics.true_positives}")
    print(f"False Positives (FP):          {metrics.false_positives}")
    print(f"False Negatives (FN):          {metrics.false_negatives}")
    print(f"Precision:                     {metrics.precision:.4f}")
    print(f"Recall:                        {metrics.recall:.4f}")
    print(f"F1 Score:                      {metrics.f1_score:.4f}")
    print(f"Critical Finding Recall:       {metrics.critical_finding_recall:.4f}")
    print("-" * 75)
    print("SEVERITY CALIBRATION METRICS")
    print("-" * 75)
    print(f"Severity Exact-Match Rate:     {metrics.severity_exact_match_rate:.4f}")
    print(f"Mean Severity Distance:        {metrics.mean_severity_distance:.4f}")
    print(f"Over-Severity Rate:            {metrics.over_severity_rate:.4f}")
    print(f"Under-Severity Rate:           {metrics.under_severity_rate:.4f}")
    print(f"One-Tier Deviation Rate:       {metrics.one_tier_deviation_rate:.4f}")
    print(f"Major Deviation Rate:          {metrics.major_deviation_rate:.4f}")
    print("=" * 75)

    if run_gate and mode == "mock":
        print("REGRESSION GATE EVALUATION")
        print("-" * 75)
        gate_config = RegressionGateConfig(
            min_precision=0.80,
            min_recall=0.80,
            min_f1=0.80,
            min_critical_finding_recall=1.0,
            min_severity_exact_match_rate=0.70,
            max_mean_severity_distance=0.50,
            max_over_severity_rate=0.20,
        )
        gate_evaluator = PromotionGateEvaluator()
        result = gate_evaluator.evaluate_gate(
            candidate_version=dataset.version,
            metrics=metrics,
            config=gate_config,
        )
        if result.passed:
            print("Promotion Gate: PASSED (All thresholds satisfied)")
        else:
            print("Promotion Gate: FAILED")
            for reason in result.reasons:
                print(f"  - {reason}")
            return 1

    print("Status: BENCHMARK RUN COMPLETED SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate golden PR benchmarks.")
    parser.add_argument("--dataset", default="data/golden_prs_v1.json", help="Path to golden dataset JSON")
    parser.add_argument("--split", default="development", choices=["development", "holdout"], help="Dataset split")
    parser.add_argument("--mode", default="mock", choices=["mock", "live"], help="Execution mode (mock or live)")
    parser.add_argument("--provider", default="groq", choices=["groq", "openai"], help="Model provider for live mode")
    parser.add_argument("--model", default=None, help="Model name override for live mode")
    parser.add_argument("--gate", action="store_true", help="Evaluate regression promotion gate")
    parser.add_argument("--output-report", default=None, help="Output path for JSON evaluation report")
    parser.add_argument("--output-markdown", default=None, help="Output path for Markdown evaluation report")
    parser.add_argument("--cases", default=None, help="Comma-separated list of case IDs to evaluate")
    args = parser.parse_args()

    case_filter = [c.strip() for c in args.cases.split(",")] if args.cases else None

    exit_code = run_evaluation(
        dataset_path=args.dataset,
        split_str=args.split,
        run_gate=args.gate,
        mode=args.mode,
        provider=args.provider,
        model_name=args.model,
        output_report=args.output_report,
        output_markdown=args.output_markdown,
        case_filter=case_filter,
    )
    sys.exit(exit_code)
