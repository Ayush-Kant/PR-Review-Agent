"""Controlled, opt-in live integration test harness for PR-Review-Agent.

This script executes a live end-to-end review on a real GitHub repository PR using
external network adapters (GitHub REST API and external LLM provider).

SAFETY INVARIANTS:
- Strictly OPT-IN: Will refuse to run unless ENABLE_LIVE_GITHUB_TEST=1 is set.
- NEVER runs as part of pytest or automated regressions.
- DRY-RUN by default: Will NOT publish comments to GitHub unless PUBLISH_LIVE_REVIEW=1 is explicitly set.
- Enforces strict no-merge / no-branch-modification invariants.
- Secrets are resolved strictly via SecurityConfig.

CONFIGURATION ENVIRONMENT VARIABLES:
- ENABLE_LIVE_GITHUB_TEST : Must be set to '1' to enable execution.
- GITHUB_TOKEN            : GitHub Personal Access Token or App Token with pull_requests:read
                            (and pull_requests:write only if PUBLISH_LIVE_REVIEW=1).
- GITHUB_REPOSITORY       : Target repository in 'owner/repo' format.
- GITHUB_PR_NUMBER        : Target pull request number.
- MODEL_PROVIDER          : 'openai' or 'groq' (defaults to 'openai').
- MODEL_NAME              : Model name override (defaults to provider default).
- OPENAI_API_KEY          : Required if MODEL_PROVIDER is 'openai'.
- GROQ_API_KEY            : Required if MODEL_PROVIDER is 'groq'.
- PUBLISH_LIVE_REVIEW     : Set to '1' to publish review comments to GitHub (defaults to 0, dry-run).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

from pr_review_agent.adapters.github import GitHubNetworkClient
from pr_review_agent.adapters.llm import LLMSpecialistAdapter
from pr_review_agent.cost_controls import ProviderPricingRegistry
from pr_review_agent.github_output import GitHubReviewPublisher, PublicationStatus
from pr_review_agent.observability import AuditEvent, AuditSpine
from pr_review_agent.orchestration import (
    ReviewOrchestrator,
    SpecialistInput,
    SpecialistStatus,
    SpecialistType,
)
from pr_review_agent.policy import (
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


def run_live_review() -> int:
    """Execute live review flow if opted in via environment."""
    if os.environ.get("ENABLE_LIVE_GITHUB_TEST") != "1":
        print("INFO: Live integration harness is disabled by default.")
        print("To run against a live GitHub PR, set:")
        print("  ENABLE_LIVE_GITHUB_TEST=1")
        print("  GITHUB_TOKEN=<token>")
        print("  GITHUB_REPOSITORY=<owner/repo>")
        print("  GITHUB_PR_NUMBER=<pr_number>")
        print("  MODEL_PROVIDER=openai|groq")
        print("  OPENAI_API_KEY=<key> or GROQ_API_KEY=<key>")
        print("  PUBLISH_LIVE_REVIEW=0 (dry run) or 1 (publish)")
        return 0

    repo_id = os.environ.get("GITHUB_REPOSITORY", "").strip()
    pr_num_str = os.environ.get("GITHUB_PR_NUMBER", "").strip()
    provider = os.environ.get("MODEL_PROVIDER", "openai").strip().lower()
    model_name = os.environ.get("MODEL_NAME")
    publish_live = os.environ.get("PUBLISH_LIVE_REVIEW") == "1"

    if not repo_id or not pr_num_str or not pr_num_str.isdigit():
        print("ERROR: GITHUB_REPOSITORY and valid GITHUB_PR_NUMBER are required.")
        return 1

    pr_number = int(pr_num_str)
    tenant = repo_id.split("/")[0] if "/" in repo_id else repo_id

    secret_registry = RuntimeSecretRegistry()
    sec_config = SecurityConfig(
        authorized_tenant=tenant,
        authorized_repositories=[repo_id],
        secret_registry=secret_registry,
        env_provider=os.environ.get,
    )

    print(f"[*] Connecting to GitHub repository {repo_id} (PR #{pr_number})...")
    github_client = GitHubNetworkClient(sec_config)

    try:
        pr_metadata = github_client.get_pull_request(repo_id, pr_number)
        head_sha = str(pr_metadata["head"]["sha"])
        base_sha = str(pr_metadata["base"]["sha"])
        title = str(pr_metadata.get("title", ""))
        print(f"[*] Retrieved PR #{pr_number}: '{title}' (head: {head_sha[:8]}, base: {base_sha[:8]})")

        diff_content = github_client.get_pull_request_diff(repo_id, pr_number)
        print(f"[*] Retrieved diff: {len(diff_content.encode('utf-8'))} bytes")

        db_path = os.environ.get("DATABASE_PATH", "pr_review_agent.db").strip()
        print(f"[*] Connecting to Review Truth and audit database: {db_path}")
        conn = sqlite3.connect(db_path)
        audit_spine = AuditSpine(conn)
        truth_store = ReviewTruthStore(conn)

        print(f"[*] Initializing {provider.upper()} specialist adapter...")
        pricing = ProviderPricingRegistry()
        specialist_adapter = LLMSpecialistAdapter(
            provider=provider,
            security_config=sec_config,
            model=model_name,
            pricing_registry=pricing,
            audit_spine=audit_spine,
        )

        # Run all four specialist evaluations: Security, Quality, Tests, Documentation
        all_candidate_findings = []
        all_specialist_types = (
            SpecialistType.SECURITY,
            SpecialistType.QUALITY,
            SpecialistType.TESTS,
            SpecialistType.DOCUMENTATION,
        )
        succeeded_specialists: list[SpecialistType] = []
        degraded_specialists: list[SpecialistType] = []
        failed_specialists: list[SpecialistType] = []
        timed_out_specialists: list[SpecialistType] = []
        skipped_specialists: list[SpecialistType] = []
        failure_reasons: dict[str, str] = {}

        run_corr_id = f"live-pr-{pr_number}-{int(time.time())}"

        for spec_type in all_specialist_types:
            print(f"[*] Running {spec_type.value.upper()} specialist...")
            if spec_type == SpecialistType.SECURITY:
                instructions = (
                    f"Analyze PR #{pr_number} changes for genuine security defects with evidence.\n"
                    "Security Review Guidelines:\n"
                    "- Generic use of random, non-cryptographic hashing, or debug logging is NOT automatically a security vulnerability.\n"
                    "- A security finding requires evidence that the behavior is security-sensitive in context.\n"
                    "- For randomness specifically, classify it as a security issue only when the code path is used for something security-sensitive such as: "
                    "authentication secrets, password reset tokens, session identifiers, CSRF tokens, cryptographic material, authorization/security tokens, or other explicitly security-sensitive values.\n"
                    "- If the code clearly uses randomness for a benign identifier, demo/test value, display value, sampling, non-security ID, etc., do not emit a high/medium security vulnerability finding merely because the API is non-cryptographic.\n"
                    "- When the context is ambiguous, prefer omission or a lower-severity informational/quality observation rather than an unsupported security claim.\n"
                    "- Never suppress a genuine security issue when surrounding code/evidence establishes security-sensitive use."
                )
            else:
                instructions = f"Analyze PR #{pr_number} changes for genuine {spec_type.value} defects with evidence."

            spec_input = SpecialistInput(
                specialist_type=spec_type,
                correlation_id=run_corr_id,
                instructions=instructions,
                changed_files=(),
                diff_content=diff_content,
                retrieved_evidence=(),
                head_sha=head_sha,
            )
            spec_output = specialist_adapter(spec_input)
            print(f"    Status: {spec_output.status}, Findings: {len(spec_output.findings)}")
            if spec_output.usage and spec_output.usage.total_tokens:
                print(f"    Tokens: {spec_output.usage.total_tokens}, Cost: ${spec_output.usage.cost_usd or 0:.4f}")

            if spec_output.status in (SpecialistStatus.COMPLETED.value, "completed"):
                succeeded_specialists.append(spec_type)
                all_candidate_findings.extend(spec_output.findings)
            elif spec_output.status in (SpecialistStatus.DEGRADED.value, "degraded"):
                degraded_specialists.append(spec_type)
                all_candidate_findings.extend(spec_output.findings)
                if spec_output.error_message:
                    failure_reasons[spec_type.value] = spec_output.error_message
            elif spec_output.status in (SpecialistStatus.TIMEOUT.value, "timeout"):
                timed_out_specialists.append(spec_type)
                failure_reasons[spec_type.value] = spec_output.error_message or "Execution timed out"
            elif spec_output.status in (SpecialistStatus.SKIPPED.value, "skipped"):
                skipped_specialists.append(spec_type)
                failure_reasons[spec_type.value] = spec_output.error_message or "Specialist skipped"
            else:
                failed_specialists.append(spec_type)
                failure_reasons[spec_type.value] = spec_output.error_message or "Execution failed"

        # Explicit coverage reporting
        total_count = len(all_specialist_types)
        succeeded_count = len(succeeded_specialists)
        is_degraded = (succeeded_count < total_count) or bool(
            degraded_specialists or failed_specialists or timed_out_specialists or skipped_specialists
        )
        print(f"\n[*] Coverage: {succeeded_count}/{total_count} specialists")
        if succeeded_specialists:
            print(f"    Succeeded: {', '.join(s.value.upper() for s in succeeded_specialists)}")
        if degraded_specialists:
            print(f"    Degraded:  {', '.join(s.value.upper() for s in degraded_specialists)}")
        if failed_specialists:
            print(f"    Failed:    {', '.join(s.value.upper() for s in failed_specialists)}")
        if timed_out_specialists:
            print(f"    Timed out: {', '.join(s.value.upper() for s in timed_out_specialists)}")
        if skipped_specialists:
            print(f"    Skipped:   {', '.join(s.value.upper() for s in skipped_specialists)}")
        if failure_reasons:
            print(f"    Diagnostics: {failure_reasons}")

        if is_degraded:
            print(f"[!] WARNING: Review coverage is degraded ({succeeded_count}/{total_count} specialists succeeded).")
            audit_spine.record_event(
                AuditEvent(
                    correlation_id=run_corr_id,
                    event_name="specialist_coverage_degraded",
                    step="coverage_evaluation",
                    timestamp=time.time(),
                    details={
                        "is_full_coverage": False,
                        "coverage_ratio": round(succeeded_count / total_count, 4),
                        "succeeded": [s.value for s in succeeded_specialists],
                        "degraded": [s.value for s in degraded_specialists],
                        "failed": [s.value for s in failed_specialists],
                        "timed_out": [s.value for s in timed_out_specialists],
                        "skipped": [s.value for s in skipped_specialists],
                        "failure_reasons": failure_reasons,
                    },
                )
            )

        if succeeded_count == 0 and not degraded_specialists:
            print("[!] ERROR: All specialists failed to execute. Review run failed.")
            conn.close()
            return 1

        # Aggregate candidate findings
        aggregator = FindingAggregator()
        canonical_findings = aggregator.aggregate(
            all_candidate_findings,
            repository_id=repo_id,
            head_sha=head_sha,
        )
        print(f"[*] Aggregated {len(canonical_findings)} canonical findings.")

        policy_engine = ReviewPolicyEngine(
            policy_version="v1",
            min_auto_approve_confidence=0.8,
            auto_approvable_severities=("low", "medium", "info"),
        )

        for finding in canonical_findings:
            evaluated = policy_engine.evaluate(finding, is_fresh=True)
            initial_state = (
                TruthState.AUTO_APPROVED
                if evaluated.disposition == FindingDisposition.AUTO_APPROVED
                else TruthState.HELD
            )
            truth_store.record_initial(evaluated, initial_state=initial_state)
            print(f"    - Finding: [{evaluated.severity.upper()}] {evaluated.summary} -> {evaluated.disposition.value}")

        # Publication
        if publish_live:
            print("[!] PUBLISH_LIVE_REVIEW=1 enabled: Publishing findings to GitHub...")
            publisher = GitHubReviewPublisher(conn, truth_store, github_client)
            for finding in canonical_findings:
                latest = truth_store.get_latest_state(finding.canonical_id)
                if latest and latest.state == TruthState.AUTO_APPROVED:
                    res = publisher.publish_finding(finding, pull_number=pr_number, diff_content=diff_content)
                    print(f"    Published {finding.canonical_id}: status={res.status.value}, review_id={res.review_id}")
        else:
            print("[*] DRY RUN: Publication skipped. Set PUBLISH_LIVE_REVIEW=1 to publish comments.")

        conn.commit()
        print("[+] Live integration execution completed successfully.")
        return 0

    finally:
        if "conn" in locals() and conn is not None:
            conn.close()
        if "github_client" in locals() and hasattr(github_client, "close"):
            github_client.close()


if __name__ == "__main__":
    sys.exit(run_live_review())
