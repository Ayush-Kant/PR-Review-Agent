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
from pr_review_agent.orchestration import (
    ReviewOrchestrator,
    SpecialistInput,
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

        print(f"[*] Initializing {provider.upper()} specialist adapter...")
        pricing = ProviderPricingRegistry()
        specialist_adapter = LLMSpecialistAdapter(
            provider=provider,
            security_config=sec_config,
            model=model_name,
            pricing_registry=pricing,
        )

        # Run Security and Quality specialist evaluations
        all_candidate_findings = []
        for spec_type in (SpecialistType.SECURITY, SpecialistType.QUALITY):
            print(f"[*] Running {spec_type.value.upper()} specialist...")
            spec_input = SpecialistInput(
                specialist_type=spec_type,
                correlation_id=f"live-pr-{pr_number}-{int(time.time())}",
                instructions=f"Analyze PR #{pr_number} changes for genuine {spec_type.value} defects with evidence.",
                changed_files=(),
                diff_content=diff_content,
                retrieved_evidence=(),
                head_sha=head_sha,
            )
            spec_output = specialist_adapter(spec_input)
            print(f"    Status: {spec_output.status}, Findings: {len(spec_output.findings)}")
            if spec_output.usage and spec_output.usage.total_tokens:
                print(f"    Tokens: {spec_output.usage.total_tokens}, Cost: ${spec_output.usage.cost_usd or 0:.4f}")
            all_candidate_findings.extend(spec_output.findings)

        # Aggregate candidate findings
        aggregator = FindingAggregator()
        canonical_findings = aggregator.aggregate(
            all_candidate_findings,
            repository_id=repo_id,
            head_sha=head_sha,
        )
        print(f"[*] Aggregated {len(canonical_findings)} canonical findings.")

        # Policy evaluation and Review Truth persistence
        conn = sqlite3.connect(":memory:")
        truth_store = ReviewTruthStore(conn)
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

        conn.close()
        print("[+] Live integration execution completed successfully.")
        return 0

    finally:
        github_client.close()


if __name__ == "__main__":
    sys.exit(run_live_review())
