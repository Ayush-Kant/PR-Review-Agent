"""Tests for policy-permitted, current-SHA-safe, idempotent GitHub reviews and inline findings."""

from __future__ import annotations

import sqlite3
import pytest

from pr_review_agent.github_output import (
    FakeGitHubClient,
    GitHubReviewPublisher,
    PublicationResult,
    PublicationStatus,
)
from pr_review_agent.policy import (
    CanonicalFinding,
    FindingDisposition,
    ReviewTruthStore,
    TruthState,
)


def _make_canonical(
    canonical_id: str = "can-test-1",
    repository_id: str = "owner/repo",
    head_sha: str = "sha-reviewed-123",
    file_path: str = "src/calculator.py",
    line_range: tuple[int, int] = (10, 15),
    category: str = "correctness",
    severity: str = "medium",
    confidence: float = 0.90,
    summary: str = "Potential off-by-one error",
    rationale: str = "Loop boundary condition allows indexing out of bounds.",
    evidence_refs: tuple[str, ...] = ("diff://src/calculator.py#L10-L15",),
    remediation: str | None = "Use < instead of <= in range boundary.",
) -> CanonicalFinding:
    return CanonicalFinding(
        canonical_id=canonical_id,
        repository_id=repository_id,
        head_sha=head_sha,
        category=category,
        severity=severity,
        confidence=confidence,
        summary=summary,
        rationale=rationale,
        file_path=file_path,
        line_range=line_range,
        contributing_candidate_ids=("cand-1",),
        contributing_specialists=("quality",),
        evidence_refs=evidence_refs,
        remediation=remediation,
        disposition=FindingDisposition.AUTO_APPROVED,
    )


SAMPLE_DIFF = """diff --git a/src/calculator.py b/src/calculator.py
index e69de29..49e29a1 100644
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -10,10 +10,12 @@ def calculate_sum(items):
+    total = 0
+    for i in range(len(items)):
+        total += items[i]
+    return total
"""


def test_approved_finding_publishes_inline_when_in_diff() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    finding = _make_canonical()
    truth_store.record_initial(finding, initial_state=TruthState.HELD)
    truth_store.record_transition(
        finding.canonical_id,
        TruthState.APPROVED,
        actor="lead_reviewer",
        actor_role="maintainer",
        rationale="Approved for publication",
    )

    result = publisher.publish_finding(
        finding,
        pull_number=42,
        diff_content=SAMPLE_DIFF,
    )

    assert result.status == PublicationStatus.PUBLISHED
    assert result.published_inline is True
    assert result.review_id is not None
    assert result.html_url is not None
    assert "Verified Evidence References" in github_client.reviews[0]["comments"][0]["body"]

    # Verify Review Truth transitioned to PUBLISHED
    latest = truth_store.get_latest_state(finding.canonical_id)
    assert latest is not None
    assert latest.state == TruthState.PUBLISHED


def test_auto_approved_finding_publishes() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    finding = _make_canonical()
    truth_store.record_initial(finding, initial_state=TruthState.AUTO_APPROVED)

    result = publisher.publish_finding(
        finding,
        pull_number=42,
        diff_content=SAMPLE_DIFF,
    )

    assert result.status == PublicationStatus.PUBLISHED
    assert result.published_inline is True
    assert len(github_client.reviews) == 1
    assert truth_store.get_latest_state(finding.canonical_id).state == TruthState.PUBLISHED


def test_held_finding_cannot_publish() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    finding = _make_canonical()
    truth_store.record_initial(finding, initial_state=TruthState.HELD)

    result = publisher.publish_finding(
        finding,
        pull_number=42,
        diff_content=SAMPLE_DIFF,
    )

    assert result.status == PublicationStatus.HELD_OR_UNAUTHORIZED
    assert len(github_client.reviews) == 0
    assert len(github_client.comments) == 0
    assert truth_store.get_latest_state(finding.canonical_id).state == TruthState.HELD


def test_suppressed_finding_cannot_publish() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    finding = _make_canonical()
    truth_store.record_initial(finding, initial_state=TruthState.SUPPRESSED)

    result = publisher.publish_finding(
        finding,
        pull_number=42,
        diff_content=SAMPLE_DIFF,
    )

    assert result.status == PublicationStatus.HELD_OR_UNAUTHORIZED
    assert len(github_client.reviews) == 0
    assert truth_store.get_latest_state(finding.canonical_id).state == TruthState.SUPPRESSED


def test_rejected_or_dismissed_finding_cannot_publish() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    # 1. Rejected
    finding_rej = _make_canonical(canonical_id="can-rej")
    truth_store.record_initial(finding_rej, initial_state=TruthState.HELD)
    truth_store.record_transition(
        "can-rej",
        TruthState.REJECTED,
        actor="maintainer",
        actor_role="maintainer",
        rationale="False positive",
    )
    res_rej = publisher.publish_finding(finding_rej, pull_number=42, diff_content=SAMPLE_DIFF)
    assert res_rej.status == PublicationStatus.HELD_OR_UNAUTHORIZED

    # 2. Dismissed
    finding_dis = _make_canonical(canonical_id="can-dis")
    truth_store.record_initial(finding_dis, initial_state=TruthState.HELD)
    truth_store.record_transition(
        "can-dis",
        TruthState.DISMISSED,
        actor="maintainer",
        actor_role="maintainer",
        rationale="Dismissed by maintainer",
    )
    res_dis = publisher.publish_finding(finding_dis, pull_number=42, diff_content=SAMPLE_DIFF)
    assert res_dis.status == PublicationStatus.HELD_OR_UNAUTHORIZED

    assert len(github_client.reviews) == 0
    assert len(github_client.comments) == 0


def test_head_sha_mismatch_prevents_publication_and_supersedes() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    # Live PR has moved to sha-new-456
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-new-456"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    # Finding was reviewed against sha-reviewed-123
    finding = _make_canonical(head_sha="sha-reviewed-123")
    truth_store.record_initial(finding, initial_state=TruthState.APPROVED)

    result = publisher.publish_finding(
        finding,
        pull_number=42,
        diff_content=SAMPLE_DIFF,
    )

    # Publication refused
    assert result.status == PublicationStatus.SUPERSEDED_SHA_MISMATCH
    assert "Head SHA mismatch" in result.reason
    assert len(github_client.reviews) == 0
    assert len(github_client.comments) == 0

    # Finding marked SUPERSEDED in Review Truth
    latest = truth_store.get_latest_state(finding.canonical_id)
    assert latest is not None
    assert latest.state == TruthState.SUPERSEDED
    assert "Head SHA mismatch" in (latest.rationale or "")


def test_unknown_or_missing_current_sha_prevents_publication() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    # No head SHA configured for PR #99
    github_client = FakeGitHubClient(pr_heads={})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    finding = _make_canonical()
    truth_store.record_initial(finding, initial_state=TruthState.APPROVED)

    result = publisher.publish_finding(
        finding,
        pull_number=99,
        diff_content=SAMPLE_DIFF,
    )

    assert result.status == PublicationStatus.FAILED
    assert "Failed to query live PR head SHA" in result.reason
    assert len(github_client.reviews) == 0


def test_invalid_unrepresentable_inline_position_safely_falls_back() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    # Finding on line 999 (which is outside SAMPLE_DIFF's lines 10-21)
    finding = _make_canonical(
        file_path="src/calculator.py",
        line_range=(999, 1005),
    )
    truth_store.record_initial(finding, initial_state=TruthState.APPROVED)

    result = publisher.publish_finding(
        finding,
        pull_number=42,
        diff_content=SAMPLE_DIFF,
    )

    # Safely published as top-level fallback comment without fabricating line position
    assert result.status == PublicationStatus.PUBLISHED
    assert result.published_inline is False
    assert result.comment_id is not None
    assert len(github_client.reviews) == 0
    assert len(github_client.comments) == 1
    comment = github_client.comments[0]
    assert "Non-inline summary" in comment["body"]
    assert "src/calculator.py:999-1005" in comment["body"]


def test_duplicate_publication_is_idempotent() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    finding = _make_canonical()
    truth_store.record_initial(finding, initial_state=TruthState.APPROVED)

    # First publication
    res1 = publisher.publish_finding(finding, pull_number=42, diff_content=SAMPLE_DIFF)
    assert res1.status == PublicationStatus.PUBLISHED
    assert len(github_client.reviews) == 1
    rev_id = res1.review_id

    # Second publication with same finding and PR
    res2 = publisher.publish_finding(finding, pull_number=42, diff_content=SAMPLE_DIFF)
    assert res2.status == PublicationStatus.ALREADY_PUBLISHED
    assert res2.review_id == rev_id
    # No second review created on GitHub
    assert len(github_client.reviews) == 1


def test_ambiguous_external_failure_fails_closed_and_is_auditable() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(
        pr_heads={("owner/repo", 42): "sha-reviewed-123"},
        should_fail_api=True,
        api_error_message="502 Bad Gateway from GitHub API",
    )
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    finding = _make_canonical()
    truth_store.record_initial(finding, initial_state=TruthState.APPROVED)

    result = publisher.publish_finding(finding, pull_number=42, diff_content=SAMPLE_DIFF)
    assert result.status == PublicationStatus.FAILED
    assert "502 Bad Gateway" in result.reason

    # State in Review Truth remains APPROVED (not transitioned to PUBLISHED)
    latest = truth_store.get_latest_state(finding.canonical_id)
    assert latest.state == TruthState.APPROVED


def test_no_code_modification_or_pr_merge_calls_occur() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    finding = _make_canonical()
    truth_store.record_initial(finding, initial_state=TruthState.APPROVED)
    publisher.publish_finding(finding, pull_number=42, diff_content=SAMPLE_DIFF)

    assert github_client.attempted_merges == 0
    assert github_client.attempted_code_modifications == 0


def test_durable_publication_state_survives_connection_reopening(tmp_path) -> None:
    db_file = tmp_path / "github_effects_durability.db"

    # Connection 1: publish
    conn1 = sqlite3.connect(str(db_file))
    store1 = ReviewTruthStore(conn1)
    client = FakeGitHubClient(pr_heads={("owner/repo", 77): "sha-reviewed-123"})
    publisher1 = GitHubReviewPublisher(conn1, store1, client)

    finding = _make_canonical(canonical_id="can-durable-pub")
    store1.record_initial(finding, initial_state=TruthState.APPROVED)
    res1 = publisher1.publish_finding(finding, pull_number=77, diff_content=SAMPLE_DIFF)
    assert res1.status == PublicationStatus.PUBLISHED
    conn1.close()

    # Connection 2: inspect and attempt second publish
    conn2 = sqlite3.connect(str(db_file))
    store2 = ReviewTruthStore(conn2)
    publisher2 = GitHubReviewPublisher(conn2, store2, client)

    # Reconstructed state shows PUBLISHED
    latest = store2.get_latest_state("can-durable-pub")
    assert latest is not None
    assert latest.state == TruthState.PUBLISHED

    # Retry publish detects idempotent effect across restart
    res2 = publisher2.publish_finding(finding, pull_number=77, diff_content=SAMPLE_DIFF)
    assert res2.status == PublicationStatus.ALREADY_PUBLISHED
    assert res2.review_id == res1.review_id
    assert len(client.reviews) == 1
    conn2.close()


def test_batch_review_publication() -> None:
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)
    github_client = FakeGitHubClient(pr_heads={("owner/repo", 42): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, github_client)

    f1 = _make_canonical(canonical_id="can-batch-1", line_range=(10, 12))
    f2 = _make_canonical(canonical_id="can-batch-2", line_range=(14, 15))
    truth_store.record_initial(f1, initial_state=TruthState.APPROVED)
    truth_store.record_initial(f2, initial_state=TruthState.APPROVED)

    results = publisher.publish_review_batch(
        [f1, f2],
        pull_number=42,
        diff_content=SAMPLE_DIFF,
    )

    assert len(results) == 2
    assert results[0].status == PublicationStatus.PUBLISHED
    assert results[1].status == PublicationStatus.PUBLISHED
    assert len(github_client.reviews) == 2


def test_crash_after_external_publication_reconciles_without_duplicate(tmp_path) -> None:
    """Simulate: GitHub accepts the publication, local persistence fails/crashes, retry occurs.

    Proves that retry does not create a duplicate visible GitHub effect.
    """
    db_file = tmp_path / "crash_recovery.db"
    conn1 = sqlite3.connect(str(db_file))
    truth_store1 = ReviewTruthStore(conn1)
    client = FakeGitHubClient(pr_heads={("owner/repo", 99): "sha-reviewed-123"})
    publisher1 = GitHubReviewPublisher(conn1, truth_store1, client)

    finding = _make_canonical(canonical_id="can-crash-recovery", head_sha="sha-reviewed-123")
    truth_store1.record_initial(finding, initial_state=TruthState.APPROVED)

    # Hook create_review: GitHub accepts publication, then process crashes before local DB commits PUBLISHED
    orig_create_review = client.create_review

    def crash_after_create_review(*args, **kwargs):
        resp = orig_create_review(*args, **kwargs)
        # GitHub has accepted and recorded the review
        assert len(client.reviews) == 1
        # Now simulate hard process crash/death immediately after GitHub accepted
        raise BaseException("Simulated process death immediately after GitHub accepted publication")

    client.create_review = crash_after_create_review

    # First publication attempt crashes after GitHub accepts
    with pytest.raises(BaseException, match="Simulated process death"):
        publisher1.publish_finding(finding, pull_number=99, diff_content=SAMPLE_DIFF)

    # Restore normal create_review on client
    client.create_review = orig_create_review

    # Verify state immediately after crash:
    # 1. GitHub has accepted the review
    assert len(client.reviews) == 1
    review_id_on_github = client.reviews[0]["review_id"]
    # 2. Local effect table holds PENDING (from pre-commit)
    pending_row = conn1.execute(
        "SELECT status FROM github_review_effects WHERE canonical_id = ?",
        (finding.canonical_id,),
    ).fetchone()
    assert pending_row is not None
    assert pending_row[0] == PublicationStatus.PENDING.value
    # 3. Review truth is still APPROVED (not yet updated to PUBLISHED)
    pre_crash_truth = truth_store1.get_latest_state(finding.canonical_id)
    assert pre_crash_truth is not None
    assert pre_crash_truth.state == TruthState.APPROVED

    conn1.close()

    # Retry occurs: new process / connection
    conn2 = sqlite3.connect(str(db_file))
    truth_store2 = ReviewTruthStore(conn2)
    publisher2 = GitHubReviewPublisher(conn2, truth_store2, client)

    retry_res = publisher2.publish_finding(finding, pull_number=99, diff_content=SAMPLE_DIFF)

    # Must reconcile with GitHub and report ALREADY_PUBLISHED
    assert retry_res.status == PublicationStatus.ALREADY_PUBLISHED
    assert retry_res.review_id == review_id_on_github
    assert len(client.reviews) == 1  # ZERO duplicate visible GitHub effect!

    # Review Truth must now be reconciled to PUBLISHED
    final_truth = truth_store2.get_latest_state(finding.canonical_id)
    assert final_truth is not None
    assert final_truth.state == TruthState.PUBLISHED

    # Durable effect table must now be PUBLISHED
    final_effect = conn2.execute(
        "SELECT status FROM github_review_effects WHERE canonical_id = ?",
        (finding.canonical_id,),
    ).fetchone()
    assert final_effect is not None
    assert final_effect[0] == PublicationStatus.PUBLISHED.value

    conn2.close()


def test_live_sha_changes_between_validation_and_creation_fails_closed() -> None:
    """Simulate: live SHA matches during validation -> SHA changes before create_review.

    Publication must not be treated as a valid current-SHA publication.
    """
    conn = sqlite3.connect(":memory:")
    truth_store = ReviewTruthStore(conn)

    class RaceClient(FakeGitHubClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sha_checks = 0

        def get_pull_request_head_sha(self, repository_id: str, pull_number: int) -> str:
            self.sha_checks += 1
            if self.sha_checks == 1:
                # Validation check matches reviewed SHA
                return "sha-reviewed-123"
            # Author pushes new commit before review creation
            self.set_head_sha(repository_id, pull_number, "sha-raced-456")
            return "sha-raced-456"

    client = RaceClient(pr_heads={("owner/repo", 55): "sha-reviewed-123"})
    publisher = GitHubReviewPublisher(conn, truth_store, client)

    finding = _make_canonical(canonical_id="can-toctou-race", head_sha="sha-reviewed-123")
    truth_store.record_initial(finding, initial_state=TruthState.APPROVED)

    result = publisher.publish_finding(finding, pull_number=55, diff_content=SAMPLE_DIFF)

    # Must NOT be treated as a valid current-SHA publication
    assert result.status == PublicationStatus.SUPERSEDED_SHA_MISMATCH
    assert "Current PR head SHA moved before publication" in result.reason
    assert len(client.reviews) == 0  # No review posted against stale commit

    # Review Truth state transitioned to SUPERSEDED
    latest_truth = truth_store.get_latest_state("can-toctou-race")
    assert latest_truth is not None
    assert latest_truth.state == TruthState.SUPERSEDED
