"""Policy-permitted, current-SHA-safe, idempotent GitHub reviews and inline findings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import sqlite3
import time

from pr_review_agent.policy import (
    CanonicalFinding,
    ReviewTruthRecord,
    ReviewTruthStore,
    TruthState,
)
from pr_review_agent.retrieval import parse_diff_file_hunks


class PublicationStatus(str, Enum):
    """Auditable publication outcome status."""

    PENDING = "pending"
    PUBLISHED = "published"
    ALREADY_PUBLISHED = "already_published"
    HELD_OR_UNAUTHORIZED = "held_or_unauthorized"
    SUPERSEDED_SHA_MISMATCH = "superseded_sha_mismatch"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


class StaleHeadShaError(RuntimeError):
    """Raised when the PR head SHA changed before or during review creation."""


@dataclass(frozen=True)
class InlineComment:
    """Structure representing a validated inline PR review comment."""

    path: str
    line: int
    start_line: int | None
    side: str
    body: str


@dataclass(frozen=True)
class GitHubReviewResponse:
    """Response returned from a GitHub review creation API call."""

    review_id: str
    html_url: str
    state: str
    comments_count: int = 0


@dataclass(frozen=True)
class GitHubCommentResponse:
    """Response returned from a GitHub issue/PR comment creation API call."""

    comment_id: str
    html_url: str


@dataclass(frozen=True)
class PublicationResult:
    """Auditable result of attempting to publish a finding or review to GitHub."""

    status: PublicationStatus
    idempotency_key: str
    canonical_id: str
    head_sha: str
    review_id: str | None = None
    comment_id: str | None = None
    html_url: str | None = None
    reason: str = ""
    published_inline: bool = False
    timestamp: float = field(default_factory=time.time)


class GitHubClient(ABC):
    """Provider boundary for GitHub interactions.

    Restricted strictly to review and comment operations.
    Does NOT support code modification or PR merging.
    """

    @abstractmethod
    def get_pull_request_head_sha(self, repository_id: str, pull_number: int) -> str:
        """Fetch the current head SHA for the pull request."""

    @abstractmethod
    def create_review(
        self,
        repository_id: str,
        pull_number: int,
        *,
        commit_sha: str,
        body: str,
        event: str = "COMMENT",
        comments: Sequence[InlineComment] = (),
    ) -> GitHubReviewResponse:
        """Create a pull request review anchored to commit_sha.

        Must fail if commit_sha is not the current head of the PR.
        """

    @abstractmethod
    def create_issue_comment(
        self,
        repository_id: str,
        pull_number: int,
        *,
        body: str,
    ) -> GitHubCommentResponse:
        """Create a top-level comment on a pull request."""

    @abstractmethod
    def list_reviews(
        self,
        repository_id: str,
        pull_number: int,
    ) -> list[dict[str, object]]:
        """List existing reviews on the pull request for reconciliation."""

    @abstractmethod
    def list_issue_comments(
        self,
        repository_id: str,
        pull_number: int,
    ) -> list[dict[str, object]]:
        """List existing issue comments on the pull request for reconciliation."""


class FakeGitHubClient(GitHubClient):
    """In-memory GitHub client for testing and offline execution."""

    def __init__(
        self,
        pr_heads: Mapping[tuple[str, int], str] | None = None,
        *,
        should_fail_api: bool = False,
        api_error_message: str = "Simulated GitHub API failure",
    ) -> None:
        self.pr_heads: dict[tuple[str, int], str] = dict(pr_heads or {})
        self.reviews: list[dict[str, object]] = []
        self.comments: list[dict[str, object]] = []
        self.should_fail_api = should_fail_api
        self.api_error_message = api_error_message

        # Explicit safety tracking: confirm no code edit or merge calls occur
        self.attempted_merges: int = 0
        self.attempted_code_modifications: int = 0

    def set_head_sha(self, repository_id: str, pull_number: int, head_sha: str) -> None:
        self.pr_heads[(repository_id, pull_number)] = head_sha

    def get_pull_request_head_sha(self, repository_id: str, pull_number: int) -> str:
        if self.should_fail_api:
            raise RuntimeError(self.api_error_message)
        key = (repository_id, pull_number)
        if key not in self.pr_heads:
            raise KeyError(f"Pull request #{pull_number} in {repository_id} not found")
        return self.pr_heads[key]

    def create_review(
        self,
        repository_id: str,
        pull_number: int,
        *,
        commit_sha: str,
        body: str,
        event: str = "COMMENT",
        comments: Sequence[InlineComment] = (),
    ) -> GitHubReviewResponse:
        if self.should_fail_api:
            raise RuntimeError(self.api_error_message)

        # TOCTOU guard: verify the review commit SHA matches the current live PR head
        current_head = self.pr_heads.get((repository_id, pull_number))
        if current_head != commit_sha:
            raise StaleHeadShaError(
                f"PR #{pull_number} head SHA is '{current_head}', but review is anchored to '{commit_sha}'"
            )

        review_id = f"rev-{len(self.reviews) + 1}"
        html_url = f"https://github.com/{repository_id}/pull/{pull_number}#pullrequestreview-{review_id}"
        record = {
            "review_id": review_id,
            "repository_id": repository_id,
            "pull_number": pull_number,
            "commit_sha": commit_sha,
            "body": body,
            "event": event,
            "comments": [asdict(c) for c in comments],
            "html_url": html_url,
        }
        self.reviews.append(record)
        return GitHubReviewResponse(
            review_id=review_id,
            html_url=html_url,
            state="COMMENTED",
            comments_count=len(comments),
        )

    def create_issue_comment(
        self,
        repository_id: str,
        pull_number: int,
        *,
        body: str,
    ) -> GitHubCommentResponse:
        if self.should_fail_api:
            raise RuntimeError(self.api_error_message)

        comment_id = f"comment-{len(self.comments) + 1}"
        html_url = f"https://github.com/{repository_id}/pull/{pull_number}#issuecomment-{comment_id}"
        record = {
            "comment_id": comment_id,
            "repository_id": repository_id,
            "pull_number": pull_number,
            "body": body,
            "html_url": html_url,
        }
        self.comments.append(record)
        return GitHubCommentResponse(
            comment_id=comment_id,
            html_url=html_url,
        )

    def list_reviews(
        self,
        repository_id: str,
        pull_number: int,
    ) -> list[dict[str, object]]:
        if self.should_fail_api:
            raise RuntimeError(self.api_error_message)
        return [
            dict(r)
            for r in self.reviews
            if r.get("repository_id") == repository_id and r.get("pull_number") == pull_number
        ]

    def list_issue_comments(
        self,
        repository_id: str,
        pull_number: int,
    ) -> list[dict[str, object]]:
        if self.should_fail_api:
            raise RuntimeError(self.api_error_message)
        return [
            dict(c)
            for c in self.comments
            if c.get("repository_id") == repository_id and c.get("pull_number") == pull_number
        ]


class GitHubReviewPublisher:
    """Manages policy-permitted, current-SHA-safe, idempotent publication to GitHub."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        truth_store: ReviewTruthStore,
        github_client: GitHubClient,
    ) -> None:
        self.connection = connection
        self.truth_store = truth_store
        self.github_client = github_client
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS github_review_effects (
                    idempotency_key TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    pull_number INTEGER NOT NULL,
                    head_sha TEXT NOT NULL,
                    canonical_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    review_id TEXT,
                    comment_id TEXT,
                    html_url TEXT,
                    published_inline INTEGER NOT NULL,
                    reason TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_github_effects_lookup
                ON github_review_effects (repository_id, pull_number, canonical_id)
                """
            )

    def compute_idempotency_key(
        self,
        repository_id: str,
        pull_number: int,
        head_sha: str,
        canonical_id: str,
        operation: str = "review",
    ) -> str:
        """Construct a deterministic publication idempotency key."""
        seed = f"{repository_id}:{pull_number}:{head_sha}:{canonical_id}:{operation}"
        return f"pub-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"

    def publish_finding(
        self,
        finding: CanonicalFinding,
        *,
        pull_number: int,
        diff_content: str = "",
        operation: str = "review",
        now: float | None = None,
    ) -> PublicationResult:
        """Publish a single policy-permitted canonical finding to GitHub with current-SHA safety."""
        current_time = time.time() if now is None else now
        idempotency_key = self.compute_idempotency_key(
            finding.repository_id,
            pull_number,
            finding.head_sha,
            finding.canonical_id,
            operation,
        )

        # 1. Idempotency check: check if already successfully published or in-flight
        existing = self._get_existing_effect(idempotency_key)
        if existing:
            status = existing["status"]
            if status == PublicationStatus.PUBLISHED.value:
                return PublicationResult(
                    status=PublicationStatus.ALREADY_PUBLISHED,
                    idempotency_key=idempotency_key,
                    canonical_id=finding.canonical_id,
                    head_sha=finding.head_sha,
                    review_id=existing.get("review_id"),
                    comment_id=existing.get("comment_id"),
                    html_url=existing.get("html_url"),
                    published_inline=bool(existing.get("published_inline")),
                    reason="Finding already published to GitHub; duplicate prevented",
                    timestamp=current_time,
                )
            elif status == PublicationStatus.AMBIGUOUS.value:
                # Ambiguous external state from prior attempt: must reconcile with GitHub or fail closed
                reconciled = self._reconcile_with_github(
                    finding,
                    pull_number,
                    idempotency_key,
                    now=current_time,
                )
                if reconciled is not None:
                    return reconciled
                return PublicationResult(
                    status=PublicationStatus.FAILED,
                    idempotency_key=idempotency_key,
                    canonical_id=finding.canonical_id,
                    head_sha=finding.head_sha,
                    reason="Prior publication attempt resulted in an ambiguous state and reconciliation found no verified GitHub effect; failing closed",
                    timestamp=current_time,
                )
            elif status == PublicationStatus.PENDING.value:
                # In-flight or crashed publication attempt: reconcile with GitHub
                reconciled = self._reconcile_with_github(
                    finding,
                    pull_number,
                    idempotency_key,
                    now=current_time,
                )
                if reconciled is not None:
                    return reconciled

        # 2. Policy authorization check: strictly verify durable state in Review Truth
        latest_truth = self.truth_store.get_latest_state(finding.canonical_id)
        if latest_truth is None:
            return self._record_effect_and_result(
                status=PublicationStatus.HELD_OR_UNAUTHORIZED,
                idempotency_key=idempotency_key,
                finding=finding,
                pull_number=pull_number,
                reason="Finding not found in Review Truth; publication disallowed",
                now=current_time,
            )

        allowed_states = {TruthState.APPROVED, TruthState.AUTO_APPROVED}
        if latest_truth.state not in allowed_states:
            return self._record_effect_and_result(
                status=PublicationStatus.HELD_OR_UNAUTHORIZED,
                idempotency_key=idempotency_key,
                finding=finding,
                pull_number=pull_number,
                reason=f"Finding in Review Truth is '{latest_truth.state.value}'; publication requires APPROVED or AUTO_APPROVED",
                now=current_time,
            )

        # 3. Current PR head SHA check: compare live PR head SHA with reviewed head SHA
        try:
            current_head_sha = self.github_client.get_pull_request_head_sha(
                finding.repository_id,
                pull_number,
            )
        except Exception as e:
            return self._record_effect_and_result(
                status=PublicationStatus.FAILED,
                idempotency_key=idempotency_key,
                finding=finding,
                pull_number=pull_number,
                reason=f"Failed to query live PR head SHA: {e}",
                now=current_time,
            )

        if not current_head_sha or current_head_sha.strip().lower() != finding.head_sha.strip().lower():
            reason = (
                f"Head SHA mismatch: reviewed snapshot is {finding.head_sha}, but current live PR head is {current_head_sha}"
            )
            # Record superseded state in Review Truth (INVARIANT-fd3d3e39, AC-09)
            self.truth_store.record_transition(
                finding.canonical_id,
                TruthState.SUPERSEDED,
                actor="github_publisher",
                actor_role="system",
                rationale=reason,
                now=current_time,
            )
            return self._record_effect_and_result(
                status=PublicationStatus.SUPERSEDED_SHA_MISMATCH,
                idempotency_key=idempotency_key,
                finding=finding,
                pull_number=pull_number,
                reason=reason,
                now=current_time,
            )

        # 4. Inline location validation against diff hunks
        can_inline, inline_comment = self._validate_inline_comment(finding, diff_content, idempotency_key)

        # 5. Pre-commit PENDING state to database before making external network call
        # This guarantees that if the process crashes after GitHub accepts the review,
        # subsequent retries detect the pending attempt and reconcile rather than duplicate.
        self._record_effect_and_result(
            status=PublicationStatus.PENDING,
            idempotency_key=idempotency_key,
            finding=finding,
            pull_number=pull_number,
            reason="Publication in-flight to GitHub",
            now=current_time,
        )

        # 6. Publication execution via GitHub API
        formatted_body = self._format_finding_body(finding, idempotency_key)
        try:
            # Re-verify live head SHA immediately before call to protect against TOCTOU race
            live_recheck = self.github_client.get_pull_request_head_sha(
                finding.repository_id,
                pull_number,
            )
            if live_recheck != finding.head_sha:
                raise StaleHeadShaError(
                    f"Live head changed to '{live_recheck}' immediately before publication; expected '{finding.head_sha}'"
                )

            if can_inline and inline_comment:
                review_resp = self.github_client.create_review(
                    finding.repository_id,
                    pull_number,
                    commit_sha=finding.head_sha,
                    body=f"### AI Code Review Finding ({finding.category.upper()} / {finding.severity.upper()})\n<!-- pr-review-agent-idempotency: {idempotency_key} -->",
                    event="COMMENT",
                    comments=[inline_comment],
                )
                review_id = review_resp.review_id
                html_url = review_resp.html_url
                comment_id = None
                published_inline = True
                pub_reason = "Published inline review comment at verified diff position"
            else:
                # Safe fallback to top-level PR comment (never fabricate inline location)
                fallback_body = (
                    f"### AI Code Review Finding (Non-inline summary)\n"
                    f"**Location:** `{finding.file_path}:{finding.line_range[0]}-{finding.line_range[1]}` (line outside modified diff hunks)\n\n"
                    f"{formatted_body}"
                )
                comment_resp = self.github_client.create_issue_comment(
                    finding.repository_id,
                    pull_number,
                    body=fallback_body,
                )
                review_id = None
                comment_id = comment_resp.comment_id
                html_url = comment_resp.html_url
                published_inline = False
                pub_reason = "Location outside modified diff hunks; safely published top-level comment"

        except StaleHeadShaError as e:
            # Race detected: PR head changed before review creation was completed
            reason = f"Current PR head SHA moved before publication: {e}"
            self.truth_store.record_transition(
                finding.canonical_id,
                TruthState.SUPERSEDED,
                actor="github_publisher",
                actor_role="system",
                rationale=reason,
                now=current_time,
            )
            return self._record_effect_and_result(
                status=PublicationStatus.SUPERSEDED_SHA_MISMATCH,
                idempotency_key=idempotency_key,
                finding=finding,
                pull_number=pull_number,
                reason=reason,
                now=current_time,
            )
        except Exception as e:
            # Ambiguous external failure: mark AMBIGUOUS in effect table and fail closed
            self._record_effect_and_result(
                status=PublicationStatus.AMBIGUOUS,
                idempotency_key=idempotency_key,
                finding=finding,
                pull_number=pull_number,
                reason=f"GitHub API publication failed ambiguously: {e}",
                now=current_time,
            )
            return PublicationResult(
                status=PublicationStatus.FAILED,
                idempotency_key=idempotency_key,
                canonical_id=finding.canonical_id,
                head_sha=finding.head_sha,
                reason=f"GitHub API publication failed: {e}",
                timestamp=current_time,
            )

        # 7. Record state transition to PUBLISHED in Review Truth
        self.truth_store.record_transition(
            finding.canonical_id,
            TruthState.PUBLISHED,
            actor="github_publisher",
            actor_role="system",
            rationale=f"Successfully published to PR #{pull_number} ({pub_reason})",
            now=current_time,
        )

        # 8. Update durable GitHub effect to PUBLISHED
        return self._record_effect_and_result(
            status=PublicationStatus.PUBLISHED,
            idempotency_key=idempotency_key,
            finding=finding,
            pull_number=pull_number,
            review_id=review_id,
            comment_id=comment_id,
            html_url=html_url,
            published_inline=published_inline,
            reason=pub_reason,
            now=current_time,
        )

    def _reconcile_with_github(
        self,
        finding: CanonicalFinding,
        pull_number: int,
        idempotency_key: str,
        *,
        now: float,
    ) -> PublicationResult | None:
        """Query GitHub to reconcile whether a pending/ambiguous publication already succeeded externally."""
        marker = f"<!-- pr-review-agent-idempotency: {idempotency_key} -->"
        try:
            reviews = self.github_client.list_reviews(finding.repository_id, pull_number)
            for r in reviews:
                body = str(r.get("body") or "")
                comments = r.get("comments") or []
                found_in_comments = (
                    any(marker in str(c.get("body") or "") for c in comments)
                    if isinstance(comments, list)
                    else False
                )
                if marker in body or found_in_comments:
                    review_id = str(r.get("review_id") or "")
                    html_url = str(r.get("html_url") or "")
                    # Transition Review Truth to PUBLISHED if needed
                    latest = self.truth_store.get_latest_state(finding.canonical_id)
                    if latest and latest.state in (TruthState.APPROVED, TruthState.AUTO_APPROVED):
                        self.truth_store.record_transition(
                            finding.canonical_id,
                            TruthState.PUBLISHED,
                            actor="github_publisher",
                            actor_role="system",
                            rationale=f"Reconciled existing review {review_id} from GitHub after uncommitted state",
                            now=now,
                        )
                    # Update local effect
                    self._record_effect_and_result(
                        status=PublicationStatus.PUBLISHED,
                        idempotency_key=idempotency_key,
                        finding=finding,
                        pull_number=pull_number,
                        review_id=review_id,
                        html_url=html_url,
                        published_inline=True,
                        reason="Reconciled existing publication from GitHub after uncommitted state",
                        now=now,
                    )
                    return PublicationResult(
                        status=PublicationStatus.ALREADY_PUBLISHED,
                        idempotency_key=idempotency_key,
                        canonical_id=finding.canonical_id,
                        head_sha=finding.head_sha,
                        review_id=review_id,
                        comment_id=None,
                        html_url=html_url,
                        published_inline=True,
                        reason="Reconciled existing publication from GitHub after uncommitted state; duplicate prevented",
                        timestamp=now,
                    )

            issue_comments = self.github_client.list_issue_comments(finding.repository_id, pull_number)
            for c in issue_comments:
                body = str(c.get("body") or "")
                if marker in body:
                    comment_id = str(c.get("comment_id") or "")
                    html_url = str(c.get("html_url") or "")
                    latest = self.truth_store.get_latest_state(finding.canonical_id)
                    if latest and latest.state in (TruthState.APPROVED, TruthState.AUTO_APPROVED):
                        self.truth_store.record_transition(
                            finding.canonical_id,
                            TruthState.PUBLISHED,
                            actor="github_publisher",
                            actor_role="system",
                            rationale=f"Reconciled existing comment {comment_id} from GitHub after uncommitted state",
                            now=now,
                        )
                    self._record_effect_and_result(
                        status=PublicationStatus.PUBLISHED,
                        idempotency_key=idempotency_key,
                        finding=finding,
                        pull_number=pull_number,
                        comment_id=comment_id,
                        html_url=html_url,
                        published_inline=False,
                        reason="Reconciled existing publication from GitHub after uncommitted state",
                        now=now,
                    )
                    return PublicationResult(
                        status=PublicationStatus.ALREADY_PUBLISHED,
                        idempotency_key=idempotency_key,
                        canonical_id=finding.canonical_id,
                        head_sha=finding.head_sha,
                        review_id=None,
                        comment_id=comment_id,
                        html_url=html_url,
                        published_inline=False,
                        reason="Reconciled existing publication from GitHub after uncommitted state; duplicate prevented",
                        timestamp=now,
                    )
        except Exception as e:
            # Ambiguous external reconciliation failure: fail closed
            return PublicationResult(
                status=PublicationStatus.FAILED,
                idempotency_key=idempotency_key,
                canonical_id=finding.canonical_id,
                head_sha=finding.head_sha,
                reason=f"Failed to reconcile external publication state with GitHub: {e}",
                timestamp=now,
            )

        # Confirmed absent on GitHub
        return None

    def publish_review_batch(
        self,
        findings: Sequence[CanonicalFinding],
        *,
        pull_number: int,
        diff_content: str = "",
        review_summary: str = "Automated AI Code Review Findings",
        now: float | None = None,
    ) -> list[PublicationResult]:
        """Publish a batch of findings for a PR as a single cohesive review where possible."""
        results: list[PublicationResult] = []
        for f in findings:
            res = self.publish_finding(
                f,
                pull_number=pull_number,
                diff_content=diff_content,
                now=now,
            )
            results.append(res)
        return results

    def _validate_inline_comment(
        self,
        finding: CanonicalFinding,
        diff_content: str,
        idempotency_key: str = "",
    ) -> tuple[bool, InlineComment | None]:
        """Validate if finding line range safely falls inside the modified diff hunks."""
        if not diff_content or not finding.file_path:
            return False, None

        file_hunks = parse_diff_file_hunks(diff_content)
        normalized_file = finding.file_path.strip().replace("\\", "/")

        if normalized_file not in file_hunks:
            return False, None

        hunks = file_hunks[normalized_file]
        start_line, end_line = finding.line_range

        # Check if the line range falls within any diff hunk
        falls_in_hunk = any(
            max(h_start, start_line) <= min(h_end, end_line)
            for h_start, h_end in hunks
        )
        if not falls_in_hunk:
            return False, None

        body = self._format_finding_body(finding, idempotency_key)
        return True, InlineComment(
            path=normalized_file,
            line=end_line,
            start_line=start_line if start_line < end_line else None,
            side="RIGHT",
            body=body,
        )

    def _format_finding_body(self, finding: CanonicalFinding, idempotency_key: str = "") -> str:
        """Format the markdown content of a GitHub finding comment with embedded idempotency tag."""
        marker = f"<!-- pr-review-agent-idempotency: {idempotency_key} -->\n" if idempotency_key else ""
        lines = [
            marker,
            f"**Category:** `{finding.category}` | **Severity:** `{finding.severity}` | **Confidence:** `{finding.confidence:.2f}`",
            "",
            f"### {finding.summary}",
            "",
            finding.rationale,
        ]

        if finding.remediation:
            lines.extend([
                "",
                "#### Suggested Remediation",
                finding.remediation,
            ])

        if finding.evidence_refs:
            lines.extend([
                "",
                "<details>",
                "<summary>Verified Evidence References</summary>",
                "",
            ])
            for ref in finding.evidence_refs:
                lines.append(f"- `{ref}`")
            lines.append("</details>")

        return "\n".join(lines)

    def _get_existing_effect(self, idempotency_key: str) -> dict[str, object] | None:
        row = self.connection.execute(
            """
            SELECT idempotency_key, repository_id, pull_number, head_sha,
                   canonical_id, status, review_id, comment_id, html_url,
                   published_inline, reason, payload_json, created_at
            FROM github_review_effects
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()
        if not row:
            return None
        return {
            "idempotency_key": row[0],
            "repository_id": row[1],
            "pull_number": row[2],
            "head_sha": row[3],
            "canonical_id": row[4],
            "status": row[5],
            "review_id": row[6],
            "comment_id": row[7],
            "html_url": row[8],
            "published_inline": bool(row[9]),
            "reason": row[10],
            "payload": json.loads(row[11]),
            "created_at": row[12],
        }

    def _record_effect_and_result(
        self,
        *,
        status: PublicationStatus,
        idempotency_key: str,
        finding: CanonicalFinding,
        pull_number: int,
        reason: str,
        review_id: str | None = None,
        comment_id: str | None = None,
        html_url: str | None = None,
        published_inline: bool = False,
        now: float,
    ) -> PublicationResult:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO github_review_effects (
                    idempotency_key, repository_id, pull_number, head_sha,
                    canonical_id, status, review_id, comment_id, html_url,
                    published_inline, reason, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    finding.repository_id,
                    pull_number,
                    finding.head_sha,
                    finding.canonical_id,
                    status.value,
                    review_id,
                    comment_id,
                    html_url,
                    1 if published_inline else 0,
                    reason,
                    json.dumps(asdict(finding), sort_keys=True),
                    now,
                ),
            )

        return PublicationResult(
            status=status,
            idempotency_key=idempotency_key,
            canonical_id=finding.canonical_id,
            head_sha=finding.head_sha,
            review_id=review_id,
            comment_id=comment_id,
            html_url=html_url,
            published_inline=published_inline,
            reason=reason,
            timestamp=now,
        )
