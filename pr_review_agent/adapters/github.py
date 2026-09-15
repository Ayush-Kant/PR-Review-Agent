"""Concrete GitHub network client implementing the existing GitHubClient contract."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import httpx

from pr_review_agent.github_output import (
    GitHubClient,
    GitHubCommentResponse,
    GitHubReviewResponse,
    InlineComment,
    StaleHeadShaError,
)
from pr_review_agent.security import SecretType, SecurityConfig


@dataclass(frozen=True)
class DiffRetrievalResult:
    """Result of bounded diff retrieval, explicitly representing degradation."""

    content: str
    is_truncated: bool
    original_bytes: int
    retained_bytes: int


class GitHubNetworkError(RuntimeError):
    """Base exception for GitHub network operations."""


class GitHubRateLimitError(GitHubNetworkError):
    """Raised when GitHub API rate limits are exceeded."""


class GitHubNotFoundError(GitHubNetworkError):
    """Raised when the requested repository or PR resource is not found."""


class GitHubTimeoutError(GitHubNetworkError):
    """Raised when a GitHub API request times out."""


class GitHubApiError(GitHubNetworkError):
    """Raised on non-transient or unexpected GitHub API error responses."""

    def __init__(self, message: str, status_code: int | None = None, response_body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GitHubNetworkClient(GitHubClient):
    """Thin concrete network adapter for GitHub API interactions.

    Strict invariants:
    - Restricted to read operations and comment/review publication.
    - Zero merge, zero push, zero branch modification, zero code writing.
    - All token resolution flows strictly through SecurityConfig.
    - Stale-head protection is verified before review publication and during race boundaries.
    - Respects configurable diff and repository file retrieval budget bounds.
    - Bounded pagination for reconciliation listings.
    """

    def __init__(
        self,
        security_config: SecurityConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        base_url: str = "https://api.github.com",
        timeout_seconds: float = 30.0,
        token_env_var: str = "GITHUB_TOKEN",
        max_diff_bytes: int = 500_000,
        max_file_bytes: int = 200_000,
    ) -> None:
        self.security_config = security_config
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.token_env_var = token_env_var
        self.max_diff_bytes = max_diff_bytes
        self.max_file_bytes = max_file_bytes
        self.last_diff_truncated: bool = False

        # Resolve GitHub credentials via the security boundary
        self._token = self.security_config.resolve_for_client(
            boundary_target="github_api_client",
            secret_type=SecretType.GITHUB_TOKEN,
            env_var_name=token_env_var,
        )

        self._client = httpx.Client(
            transport=transport,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            headers={
                "Authorization": f"token {self._token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "PR-Review-Agent/1.0",
            },
        )

    def _check_repo_authorized(self, repository_id: str) -> None:
        """Verify repository belongs to authorized tenant."""
        if not self.security_config.is_repository_authorized(repository_id):
            raise PermissionError(
                f"Repository '{repository_id}' is not authorized under the active tenant configuration."
            )

    def _handle_response_error(self, exc: Exception | None, resp: httpx.Response | None = None) -> None:
        """Map HTTP and network failure modes to typed exceptions."""
        if isinstance(exc, httpx.TimeoutException):
            raise GitHubTimeoutError(f"GitHub API request timed out after {self.timeout_seconds}s") from exc

        if isinstance(exc, httpx.RequestError):
            raise GitHubNetworkError(f"GitHub network communication error: {exc}") from exc

        if resp is not None and not resp.is_success:
            if resp.status_code in (403, 429):
                rate_remaining = resp.headers.get("x-ratelimit-remaining")
                if rate_remaining == "0" or "rate limit" in resp.text.lower():
                    raise GitHubRateLimitError(
                        f"GitHub API rate limit exceeded (status {resp.status_code}): {resp.text}"
                    )
            if resp.status_code == 404:
                raise GitHubNotFoundError(f"GitHub resource not found (status 404): {resp.text}")

            raise GitHubApiError(
                f"GitHub API responded with error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

    def get_pull_request(self, repository_id: str, pull_number: int) -> dict[str, Any]:
        """Fetch PR metadata from GitHub."""
        self._check_repo_authorized(repository_id)
        endpoint = f"/repos/{repository_id}/pulls/{pull_number}"
        try:
            resp = self._client.get(endpoint)
            self._handle_response_error(None, resp)
            data = resp.json()
            if not isinstance(data, dict) or "head" not in data or "sha" not in data["head"]:
                raise GitHubApiError("Malformed GitHub response: missing head SHA", resp.status_code, resp.text)
            return data
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._handle_response_error(exc)
            raise

    def get_pull_request_head_sha(self, repository_id: str, pull_number: int) -> str:
        """Fetch the current head SHA for the pull request."""
        pr_data = self.get_pull_request(repository_id, pull_number)
        return str(pr_data["head"]["sha"])

    def get_pull_request_diff_result(self, repository_id: str, pull_number: int) -> DiffRetrievalResult:
        """Retrieve unified diff with explicit tracking of degradation/truncation."""
        self._check_repo_authorized(repository_id)
        endpoint = f"/repos/{repository_id}/pulls/{pull_number}"
        try:
            resp = self._client.get(
                endpoint,
                headers={"Accept": "application/vnd.github.v3.diff"},
            )
            self._handle_response_error(None, resp)
            content_bytes = resp.content
            original_bytes = len(content_bytes)

            if original_bytes > self.max_diff_bytes:
                self.last_diff_truncated = True
                retained = content_bytes[: self.max_diff_bytes].decode("utf-8", errors="replace")
                warning_marker = (
                    f"\n\n[WARNING: DIFF TRUNCATED - original diff ({original_bytes} bytes) "
                    f"exceeds configured max_diff_bytes cap ({self.max_diff_bytes} bytes). "
                    f"Context is degraded/incomplete.]\n"
                )
                degraded_content = retained + warning_marker
                return DiffRetrievalResult(
                    content=degraded_content,
                    is_truncated=True,
                    original_bytes=original_bytes,
                    retained_bytes=len(retained.encode("utf-8")),
                )

            self.last_diff_truncated = False
            return DiffRetrievalResult(
                content=resp.text,
                is_truncated=False,
                original_bytes=original_bytes,
                retained_bytes=original_bytes,
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._handle_response_error(exc)
            raise

    def get_pull_request_diff(self, repository_id: str, pull_number: int) -> str:
        """Retrieve unified diff for a pull request, bounded and explicitly marked if truncated."""
        result = self.get_pull_request_diff_result(repository_id, pull_number)
        return result.content

    def get_repository_file_content(self, repository_id: str, path: str, ref: str) -> str:
        """Retrieve a specific file content from GitHub repository at ref."""
        self._check_repo_authorized(repository_id)
        endpoint = f"/repos/{repository_id}/contents/{path.lstrip('/')}"
        try:
            resp = self._client.get(endpoint, params={"ref": ref})
            self._handle_response_error(None, resp)
            data = resp.json()
            if not isinstance(data, dict):
                raise GitHubApiError("Malformed repository contents response", resp.status_code, resp.text)

            encoding = data.get("encoding")
            raw_content = data.get("content", "")
            if encoding == "base64" and raw_content:
                decoded_bytes = base64.b64decode(raw_content)
            else:
                decoded_bytes = raw_content.encode("utf-8")

            if len(decoded_bytes) > self.max_file_bytes:
                return decoded_bytes[: self.max_file_bytes].decode("utf-8", errors="replace")
            return decoded_bytes.decode("utf-8", errors="replace")
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._handle_response_error(exc)
            raise

    def _paginate_list(self, endpoint: str, max_pages: int = 10, per_page: int = 100) -> list[dict[str, object]]:
        """Safely fetch and combine paginated resources up to bounded max_pages."""
        all_items: list[dict[str, object]] = []
        page = 1
        while page <= max_pages:
            resp = self._client.get(endpoint, params={"per_page": per_page, "page": page})
            self._handle_response_error(None, resp)
            data = resp.json()
            if not isinstance(data, list):
                raise GitHubApiError("Malformed paginated listing response", resp.status_code, resp.text)
            if not data:
                break
            all_items.extend(data)
            # Stop if page returned fewer items than requested or Link header shows no next page
            if len(data) < per_page:
                break
            link_header = resp.headers.get("link", "")
            if link_header and 'rel="next"' not in link_header:
                break
            page += 1
        return all_items

    def list_reviews(self, repository_id: str, pull_number: int) -> list[dict[str, object]]:
        """List existing reviews on the pull request for reconciliation with safe pagination."""
        self._check_repo_authorized(repository_id)
        endpoint = f"/repos/{repository_id}/pulls/{pull_number}/reviews"
        try:
            return self._paginate_list(endpoint)
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._handle_response_error(exc)
            raise

    def list_issue_comments(self, repository_id: str, pull_number: int) -> list[dict[str, object]]:
        """List existing issue comments on the pull request for reconciliation with safe pagination."""
        self._check_repo_authorized(repository_id)
        endpoint = f"/repos/{repository_id}/issues/{pull_number}/comments"
        try:
            return self._paginate_list(endpoint)
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._handle_response_error(exc)
            raise

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

        Strictly enforces that commit_sha matches the current head of the PR both
        before and across the publication boundary to prevent race conditions.
        Fails closed with StaleHeadShaError if the PR head has changed or moved.
        """
        self._check_repo_authorized(repository_id)

        # 1. Pre-publication check: Enforce current head SHA matches commit_sha
        current_sha = self.get_pull_request_head_sha(repository_id, pull_number)
        if current_sha != commit_sha:
            raise StaleHeadShaError(
                f"PR {pull_number} in {repository_id} head SHA moved: expected {commit_sha}, current is {current_sha}"
            )

        endpoint = f"/repos/{repository_id}/pulls/{pull_number}/reviews"
        payload_comments = []
        for c in comments:
            item = {
                "path": c.path,
                "line": c.line,
                "side": c.side,
                "body": c.body,
            }
            if c.start_line is not None and c.start_line != c.line:
                item["start_line"] = c.start_line
                item["start_side"] = c.side
            payload_comments.append(item)

        payload: dict[str, Any] = {
            "commit_id": commit_sha,
            "body": body,
            "event": event,
        }
        if payload_comments:
            payload["comments"] = payload_comments

        try:
            resp = self._client.post(endpoint, json=payload)
            if resp.status_code == 422:
                err_text_lower = resp.text.lower()
                if any(kw in err_text_lower for kw in ("head", "commit", "sha", "stale", "validation failed")):
                    raise StaleHeadShaError(
                        f"GitHub API rejected review creation for stale commit '{commit_sha}': {resp.text}"
                    )
            self._handle_response_error(None, resp)
            res_data = resp.json()

            # 2. Post-publication race verification: ensure head did not move during call
            post_sha = self.get_pull_request_head_sha(repository_id, pull_number)
            if post_sha != commit_sha:
                raise StaleHeadShaError(
                    f"PR head SHA moved during review publication: expected {commit_sha}, became {post_sha}"
                )

            return GitHubReviewResponse(
                review_id=str(res_data.get("id", "")),
                html_url=str(res_data.get("html_url", "")),
                state=str(res_data.get("state", event)),
                comments_count=len(payload_comments),
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._handle_response_error(exc)
            raise

    def create_issue_comment(
        self,
        repository_id: str,
        pull_number: int,
        *,
        body: str,
    ) -> GitHubCommentResponse:
        """Create a top-level issue/PR comment."""
        self._check_repo_authorized(repository_id)
        endpoint = f"/repos/{repository_id}/issues/{pull_number}/comments"
        payload = {"body": body}
        try:
            resp = self._client.post(endpoint, json=payload)
            self._handle_response_error(None, resp)
            res_data = resp.json()
            return GitHubCommentResponse(
                comment_id=str(res_data.get("id", "")),
                html_url=str(res_data.get("html_url", "")),
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._handle_response_error(exc)
            raise

    def close(self) -> None:
        """Close underlying HTTP client connection."""
        self._client.close()

    def __enter__(self) -> GitHubNetworkClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
