"""Secure, durable intake primitives for GitHub pull-request webhooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import sqlite3
from collections.abc import Mapping, Sequence


SUPPORTED_ACTIONS = frozenset({"opened", "reopened", "synchronize"})


@dataclass(frozen=True)
class ReviewSnapshot:
    """The immutable inputs that identify a review run."""

    repository_id: str
    repository_full_name: str
    pull_request_number: int
    base_sha: str
    head_sha: str
    changed_files: tuple[str, ...]
    policy_version: str
    prompt_version: str
    retrieval_index_version: str
    model_configuration: Mapping[str, str]


@dataclass(frozen=True)
class IntakeResult:
    status: str
    delivery_id: str | None = None
    snapshot: ReviewSnapshot | None = None


class WebhookIntake:
    """Validate, deduplicate, and persist supported GitHub PR deliveries."""

    def __init__(self, connection: sqlite3.Connection, webhook_secret: bytes) -> None:
        self.connection = connection
        self.webhook_secret = webhook_secret
        self._create_schema()

    def accept(
        self,
        headers: Mapping[str, str],
        body: bytes,
        *,
        policy_version: str,
        prompt_version: str,
        retrieval_index_version: str,
        model_configuration: Mapping[str, str],
        changed_files: Sequence[str] = (),
    ) -> IntakeResult:
        """Accept one GitHub delivery only after signature verification."""
        normalized_headers = {str(k).lower(): str(v) for k, v in headers.items()}
        signature = normalized_headers.get("x-hub-signature-256")
        delivery_id = normalized_headers.get("x-github-delivery")
        if not delivery_id or not self._valid_signature(signature, body):
            return IntakeResult("rejected")

        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return IntakeResult("rejected", delivery_id)

        if (
            normalized_headers.get("x-github-event") != "pull_request"
            or payload.get("action") not in SUPPORTED_ACTIONS
        ):
            return IntakeResult("ignored", delivery_id)

        try:
            snapshot = self._snapshot(
                payload,
                policy_version=policy_version,
                prompt_version=prompt_version,
                retrieval_index_version=retrieval_index_version,
                model_configuration=model_configuration,
                changed_files=changed_files,
            )
        except (KeyError, TypeError, ValueError):
            return IntakeResult("rejected", delivery_id)

        snapshot_json = json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"))
        with self.connection:
            inserted = self.connection.execute(
                "INSERT OR IGNORE INTO deliveries(delivery_id, state) VALUES (?, 'accepted')",
                (delivery_id,),
            ).rowcount
            if not inserted:
                return IntakeResult("duplicate", delivery_id)
            self.connection.execute(
                "INSERT INTO review_snapshots(delivery_id, snapshot_json) VALUES (?, ?)",
                (delivery_id, snapshot_json),
            )
        return IntakeResult("accepted", delivery_id, snapshot)

    def snapshot_is_current(self, delivery_id: str, head_sha: str) -> bool:
        """Return whether a persisted review snapshot still matches a PR head SHA."""
        row = self.connection.execute(
            "SELECT snapshot_json FROM review_snapshots WHERE delivery_id = ?", (delivery_id,)
        ).fetchone()
        return bool(row and json.loads(row[0])["head_sha"] == head_sha)

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS deliveries (delivery_id TEXT PRIMARY KEY, state TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS review_snapshots ("
                "delivery_id TEXT PRIMARY KEY REFERENCES deliveries(delivery_id), "
                "snapshot_json TEXT NOT NULL)"
            )

    def _valid_signature(self, signature: str | None, body: bytes) -> bool:
        if not signature or not signature.startswith("sha256="):
            return False
        expected = "sha256=" + hmac.new(self.webhook_secret, body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def _snapshot(
        payload: Mapping[str, object],
        *,
        policy_version: str,
        prompt_version: str,
        retrieval_index_version: str,
        model_configuration: Mapping[str, str],
        changed_files: Sequence[str] = (),
    ) -> ReviewSnapshot:
        repository = payload["repository"]
        pull_request = payload["pull_request"]
        if not isinstance(repository, Mapping) or not isinstance(pull_request, Mapping):
            raise TypeError("GitHub payload has invalid repository or pull request")
        base = pull_request["base"]
        head = pull_request["head"]
        if not isinstance(base, Mapping) or not isinstance(head, Mapping):
            raise TypeError("GitHub payload has invalid revision data")
        if not isinstance(changed_files, Sequence) or isinstance(changed_files, (str, bytes)):
            raise TypeError("changed_files must be a sequence of paths")
        if not all(isinstance(path, str) for path in changed_files):
            raise TypeError("changed_files must contain only string paths")
        return ReviewSnapshot(
            repository_id=str(repository["id"]),
            repository_full_name=str(repository["full_name"]),
            pull_request_number=int(pull_request["number"]),
            base_sha=str(base["sha"]),
            head_sha=str(head["sha"]),
            changed_files=tuple(changed_files),
            policy_version=policy_version,
            prompt_version=prompt_version,
            retrieval_index_version=retrieval_index_version,
            model_configuration=dict(model_configuration),
        )
