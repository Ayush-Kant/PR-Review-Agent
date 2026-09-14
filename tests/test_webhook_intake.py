import hashlib
import hmac
import json
import sqlite3
import unittest

from pr_review_agent.intake import WebhookIntake


SECRET = b"test-secret"


def signed_headers(body: bytes, delivery_id: str = "delivery-1", lowercase: bool = False) -> dict[str, str]:
    sig = "sha256=" + hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    if lowercase:
        return {
            "x-github-delivery": delivery_id,
            "x-github-event": "pull_request",
            "x-hub-signature-256": sig,
        }
    return {
        "X-GitHub-Delivery": delivery_id,
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": sig,
    }


def payload(action: str = "opened", head_sha: str = "head-sha") -> bytes:
    return json.dumps(
        {
            "action": action,
            "repository": {"id": 42, "full_name": "ayush/example"},
            "pull_request": {
                "number": 7,
                "base": {"sha": "base-sha"},
                "head": {"sha": head_sha},
                "changed_files": 1,
            },
        }
    ).encode()


class WebhookIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intake = WebhookIntake(sqlite3.connect(":memory:"), SECRET)

    def accept(
        self,
        body: bytes,
        delivery_id: str = "delivery-1",
        headers: dict[str, str] | None = None,
        changed_files: tuple[str, ...] | list[str] = (),
    ):
        h = headers if headers is not None else signed_headers(body, delivery_id)
        return self.intake.accept(
            h,
            body,
            policy_version="policy-v1",
            prompt_version="prompt-v1",
            retrieval_index_version="index-v1",
            model_configuration={"provider": "configured"},
            changed_files=changed_files,
        )

    def test_rejects_invalid_signature_before_payload_parsing(self) -> None:
        result = self.intake.accept(
            {"X-GitHub-Delivery": "delivery-1", "X-Hub-Signature-256": "sha256=invalid"},
            b"not-json",
            policy_version="policy-v1",
            prompt_version="prompt-v1",
            retrieval_index_version="index-v1",
            model_configuration={},
        )
        self.assertEqual("rejected", result.status)

    def test_persists_immutable_snapshot_and_deduplicates_delivery(self) -> None:
        body = payload()
        result = self.accept(body)
        self.assertEqual("accepted", result.status)
        self.assertEqual("head-sha", result.snapshot.head_sha)
        self.assertEqual((), result.snapshot.changed_files)
        self.assertTrue(self.intake.snapshot_is_current("delivery-1", "head-sha"))
        self.assertFalse(self.intake.snapshot_is_current("delivery-1", "new-head-sha"))
        self.assertEqual("duplicate", self.accept(body).status)

    def test_ignores_unsupported_pull_request_actions(self) -> None:
        body = payload("closed")
        self.assertEqual("ignored", self.accept(body).status)

    def test_accepts_lowercase_and_normalized_headers(self) -> None:
        body = payload()
        headers = signed_headers(body, delivery_id="delivery-lower", lowercase=True)
        result = self.accept(body, delivery_id="delivery-lower", headers=headers)
        self.assertEqual("accepted", result.status)
        self.assertEqual("delivery-lower", result.delivery_id)

    def test_accepts_explicit_changed_files_when_supplied(self) -> None:
        body = payload()
        result = self.accept(body, delivery_id="delivery-files", changed_files=["app.py", "utils.py"])
        self.assertEqual("accepted", result.status)
        self.assertEqual(("app.py", "utils.py"), result.snapshot.changed_files)


if __name__ == "__main__":
    unittest.main()
