import sqlite3
import unittest

from pr_review_agent.orchestration import CandidateFinding, SpecialistType
from pr_review_agent.retrieval import (
    CodeMemoryStore,
    FindingEvidenceValidator,
    HybridRetriever,
)


def sample_finding(
    finding_id: str = "find-1",
    evidence_refs: tuple[str, ...] = ("diff://auth.py#L10-L14",),
) -> CandidateFinding:
    return CandidateFinding(
        finding_id=finding_id,
        correlation_id="deliv-1:run-1",
        specialist_type=SpecialistType.SECURITY,
        category="security",
        severity="high",
        confidence=0.95,
        summary="Insecure token verification",
        rationale="Token signature is not verified against public key",
        file_path="auth.py",
        line_range=(10, 14),
        evidence_refs=evidence_refs,
        remediation="Use jwt.verify(token, public_key)",
    )


class CodeMemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.store = CodeMemoryStore(self.conn)

    def test_initial_indexing_and_tenancy_isolation(self) -> None:
        files_repo_a = {
            "auth.py": "def authenticate():\n    return True\n",
            "utils.py": "def helper():\n    pass\n",
        }
        files_repo_b = {
            "secret.py": "API_KEY = 'secret-b'\n",
        }

        # Index Repo A
        count_a = self.store.index_repository("repo-a", "rev-a1", files_repo_a)
        self.assertEqual(2, count_a)
        self.assertTrue(self.store.is_fresh("repo-a", "rev-a1"))

        # Index Repo B
        count_b = self.store.index_repository("repo-b", "rev-b1", files_repo_b)
        self.assertEqual(1, count_b)
        self.assertTrue(self.store.is_fresh("repo-b", "rev-b1"))

        # Tenancy check: Repo A chunks never returned when querying Repo B
        chunks_a = self.store.get_chunks("repo-a", "rev-a1")
        self.assertEqual(2, len(chunks_a))
        self.assertTrue(all(c.repository_id == "repo-a" for c in chunks_a))

        chunks_b = self.store.get_chunks("repo-b", "rev-b1")
        self.assertEqual(1, len(chunks_b))
        self.assertEqual("repo-b", chunks_b[0].repository_id)
        self.assertEqual("secret.py", chunks_b[0].file_path)

        # Cross-tenant query must return empty list
        self.assertEqual([], self.store.get_chunks("repo-a", "rev-b1"))
        self.assertEqual([], self.store.get_chunks("repo-b", "rev-a1"))

    def test_incremental_indexing_reuses_unchanged_chunks(self) -> None:
        v1_files = {
            "auth.py": "def auth_v1():\n    pass\n",
            "unchanged.py": "def common():\n    return 42\n",
        }
        self.store.index_repository("repo-1", "rev-1", v1_files)
        self.assertTrue(self.store.is_fresh("repo-1", "rev-1"))

        # Fetch unchanged chunk content hash
        rev1_chunks = {c.file_path: c for c in self.store.get_chunks("repo-1", "rev-1")}
        unchanged_hash = rev1_chunks["unchanged.py"].content_hash

        # Index revision 2: auth.py changes, unchanged.py stays identical
        v2_files = {
            "auth.py": "def auth_v2():\n    pass\n",
            "unchanged.py": "def common():\n    return 42\n",
        }
        self.store.index_repository("repo-1", "rev-2", v2_files)

        # Rev-1 is no longer fresh, Rev-2 is fresh
        self.assertFalse(self.store.is_fresh("repo-1", "rev-1"))
        self.assertTrue(self.store.is_fresh("repo-1", "rev-2"))

        rev2_chunks = {c.file_path: c for c in self.store.get_chunks("repo-1", "rev-2")}
        self.assertEqual(unchanged_hash, rev2_chunks["unchanged.py"].content_hash)
        self.assertNotEqual(rev1_chunks["auth.py"].content_hash, rev2_chunks["auth.py"].content_hash)


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.store = CodeMemoryStore(self.conn)
        self.retriever = HybridRetriever(self.store)

        self.files = {
            "auth/jwt.py": "def verify_jwt_token(token, secret):\n    # JWT signature authentication check\n    return True\n",
            "db/models.py": "class UserAccount:\n    id = 1\n    username = 'alice'\n",
            "api/routes.py": "def handle_login(request):\n    verify_jwt_token(request.token, secret)\n",
        }
        self.store.index_repository("repo-1", "head-sha", self.files)

    def test_lexical_retrieval(self) -> None:
        results = self.retriever.retrieve("repo-1", "head-sha", "verify_jwt_token")
        self.assertTrue(len(results) >= 1)
        top = results[0]
        self.assertIn("jwt", top.file_path)
        self.assertGreater(top.relevance_score, 0.0)

    def test_semantic_retrieval_signal(self) -> None:
        # Query uses terms not verbatim in the text (authenticating login credentials)
        results = self.retriever.retrieve("repo-1", "head-sha", "authenticating login credentials")
        self.assertTrue(len(results) >= 1)
        self.assertTrue(any(c.relevance_score > 0 for c in results))

    def test_hybrid_ranking_and_capping(self) -> None:
        # Max-k capping
        results_k1 = self.retriever.retrieve("repo-1", "head-sha", "token", max_k=1)
        self.assertEqual(1, len(results_k1))

        # Citation metadata correctness
        cite = results_k1[0]
        self.assertEqual("repo-1", cite.repository_id)
        self.assertEqual("head-sha", cite.revision)
        self.assertTrue(cite.file_path in self.files)
        self.assertEqual("v1", cite.index_version)

    def test_hard_max_token_cap_never_exceeded_by_first_result(self) -> None:
        long_content = "def sensitive_action():\n" + ("    # very long code explanation\n" * 30)
        self.store.index_repository("repo-1", "head-sha", {"long.py": long_content})

        # Cap at 10 tokens (~40 chars)
        results = self.retriever.retrieve("repo-1", "head-sha", "sensitive", max_tokens=10)
        self.assertEqual(1, len(results))
        # Content must be truncated and never exceed max_tokens (10 * 4 = 40 chars)
        self.assertLessEqual(len(results[0].excerpt), 45)
        self.assertTrue(results[0].excerpt.endswith("..."))

        # Zero max_tokens must return empty
        self.assertEqual((), self.retriever.retrieve("repo-1", "head-sha", "sensitive", max_tokens=0))

    def test_empty_query_or_no_match_returns_empty(self) -> None:
        self.assertEqual((), self.retriever.retrieve("repo-1", "head-sha", ""))
        self.assertEqual((), self.retriever.retrieve("repo-1", "head-sha", "   "))
        self.assertEqual((), self.retriever.retrieve("nonexistent-repo", "head-sha", "jwt"))


class FindingEvidenceValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.store = CodeMemoryStore(self.conn)
        self.validator = FindingEvidenceValidator(self.store)

        self.diff = (
            "--- a/auth.py\n"
            "+++ b/auth.py\n"
            "@@ -10,5 +10,5 @@\n"
            "- old_code\n"
            "+ new_code\n"
        )
        auth_repo_code = "\n".join(f"line_{i} = {i}" for i in range(1, 31)) + "\n"
        self.store.index_repository("repo-1", "head-1", {"auth.py": auth_repo_code})

    def test_valid_diff_and_repo_evidence_passes(self) -> None:
        finding = sample_finding(
            evidence_refs=("diff://auth.py#L10-L14", "repo://auth.py#L1-L25@head-1")
        )
        result = self.validator.validate_finding(
            finding,
            repository_id="repo-1",
            head_sha="head-1",
            diff_content=self.diff,
        )
        self.assertEqual("verified", result.status)
        self.assertFalse(result.is_suppressed)
        self.assertIsNone(result.suppression_reason)
        self.assertEqual(2, len(result.valid_evidence_refs))
        self.assertEqual((), result.invalid_evidence_refs)

    def test_out_of_bounds_diff_line_range_suppressed(self) -> None:
        # Diff only modifies lines 10 to 14
        finding = sample_finding(evidence_refs=("diff://auth.py#L900-L950",))
        result = self.validator.validate_finding(
            finding,
            repository_id="repo-1",
            head_sha="head-1",
            diff_content=self.diff,
        )
        self.assertEqual("suppressed", result.status)
        self.assertTrue(result.is_suppressed)
        self.assertIn("diff://auth.py#L900-L950", result.invalid_evidence_refs)

    def test_out_of_bounds_repository_line_range_suppressed(self) -> None:
        # auth.py only has 30 lines in repository
        finding = sample_finding(evidence_refs=("repo://auth.py#L500-L600@head-1",))
        result = self.validator.validate_finding(
            finding,
            repository_id="repo-1",
            head_sha="head-1",
            diff_content=self.diff,
        )
        self.assertEqual("suppressed", result.status)
        self.assertTrue(result.is_suppressed)
        self.assertIn("repo://auth.py#L500-L600@head-1", result.invalid_evidence_refs)

    def test_exact_diff_filename_matching_rejects_substring_collision(self) -> None:
        # Diff modifies unauth.py and auth.pyc, NOT auth.py
        collision_diff = (
            "--- a/unauth.py\n"
            "+++ b/unauth.py\n"
            "@@ -1,5 +1,5 @@\n"
            "+ change\n"
            "--- a/auth.pyc\n"
            "+++ b/auth.pyc\n"
            "@@ -1,5 +1,5 @@\n"
            "+ change\n"
        )
        finding = sample_finding(evidence_refs=("diff://auth.py#L1-L5",))
        result = self.validator.validate_finding(
            finding,
            repository_id="repo-1",
            head_sha="head-1",
            diff_content=collision_diff,
        )
        self.assertEqual("suppressed", result.status)
        self.assertTrue(result.is_suppressed)
        self.assertIn("diff://auth.py#L1-L5", result.invalid_evidence_refs)

    def test_missing_evidence_is_suppressed(self) -> None:
        finding = sample_finding(evidence_refs=())
        result = self.validator.validate_finding(
            finding,
            repository_id="repo-1",
            head_sha="head-1",
            diff_content=self.diff,
        )
        self.assertEqual("suppressed", result.status)
        self.assertTrue(result.is_suppressed)
        self.assertIn("mandatory evidence", result.suppression_reason.lower())

    def test_invalid_diff_evidence_is_suppressed(self) -> None:
        finding = sample_finding(evidence_refs=("diff://nonexistent_file.py#L1-L5",))
        result = self.validator.validate_finding(
            finding,
            repository_id="repo-1",
            head_sha="head-1",
            diff_content=self.diff,
        )
        self.assertEqual("suppressed", result.status)
        self.assertTrue(result.is_suppressed)
        self.assertEqual(1, len(result.invalid_evidence_refs))

    def test_stale_or_cross_tenant_evidence_fails_closed(self) -> None:
        # 1. Cites a stale revision
        finding_stale = sample_finding(evidence_refs=("repo://auth.py#L1-L10@stale-revision",))
        result_stale = self.validator.validate_finding(
            finding_stale,
            repository_id="repo-1",
            head_sha="head-1",
            diff_content=self.diff,
        )
        self.assertEqual("suppressed", result_stale.status)
        self.assertTrue(result_stale.is_suppressed)

        # 2. Cites wrong repository ID
        finding_tenant = sample_finding(evidence_refs=("cite-other-repo:head-1:auth.py:1",))
        result_tenant = self.validator.validate_finding(
            finding_tenant,
            repository_id="repo-1",
            head_sha="head-1",
            diff_content=self.diff,
        )
        self.assertEqual("suppressed", result_tenant.status)
        self.assertTrue(result_tenant.is_suppressed)


if __name__ == "__main__":
    unittest.main()
