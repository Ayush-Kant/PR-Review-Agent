"""Finding aggregation, confidence/risk policy, Review Truth, and maintainer HITL workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import re
import sqlite3
import time

from pr_review_agent.orchestration import CandidateFinding
from pr_review_agent.retrieval import FindingValidationResult


class FindingDisposition(str, Enum):
    HELD = "held"
    AUTO_APPROVED = "auto_approved"
    SUPPRESSED = "suppressed"


class TruthState(str, Enum):
    CANDIDATE = "candidate"
    MERGED = "merged"
    HELD = "held"
    AUTO_APPROVED = "auto_approved"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISMISSED = "dismissed"
    DISPUTED = "disputed"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"


# Strictly defined state machine transitions
ALLOWED_TRANSITIONS: dict[TruthState, frozenset[TruthState]] = {
    TruthState.CANDIDATE: frozenset({TruthState.MERGED, TruthState.SUPPRESSED}),
    TruthState.MERGED: frozenset({TruthState.HELD, TruthState.AUTO_APPROVED, TruthState.SUPPRESSED}),
    TruthState.SUPPRESSED: frozenset({TruthState.SUPERSEDED}),
    TruthState.HELD: frozenset({
        TruthState.HELD,  # for human edits while held
        TruthState.APPROVED,
        TruthState.REJECTED,
        TruthState.DISMISSED,
        TruthState.DISPUTED,
        TruthState.SUPERSEDED,
    }),
    TruthState.AUTO_APPROVED: frozenset({
        TruthState.DISMISSED,
        TruthState.DISPUTED,
        TruthState.SUPERSEDED,
        TruthState.PUBLISHED,
    }),
    TruthState.APPROVED: frozenset({
        TruthState.PUBLISHED,
        TruthState.SUPERSEDED,
    }),
    TruthState.DISPUTED: frozenset({
        TruthState.DISPUTED,  # for human edits while disputed
        TruthState.APPROVED,
        TruthState.REJECTED,
        TruthState.RESOLVED,
        TruthState.DISMISSED,
        TruthState.SUPERSEDED,
    }),
    TruthState.RESOLVED: frozenset({TruthState.SUPERSEDED}),
    TruthState.REJECTED: frozenset({TruthState.SUPERSEDED}),
    TruthState.DISMISSED: frozenset({TruthState.SUPERSEDED}),
    TruthState.PUBLISHED: frozenset({
        TruthState.DISPUTED,
        TruthState.DISMISSED,
        TruthState.SUPERSEDED,
    }),
    TruthState.SUPERSEDED: frozenset(),  # Terminal
}


AUTHORIZED_HITL_ROLES = frozenset({"maintainer", "reviewer", "admin"})

SEVERITY_WEIGHTS: dict[str, int] = {
    "critical": 5,
    "high": 4,
    "blocking": 4,
    "medium": 3,
    "low": 2,
    "info": 1,
}


@dataclass(frozen=True)
class CanonicalFinding:
    """A deduplicated canonical finding representing one or more candidate findings."""

    canonical_id: str
    repository_id: str
    head_sha: str
    category: str
    severity: str
    confidence: float
    summary: str
    rationale: str
    file_path: str
    line_range: tuple[int, int]
    contributing_candidate_ids: tuple[str, ...]
    contributing_specialists: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    remediation: str | None = None
    disposition: FindingDisposition = FindingDisposition.HELD
    disposition_reason: str = ""
    merge_rationale: str = ""
    policy_version: str = "v1"
    delivery_id: str = ""
    run_id: str = ""
    raw_severity: str = ""
    calibrated_severity: str = ""
    calibration_rule: str = ""
    calibration_reason: str = ""

    def __post_init__(self) -> None:
        if not self.raw_severity:
            object.__setattr__(self, "raw_severity", self.severity)
        if not self.calibrated_severity:
            object.__setattr__(self, "calibrated_severity", self.severity)



@dataclass(frozen=True)
class ReviewTruthRecord:
    """An immutable, auditable record in the Review Truth lifecycle."""

    record_id: str
    canonical_id: str
    sequence_id: int
    repository_id: str
    head_sha: str
    delivery_id: str
    run_id: str
    state: TruthState
    actor: str | None
    actor_role: str | None
    rationale: str | None
    finding_data: Mapping[str, object]
    timestamp: float


CATEGORY_FAMILY_MAP: dict[str, str] = {
    # QUALITY_DEFECT
    "quality": "quality_defect",
    "correctness": "quality_defect",
    "defect": "quality_defect",
    "code_smell": "quality_defect",
    "bug": "quality_defect",
    "style": "quality_defect",
    "maintainability": "quality_defect",
    "best_practice": "quality_defect",
    # SECURITY
    "security": "security",
    "security_vulnerability": "security",
    "vulnerability": "security",
    # TEST_GAP
    "tests": "test_gap",
    "test_gap": "test_gap",
    "coverage": "test_gap",
    "untested": "test_gap",
    # DOCUMENTATION
    "documentation": "documentation",
    "doc_issue": "documentation",
    "docs": "documentation",
    "docstring": "documentation",
}


def normalize_category_family(category: str) -> str:
    """Normalize raw finding category into its canonical category family."""
    cleaned = (category or "").strip().lower().replace("-", "_").replace(" ", "_")
    return CATEGORY_FAMILY_MAP.get(cleaned, cleaned)


_DEFECT_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "uses", "using", "used", "use",
    "leads", "lead", "leading", "across", "code", "issue", "defect", "finding",
    "candidate", "potential", "detected", "observed", "found", "error", "line",
    "lines", "this", "that", "there", "has", "have", "had", "should", "could",
    "would", "may", "might", "will", "can", "not", "no", "and", "or", "but",
    "as", "if", "when", "than", "so", "such", "all", "any", "each", "every",
    "both", "either", "neither", "one", "two", "into", "onto", "under", "over"
})


def extract_semantic_tokens(text: str) -> set[str]:
    """Extract normalized semantic tokens from text, filtering stopwords and volatile IDs."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    tokens = set()
    for w in words:
        if w.startswith(("cand", "corr", "can_")) or (len(w) >= 8 and all(c in "0123456789abcdef" for c in w)):
            continue
        if w not in _DEFECT_STOPWORDS and len(w) > 1:
            tokens.add(w)
    return tokens


def compute_defect_fingerprint(text: str) -> str:
    """Compute a deterministic semantic defect fingerprint string."""
    tokens = sorted(extract_semantic_tokens(text))
    if not tokens:
        cleaned = re.sub(r"[^a-z0-9]", "_", (text or "").lower().strip())
        return cleaned[:32] or "defect"
    return "_".join(tokens[:8])


class SeverityCalibrator:
    """Deterministic, auditable calibration layer between raw model severity and policy severity."""

    @classmethod
    def calibrate(
        cls,
        *,
        raw_severity: str,
        category: str,
        summary: str,
        rationale: str,
        contributing_specialists: Sequence[str] = (),
    ) -> tuple[str, str, str]:
        """Calibrate raw model severity based on category family, defect pattern, and evidence.

        Returns:
            (calibrated_severity, calibration_rule, calibration_reason)
        """
        raw_sev = (raw_severity or "medium").strip().lower()
        if raw_sev not in ("critical", "high", "blocking", "medium", "low", "info"):
            raw_sev = "medium"
        norm_fam = normalize_category_family(category)
        text = f"{summary} {rationale}".lower()

        # 1. CRITICAL checks: reserve critical for catastrophic security or unrecoverable data loss
        if raw_sev in ("critical", "blocking"):
            is_catastrophic_sec = norm_fam == "security" and any(
                term in text for term in (
                    "remote code execution", "rce", "sql injection", "command injection",
                    "hardcoded private key", "arbitrary code execution"
                )
            )
            is_data_loss = any(term in text for term in ("data loss", "data corruption", "unrecoverable"))
            if is_catastrophic_sec:
                return ("critical", "critical_security_catastrophic", "Verified catastrophic security impact; critical severity preserved")
            if is_data_loss:
                return ("critical", "critical_data_loss", "Verified catastrophic data loss risk; critical severity preserved")
            # If not catastrophic, normalize down to high
            return ("high", "critical_capped_to_high", "Critical severity requires catastrophic compromise evidence; calibrated to high")

        # 2. DOCUMENTATION: non-executable, cannot directly cause runtime crash
        if norm_fam == "documentation":
            is_material_contradiction = any(
                term in text for term in (
                    "contradict", "incorrectly states", "claims safe", "claims", "mismatch",
                    "wrong return", "wrong parameter", "doc contradiction", "inconsistent"
                )
            )
            if is_material_contradiction:
                return (
                    "medium",
                    "doc_material_contradiction_medium",
                    "Documentation materially contradicts runtime code behavior; calibrated to medium severity",
                )
            if raw_sev == "high":
                return (
                    "low",
                    "doc_high_capped_to_low",
                    "Documentation defect cannot cause direct runtime failure; calibrated from high down to low",
                )
            if raw_sev == "medium":
                return (
                    "medium",
                    "doc_medium_preserved",
                    "Documentation discrepancy preserved as medium severity",
                )
            return (
                "low",
                "doc_standard_low",
                "Documentation improvement / clarification calibrated to low severity",
            )

        # 3. QUALITY / MAINTAINABILITY / CORRECTNESS
        if norm_fam == "quality_defect":
            # Check for mutable default argument pattern
            is_mutable_default = any(
                term in text for term in (
                    "mutable default", "default argument", "list = []", "dict = {}", "default parameter"
                )
            )
            if is_mutable_default:
                return (
                    "medium",
                    "quality_mutable_default_medium",
                    "Mutable default argument is a quality/anti-pattern defect; calibrated to medium severity",
                )

            # Check for non-critical code smells
            is_code_smell = any(
                term in text for term in (
                    "code smell", "dead code", "unused", "naming", "refactor",
                    "style", "formatting", "complexity", "duplicate code"
                )
            )
            if is_code_smell:
                if raw_sev in ("high", "critical"):
                    return (
                        "medium",
                        "quality_smell_capped_medium",
                        "Code smell / maintainability issue cannot be high severity; calibrated down to medium",
                    )
                return (
                    raw_sev if raw_sev in ("medium", "low", "info") else "low",
                    "quality_smell_preserved",
                    f"Code quality smell preserved as {raw_sev}",
                )

            # Check for genuine runtime crashes / unhandled exceptions
            is_runtime_crash = any(
                term in text for term in (
                    "zerodivisionerror", "division by zero", "nullpointer", "unhandled exception",
                    "raises ", "crash", "indexerror", "keyerror", "attributeerror", "typeerror",
                    "infinite loop", "deadlock", "memory leak", "resource leak"
                )
            )
            if is_runtime_crash:
                if raw_sev in ("high", "critical"):
                    return (
                        "high",
                        "correctness_runtime_crash_high",
                        "Genuine runtime crash/unhandled exception constitutes high severity correctness defect",
                    )
                return (
                    "medium",
                    "correctness_runtime_crash_medium",
                    "Runtime error under specific conditions preserved as medium severity",
                )

            # General correctness / defect: if raw is high, preserve high
            if raw_sev == "high":
                return ("high", "correctness_high_preserved", "Correctness defect preserved as high severity")

            return (raw_sev, "quality_preserved", f"Quality defect preserved as {raw_sev}")

        # 4. TEST GAP
        if norm_fam == "test_gap":
            is_ineffective_assertion = any(
                term in text for term in (
                    "smoke test", "ineffective assertion", "no assertion", "useless test",
                    "assert true", "tautological", "does not verify output"
                )
            )
            if is_ineffective_assertion:
                return (
                    "low",
                    "test_gap_ineffective_assertion_low",
                    "Smoke test lacking behavioral assertions; calibrated to low severity",
                )

            is_critical_gap = any(
                term in text for term in (
                    "critical security", "auth test", "safety critical", "authentication boundary"
                )
            )
            if is_critical_gap and raw_sev == "high":
                return (
                    "high",
                    "test_gap_critical_boundary_high",
                    "Missing test for critical security/safety boundary; high severity preserved",
                )

            if raw_sev in ("high", "critical"):
                return (
                    "medium",
                    "test_gap_high_capped_to_medium",
                    "Missing test coverage is quality risk; calibrated from high down to medium",
                )
            if raw_sev == "medium":
                return ("medium", "test_gap_medium_preserved", "Test gap preserved as medium severity")
            return ("low", "test_gap_low_preserved", "Minor test gap preserved as low severity")

        # 5. SECURITY
        if norm_fam == "security":
            is_security_context = any(
                term in text for term in (
                    "password", "token", "secret", "reset", "session", "auth", "csrf",
                    "credential", "crypto", "encryption", "privilege", "injection", "rce", "cwe",
                    "tamper", "leak", "authorization", "bypass"
                )
            )
            is_benign_randomness = any(
                term in text for term in (
                    "ui badge", "cosmetic", "display", "sample", "color", "non-security", "benign"
                )
            )
            if is_benign_randomness:
                return (
                    "low",
                    "security_benign_context_low",
                    "Non-security context with cosmetic/benign randomness; calibrated down to low",
                )
            if is_security_context:
                if raw_sev in ("high", "critical"):
                    return (
                        "high" if raw_sev == "high" else "critical",
                        "security_sensitive_context_high",
                        "Verified security-sensitive context; high/critical severity preserved",
                    )
                return ("medium", "security_sensitive_context_medium", "Security finding in sensitive context preserved as medium")
            # Ambiguous security context lacking verified sensitive evidence:
            if raw_sev in ("high", "critical"):
                return (
                    "medium",
                    "security_ambiguous_capped_medium",
                    "Security finding lacking verified sensitive context; calibrated down to medium pending review",
                )
            return (raw_sev, "security_preserved", f"Security finding preserved as {raw_sev}")

        # Default fallback
        return (raw_sev, "default_preserved", f"Severity preserved as {raw_sev}")


class FindingAggregator:
    """Aggregates and deduplicates specialist CandidateFindings into CanonicalFindings."""

    def aggregate(
        self,
        candidates: Sequence[CandidateFinding],
        *,
        repository_id: str,
        head_sha: str,
        delivery_id: str = "",
        run_id: str = "",
    ) -> list[CanonicalFinding]:
        """Deduplicate candidates by file path and overlapping line ranges deterministically."""
        if not candidates:
            return []

        # Sort candidates deterministically by finding_id
        sorted_candidates = sorted(candidates, key=lambda c: c.finding_id)

        # Group candidates by normalized file_path
        by_file: dict[str, list[CandidateFinding]] = {}
        for c in sorted_candidates:
            file_key = (c.file_path or "").strip().replace("\\", "/")
            by_file.setdefault(file_key, []).append(c)

        canonical_findings: list[CanonicalFinding] = []

        for file_path in sorted(by_file.keys()):
            file_candidates = by_file[file_path]
            # Cluster candidates that overlap in line range and category family
            clusters: list[list[CandidateFinding]] = []
            for cand in file_candidates:
                placed = False
                for cluster in clusters:
                    if any(self._should_merge(cand, existing) for existing in cluster):
                        cluster.append(cand)
                        placed = True
                        break
                if not placed:
                    clusters.append([cand])

            for cluster in clusters:
                canonical = self._create_canonical(
                    cluster,
                    repository_id=repository_id,
                    head_sha=head_sha,
                    delivery_id=delivery_id,
                    run_id=run_id,
                )
                canonical_findings.append(canonical)

        # Sort canonical findings deterministically by file_path and start line
        canonical_findings.sort(key=lambda c: (c.file_path, c.line_range[0], c.canonical_id))
        return canonical_findings

    def _should_merge(self, a: CandidateFinding, b: CandidateFinding) -> bool:
        path_a = (a.file_path or "").strip().replace("\\", "/")
        path_b = (b.file_path or "").strip().replace("\\", "/")
        if path_a != path_b:
            return False

        # Check line range overlap (with 2-line proximity tolerance)
        r_a = a.line_range or (0, 0)
        r_b = b.line_range or (0, 0)
        overlaps = max(r_a[0], r_b[0]) <= min(r_a[1], r_b[1]) + 2
        if not overlaps:
            return False

        # Check category alignment via normalized category family
        fam_a = normalize_category_family(a.category)
        fam_b = normalize_category_family(b.category)
        if fam_a != fam_b:
            return False

        # Check semantic defect identity: ensure two distinct defects on the same line remain separate
        return self._are_defects_semantically_equivalent(a, b)

    def _are_defects_semantically_equivalent(self, a: CandidateFinding, b: CandidateFinding) -> bool:
        """Determine if two findings in the same category family represent the same defect."""
        toks_a = extract_semantic_tokens(a.summary)
        toks_b = extract_semantic_tokens(b.summary)

        # If both summaries contain semantic tokens, evaluate their overlap
        if toks_a and toks_b:
            common = toks_a & toks_b
            if not common:
                # Fallback: check if rationales establish strong semantic overlap
                rat_a = extract_semantic_tokens(a.rationale)
                rat_b = extract_semantic_tokens(b.rationale)
                common_rat = rat_a & rat_b
                return len(common_rat) >= 2

            # They share tokens. If identical token sets, definitely same defect
            if toks_a == toks_b:
                return True

            # If multiple tokens are shared, they represent the same defect
            if len(common) >= 2:
                return True

            # If only 1 token is shared, verify if it's a substantive overlap or just a shared code symbol
            jaccard = len(common) / len(toks_a | toks_b)
            min_ratio = len(common) / min(len(toks_a), len(toks_b))
            return jaccard >= 0.3 or min_ratio >= 0.5

        # If either summary lacked extractable semantic tokens, fall back to rationale or conservative merge
        rat_a = extract_semantic_tokens(a.rationale)
        rat_b = extract_semantic_tokens(b.rationale)
        if rat_a and rat_b:
            return bool(rat_a & rat_b)

        return True

    def _create_canonical(
        self,
        cluster: list[CandidateFinding],
        *,
        repository_id: str,
        head_sha: str,
        delivery_id: str = "",
        run_id: str = "",
    ) -> CanonicalFinding:
        # Determine highest severity and primary candidate deterministically
        cluster_sorted = sorted(
            cluster,
            key=lambda c: (
                SEVERITY_WEIGHTS.get(c.severity.lower(), 1),
                c.confidence,
                c.summary,
            ),
            reverse=True,
        )
        top = cluster_sorted[0]
        raw_severity = top.severity.lower()

        # Merged category preserves original category of top candidate
        merged_category = top.category.lower()
        norm_family = normalize_category_family(merged_category)

        # Confidence calculation: calibrated maximum of contributing candidates
        merged_confidence = min(1.0, round(max(c.confidence for c in cluster), 4))

        # Min start line, max end line
        start_lines = [c.line_range[0] for c in cluster if c.line_range]
        end_lines = [c.line_range[1] for c in cluster if c.line_range]
        line_range = (min(start_lines) if start_lines else 1, max(end_lines) if end_lines else 1)

        file_path = (top.file_path or "").strip().replace("\\", "/")
        contributing_ids = tuple(sorted(set(c.finding_id for c in cluster)))
        contributing_specs = tuple(
            sorted(
                set(
                    c.specialist_type.value if hasattr(c.specialist_type, "value") else str(c.specialist_type)
                    for c in cluster
                )
            )
        )

        # Collect and deduplicate evidence references preserving order
        all_refs: list[str] = []
        for c in cluster:
            for r in c.evidence_refs:
                if r not in all_refs:
                    all_refs.append(r)

        # Deterministic semantic defect fingerprint
        defect_fp = compute_defect_fingerprint(top.summary)

        # Deterministic canonical finding ID derived strictly from stable attributes:
        # repository + file path + normalized line/span + normalized category family + semantic defect fingerprint
        # Notice: contributing_ids (volatile candidate UUIDs) are intentionally EXCLUDED.
        seed = f"{repository_id}:{file_path}:{line_range[0]}_{line_range[1]}:{norm_family}:{defect_fp}"
        canonical_id = f"can-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"

        merge_rationale = (
            f"Merged {len(cluster)} specialist finding(s) from {list(contributing_specs)} "
            f"on {file_path}:{line_range[0]}-{line_range[1]}"
        )

        # Deterministic severity calibration
        calibrated_sev, cal_rule, cal_reason = SeverityCalibrator.calibrate(
            raw_severity=raw_severity,
            category=merged_category,
            summary=top.summary,
            rationale=top.rationale,
            contributing_specialists=contributing_specs,
        )

        return CanonicalFinding(
            canonical_id=canonical_id,
            repository_id=repository_id,
            head_sha=head_sha,
            category=merged_category,
            severity=calibrated_sev,
            confidence=merged_confidence,
            summary=top.summary,
            rationale=top.rationale,
            file_path=file_path,
            line_range=line_range,
            contributing_candidate_ids=contributing_ids,
            contributing_specialists=contributing_specs,
            evidence_refs=tuple(all_refs),
            remediation=top.remediation,
            merge_rationale=merge_rationale,
            delivery_id=delivery_id,
            run_id=run_id,
            raw_severity=raw_severity,
            calibrated_severity=calibrated_sev,
            calibration_rule=cal_rule,
            calibration_reason=cal_reason,
        )


class ReviewPolicyEngine:
    """Configurable policy engine evaluating canonical findings against confidence and risk rules."""

    def __init__(
        self,
        *,
        policy_version: str = "v1",
        min_auto_approve_confidence: float = 0.8,
        blocking_severities: Sequence[str] = ("critical", "high", "blocking"),
        blocking_categories: Sequence[str] = ("security",),
        auto_approvable_severities: Sequence[str] = ("medium", "low", "info"),
    ) -> None:
        self.policy_version = policy_version
        self.min_auto_approve_confidence = min_auto_approve_confidence
        self.blocking_severities = frozenset(s.lower() for s in blocking_severities)
        self.blocking_categories = frozenset(c.lower() for c in blocking_categories)
        self.auto_approvable_severities = frozenset(s.lower() for s in auto_approvable_severities)

    def evaluate(
        self,
        finding: CanonicalFinding,
        *,
        is_fresh: bool = True,
        evidence_results: Sequence[FindingValidationResult] = (),
    ) -> CanonicalFinding:
        """Evaluate policy disposition for a canonical finding, strictly failing closed on uncertainty."""
        # Rule 1: Freshness verification (NFR-01, INVARIANT-fd3d3e39)
        if not is_fresh:
            return self._with_disposition(
                finding,
                FindingDisposition.HELD,
                "Stale or uncertain repository revision; fails closed to HELD",
            )

        # Rule 2: Evidence validity check (AC-05, INVARIANT-dae4200e)
        contributing_set = set(finding.contributing_candidate_ids)
        matching_validations = [v for v in evidence_results if v.finding_id in contributing_set]
        if matching_validations and any(v.is_suppressed or v.status != "verified" for v in matching_validations):
            return self._with_disposition(
                finding,
                FindingDisposition.SUPPRESSED,
                "Contributing candidate finding contains unverified or suppressed evidence",
            )

        if not finding.evidence_refs:
            return self._with_disposition(
                finding,
                FindingDisposition.SUPPRESSED,
                "Finding lacks verified evidence references (AC-05 / FR-08)",
            )

        # Rule 3: High-impact or blocking risk policy (AC-07, DECISION-cf2aec81)
        if (
            finding.severity.lower() in self.blocking_severities
            or finding.category.lower() in self.blocking_categories
        ):
            return self._with_disposition(
                finding,
                FindingDisposition.HELD,
                f"High-impact finding ({finding.severity}/{finding.category}) requires human maintainer approval",
            )

        # Rule 4: Confidence threshold check (FR-10)
        if finding.confidence < self.min_auto_approve_confidence:
            return self._with_disposition(
                finding,
                FindingDisposition.HELD,
                f"Confidence {finding.confidence} below threshold {self.min_auto_approve_confidence}; held for review",
            )

        # Rule 5: Explicit auto-approval criteria (NFR-10)
        if finding.severity.lower() in self.auto_approvable_severities:
            return self._with_disposition(
                finding,
                FindingDisposition.AUTO_APPROVED,
                f"Lower-risk finding ({finding.severity}) meets confidence threshold {finding.confidence} with verified evidence",
            )

        # Default fail-closed fallback
        return self._with_disposition(
            finding,
            FindingDisposition.HELD,
            "Ambiguous policy disposition; failing closed to HELD",
        )

    def _with_disposition(
        self,
        finding: CanonicalFinding,
        disposition: FindingDisposition,
        reason: str,
    ) -> CanonicalFinding:
        return CanonicalFinding(
            canonical_id=finding.canonical_id,
            repository_id=finding.repository_id,
            head_sha=finding.head_sha,
            category=finding.category,
            severity=finding.severity,
            confidence=finding.confidence,
            summary=finding.summary,
            rationale=finding.rationale,
            file_path=finding.file_path,
            line_range=finding.line_range,
            contributing_candidate_ids=finding.contributing_candidate_ids,
            contributing_specialists=finding.contributing_specialists,
            evidence_refs=finding.evidence_refs,
            remediation=finding.remediation,
            disposition=disposition,
            disposition_reason=reason,
            merge_rationale=finding.merge_rationale,
            policy_version=self.policy_version,
            delivery_id=finding.delivery_id,
            run_id=finding.run_id,
            raw_severity=finding.raw_severity,
            calibrated_severity=finding.calibrated_severity,
            calibration_rule=finding.calibration_rule,
            calibration_reason=finding.calibration_reason,
        )


class ReviewTruthStore:
    """Durable SQLite-backed Review Truth store tracking the append-only finding lifecycle."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_truth (
                    record_id TEXT PRIMARY KEY,
                    canonical_id TEXT NOT NULL,
                    sequence_id INTEGER NOT NULL,
                    repository_id TEXT NOT NULL,
                    head_sha TEXT NOT NULL,
                    delivery_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    actor TEXT,
                    actor_role TEXT,
                    rationale TEXT,
                    finding_json TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_truth_canonical ON review_truth (canonical_id, sequence_id ASC)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_truth_repo_head ON review_truth (repository_id, head_sha)"
            )

    def record_initial(
        self,
        finding: CanonicalFinding,
        *,
        delivery_id: str | None = None,
        run_id: str | None = None,
        initial_state: TruthState,
        actor: str | None = None,
        actor_role: str | None = None,
        rationale: str | None = None,
        now: float | None = None,
    ) -> ReviewTruthRecord:
        """Record the initial finding in Review Truth."""
        current_time = time.time() if now is None else now
        deliv = delivery_id or finding.delivery_id or "delivery-unknown"
        r_id = run_id or finding.run_id or "run-unknown"
        sequence_id = 1
        record_id = f"truth-{finding.canonical_id}-{sequence_id}"
        finding_dict = asdict(finding)
        # Convert enums to string values
        finding_dict["disposition"] = finding.disposition.value
        finding_json = json.dumps(finding_dict, sort_keys=True)

        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO review_truth (
                    record_id, canonical_id, sequence_id, repository_id, head_sha,
                    delivery_id, run_id, state, actor, actor_role,
                    rationale, finding_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    finding.canonical_id,
                    sequence_id,
                    finding.repository_id,
                    finding.head_sha,
                    deliv,
                    r_id,
                    initial_state.value,
                    actor,
                    actor_role,
                    rationale or finding.disposition_reason,
                    finding_json,
                    current_time,
                ),
            )

        return ReviewTruthRecord(
            record_id=record_id,
            canonical_id=finding.canonical_id,
            sequence_id=sequence_id,
            repository_id=finding.repository_id,
            head_sha=finding.head_sha,
            delivery_id=deliv,
            run_id=r_id,
            state=initial_state,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale or finding.disposition_reason,
            finding_data=finding_dict,
            timestamp=current_time,
        )

    def record_transition(
        self,
        canonical_id: str,
        new_state: TruthState,
        *,
        actor: str | None = None,
        actor_role: str | None = None,
        rationale: str | None = None,
        updated_finding: CanonicalFinding | None = None,
        now: float | None = None,
    ) -> ReviewTruthRecord:
        """Validate and record a state transition in the append-only lifecycle."""
        current_time = time.time() if now is None else now
        latest = self.get_latest_state(canonical_id)
        if latest is None:
            raise KeyError(f"Canonical finding {canonical_id} not found in Review Truth")

        current_state = latest.state
        allowed = ALLOWED_TRANSITIONS.get(current_state, frozenset())
        if new_state not in allowed:
            raise ValueError(
                f"Illegal state transition from {current_state.value} to {new_state.value} for {canonical_id}"
            )

        row = self.connection.execute(
            "SELECT COALESCE(MAX(sequence_id), 0) + 1 FROM review_truth WHERE canonical_id = ?",
            (canonical_id,),
        ).fetchone()
        seq = row[0] if row else latest.sequence_id + 1
        record_id = f"truth-{canonical_id}-{seq}"

        finding_dict = asdict(updated_finding) if updated_finding else dict(latest.finding_data)
        if updated_finding:
            finding_dict["disposition"] = updated_finding.disposition.value
        finding_json = json.dumps(finding_dict, sort_keys=True)

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO review_truth (
                    record_id, canonical_id, sequence_id, repository_id, head_sha,
                    delivery_id, run_id, state, actor, actor_role,
                    rationale, finding_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    canonical_id,
                    seq,
                    latest.repository_id,
                    latest.head_sha,
                    latest.delivery_id,
                    latest.run_id,
                    new_state.value,
                    actor,
                    actor_role,
                    rationale,
                    finding_json,
                    current_time,
                ),
            )

        return ReviewTruthRecord(
            record_id=record_id,
            canonical_id=canonical_id,
            sequence_id=seq,
            repository_id=latest.repository_id,
            head_sha=latest.head_sha,
            delivery_id=latest.delivery_id,
            run_id=latest.run_id,
            state=new_state,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
            finding_data=finding_dict,
            timestamp=current_time,
        )

    def get_latest_state(self, canonical_id: str) -> ReviewTruthRecord | None:
        row = self.connection.execute(
            """
            SELECT record_id, canonical_id, sequence_id, repository_id, head_sha,
                   delivery_id, run_id, state, actor, actor_role,
                   rationale, finding_json, timestamp
            FROM review_truth
            WHERE canonical_id = ?
            ORDER BY sequence_id DESC
            LIMIT 1
            """,
            (canonical_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_history(self, canonical_id: str) -> list[ReviewTruthRecord]:
        rows = self.connection.execute(
            """
            SELECT record_id, canonical_id, sequence_id, repository_id, head_sha,
                   delivery_id, run_id, state, actor, actor_role,
                   rationale, finding_json, timestamp
            FROM review_truth
            WHERE canonical_id = ?
            ORDER BY sequence_id ASC
            """,
            (canonical_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_state(self, state: TruthState) -> list[ReviewTruthRecord]:
        """List the latest records matching a given state."""
        canonical_ids = self.list_all_canonical()
        results: list[ReviewTruthRecord] = []
        for cid in canonical_ids:
            latest = self.get_latest_state(cid)
            if latest and latest.state == state:
                results.append(latest)
        return results

    def list_all_canonical(self) -> list[str]:
        """List all distinct canonical finding IDs in Review Truth."""
        rows = self.connection.execute(
            "SELECT DISTINCT canonical_id FROM review_truth ORDER BY canonical_id ASC"
        ).fetchall()
        return [r[0] for r in rows]

    def _row_to_record(self, row: tuple) -> ReviewTruthRecord:
        return ReviewTruthRecord(
            record_id=row[0],
            canonical_id=row[1],
            sequence_id=row[2],
            repository_id=row[3],
            head_sha=row[4],
            delivery_id=row[5],
            run_id=row[6],
            state=TruthState(row[7]),
            actor=row[8],
            actor_role=row[9],
            rationale=row[10],
            finding_data=json.loads(row[11]),
            timestamp=row[12],
        )


class MaintainerWorkflow:
    """Provides maintainer human-in-the-loop review actions with strict authorization."""

    def __init__(self, truth_store: ReviewTruthStore) -> None:
        self.truth_store = truth_store

    def _check_auth(self, actor: str, actor_role: str) -> None:
        if not actor or not actor.strip():
            raise ValueError("Maintainer action requires a verified actor identity")
        if not actor_role or actor_role.lower() not in AUTHORIZED_HITL_ROLES:
            raise PermissionError(
                f"Actor role '{actor_role}' is not authorized to perform HITL decisions. Allowed: {sorted(AUTHORIZED_HITL_ROLES)}"
            )

    def approve(
        self,
        canonical_id: str,
        *,
        actor: str,
        actor_role: str,
        rationale: str,
    ) -> ReviewTruthRecord:
        """Approve a held or disputed finding for publication."""
        self._check_auth(actor, actor_role)
        if not rationale or not rationale.strip():
            raise ValueError("Human approval requires an explicit rationale")
        return self.truth_store.record_transition(
            canonical_id,
            TruthState.APPROVED,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
        )

    def reject(
        self,
        canonical_id: str,
        *,
        actor: str,
        actor_role: str,
        rationale: str,
    ) -> ReviewTruthRecord:
        """Reject a finding."""
        self._check_auth(actor, actor_role)
        if not rationale or not rationale.strip():
            raise ValueError("Human rejection requires an explicit rationale")
        return self.truth_store.record_transition(
            canonical_id,
            TruthState.REJECTED,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
        )

    def edit(
        self,
        canonical_id: str,
        updated_finding: CanonicalFinding,
        *,
        actor: str,
        actor_role: str,
        rationale: str,
    ) -> ReviewTruthRecord:
        """Edit a held or disputed finding's content or metadata."""
        self._check_auth(actor, actor_role)
        if not rationale or not rationale.strip():
            raise ValueError("Human edit requires an explicit rationale")
        latest = self.truth_store.get_latest_state(canonical_id)
        if latest is None:
            raise KeyError(f"Canonical finding {canonical_id} not found in Review Truth")
        if latest.state not in {TruthState.HELD, TruthState.DISPUTED}:
            raise ValueError(f"Cannot edit finding in state {latest.state.value}")
        return self.truth_store.record_transition(
            canonical_id,
            latest.state,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
            updated_finding=updated_finding,
        )

    def dismiss(
        self,
        canonical_id: str,
        *,
        actor: str,
        actor_role: str,
        rationale: str,
    ) -> ReviewTruthRecord:
        """Dismiss a finding."""
        self._check_auth(actor, actor_role)
        if not rationale or not rationale.strip():
            raise ValueError("Dismissing a finding requires an explicit rationale")
        return self.truth_store.record_transition(
            canonical_id,
            TruthState.DISMISSED,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
        )

    def dispute(
        self,
        canonical_id: str,
        *,
        actor: str,
        actor_role: str,
        rationale: str,
    ) -> ReviewTruthRecord:
        """Dispute an auto-approved or held finding."""
        self._check_auth(actor, actor_role)
        if not rationale or not rationale.strip():
            raise ValueError("Disputing a finding requires an explicit rationale")
        return self.truth_store.record_transition(
            canonical_id,
            TruthState.DISPUTED,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
        )

    def resolve(
        self,
        canonical_id: str,
        *,
        actor: str,
        actor_role: str,
        rationale: str,
    ) -> ReviewTruthRecord:
        """Resolve a previously disputed finding."""
        self._check_auth(actor, actor_role)
        if not rationale or not rationale.strip():
            raise ValueError("Resolving a disputed finding requires an explicit rationale")
        return self.truth_store.record_transition(
            canonical_id,
            TruthState.RESOLVED,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
        )

    def supersede(
        self,
        canonical_id: str,
        *,
        actor: str = "system",
        actor_role: str = "maintainer",
        rationale: str = "New head SHA revision arrived",
    ) -> ReviewTruthRecord:
        """Supersede a finding when head SHA changes."""
        return self.truth_store.record_transition(
            canonical_id,
            TruthState.SUPERSEDED,
            actor=actor,
            actor_role=actor_role,
            rationale=rationale,
        )
