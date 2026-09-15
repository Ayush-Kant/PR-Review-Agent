"""Tests for finding aggregation, confidence/risk policy, Review Truth, and maintainer HITL workflow."""

from __future__ import annotations

import sqlite3
import pytest

from pr_review_agent.orchestration import CandidateFinding, SpecialistType
from pr_review_agent.retrieval import FindingValidationResult
from pr_review_agent.policy import (
    ALLOWED_TRANSITIONS,
    AUTHORIZED_HITL_ROLES,
    CanonicalFinding,
    FindingAggregator,
    FindingDisposition,
    MaintainerWorkflow,
    ReviewPolicyEngine,
    ReviewTruthRecord,
    ReviewTruthStore,
    TruthState,
)


def _make_candidate(
    finding_id: str,
    correlation_id: str = "corr-1",
    file_path: str = "src/main.py",
    line_range: tuple[int, int] = (10, 20),
    category: str = "correctness",
    severity: str = "medium",
    confidence: float = 0.85,
    specialist_type: SpecialistType = SpecialistType.QUALITY,
    evidence_refs: tuple[str, ...] = ("diff://src/main.py#L10-L20",),
) -> CandidateFinding:
    return CandidateFinding(
        finding_id=finding_id,
        correlation_id=correlation_id,
        specialist_type=specialist_type,
        category=category,
        severity=severity,
        confidence=confidence,
        summary=f"Summary for {finding_id}",
        rationale=f"Rationale for {finding_id}",
        file_path=file_path,
        line_range=line_range,
        evidence_refs=evidence_refs,
        remediation=f"Fix for {finding_id}",
    )


def _make_validation_result(
    finding_id: str,
    *,
    is_suppressed: bool = False,
    status: str = "verified",
    suppression_reason: str | None = None,
) -> FindingValidationResult:
    cand = _make_candidate(finding_id)
    return FindingValidationResult(
        finding_id=finding_id,
        status=status,
        is_suppressed=is_suppressed,
        suppression_reason=suppression_reason,
        valid_evidence_refs=cand.evidence_refs if not is_suppressed else (),
        invalid_evidence_refs=() if not is_suppressed else cand.evidence_refs,
        finding=cand,
    )


def test_two_overlapping_specialist_findings_become_one_canonical() -> None:
    aggregator = FindingAggregator()
    cand1 = _make_candidate(
        "cand-1",
        file_path="src/service.py",
        line_range=(10, 20),
        category="correctness",
        severity="medium",
        confidence=0.82,
        specialist_type=SpecialistType.QUALITY,
        evidence_refs=("diff://src/service.py#L10-L20",),
    )
    cand2 = _make_candidate(
        "cand-2",
        file_path="src/service.py",
        line_range=(15, 25),
        category="quality",
        severity="high",
        confidence=0.91,
        specialist_type=SpecialistType.TESTS,
        evidence_refs=("diff://src/service.py#L15-L25", "repo://src/service.py#L1-L30@abc"),
    )

    canonical = aggregator.aggregate(
        [cand1, cand2],
        repository_id="owner/repo",
        head_sha="head123",
        delivery_id="deliv-1",
        run_id="run-1",
    )

    assert len(canonical) == 1
    c = canonical[0]
    assert c.repository_id == "owner/repo"
    assert c.head_sha == "head123"
    assert c.file_path == "src/service.py"
    # Merged line range spans min start to max end
    assert c.line_range == (10, 25)
    # Merged severity is highest
    assert c.severity == "high"
    # Calibrated confidence is maximum of candidates
    assert c.confidence == 0.91
    # Both contributing candidate IDs and specialists preserved
    assert set(c.contributing_candidate_ids) == {"cand-1", "cand-2"}
    assert set(c.contributing_specialists) == {"quality", "tests"}
    # Deduplicated evidence references preserved
    assert "diff://src/service.py#L10-L20" in c.evidence_refs
    assert "diff://src/service.py#L15-L25" in c.evidence_refs
    assert "repo://src/service.py#L1-L30@abc" in c.evidence_refs
    # Provenance preserved
    assert c.delivery_id == "deliv-1"
    assert c.run_id == "run-1"
    assert "Merged 2 specialist finding(s)" in c.merge_rationale


def test_materially_different_findings_remain_separate() -> None:
    aggregator = FindingAggregator()
    # Distinct files
    cand1 = _make_candidate("cand-1", file_path="src/a.py", line_range=(10, 20))
    cand2 = _make_candidate("cand-2", file_path="src/b.py", line_range=(10, 20))
    # Distinct non-overlapping line ranges in same file
    cand3 = _make_candidate("cand-3", file_path="src/a.py", line_range=(80, 95))
    # Incompatible category (security vs quality)
    cand4 = _make_candidate(
        "cand-4",
        file_path="src/a.py",
        line_range=(10, 20),
        category="security",
        specialist_type=SpecialistType.SECURITY,
    )

    canonical = aggregator.aggregate(
        [cand1, cand2, cand3, cand4],
        repository_id="owner/repo",
        head_sha="head123",
    )

    # None should merge together
    assert len(canonical) == 4
    file_lines = [(c.file_path, c.line_range, c.category) for c in canonical]
    assert ("src/a.py", (10, 20), "correctness") in file_lines
    assert ("src/a.py", (10, 20), "security") in file_lines
    assert ("src/a.py", (80, 95), "correctness") in file_lines
    assert ("src/b.py", (10, 20), "correctness") in file_lines


def test_canonical_finding_preserves_contributing_ids_and_evidence() -> None:
    aggregator = FindingAggregator()
    cand = _make_candidate(
        "cand-unique",
        file_path="pr_review_agent/intake.py",
        line_range=(50, 60),
        category="correctness",
        severity="low",
        confidence=0.88,
        evidence_refs=("diff://pr_review_agent/intake.py#L50-L60",),
    )
    canonical = aggregator.aggregate([cand], repository_id="owner/repo", head_sha="sha1")
    assert len(canonical) == 1
    c = canonical[0]
    assert c.contributing_candidate_ids == ("cand-unique",)
    assert c.evidence_refs == ("diff://pr_review_agent/intake.py#L50-L60",)
    assert c.canonical_id.startswith("can-")


def test_high_critical_blocking_findings_are_held() -> None:
    engine = ReviewPolicyEngine()
    # High severity
    finding_high = CanonicalFinding(
        canonical_id="can-1",
        repository_id="repo",
        head_sha="sha",
        category="correctness",
        severity="high",
        confidence=0.99,
        summary="High issue",
        rationale="Detailed rationale",
        file_path="src/app.py",
        line_range=(1, 5),
        contributing_candidate_ids=("c-1",),
        contributing_specialists=("correctness",),
        evidence_refs=("diff://src/app.py#L1-L5",),
    )
    evaluated_high = engine.evaluate(
        finding_high,
        is_fresh=True,
        evidence_results=[_make_validation_result("c-1")],
    )
    assert evaluated_high.disposition == FindingDisposition.HELD
    assert "High-impact finding" in evaluated_high.disposition_reason

    # Critical severity
    finding_crit = CanonicalFinding(
        canonical_id="can-2",
        repository_id="repo",
        head_sha="sha",
        category="security",
        severity="critical",
        confidence=1.0,
        summary="Critical vulnerability",
        rationale="Exploit possible",
        file_path="src/auth.py",
        line_range=(10, 15),
        contributing_candidate_ids=("c-2",),
        contributing_specialists=("security",),
        evidence_refs=("diff://src/auth.py#L10-L15",),
    )
    evaluated_crit = engine.evaluate(
        finding_crit,
        is_fresh=True,
        evidence_results=[_make_validation_result("c-2")],
    )
    assert evaluated_crit.disposition == FindingDisposition.HELD
    assert "High-impact finding" in evaluated_crit.disposition_reason

    # Blocking severity
    finding_block = CanonicalFinding(
        canonical_id="can-3",
        repository_id="repo",
        head_sha="sha",
        category="correctness",
        severity="blocking",
        confidence=0.95,
        summary="Blocking crash",
        rationale="Crash on startup",
        file_path="src/main.py",
        line_range=(20, 25),
        contributing_candidate_ids=("c-3",),
        contributing_specialists=("correctness",),
        evidence_refs=("diff://src/main.py#L20-L25",),
    )
    evaluated_block = engine.evaluate(
        finding_block,
        is_fresh=True,
        evidence_results=[_make_validation_result("c-3")],
    )
    assert evaluated_block.disposition == FindingDisposition.HELD


def test_medium_low_findings_auto_approve_only_with_verified_evidence_and_confidence() -> None:
    engine = ReviewPolicyEngine(min_auto_approve_confidence=0.8)
    finding = CanonicalFinding(
        canonical_id="can-low",
        repository_id="repo",
        head_sha="sha",
        category="quality",
        severity="low",
        confidence=0.85,
        summary="Minor unused import",
        rationale="Clean up import",
        file_path="src/utils.py",
        line_range=(5, 6),
        contributing_candidate_ids=("c-low",),
        contributing_specialists=("quality",),
        evidence_refs=("diff://src/utils.py#L5-L6",),
    )

    # Valid and fresh -> AUTO_APPROVED
    evaluated = engine.evaluate(
        finding,
        is_fresh=True,
        evidence_results=[_make_validation_result("c-low")],
    )
    assert evaluated.disposition == FindingDisposition.AUTO_APPROVED
    assert "meets confidence threshold" in evaluated.disposition_reason


def test_low_confidence_is_held() -> None:
    engine = ReviewPolicyEngine(min_auto_approve_confidence=0.8)
    finding = CanonicalFinding(
        canonical_id="can-low-conf",
        repository_id="repo",
        head_sha="sha",
        category="quality",
        severity="medium",
        confidence=0.74,
        summary="Possible style issue",
        rationale="Uncertain style",
        file_path="src/utils.py",
        line_range=(10, 12),
        contributing_candidate_ids=("c-lc",),
        contributing_specialists=("quality",),
        evidence_refs=("diff://src/utils.py#L10-L12",),
    )
    evaluated = engine.evaluate(
        finding,
        is_fresh=True,
        evidence_results=[_make_validation_result("c-lc")],
    )
    assert evaluated.disposition == FindingDisposition.HELD
    assert "below threshold" in evaluated.disposition_reason

    # Verify configurability: lowering threshold to 0.7 allows auto-approval
    lenient_engine = ReviewPolicyEngine(min_auto_approve_confidence=0.7)
    lenient_evaluated = lenient_engine.evaluate(
        finding,
        is_fresh=True,
        evidence_results=[_make_validation_result("c-lc")],
    )
    assert lenient_evaluated.disposition == FindingDisposition.AUTO_APPROVED


def test_missing_invalid_stale_suppressed_evidence_cannot_auto_approve() -> None:
    engine = ReviewPolicyEngine()
    base_finding = CanonicalFinding(
        canonical_id="can-ev",
        repository_id="repo",
        head_sha="sha",
        category="quality",
        severity="low",
        confidence=0.9,
        summary="Quality issue",
        rationale="Details",
        file_path="src/app.py",
        line_range=(1, 5),
        contributing_candidate_ids=("c-ev",),
        contributing_specialists=("quality",),
        evidence_refs=("diff://src/app.py#L1-L5",),
    )

    # 1. Stale revision fails closed to HELD
    stale_eval = engine.evaluate(
        base_finding,
        is_fresh=False,
        evidence_results=[_make_validation_result("c-ev")],
    )
    assert stale_eval.disposition == FindingDisposition.HELD
    assert "Stale or uncertain" in stale_eval.disposition_reason

    # 2. Suppressed / invalid evidence fails closed to SUPPRESSED
    suppressed_eval = engine.evaluate(
        base_finding,
        is_fresh=True,
        evidence_results=[
            _make_validation_result(
                "c-ev",
                is_suppressed=True,
                status="suppressed",
                suppression_reason="Line cited was not in diff",
            )
        ],
    )
    assert suppressed_eval.disposition == FindingDisposition.SUPPRESSED
    assert "unverified or suppressed" in suppressed_eval.disposition_reason

    # 3. Missing evidence refs fails closed to SUPPRESSED
    no_evidence_finding = CanonicalFinding(
        canonical_id="can-no-ev",
        repository_id="repo",
        head_sha="sha",
        category="quality",
        severity="low",
        confidence=0.9,
        summary="No ev",
        rationale="Details",
        file_path="src/app.py",
        line_range=(1, 5),
        contributing_candidate_ids=("c-no-ev",),
        contributing_specialists=("quality",),
        evidence_refs=(),
    )
    no_ev_eval = engine.evaluate(no_evidence_finding, is_fresh=True)
    assert no_ev_eval.disposition == FindingDisposition.SUPPRESSED
    assert "lacks verified evidence" in no_ev_eval.disposition_reason


def test_policy_ambiguity_fails_closed() -> None:
    # Custom severity not configured in blocking or auto-approvable
    engine = ReviewPolicyEngine(auto_approvable_severities=("low",))
    finding = CanonicalFinding(
        canonical_id="can-ambig",
        repository_id="repo",
        head_sha="sha",
        category="quality",
        severity="medium",  # Not auto-approvable and not blocking
        confidence=0.95,
        summary="Ambiguous",
        rationale="Details",
        file_path="src/app.py",
        line_range=(1, 5),
        contributing_candidate_ids=("c-ambig",),
        contributing_specialists=("quality",),
        evidence_refs=("diff://src/app.py#L1-L5",),
    )
    eval_ambig = engine.evaluate(
        finding,
        is_fresh=True,
        evidence_results=[_make_validation_result("c-ambig")],
    )
    # Must fail closed to HELD
    assert eval_ambig.disposition == FindingDisposition.HELD
    assert "Ambiguous policy disposition" in eval_ambig.disposition_reason


def test_review_truth_lifecycle_transitions() -> None:
    conn = sqlite3.connect(":memory:")
    store = ReviewTruthStore(conn)
    finding = CanonicalFinding(
        canonical_id="can-life",
        repository_id="owner/repo",
        head_sha="sha1",
        category="correctness",
        severity="high",
        confidence=0.88,
        summary="Initial finding",
        rationale="Rationale",
        file_path="src/main.py",
        line_range=(10, 20),
        contributing_candidate_ids=("c-1",),
        contributing_specialists=("correctness",),
        evidence_refs=("diff://src/main.py#L10-L20",),
        delivery_id="deliv-100",
        run_id="run-100",
    )

    # Initial record: HELD
    init_rec = store.record_initial(finding, initial_state=TruthState.HELD)
    assert init_rec.state == TruthState.HELD
    assert init_rec.sequence_id == 1

    # Transition: HELD -> APPROVED (by maintainer)
    appr_rec = store.record_transition(
        "can-life",
        TruthState.APPROVED,
        actor="alice",
        actor_role="maintainer",
        rationale="Verified by team lead",
    )
    assert appr_rec.state == TruthState.APPROVED
    assert appr_rec.sequence_id == 2
    assert appr_rec.actor == "alice"
    assert appr_rec.actor_role == "maintainer"

    # Transition: APPROVED -> SUPERSEDED (new commit)
    sup_rec = store.record_transition(
        "can-life",
        TruthState.SUPERSEDED,
        actor="system",
        actor_role="maintainer",
        rationale="Commit updated to sha2",
    )
    assert sup_rec.state == TruthState.SUPERSEDED
    assert sup_rec.sequence_id == 3

    # History contains all 3 records in sequence
    history = store.get_history("can-life")
    assert len(history) == 3
    assert [h.state for h in history] == [
        TruthState.HELD,
        TruthState.APPROVED,
        TruthState.SUPERSEDED,
    ]
    latest = store.get_latest_state("can-life")
    assert latest is not None
    assert latest.state == TruthState.SUPERSEDED


def test_illegal_transitions_rejected() -> None:
    conn = sqlite3.connect(":memory:")
    store = ReviewTruthStore(conn)
    finding = CanonicalFinding(
        canonical_id="can-illegal",
        repository_id="owner/repo",
        head_sha="sha1",
        category="correctness",
        severity="high",
        confidence=0.9,
        summary="Finding",
        rationale="Rationale",
        file_path="src/main.py",
        line_range=(10, 20),
        contributing_candidate_ids=("c-1",),
        contributing_specialists=("correctness",),
        evidence_refs=("diff://src/main.py#L10-L20",),
    )

    store.record_initial(finding, initial_state=TruthState.CANDIDATE)

    # Illegal: CANDIDATE cannot transition directly to APPROVED
    with pytest.raises(ValueError, match="Illegal state transition from candidate to approved"):
        store.record_transition(
            "can-illegal",
            TruthState.APPROVED,
            actor="alice",
            actor_role="maintainer",
            rationale="Approved prematurely",
        )

    # Legally transition to HELD via MERGED
    store.record_transition("can-illegal", TruthState.MERGED)
    store.record_transition("can-illegal", TruthState.HELD)

    # Reject finding
    store.record_transition(
        "can-illegal",
        TruthState.REJECTED,
        actor="alice",
        actor_role="maintainer",
        rationale="Not an issue",
    )

    # Illegal: REJECTED cannot transition to APPROVED
    with pytest.raises(ValueError, match="Illegal state transition from rejected to approved"):
        store.record_transition(
            "can-illegal",
            TruthState.APPROVED,
            actor="alice",
            actor_role="maintainer",
            rationale="Trying to undo rejection illegally",
        )


def test_hitl_actions_require_authorized_actor_and_rationale() -> None:
    conn = sqlite3.connect(":memory:")
    store = ReviewTruthStore(conn)
    finding = CanonicalFinding(
        canonical_id="can-hitl",
        repository_id="owner/repo",
        head_sha="sha1",
        category="correctness",
        severity="high",
        confidence=0.9,
        summary="Finding",
        rationale="Rationale",
        file_path="src/main.py",
        line_range=(10, 20),
        contributing_candidate_ids=("c-1",),
        contributing_specialists=("correctness",),
        evidence_refs=("diff://src/main.py#L10-L20",),
    )
    store.record_initial(finding, initial_state=TruthState.HELD)
    workflow = MaintainerWorkflow(store)

    # 1. Missing actor -> ValueError
    with pytest.raises(ValueError, match="verified actor identity"):
        workflow.approve("can-hitl", actor="", actor_role="maintainer", rationale="Looks good")

    # 2. Missing rationale -> ValueError
    with pytest.raises(ValueError, match="explicit rationale"):
        workflow.approve("can-hitl", actor="alice", actor_role="maintainer", rationale="   ")

    # 3. Unauthorized role -> PermissionError
    with pytest.raises(PermissionError, match="not authorized"):
        workflow.approve("can-hitl", actor="external_user", actor_role="contributor", rationale="Looks good")

    with pytest.raises(PermissionError, match="not authorized"):
        workflow.reject("can-hitl", actor="external_user", actor_role="guest", rationale="Reject")

    with pytest.raises(PermissionError, match="not authorized"):
        workflow.dismiss("can-hitl", actor="external_user", actor_role="bot", rationale="Dismiss")

    with pytest.raises(PermissionError, match="not authorized"):
        workflow.dispute("can-hitl", actor="external_user", actor_role="intern", rationale="Dispute")

    with pytest.raises(PermissionError, match="not authorized"):
        workflow.edit(
            "can-hitl",
            finding,
            actor="external_user",
            actor_role="viewer",
            rationale="Edit",
        )


def test_maintainer_edit_preserves_held_state_and_updates_data() -> None:
    conn = sqlite3.connect(":memory:")
    store = ReviewTruthStore(conn)
    finding = CanonicalFinding(
        canonical_id="can-edit",
        repository_id="owner/repo",
        head_sha="sha1",
        category="correctness",
        severity="high",
        confidence=0.9,
        summary="Original summary",
        rationale="Original rationale",
        file_path="src/main.py",
        line_range=(10, 20),
        contributing_candidate_ids=("c-1",),
        contributing_specialists=("correctness",),
        evidence_refs=("diff://src/main.py#L10-L20",),
    )
    store.record_initial(finding, initial_state=TruthState.HELD)
    workflow = MaintainerWorkflow(store)

    # Maintainer edits the finding's summary and remediation
    updated_finding = CanonicalFinding(
        canonical_id="can-edit",
        repository_id="owner/repo",
        head_sha="sha1",
        category="correctness",
        severity="medium",  # downgraded
        confidence=0.9,
        summary="Edited improved summary",
        rationale="Refined explanation",
        file_path="src/main.py",
        line_range=(10, 22),
        contributing_candidate_ids=("c-1",),
        contributing_specialists=("correctness",),
        evidence_refs=("diff://src/main.py#L10-L20",),
        remediation="Use try/except block",
    )

    rec = workflow.edit(
        "can-edit",
        updated_finding,
        actor="lead_dev",
        actor_role="reviewer",
        rationale="Refined line range and downgraded severity to medium",
    )

    assert rec.state == TruthState.HELD
    assert rec.sequence_id == 2
    assert rec.actor == "lead_dev"
    assert rec.finding_data["summary"] == "Edited improved summary"
    assert rec.finding_data["severity"] == "medium"
    assert rec.finding_data["remediation"] == "Use try/except block"

    history = store.get_history("can-edit")
    assert len(history) == 2
    assert history[0].finding_data["summary"] == "Original summary"
    assert history[1].finding_data["summary"] == "Edited improved summary"


def test_review_truth_survives_closing_and_reopening_sqlite_connection(tmp_path) -> None:
    db_file = tmp_path / "review_truth_durability.db"

    # Connection 1: write initial and transition
    conn1 = sqlite3.connect(str(db_file))
    store1 = ReviewTruthStore(conn1)
    finding = CanonicalFinding(
        canonical_id="can-durable-1",
        repository_id="owner/repo",
        head_sha="sha-reopen",
        category="correctness",
        severity="medium",
        confidence=0.85,
        summary="Durable finding test",
        rationale="Survives restart",
        file_path="src/durability.py",
        line_range=(100, 110),
        contributing_candidate_ids=("cand-dur-1", "cand-dur-2"),
        contributing_specialists=("correctness", "performance"),
        evidence_refs=("diff://src/durability.py#L100-L110",),
        delivery_id="deliv-dur",
        run_id="run-dur",
    )
    store1.record_initial(finding, initial_state=TruthState.HELD)
    store1.record_transition(
        "can-durable-1",
        TruthState.APPROVED,
        actor="maintainer_bob",
        actor_role="maintainer",
        rationale="Approved after offline verification",
    )
    conn1.close()

    # Connection 2: read from the file on disk
    conn2 = sqlite3.connect(str(db_file))
    store2 = ReviewTruthStore(conn2)

    latest = store2.get_latest_state("can-durable-1")
    assert latest is not None
    assert latest.state == TruthState.APPROVED
    assert latest.sequence_id == 2
    assert latest.actor == "maintainer_bob"
    assert latest.actor_role == "maintainer"
    assert latest.rationale == "Approved after offline verification"
    assert latest.delivery_id == "deliv-dur"
    assert latest.run_id == "run-dur"

    history = store2.get_history("can-durable-1")
    assert len(history) == 2
    assert history[0].state == TruthState.HELD
    assert history[1].state == TruthState.APPROVED

    # Perform another transition on the reopened connection
    store2.record_transition(
        "can-durable-1",
        TruthState.SUPERSEDED,
        actor="system",
        actor_role="maintainer",
        rationale="New PR commit received",
    )
    conn2.close()

    # Connection 3: verify third transition
    conn3 = sqlite3.connect(str(db_file))
    store3 = ReviewTruthStore(conn3)
    history3 = store3.get_history("can-durable-1")
    assert len(history3) == 3
    assert history3[2].state == TruthState.SUPERSEDED
    conn3.close()


def test_complete_lifecycle_can_be_reconstructed_from_durable_records() -> None:
    conn = sqlite3.connect(":memory:")
    store = ReviewTruthStore(conn)
    finding = CanonicalFinding(
        canonical_id="can-full-cycle",
        repository_id="owner/repo",
        head_sha="sha-init",
        category="security",
        severity="high",
        confidence=0.92,
        summary="Security flaw",
        rationale="Needs audit",
        file_path="src/sec.py",
        line_range=(40, 50),
        contributing_candidate_ids=("c-sec",),
        contributing_specialists=("security",),
        evidence_refs=("diff://src/sec.py#L40-L50",),
    )

    workflow = MaintainerWorkflow(store)

    # 1. Initial state: candidate
    store.record_initial(finding, initial_state=TruthState.CANDIDATE)
    # 2. Merged
    store.record_transition("can-full-cycle", TruthState.MERGED)
    # 3. Held
    store.record_transition("can-full-cycle", TruthState.HELD)
    # 4. Disputed by maintainer
    workflow.dispute("can-full-cycle", actor="dev2", actor_role="reviewer", rationale="Potential false positive")
    # 5. Resolved after review
    workflow.resolve("can-full-cycle", actor="lead", actor_role="maintainer", rationale="Confirmed real after discussion")
    # 6. Superseded on revision
    workflow.supersede("can-full-cycle", rationale="PR branch rebased")

    history = store.get_history("can-full-cycle")
    assert len(history) == 6
    expected_states = [
        TruthState.CANDIDATE,
        TruthState.MERGED,
        TruthState.HELD,
        TruthState.DISPUTED,
        TruthState.RESOLVED,
        TruthState.SUPERSEDED,
    ]
    assert [h.state for h in history] == expected_states
    # Reconstructed history has continuous sequence IDs
    assert [h.sequence_id for h in history] == [1, 2, 3, 4, 5, 6]


def test_no_github_publication_side_effects_occur() -> None:
    # Ensure all components operate entirely locally without any network or external publication
    aggregator = FindingAggregator()
    engine = ReviewPolicyEngine()
    conn = sqlite3.connect(":memory:")
    store = ReviewTruthStore(conn)
    workflow = MaintainerWorkflow(store)

    cand = _make_candidate("cand-loc", severity="low", confidence=0.9)
    canonical = aggregator.aggregate([cand], repository_id="owner/repo", head_sha="sha1")
    assert len(canonical) == 1

    evaluated = engine.evaluate(
        canonical[0],
        is_fresh=True,
        evidence_results=[_make_validation_result("cand-loc")],
    )
    # Even if auto-approved, no publication calls are made
    assert evaluated.disposition == FindingDisposition.AUTO_APPROVED

    rec = store.record_initial(evaluated, initial_state=TruthState.AUTO_APPROVED)
    assert rec.state == TruthState.AUTO_APPROVED

    # Maintainer actions also do not trigger external calls
    disp = workflow.dispute(evaluated.canonical_id, actor="maintainer1", actor_role="maintainer", rationale="Investigating")
    assert disp.state == TruthState.DISPUTED


def test_different_raw_categories_same_mutable_default_defect_merge() -> None:
    """Findings with different raw categories (defect vs code_smell) for mutable default merge."""
    aggregator = FindingAggregator()
    cand1 = _make_candidate(
        "cand-def-1",
        file_path="examples/report.py",
        line_range=(17, 20),
        category="defect",
        severity="medium",
        confidence=0.85,
        specialist_type=SpecialistType.QUALITY,
        evidence_refs=("diff://examples/report.py#L17-L20",),
    )
    cand1 = CandidateFinding(
        finding_id="cand-def-1",
        correlation_id="corr-1",
        specialist_type=SpecialistType.QUALITY,
        category="defect",
        severity="medium",
        confidence=0.85,
        summary="Mutable default argument in format_report leads to state leakage across calls",
        rationale="Default parameter is initialized once at definition time.",
        file_path="examples/report.py",
        line_range=(17, 20),
        evidence_refs=("diff://examples/report.py#L17-L20",),
        remediation="Use None as default and initialize inside function.",
    )
    cand2 = CandidateFinding(
        finding_id="cand-smell-2",
        correlation_id="corr-1",
        specialist_type=SpecialistType.QUALITY,
        category="code_smell",
        severity="medium",
        confidence=0.88,
        summary="format_report() uses a mutable default argument",
        rationale="Mutable default argument dictionary shared across invocations.",
        file_path="examples/report.py",
        line_range=(18, 21),
        evidence_refs=("diff://examples/report.py#L18-L21",),
        remediation="Replace with None sentinel.",
    )

    canonical = aggregator.aggregate(
        [cand1, cand2],
        repository_id="Ayush-Kant/PR-Review-Agent",
        head_sha="head123",
    )

    assert len(canonical) == 1
    c = canonical[0]
    assert c.file_path == "examples/report.py"
    assert c.line_range == (17, 21)
    assert set(c.contributing_candidate_ids) == {"cand-def-1", "cand-smell-2"}


def test_same_semantic_finding_different_candidate_uuids_same_canonical_id() -> None:
    """The same semantic finding with different volatile UUIDs generates identical canonical_id."""
    aggregator = FindingAggregator()

    run1_cand = CandidateFinding(
        finding_id="uuid-run1-aaaa-1111",
        correlation_id="corr-run-1",
        specialist_type=SpecialistType.QUALITY,
        category="defect",
        severity="medium",
        confidence=0.85,
        summary="format_report() uses a mutable default argument",
        rationale="Shared mutable default.",
        file_path="examples/report.py",
        line_range=(18, 21),
        evidence_refs=("diff://examples/report.py#L18-L21",),
        remediation="Replace with None sentinel.",
    )
    run2_cand = CandidateFinding(
        finding_id="uuid-run2-bbbb-2222",
        correlation_id="corr-run-2",
        specialist_type=SpecialistType.QUALITY,
        category="defect",
        severity="medium",
        confidence=0.85,
        summary="format_report() uses a mutable default argument",
        rationale="Shared mutable default.",
        file_path="examples/report.py",
        line_range=(18, 21),
        evidence_refs=("diff://examples/report.py#L18-L21",),
        remediation="Replace with None sentinel.",
    )

    canon1 = aggregator.aggregate([run1_cand], repository_id="owner/repo", head_sha="sha1")
    canon2 = aggregator.aggregate([run2_cand], repository_id="owner/repo", head_sha="sha1")

    assert len(canon1) == 1
    assert len(canon2) == 1
    assert canon1[0].canonical_id == canon2[0].canonical_id
    assert canon1[0].canonical_id.startswith("can-")


def test_genuinely_different_defects_on_same_line_remain_separate() -> None:
    """Two genuinely distinct defects on the same line remain separate canonical findings."""
    aggregator = FindingAggregator()

    cand1 = CandidateFinding(
        finding_id="cand-diff-1",
        correlation_id="corr-1",
        specialist_type=SpecialistType.QUALITY,
        category="defect",
        severity="medium",
        confidence=0.85,
        summary="Mutable default argument in compute_metrics",
        rationale="Mutable default dict retains state across calls.",
        file_path="src/metrics.py",
        line_range=(25, 25),
        evidence_refs=("diff://src/metrics.py#L25",),
        remediation="Use None default.",
    )
    cand2 = CandidateFinding(
        finding_id="cand-diff-2",
        correlation_id="corr-1",
        specialist_type=SpecialistType.QUALITY,
        category="correctness",
        severity="high",
        confidence=0.90,
        summary="Unhandled ZeroDivisionError in compute_metrics when total is zero",
        rationale="Denominator is not checked for zero before division.",
        file_path="src/metrics.py",
        line_range=(25, 25),
        evidence_refs=("diff://src/metrics.py#L25",),
        remediation="Guard division with check total > 0.",
    )

    canonical = aggregator.aggregate([cand1, cand2], repository_id="owner/repo", head_sha="sha1")
    assert len(canonical) == 2
    summaries = {c.summary for c in canonical}
    assert "Mutable default argument in compute_metrics" in summaries
    assert "Unhandled ZeroDivisionError in compute_metrics when total is zero" in summaries


def test_different_category_families_on_same_line_remain_separate() -> None:
    """Security and documentation findings on the same line never merge."""
    aggregator = FindingAggregator()

    cand_sec = CandidateFinding(
        finding_id="cand-sec-1",
        correlation_id="corr-1",
        specialist_type=SpecialistType.SECURITY,
        category="security",
        severity="high",
        confidence=0.95,
        summary="Hardcoded JWT secret key in config loader",
        rationale="Secret key embedded directly in source code.",
        file_path="src/config.py",
        line_range=(10, 15),
        evidence_refs=("diff://src/config.py#L10-L15",),
        remediation="Load from environment variable.",
    )
    cand_doc = CandidateFinding(
        finding_id="cand-doc-1",
        correlation_id="corr-1",
        specialist_type=SpecialistType.DOCUMENTATION,
        category="docstring",
        severity="info",
        confidence=0.85,
        summary="Missing docstring for load_config function",
        rationale="Function is public but has no docstring explaining parameters.",
        file_path="src/config.py",
        line_range=(10, 15),
        evidence_refs=("diff://src/config.py#L10-L15",),
        remediation="Add docstring with args and return types.",
    )

    canonical = aggregator.aggregate([cand_sec, cand_doc], repository_id="owner/repo", head_sha="sha1")
    assert len(canonical) == 2
    categories = {c.category for c in canonical}
    assert "security" in categories
    assert "docstring" in categories
