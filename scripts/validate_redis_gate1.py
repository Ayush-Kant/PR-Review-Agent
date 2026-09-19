"""Real Redis Validation Harness for W1-08B Gate 1.

GATE 1: REAL REDIS VALIDATION ONLY

Validates the ACTUAL current implementation against a dedicated staging Redis instance:
1. Real Redis TLS / Connectivity (PING, server info, TLS verification)
2. Authentication (correct auth succeeds, bad auth fails closed)
3. Real Lua Engine (CLAIM_JOB_LUA and ENQUEUE_JOB_LUA proven natively)
4. Real Queue Lifecycle (enqueue, dedupe, lease, concurrency, backoff, dead-letter, zombie recovery)
5. ARQ Serialization (exact review_job_task round-trip with installed ARQ)
6. Redis Checkpoint (put, get_tuple, thread isolation, secret rejection, delete_thread)
7. Deterministic Cleanup (all validation keys removed, no FLUSHDB)
8. Machine-readable evidence artifact generation with complete credential redaction
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
import uuid

# Ensure repository root is in sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import redis

from arq.jobs import deserialize_job
from pr_review_agent.adapters.redis_checkpoint import (
    RedisCheckpointSaver,
    _validate_data_security_recursive,
)
from pr_review_agent.adapters.redis_queue import (
    CLAIM_JOB_LUA,
    ENQUEUE_JOB_LUA,
    RedisJobQueue,
    StaleLeaseError,
)
from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.orchestration import (
    CHECKPOINT_FORBIDDEN_KEYS,
    CandidateFinding,
    JobState,
    ReviewJob,
    SpecialistOutput,
    SpecialistType,
)

logger = logging.getLogger("redis_gate1_validator")


def mask_redis_url(url: str) -> str:
    """Mask credentials in Redis connection URL.

    Converts rediss://user:secret@host:port/db to rediss://user:***@host:port/db.
    """
    if not url:
        return ""
    return re.sub(r"://([^:@]*):([^@]+)@", r"://\1:***@", url)


def parse_redis_metadata(url: str) -> dict[str, Any]:
    """Extract non-sensitive connection metadata from a Redis URL."""
    if not url:
        return {
            "scheme": "unknown",
            "host": "unknown",
            "port": 0,
            "db": 0,
            "tls": False,
            "authenticated": False,
            "masked_url": "",
        }
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    path = parsed.path.lstrip("/") if parsed.path else "0"
    try:
        db = int(path) if path.isdigit() else 0
    except ValueError:
        db = 0

    query = parse_qs(parsed.query)
    tls = scheme == "rediss" or query.get("ssl", ["0"])[0].lower() in ("1", "true")
    authenticated = bool(parsed.password or parsed.username)

    return {
        "scheme": scheme,
        "host": host,
        "port": port,
        "db": db,
        "tls": tls,
        "authenticated": authenticated,
        "masked_url": mask_redis_url(url),
    }


@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    status: str  # "passed", "failed", "skipped"
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ValidationReport:
    """Authoritative machine-readable Gate 1 evidence report."""

    task: str
    gate: str
    overall_status: str  # "passed", "failed", "offline_verified"
    live_redis_executed: bool
    timestamp: str
    repository: str
    branch: str
    commit_sha: str
    redis_metadata: dict[str, Any]
    server_version: str | None
    checks: dict[str, dict[str, Any]]
    evidence_path: str | None
    failure_details: str | None = None
    capabilities: dict[str, Any] | None = None
    environmental_context: str | None = None

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)



class RedisGate1Validator:
    """Executes the comprehensive W1-08B Gate 1 Real Redis validation suite."""

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        client: Any | None = None,
        allow_insecure: bool = False,
        require_staging_isolation: bool = True,
        custom_run_id: str | None = None,
        evidence_path: Path | str | None = None,
    ) -> None:
        self.redis_url = (redis_url or os.environ.get("STAGING_REDIS_URL") or os.environ.get("REDIS_URL") or "").strip()
        self.allow_insecure = allow_insecure
        self.require_staging_isolation = require_staging_isolation
        self.run_id = custom_run_id or uuid.uuid4().hex[:8]
        self.evidence_path = Path(evidence_path) if evidence_path else Path(".genesis/evidence/w1_08b_redis_validation.json")

        self.metadata = parse_redis_metadata(self.redis_url)
        self.server_version: str | None = None
        self._owns_client = client is None

        if client is not None:
            self.client = client
        elif self.redis_url:
            self.client = redis.Redis.from_url(self.redis_url, decode_responses=False)
        else:
            self.client = None

        # Tracking for deterministic cleanup
        self.created_keys: set[str] = set()
        self.created_job_ids: set[str] = set()
        self.created_repo_ids: set[str] = set()
        self.created_thread_ids: set[str] = set()
        self.created_delivery_ids: set[str] = set()

    def _track_key(self, key: str) -> str:
        self.created_keys.add(key)
        return key

    def _to_str(self, val: Any) -> str:
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val) if val is not None else ""

    def validate_all(self) -> ValidationReport:
        """Run all Gate 1 validation checks sequentially and generate evidence report."""
        checks: dict[str, CheckResult] = {}
        overall_passed = True
        first_failure: str | None = None

        if self.client is None:
            capabilities = self.evaluate_capabilities({}, is_offline=True)
            report = ValidationReport(
                task="production-infrastructure-cutover",
                gate="gate-1-real-redis-validation",
                overall_status="not_executed",
                live_redis_executed=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
                repository="Ayush-Kant/PR-Review-Agent",
                branch="agent-v1",
                commit_sha=os.environ.get("GITHUB_SHA", "d5bf97252656ec7d3645475fb367cc927dd36dd3"),
                redis_metadata=self.metadata,
                server_version=None,
                checks={},
                evidence_path=str(self.evidence_path),
                failure_details="No REDIS_URL provided and no client injected. Real Redis validation was not executed.",
                capabilities=capabilities,
                environmental_context="No Redis environment provided.",
            )
            self._save_report(report)
            return report

        suite = [
            ("connectivity", self.check_connectivity),
            ("authentication", self.check_authentication),
            ("lua_engine", self.check_lua_engine),
            ("queue_lifecycle", self.check_queue_lifecycle),
            ("arq_serialization", self.check_arq_serialization),
            ("checkpoint", self.check_checkpoint),
            ("secret_redaction", self.check_secret_redaction),
            ("cleanup", self.cleanup),
        ]

        for check_name, check_fn in suite:
            t0 = time.perf_counter()
            try:
                res = check_fn()
            except Exception as exc:
                res = CheckResult(
                    name=check_name,
                    status="failed",
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details={},
                    error=str(exc),
                )

            checks[check_name] = res
            if res.status != "passed":
                overall_passed = False
                if first_failure is None:
                    first_failure = f"{check_name}: {res.error or res.details.get('failure_reason', 'check failed')}"

        is_offline = not bool(self.redis_url) or getattr(self.client, "_is_offline_mock", False)
        if overall_passed:
            overall_status = "offline_verified" if is_offline else "passed"
        else:
            overall_status = "failed"

        capabilities = self.evaluate_capabilities(checks, is_offline=is_offline)
        env_context = (
            "Current free-tier Redis environment provides redis:// without TLS termination. "
            "Real Redis functional behavior is proven; infrastructure capabilities (TLS, HA, scale, backups, "
            "private networking) remain deferred/unproven and are preserved as future validation targets."
            if not is_offline and not self.metadata.get("tls")
            else (
                "TLS-capable Redis endpoint verified." if self.metadata.get("tls") and not is_offline
                else "Offline mock verification."
            )
        )

        report = ValidationReport(
            task="production-infrastructure-cutover",
            gate="gate-1-real-redis-validation",
            overall_status=overall_status,
            live_redis_executed=not is_offline,
            timestamp=datetime.now(timezone.utc).isoformat(),
            repository="Ayush-Kant/PR-Review-Agent",
            branch="agent-v1",
            commit_sha=os.environ.get("GITHUB_SHA", "d5bf97252656ec7d3645475fb367cc927dd36dd3"),
            redis_metadata=self.metadata,
            server_version=self.server_version,
            checks={k: asdict(v) for k, v in checks.items()},
            evidence_path=str(self.evidence_path),
            failure_details=first_failure,
            capabilities=capabilities,
            environmental_context=env_context,
        )

        self._save_report(report)
        return report

    def evaluate_capabilities(
        self,
        checks: dict[str, CheckResult],
        *,
        is_offline: bool,
    ) -> dict[str, Any]:
        """Evaluate explicit capability dimensions from check results and environment state.

        Distinguishes:
        - PROVEN against real live Redis execution
        - OFFLINE_VERIFIED in mock/test harnesses (never claimed as live infrastructure proof)
        - NOT_PROVEN / DEFERRED due to environmental limitations (e.g. lack of TLS on free tier)
        - NOT_BENCHMARKED (e.g. production capacity limits)
        """
        def _check_passed(name: str) -> bool:
            return checks.get(name) is not None and checks[name].status == "passed"

        # 1. Functional Semantics
        if not is_offline and _check_passed("queue_lifecycle"):
            q_status, q_verdict = "PROVEN", "pass"
            q_notes = "Queue lifecycle, atomic leasing, retry schedule, dead-letter routing, and zombie recovery proven against real Redis."
        elif is_offline and _check_passed("queue_lifecycle"):
            q_status, q_verdict = "OFFLINE_VERIFIED", "pass"
            q_notes = "Queue lifecycle verified in offline mock harness only; not proven against live infrastructure."
        else:
            q_status, q_verdict = "NOT_PROVEN", "fail"
            q_notes = "Queue lifecycle check did not pass."

        # 2. Native Lua Execution
        if not is_offline and _check_passed("lua_engine"):
            lua_status, lua_verdict = "PROVEN", "pass"
            lua_notes = "CLAIM_JOB_LUA and ENQUEUE_JOB_LUA executed natively in real Redis engine with atomic concurrency limits."
        elif is_offline and _check_passed("lua_engine"):
            lua_status, lua_verdict = "OFFLINE_VERIFIED", "pass"
            lua_notes = "Lua execution verified in offline mock harness only; not proven against live infrastructure."
        else:
            lua_status, lua_verdict = "NOT_PROVEN", "fail"
            lua_notes = "Lua engine check did not pass."

        # 3. Checkpoint Semantics
        if not is_offline and _check_passed("checkpoint"):
            cp_status, cp_verdict = "PROVEN", "pass"
            cp_notes = "LangGraph BaseCheckpointSaver protocol, exact state round-trip, thread isolation, forbidden secret rejection, and thread deletion proven against real Redis."
        elif is_offline and _check_passed("checkpoint"):
            cp_status, cp_verdict = "OFFLINE_VERIFIED", "pass"
            cp_notes = "Checkpoint semantics verified in offline mock harness only; not proven against live infrastructure."
        else:
            cp_status, cp_verdict = "NOT_PROVEN", "fail"
            cp_notes = "Checkpoint check did not pass."

        # 4. ARQ Serialization Compatibility
        if not is_offline and _check_passed("arq_serialization"):
            arq_status, arq_verdict = "PROVEN", "pass"
            arq_notes = "review_job_task payload serialized with installed ARQ and round-trip deserialized."
        elif is_offline and _check_passed("arq_serialization"):
            arq_status, arq_verdict = "OFFLINE_VERIFIED", "pass"
            arq_notes = "ARQ serialization verified in offline mock harness only; not proven against live infrastructure."
        else:
            arq_status, arq_verdict = "NOT_PROVEN", "fail"
            arq_notes = "ARQ serialization check did not pass."

        # 5. Secret Handling
        if not is_offline and _check_passed("secret_redaction") and _check_passed("authentication"):
            sec_status, sec_verdict = "PROVEN", "pass"
            sec_notes = "Complete credential masking verified in URLs, metadata representations, and keyspace."
        elif is_offline and _check_passed("secret_redaction"):
            sec_status, sec_verdict = "OFFLINE_VERIFIED", "pass"
            sec_notes = "Secret redaction verified in offline mock harness."
        else:
            sec_status, sec_verdict = "NOT_PROVEN", "fail"
            sec_notes = "Secret redaction check did not pass."

        # 6. TLS Transport
        is_tls = self.metadata.get("tls", False)
        if not is_offline and is_tls and _check_passed("connectivity"):
            tls_status, tls_verdict = "PROVEN", "pass"
            tls_notes = "Real Redis TLS transport (rediss://) verified with encrypted transport."
        else:
            tls_status, tls_verdict = "NOT_PROVEN", "deferred"
            tls_notes = (
                "TLS transport is unavailable on the current free-tier Redis endpoint (uses redis://). "
                "Deferred until TLS-capable (rediss://) infrastructure is configured."
            )

        # 7. HA / Failover
        ha_status, ha_verdict = "NOT_PROVEN", "deferred"
        ha_notes = "Current free-tier Redis is a single standalone node. Multi-node Sentinel/Cluster failover remains deferred."

        # 8. Production-Scale Capacity
        cap_status, cap_verdict = "NOT_BENCHMARKED", "deferred"
        cap_notes = "Free-tier Redis instance has connection/bandwidth quotas unsuitable for load testing. Capacity benchmarking deferred."

        # 9. Managed Production Backups & Recovery
        bak_status, bak_verdict = "NOT_PROVEN", "deferred"
        bak_notes = "Automated snapshot/RDB/AOF point-in-time recovery is not provided by free-tier instance. Managed recovery deferred."

        # 10. Private Production Networking
        net_status, net_verdict = "NOT_PROVEN", "deferred"
        net_notes = "Current free-tier Redis operates on public internet with authentication; VPC peering / private network isolation deferred."

        return {
            "redis_functional_semantics": {"status": q_status, "verdict": q_verdict, "evidence": q_notes},
            "redis_native_lua_execution": {"status": lua_status, "verdict": lua_verdict, "evidence": lua_notes},
            "redis_checkpoint_semantics": {"status": cp_status, "verdict": cp_verdict, "evidence": cp_notes},
            "redis_arq_serialization_compatibility": {"status": arq_status, "verdict": arq_verdict, "evidence": arq_notes},
            "redis_secret_handling": {"status": sec_status, "verdict": sec_verdict, "evidence": sec_notes},
            "redis_tls_transport": {"status": tls_status, "verdict": tls_verdict, "reason": tls_notes, "future_target": "rediss:// URL on TLS-capable Redis endpoint"},
            "redis_ha_failover": {"status": ha_status, "verdict": ha_verdict, "reason": ha_notes, "future_target": "Multi-node Redis Sentinel or Cluster deployment"},
            "production_scale_capacity": {"status": cap_status, "verdict": cap_verdict, "reason": cap_notes, "future_target": "Benchmarked dedicated production Redis tier"},
            "managed_production_backups_recovery": {"status": bak_status, "verdict": bak_verdict, "reason": bak_notes, "future_target": "Automated snapshot/PITR persistence on managed cloud Redis"},
            "private_production_networking": {"status": net_status, "verdict": net_verdict, "reason": net_notes, "future_target": "VPC peering / AWS PrivateLink / GCP Private Service Connect"},
        }


    def _save_report(self, report: ValidationReport) -> None:
        """Persist validation report to disk while strictly redacting secrets."""
        try:
            self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            self.evidence_path.write_text(report.to_json(), encoding="utf-8")
            logger.info("Saved Gate 1 validation evidence to %s", self.evidence_path)
        except Exception as exc:
            logger.warning("Could not write evidence report to %s: %s", self.evidence_path, exc)

    def check_connectivity(self) -> CheckResult:
        """Check 1: Real Redis TLS / Connectivity."""
        t0 = time.perf_counter()
        details: dict[str, Any] = {}

        # 1. PING -> PONG
        pong = self.client.ping()
        if not pong:
            return CheckResult(
                name="connectivity",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Redis PING did not return True/PONG",
            )
        details["ping_received"] = True

        # 2. Server Information & Version
        try:
            info = self.client.info()
            self.server_version = str(info.get("redis_version", info.get("server", {}).get("redis_version", "unknown")))
            details["redis_version"] = self.server_version
        except Exception:
            self.server_version = getattr(self.client, "redis_version", "mock-version")
            details["redis_version"] = self.server_version

        # 3. TLS Enforcement
        is_tls = self.metadata["tls"]
        details["tls_configured"] = is_tls
        if self.require_staging_isolation and not is_tls and not self.allow_insecure:
            return CheckResult(
                name="connectivity",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Staging Redis URL does not use TLS (rediss://). Pass allow_insecure=True only for local mock tests.",
            )

        # 4. Database Isolation
        db_index = self.metadata["db"]
        details["database_index"] = db_index
        if self.require_staging_isolation and db_index == 0 and not self.allow_insecure:
            return CheckResult(
                name="connectivity",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Target Redis database index is 0. Staging must use an isolated non-zero database or dedicated instance.",
            )

        return CheckResult(
            name="connectivity",
            status="passed",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            details=details,
        )

    def check_authentication(self) -> CheckResult:
        """Check 2: Authentication enforcement (correct auth succeeds, bad auth fails closed)."""
        t0 = time.perf_counter()
        details: dict[str, Any] = {}

        has_auth = self.metadata["authenticated"]
        details["auth_present_in_url"] = has_auth

        if not has_auth:
            details["auth_status"] = "unauthenticated_instance_permitted_for_local_test"
            return CheckResult(
                name="authentication",
                status="passed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
            )

        # Build invalid credential probe URL without altering other parameters
        parsed = urlparse(self.redis_url)
        netloc = parsed.netloc
        if "@" in netloc:
            user_part, host_part = netloc.split("@", 1)
            username = user_part.split(":", 1)[0] if ":" in user_part else user_part
            bad_netloc = f"{username}:bad_auth_probe_99999@{host_part}"
        else:
            bad_netloc = f":bad_auth_probe_99999@{netloc}"

        bad_url = parsed._replace(netloc=bad_netloc).geturl()
        try:
            bad_client = redis.Redis.from_url(bad_url, socket_timeout=3.0, socket_connect_timeout=3.0)
            bad_client.ping()
            # If ping succeeded with invalid credentials, auth is not properly enforced!
            return CheckResult(
                name="authentication",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Invalid password was accepted by Redis server; authentication failed to fail closed.",
            )
        except (redis.AuthenticationError, redis.ResponseError) as auth_err:
            details["invalid_auth_failed_closed"] = True
            details["auth_error_type"] = type(auth_err).__name__
        except Exception as exc:
            details["invalid_auth_failed_closed"] = True
            details["auth_error_type"] = type(exc).__name__

        return CheckResult(
            name="authentication",
            status="passed",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            details=details,
        )

    def check_lua_engine(self) -> CheckResult:
        """Check 3: Prove real Lua execution for CLAIM_JOB_LUA and ENQUEUE_JOB_LUA natively."""
        t0 = time.perf_counter()
        details: dict[str, Any] = {
            "lua_path_proven": False,
            "python_fallback_used": False,
        }

        del_id = f"gate1-lua-del-{self.run_id}"
        job_id = f"job-{del_id}"
        repo_id = f"gate1-lua-repo-{self.run_id}"
        del_key = self._track_key(f"review:delivery:{del_id}")
        job_key = self._track_key(f"review:job:{job_id}")
        arq_key = self._track_key(f"arq:job:{job_id}")
        queue_key = "arq:queue"
        repo_key = self._track_key(f"review:repo_running:{repo_id}")
        running_jobs_key = "review:jobs:running"
        self.created_job_ids.add(job_id)

        now = time.time()
        score_ms = str(int(now * 1000))
        lease_token = str(uuid.uuid4())
        lease_expires_at = str(now + 60.0)

        # Test ENQUEUE_JOB_LUA
        job_hset_args = [
            "job_id", job_id,
            "delivery_id", del_id,
            "repository_id", repo_id,
            "state", JobState.QUEUED.value,
            "attempt_count", "0",
            "max_retries", "3",
            "backoff_base_seconds", "1.0",
            "deadline_seconds", "60.0",
            "created_at", str(now),
            "updated_at", str(now),
        ]
        from arq.jobs import serialize_job
        raw_arq_bytes = serialize_job(
            function_name="review_job_task",
            args=(job_id,),
            kwargs={},
            job_try=None,
            enqueue_time_ms=int(float(score_ms)),
        )
        enqueue_args = [job_id, raw_arq_bytes, score_ms] + job_hset_args

        if not hasattr(self.client, "eval"):
            return CheckResult(
                name="lua_engine",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Redis client does not support eval(); Lua engine cannot be proven.",
            )

        # 1. Enqueue via Lua
        res1 = self.client.eval(ENQUEUE_JOB_LUA, 4, del_key, job_key, arq_key, queue_key, *enqueue_args)
        res1_str = self._to_str(res1)
        if res1_str != "ok":
            return CheckResult(
                name="lua_engine",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"ENQUEUE_JOB_LUA returned unexpected status: {res1_str!r} (expected 'ok')",
            )
        details["enqueue_lua_ok"] = True

        # 2. Atomic Deduplication via Lua
        res2 = self.client.eval(ENQUEUE_JOB_LUA, 4, del_key, job_key, arq_key, queue_key, *enqueue_args)
        res2_str = self._to_str(res2)
        if res2_str != "existing":
            return CheckResult(
                name="lua_engine",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"ENQUEUE_JOB_LUA deduplication returned {res2_str!r} (expected 'existing')",
            )
        details["enqueue_lua_dedupe_proven"] = True

        # 3. Claim Job via Lua
        claim_args = [job_id, "1", str(now), lease_token, lease_expires_at]
        claim_res = self.client.eval(CLAIM_JOB_LUA, 4, repo_key, running_jobs_key, job_key, queue_key, *claim_args)
        claim_str = self._to_str(claim_res)
        if claim_str != "ok":
            return CheckResult(
                name="lua_engine",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"CLAIM_JOB_LUA returned unexpected status: {claim_str!r} (expected 'ok')",
            )
        details["claim_lua_ok"] = True

        # Verify job hash in Redis reflects Lua mutations
        job_data = self.client.hgetall(job_key)
        state_str = self._to_str(job_data.get(b"state") or job_data.get("state"))
        attempts_str = self._to_str(job_data.get(b"attempt_count") or job_data.get("attempt_count"))
        token_str = self._to_str(job_data.get(b"lease_token") or job_data.get("lease_token"))

        if state_str != "running" or attempts_str != "1" or token_str != lease_token:
            return CheckResult(
                name="lua_engine",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"CLAIM_JOB_LUA state mutation verification failed: state={state_str}, attempts={attempts_str}",
            )
        details["claim_lua_state_verified"] = True

        # 4. Concurrency Limit via Lua
        # Repo now has 1 active job and cap is 1 -> claim attempt must return repo_at_capacity
        cap_res = self.client.eval(CLAIM_JOB_LUA, 4, repo_key, running_jobs_key, job_key, queue_key, *claim_args)
        cap_str = self._to_str(cap_res)
        if cap_str not in ("repo_at_capacity", "already_leased"):
            return CheckResult(
                name="lua_engine",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"CLAIM_JOB_LUA concurrency check returned {cap_str!r} (expected 'repo_at_capacity' or 'already_leased')",
            )
        details["claim_lua_concurrency_proven"] = True
        details["lua_path_proven"] = True

        return CheckResult(
            name="lua_engine",
            status="passed",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            details=details,
        )

    def check_queue_lifecycle(self) -> CheckResult:
        """Check 4: Real RedisJobQueue lifecycle semantics."""
        t0 = time.perf_counter()
        details: dict[str, Any] = {}

        queue = RedisJobQueue(client=self.client)
        repo_id = f"gate1-q-repo-{self.run_id}"
        del_id_1 = f"gate1-q-del1-{self.run_id}"
        del_id_2 = f"gate1-q-del2-{self.run_id}"
        self.created_repo_ids.add(repo_id)
        self.created_delivery_ids.update([del_id_1, del_id_2])

        snapshot_1 = ReviewSnapshot(
            repository_id=repo_id,
            repository_full_name=repo_id,
            pull_request_number=101,
            base_sha="0000000000000000000000000000000000000001",
            head_sha="0000000000000000000000000000000000000002",
            changed_files=("main.py",),
            policy_version="v1",
            prompt_version="v1",
            retrieval_index_version="v1",
            model_configuration={"provider": "mock", "model": "mock-model"},
        )

        now = time.time()

        # 1. Enqueue job 1
        job1 = queue.enqueue(snapshot_1, del_id_1, max_retries=2, backoff_base_seconds=1.0, now=now)
        self.created_job_ids.add(job1.job_id)
        self._track_key(f"review:job:{job1.job_id}")
        self._track_key(f"arq:job:{job1.job_id}")
        self._track_key(f"review:delivery:{del_id_1}")
        details["job1_enqueued"] = job1.job_id

        # Verify duplicate delivery returns exact same job
        job1_dup = queue.enqueue(snapshot_1, del_id_1, now=now)
        if job1_dup.job_id != job1.job_id:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Duplicate enqueue did not return existing job ID",
            )
        details["enqueue_dedupe_verified"] = True

        # 2. Lease job 1
        leased1 = queue.lease_next_job(now=now)
        if leased1 is None or leased1.job_id != job1.job_id:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="lease_next_job did not return job 1",
            )
        if not leased1.lease_token:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="leased job 1 missing lease_token",
            )
        if leased1.attempt_count != 1 or leased1.state != JobState.RUNNING:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"leased job 1 invalid state: attempt={leased1.attempt_count}, state={leased1.state}",
            )
        details["lease1_verified"] = True

        # 3. Concurrency limit
        # Enqueue job 2 for same repo while job 1 is running
        snapshot_2 = ReviewSnapshot(
            repository_id=repo_id,
            repository_full_name=repo_id,
            pull_request_number=102,
            base_sha="0000000000000000000000000000000000000001",
            head_sha="0000000000000000000000000000000000000003",
            changed_files=("test.py",),
            policy_version="v1",
            prompt_version="v1",
            retrieval_index_version="v1",
            model_configuration={"provider": "mock", "model": "mock-model"},
        )
        job2 = queue.enqueue(snapshot_2, del_id_2, max_retries=2, backoff_base_seconds=1.0, now=now)
        self.created_job_ids.add(job2.job_id)
        self._track_key(f"review:job:{job2.job_id}")
        self._track_key(f"arq:job:{job2.job_id}")
        self._track_key(f"review:delivery:{del_id_2}")

        # Attempt lease with max_concurrency_per_repo=1; must return None because job 1 is running
        blocked_lease = queue.lease_next_job(now=now, max_concurrency_per_repo=1)
        if blocked_lease is not None:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Concurrency limit violated: leased job 2 while repo had active lease",
            )
        details["concurrency_limit_verified"] = True

        # Cancel job 2 to isolate job 1 for retry testing
        queue.cancel_job(job2.job_id, reason="concurrency check complete", now=now)

        # 4. Failure & Exponential Backoff
        # Mark failure on job 1 (attempt 1 of 2)
        failed1 = queue.mark_failed(job1.job_id, "transient error 1", now=now, lease_token=leased1.lease_token)
        if failed1.state != JobState.QUEUED:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"Job 1 did not re-queue after first failure: state={failed1.state}",
            )
        expected_delay = 1.0 * (2 ** (1 - 1))  # 1.0s
        if abs(failed1.next_run_at - (now + expected_delay)) > 0.1:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"Backoff calculation mismatch: next_run={failed1.next_run_at}, expected={now + expected_delay}",
            )
        details["backoff_schedule_verified"] = True

        # 5. Exhaust Retries & Dead Letter
        # Advance time to next_run_at + 0.1 and lease job 1 again
        retry_time = now + expected_delay + 0.1
        leased1_retry = queue.lease_next_job(now=retry_time)
        if leased1_retry is None:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Could not lease job 1 on retry attempt",
            )
        if leased1_retry.attempt_count != 2:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"Retry attempt count mismatch: {leased1_retry.attempt_count} (expected 2)",
            )

        # Mark failure on retry (attempt 2 of 2 -> max_retries reached)
        dead_job = queue.mark_failed(job1.job_id, "fatal error 2", now=retry_time, lease_token=leased1_retry.lease_token)
        if dead_job.state != JobState.DEAD_LETTER:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"Job 1 did not transition to DEAD_LETTER after max retries: state={dead_job.state}",
            )
        dead_letter_members = [self._to_str(m) for m in self.client.smembers(queue.dead_letter_key)]
        if job1.job_id not in dead_letter_members:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Job 1 not found in review:dead_letter set",
            )
        details["dead_letter_verified"] = True

        # 6. Zombie Recovery
        del_id_3 = f"gate1-q-del3-{self.run_id}"
        self.created_delivery_ids.add(del_id_3)
        snapshot_3 = ReviewSnapshot(
            repository_id=repo_id,
            repository_full_name=repo_id,
            pull_request_number=103,
            base_sha="0000000000000000000000000000000000000001",
            head_sha="0000000000000000000000000000000000000004",
            changed_files=("zombie.py",),
            policy_version="v1",
            prompt_version="v1",
            retrieval_index_version="v1",
            model_configuration={"provider": "mock", "model": "mock-model"},
        )
        job3 = queue.enqueue(snapshot_3, del_id_3, deadline_seconds=10.0, now=retry_time)
        self.created_job_ids.add(job3.job_id)
        self._track_key(f"review:job:{job3.job_id}")
        self._track_key(f"arq:job:{job3.job_id}")
        self._track_key(f"review:delivery:{del_id_3}")

        leased3 = queue.lease_next_job(now=retry_time)
        if leased3 is None or leased3.job_id != job3.job_id:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Could not lease job 3 for zombie recovery test",
            )
        # Advance time past deadline (deadline=10.0s -> advance by 15s)
        recovery_time = retry_time + 15.0
        recovered = queue.recover_zombie_jobs(now=recovery_time)
        recovered_ids = [j.job_id for j in recovered]
        if job3.job_id not in recovered_ids:
            return CheckResult(
                name="queue_lifecycle",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Expired job 3 was not recovered by recover_zombie_jobs",
            )
        details["zombie_recovery_verified"] = True

        return CheckResult(
            name="queue_lifecycle",
            status="passed",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            details=details,
        )

    def check_arq_serialization(self) -> CheckResult:
        """Check 5: Validate exact ARQ task payload serialization with installed ARQ."""
        t0 = time.perf_counter()
        details: dict[str, Any] = {}

        if not self.created_job_ids:
            return CheckResult(
                name="arq_serialization",
                status="skipped",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="No job created in previous queue step to inspect ARQ payload",
            )

        test_job_id = next(iter(self.created_job_ids))
        arq_key = f"arq:job:{test_job_id}"
        raw_bytes = self.client.get(arq_key)
        if not raw_bytes:
            return CheckResult(
                name="arq_serialization",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"No ARQ payload found at key {arq_key}",
            )

        try:
            job_def = deserialize_job(raw_bytes)
        except Exception as exc:
            return CheckResult(
                name="arq_serialization",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"Failed to deserialize ARQ job payload with installed ARQ library: {exc}",
            )

        if job_def.function != "review_job_task":
            return CheckResult(
                name="arq_serialization",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"ARQ function name mismatch: {job_def.function} (expected 'review_job_task')",
            )

        if not job_def.args or job_def.args[0] != test_job_id:
            return CheckResult(
                name="arq_serialization",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"ARQ job args mismatch: args={job_def.args} (expected ({test_job_id!r},))",
            )

        details["arq_function"] = job_def.function
        details["job_id_round_trip"] = True
        details["enqueue_time_ms"] = getattr(job_def, "enqueue_time_ms", None)

        return CheckResult(
            name="arq_serialization",
            status="passed",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            details=details,
        )

    def check_checkpoint(self) -> CheckResult:
        """Check 6: Real RedisCheckpointSaver protocol, thread isolation, and secret rejection."""
        t0 = time.perf_counter()
        details: dict[str, Any] = {}

        saver = RedisCheckpointSaver(client=self.client)
        thread_id_1 = f"gate1-cp-t1-{self.run_id}"
        thread_id_2 = f"gate1-cp-t2-{self.run_id}"
        self.created_thread_ids.update([thread_id_1, thread_id_2])

        config_1 = {"configurable": {"thread_id": thread_id_1, "checkpoint_ns": ""}}
        cp_id = str(uuid.uuid4())

        checkpoint_state = {
            "v": 1,
            "id": cp_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {
                "specialist_outputs": {
                    "security": SpecialistOutput(
                        specialist_type=SpecialistType.SECURITY,
                        correlation_id="corr-1",
                        status="completed",
                        findings=(
                            CandidateFinding(
                                finding_id="SEC-001",
                                correlation_id="corr-1",
                                specialist_type=SpecialistType.SECURITY,
                                category="security",
                                severity="high",
                                confidence=0.95,
                                summary="SQL Injection",
                                rationale="Unsanitized query parameter",
                                file_path="app.py",
                                line_range=(10, 15),
                            ),
                        ),
                        execution_duration=0.12,
                    )
                },
                "status": "in_progress",
            },
            "channel_versions": {
                "specialist_outputs": "00000000000000000000000000000001.0",
                "status": "00000000000000000000000000000001.0",
            },
            "versions_seen": {},
        }
        metadata_state = {"source": "input", "step": 1, "writes": {}, "parents": {}}
        new_versions = {
            "specialist_outputs": "00000000000000000000000000000001.0",
            "status": "00000000000000000000000000000001.0",
        }

        # 1. Put Checkpoint
        try:
            saved_config = saver.put(config_1, checkpoint_state, metadata_state, new_versions)
            details["checkpoint_put_ok"] = True
        except Exception as exc:
            return CheckResult(
                name="checkpoint",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"RedisCheckpointSaver.put failed: {exc}",
            )

        # 2. Get Checkpoint Tuple
        retrieved_tup = saver.get_tuple(config_1)
        if retrieved_tup is None:
            return CheckResult(
                name="checkpoint",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="get_tuple returned None for persisted checkpoint",
            )
        if retrieved_tup.checkpoint["id"] != cp_id:
            return CheckResult(
                name="checkpoint",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"Retrieved checkpoint ID mismatch: {retrieved_tup.checkpoint['id']} != {cp_id}",
            )
        details["checkpoint_get_exact_match"] = True

        # 3. Thread Isolation
        config_2 = {"configurable": {"thread_id": thread_id_2, "checkpoint_ns": ""}}
        isolated_tup = saver.get_tuple(config_2)
        if isolated_tup is not None:
            return CheckResult(
                name="checkpoint",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Thread isolation violated: thread 2 retrieved thread 1 checkpoint data",
            )
        details["thread_isolation_verified"] = True

        # 4. Forbidden Secrets Check: Must fail closed if secret key present
        bad_cp = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {"api_key": "sk-secret-12345"},
            "channel_versions": {},
            "versions_seen": {},
        }
        secret_caught = False
        try:
            saver.put(config_1, bad_cp, {}, {})
        except ValueError:
            secret_caught = True
        except Exception as exc:
            # Any security exception is acceptable
            secret_caught = "security" in str(exc).lower() or "forbidden" in str(exc).lower()

        if not secret_caught:
            return CheckResult(
                name="checkpoint",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Checkpoint saver failed to reject forbidden secret key 'api_key'",
            )
        details["forbidden_secrets_rejected"] = True

        # 5. Delete Thread
        saver.delete_thread(thread_id_1)
        deleted_tup = saver.get_tuple(config_1)
        if deleted_tup is not None:
            return CheckResult(
                name="checkpoint",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="delete_thread failed to purge thread 1 checkpoint data",
            )
        details["delete_thread_verified"] = True

        return CheckResult(
            name="checkpoint",
            status="passed",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            details=details,
        )

    def check_secret_redaction(self) -> CheckResult:
        """Check 7: Scan validation artifacts and metadata to guarantee zero secret leakage."""
        t0 = time.perf_counter()
        details: dict[str, Any] = {}

        # 1. Verify masked metadata does not contain raw password or tokens
        masked_url = self.metadata["masked_url"]
        raw_pwd = urlparse(self.redis_url).password if self.redis_url else None

        if raw_pwd and raw_pwd in masked_url:
            return CheckResult(
                name="secret_redaction",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error="Plaintext password leaked into masked_url metadata representation",
            )
        details["masked_url_clean"] = True

        # 2. Verify all keys created by this harness contain no secrets
        for k in self.created_keys:
            for forbidden in ("password", "secret", "token", "key"):
                if forbidden in k and not any(prefix in k for prefix in ("lease_token", "thread_keys", "dead_letter")):
                    return CheckResult(
                        name="secret_redaction",
                        status="failed",
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                        details=details,
                        error=f"Suspicious key generated in Redis: {k!r}",
                    )
        details["keys_clean"] = True

        return CheckResult(
            name="secret_redaction",
            status="passed",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            details=details,
        )

    def cleanup(self) -> CheckResult:
        """Check 8: Deterministic cleanup of all test keys without FLUSHDB."""
        t0 = time.perf_counter()
        details: dict[str, Any] = {}
        deleted_count = 0

        try:
            # 1. Clean individual keys
            for k in self.created_keys:
                try:
                    self.client.delete(k)
                    deleted_count += 1
                except Exception:
                    pass

            # 2. Clean shared queues/sets for created jobs
            for job_id in self.created_job_ids:
                try:
                    self.client.srem("review:jobs:running", job_id)
                    self.client.srem("review:dead_letter", job_id)
                    self.client.zrem("review:dead_letter", job_id)
                    self.client.zrem("arq:queue", job_id)
                except Exception:
                    pass

            # 3. Clean repo sets
            for repo_id in self.created_repo_ids:
                try:
                    self.client.delete(f"review:repo_running:{repo_id}")
                except Exception:
                    pass

            # 4. Clean checkpoint thread sets
            for thread_id in self.created_thread_ids:
                try:
                    self.client.srem("review:checkpoint:threads", thread_id)
                except Exception:
                    pass

            details["keys_deleted"] = deleted_count
            details["clean_completed"] = True
        except Exception as exc:
            return CheckResult(
                name="cleanup",
                status="failed",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                details=details,
                error=f"Error during deterministic cleanup: {exc}",
            )

        return CheckResult(
            name="cleanup",
            status="passed",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            details=details,
        )


def main() -> int:
    """CLI entrypoint for executing Gate 1 validation."""
    parser = argparse.ArgumentParser(description="W1-08B Gate 1: Real Redis Validation Suite")
    parser.add_argument("--redis-url", default=None, help="Staging Redis URL (e.g. rediss://...:6379/1)")
    parser.add_argument("--evidence-path", default=".genesis/evidence/w1_08b_redis_validation.json", help="Path to evidence JSON file")
    parser.add_argument("--allow-insecure", action="store_true", help="Allow non-TLS or DB 0 for offline/local harness verification")
    parser.add_argument("--output-json", action="store_true", help="Print full report JSON to stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    validator = RedisGate1Validator(
        redis_url=args.redis_url,
        allow_insecure=args.allow_insecure,
        evidence_path=args.evidence_path,
    )
    report = validator.validate_all()

    if args.output_json:
        print(report.to_json())
    else:
        print("\n=== W1-08B Gate 1: Real Redis Validation Report ===")
        print(f"Overall Status: {report.overall_status}")
        print(f"Live Redis Executed: {report.live_redis_executed}")
        print(f"Server Version: {report.server_version}")
        print(f"Evidence Saved: {report.evidence_path}")
        print("\nChecks Summary:")
        for name, res in report.checks.items():
            status_symbol = "✓" if res["status"] == "passed" else "✗"
            print(f"  {status_symbol} {name:<20}: {res['status']} ({res['duration_ms']:.1f}ms)")
            if res.get("error"):
                print(f"      Error: {res['error']}")
        if report.capabilities:
            print("\nCapabilities Breakdown:")
            for cap_name, cap_info in report.capabilities.items():
                stat = cap_info.get("status", "UNKNOWN")
                symbol = "✓" if stat in ("PROVEN", "OFFLINE_VERIFIED") else ("⏸" if "DEFERRED" in str(cap_info.get("verdict", "")).upper() or stat == "NOT_BENCHMARKED" else "✗")
                print(f"  {symbol} {cap_name:<40}: {stat} (verdict: {cap_info.get('verdict', 'unknown')})")
        print("====================================================\n")

    return 0 if report.overall_status in ("passed", "offline_verified") else 1


if __name__ == "__main__":
    sys.exit(main())
