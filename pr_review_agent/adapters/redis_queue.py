"""Redis / ARQ distributed execution queue adapter conforming to DurableQueueProtocol."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
import uuid

from arq.constants import abort_jobs_ss, default_queue_name
from arq.jobs import serialize_job
import redis

from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.orchestration import (
    DurableQueueProtocol,
    JobState,
    ReviewJob,
)

logger = logging.getLogger(__name__)

# Atomic Lua script for repository concurrency checking and slot reservation.
# Guarantees that two workers racing cannot both observe capacity and exceed max_concurrency_per_repo.
CLAIM_JOB_LUA = """
local cap = tonumber(ARGV[2])
if cap and cap > 0 then
    local current_count = redis.call('SCARD', KEYS[1])
    if current_count >= cap then
        return 'repo_at_capacity'
    end
end

if KEYS[4] ~= '' then
    local removed = redis.call('ZREM', KEYS[4], ARGV[1])
    if removed == 0 then
        return 'already_leased'
    end
end

redis.call('SADD', KEYS[1], ARGV[1])
redis.call('SADD', KEYS[2], ARGV[1])

local current_attempts = tonumber(redis.call('HGET', KEYS[3], 'attempt_count') or '0')
local new_attempts = current_attempts + 1

redis.call('HSET', KEYS[3],
    'state', 'running',
    'attempt_count', tostring(new_attempts),
    'lease_token', ARGV[4],
    'lease_expires_at', ARGV[5],
    'updated_at', ARGV[3]
)

return 'ok'
"""

# Atomic Lua script for first-time concurrent delivery/job enqueue (Concern 3).
# Guarantees that racing enqueuers with duplicate delivery_id cannot create competing
# logical queue states or overwrite an in-flight job.
ENQUEUE_JOB_LUA = """
local existing = redis.call('GET', KEYS[1])
if existing then
    return 'existing'
end

redis.call('SET', KEYS[1], ARGV[1])

local hset_args = {'HSET', KEYS[2]}
for i = 4, #ARGV do
    table.insert(hset_args, ARGV[i])
end
redis.call(unpack(hset_args))

redis.call('SET', KEYS[3], ARGV[2])
redis.call('ZADD', KEYS[4], ARGV[3], ARGV[1])

return 'ok'
"""


class StaleLeaseError(RuntimeError):
    """Raised when a worker attempts to mutate a job whose lease has expired or been reassigned."""


class RedisJobQueue(DurableQueueProtocol):
    """Redis / ARQ execution queue adapter satisfying DurableQueueProtocol.

    Architecture & Invariants:
    1. Real ARQ Job Lifecycle: 'arq:queue' is the sole scheduling queue; 'arq:job:{job_id}' contains
       valid ARQ 0.28.0 serialized jobs consumable by standard ARQ workers.
    2. Durable Review State: Domain job state and snapshot provenance are persisted in 'review:job:{job_id}'.
       ARQ execution is strictly coordinated without replacing business truth (ReviewTruth / AuditSpine).
    3. Delivery Idempotency: 'review:delivery:{delivery_id}' ensures O(1) duplicate protection.
    4. Logical Attempt Authority: ReviewJob.attempt_count and max_retries govern retry limits and exponential backoff.
    5. Lease Ownership: Atomic lease_token prevents stale workers from completing expired/reclaimed jobs.
    6. Atomic Repository Concurrency (NFR-05): Atomic Lua script enforces max_concurrency_per_repo.
    7. Indexed Zombie Recovery: 'review:jobs:running' set tracks active jobs without full keyspace scans.
    """

    def __init__(
        self,
        client: redis.Redis[Any] | None = None,
        *,
        redis_url: str | None = None,
        queue_name: str = default_queue_name,
        budget_config: Any | None = None,
    ) -> None:
        self.queue_name = queue_name
        self.budget_config = budget_config
        self._owns_client = client is None
        self.redis_url = redis_url or ""

        if client is not None:
            self.client = client
        else:
            target_url = redis_url or "redis://127.0.0.1:6379/0"
            self.client = redis.Redis.from_url(target_url, decode_responses=False)

    def _job_key(self, job_id: str) -> str:
        return f"review:job:{job_id}"

    def _arq_job_key(self, job_id: str) -> str:
        return f"arq:job:{job_id}"

    def _delivery_key(self, delivery_id: str) -> str:
        return f"review:delivery:{delivery_id}"

    def _repo_running_key(self, repo_id: str) -> str:
        return f"review:repo_running:{repo_id}"

    @property
    def running_jobs_key(self) -> str:
        return "review:jobs:running"

    @property
    def dead_letter_key(self) -> str:
        return "review:dead_letter"

    def _to_str(self, val: Any) -> str:
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val) if val is not None else ""

    def _to_float(self, val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _to_int(self, val: Any, default: int = 0) -> int:
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def _deserialize_job(self, data: dict[Any, Any]) -> ReviewJob:
        state_str = self._to_str(data.get(b"state") or data.get("state", "queued"))
        lease_token = self._to_str(data.get(b"lease_token") or data.get("lease_token")) or None
        return ReviewJob(
            job_id=self._to_str(data.get(b"job_id") or data.get("job_id")),
            delivery_id=self._to_str(data.get(b"delivery_id") or data.get("delivery_id")),
            repository_id=self._to_str(data.get(b"repository_id") or data.get("repository_id")),
            pull_request_number=self._to_int(data.get(b"pull_request_number") or data.get("pull_request_number")),
            base_sha=self._to_str(data.get(b"base_sha") or data.get("base_sha")),
            head_sha=self._to_str(data.get(b"head_sha") or data.get("head_sha")),
            state=JobState(state_str),
            attempt_count=self._to_int(data.get(b"attempt_count") or data.get("attempt_count")),
            max_retries=self._to_int(data.get(b"max_retries") or data.get("max_retries"), default=3),
            backoff_base_seconds=self._to_float(data.get(b"backoff_base_seconds") or data.get("backoff_base_seconds"), default=1.0),
            deadline_seconds=self._to_float(data.get(b"deadline_seconds") or data.get("deadline_seconds"), default=60.0),
            last_error=self._to_str(data.get(b"last_error") or data.get("last_error")) or None,
            created_at=self._to_float(data.get(b"created_at") or data.get("created_at")),
            updated_at=self._to_float(data.get(b"updated_at") or data.get("updated_at")),
            next_run_at=self._to_float(data.get(b"next_run_at") or data.get("next_run_at")),
            lease_token=lease_token,
        )

    def enqueue(
        self,
        snapshot: ReviewSnapshot,
        delivery_id: str,
        *,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        deadline_seconds: float = 60.0,
        now: float | None = None,
    ) -> ReviewJob:
        """Enqueue a review job atomically with durable identity and real ARQ task serialization.

        Enqueue Atomicity (Concern 3):
        Uses atomic Redis primitive / Lua script to guarantee that duplicate delivery_id
        cannot create competing logical queue states or overwrite in-flight execution.
        """
        current_time = time.time() if now is None else now
        job_id = f"job-{delivery_id}"
        delivery_key = self._delivery_key(delivery_id)
        job_key = self._job_key(job_id)
        arq_job_key = self._arq_job_key(job_id)

        # Fast idempotency check: if delivery already exists, return existing job
        existing_job_id = self.client.get(delivery_key)
        if existing_job_id:
            existing = self.get_job(self._to_str(existing_job_id))
            if existing is not None:
                return existing

        payload = {
            "repository_id": snapshot.repository_id,
            "repository_full_name": snapshot.repository_full_name,
            "pull_request_number": snapshot.pull_request_number,
            "base_sha": snapshot.base_sha,
            "head_sha": snapshot.head_sha,
            "changed_files": list(snapshot.changed_files),
            "policy_version": snapshot.policy_version,
            "prompt_version": snapshot.prompt_version,
            "retrieval_index_version": snapshot.retrieval_index_version,
            "model_configuration": dict(snapshot.model_configuration),
        }
        payload_json = json.dumps(payload, sort_keys=True)

        job_data = {
            "job_id": job_id,
            "delivery_id": delivery_id,
            "repository_id": snapshot.repository_id,
            "pull_request_number": str(snapshot.pull_request_number),
            "base_sha": snapshot.base_sha,
            "head_sha": snapshot.head_sha,
            "state": JobState.QUEUED.value,
            "attempt_count": "0",
            "max_retries": str(max_retries),
            "backoff_base_seconds": str(backoff_base_seconds),
            "deadline_seconds": str(deadline_seconds),
            "last_error": "",
            "created_at": str(current_time),
            "updated_at": str(current_time),
            "next_run_at": str(current_time),
            "payload_json": payload_json,
            "lease_token": "",
            "lease_expires_at": "0.0",
        }

        # Real ARQ 0.28.0 job serialization for review_job_task
        score_ms = int(current_time * 1000)
        arq_payload = serialize_job(
            function_name="review_job_task",
            args=(job_id,),
            kwargs={},
            job_try=None,
            enqueue_time_ms=score_ms,
        )

        # Atomic execution via Lua script if supported
        if hasattr(self.client, "eval"):
            try:
                hset_pairs: list[str] = []
                for k, v in job_data.items():
                    hset_pairs.extend([k, str(v)])

                lua_args = [job_id, arq_payload, str(score_ms)] + hset_pairs
                res = self.client.eval(
                    ENQUEUE_JOB_LUA,
                    4,
                    delivery_key,
                    job_key,
                    arq_job_key,
                    self.queue_name,
                    *lua_args,
                )
                res_str = self._to_str(res)
                if res_str == "existing":
                    ex_id = self.client.get(delivery_key)
                    ex_job = self.get_job(self._to_str(ex_id))
                    if ex_job is not None:
                        return ex_job
                job = self.get_job(job_id)
                if job is not None:
                    return job
            except Exception:
                pass  # Fall through to atomic SET NX fallback

        # Atomic SET NX fallback: guarantees exactly one enqueuer acquires delivery claim
        acquired = self.client.set(delivery_key, job_id, nx=True)
        if not acquired:
            ex_id = self.client.get(delivery_key)
            ex_job = self.get_job(self._to_str(ex_id))
            if ex_job is not None:
                return ex_job
            # Brief poll in case winning transaction is in flight
            for _ in range(20):
                time.sleep(0.01)
                ex_job = self.get_job(job_id)
                if ex_job is not None:
                    return ex_job
            raise RuntimeError(f"Concurrent enqueue race condition on delivery {delivery_id}")

        pipe = self.client.pipeline(transaction=True)
        pipe.hset(job_key, mapping=job_data)
        pipe.set(arq_job_key, arq_payload)
        pipe.zadd(self.queue_name, {job_id: score_ms})
        pipe.execute()

        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Failed to persist or fetch job {job_id}")
        return job

    def lease_next_job(
        self,
        now: float | None = None,
        *,
        max_concurrency_per_repo: int | None = None,
    ) -> ReviewJob | None:
        """Atomically lease the next scheduled job from arq:queue respecting repo concurrency."""
        current_time = time.time() if now is None else now
        score_ms = int(current_time * 1000)

        effective_cap = (
            max_concurrency_per_repo
            if max_concurrency_per_repo is not None
            else (
                getattr(self.budget_config, "max_concurrent_reviews_per_repo", None)
                if self.budget_config is not None
                else None
            )
        )

        # Candidate jobs eligible for execution: score <= current_time in ms
        candidate_ids = self.client.zrangebyscore(
            self.queue_name,
            min="-inf",
            max=score_ms,
            start=0,
            num=50,
        )

        for raw_cand_id in candidate_ids:
            cand_id = self._to_str(raw_cand_id)
            cand_job = self.get_job(cand_id)
            if cand_job is None or cand_job.state != JobState.QUEUED:
                # Stale or non-queued entry, remove from queue
                self.client.zrem(self.queue_name, cand_id)
                continue

            # Check per-repository concurrency cap (NFR-05)
            if effective_cap is not None:
                running_key = self._repo_running_key(cand_job.repository_id)
                running_count = self.client.scard(running_key)
                if running_count >= effective_cap:
                    # Repo is at capacity; skip to next candidate to prevent starvation
                    continue

            # Atomically attempt to lease this candidate
            claimed, _ = self._attempt_claim(
                cand_job,
                current_time,
                max_concurrency_per_repo=effective_cap,
                from_queue=True,
            )
            if claimed is not None:
                return claimed

        return None

    def _attempt_claim(
        self,
        job: ReviewJob,
        now: float,
        *,
        max_concurrency_per_repo: int | None = None,
        from_queue: bool = True,
    ) -> tuple[ReviewJob | None, str | None]:
        """Atomically check capacity, pop from arq:queue if needed, and claim lease."""
        job_key = self._job_key(job.job_id)
        running_key = self._repo_running_key(job.repository_id)
        queue_key = self.queue_name if from_queue else ""
        cap_val = max_concurrency_per_repo if max_concurrency_per_repo is not None else -1
        lease_token = str(uuid.uuid4())
        lease_expires_at = now + job.deadline_seconds

        # Attempt atomic Lua script execution if supported
        if hasattr(self.client, "eval"):
            try:
                res = self.client.eval(
                    CLAIM_JOB_LUA,
                    4,
                    running_key,
                    self.running_jobs_key,
                    job_key,
                    queue_key,
                    job.job_id,
                    str(cap_val),
                    str(now),
                    lease_token,
                    str(lease_expires_at),
                )
                res_str = self._to_str(res)
                if res_str != "ok":
                    return None, res_str
                claimed_job = self.get_job(job.job_id)
                return claimed_job, lease_token
            except Exception:
                pass  # Fall through to transactional path if eval unsupported

        # Transactional fallback
        if cap_val > 0:
            current_count = self.client.scard(running_key)
            if current_count >= cap_val:
                return None, "repo_at_capacity"

        if from_queue:
            removed = self.client.zrem(self.queue_name, job.job_id)
            if not removed:
                return None, "already_leased"

        new_attempt = job.attempt_count + 1
        pipe = self.client.pipeline(transaction=True)
        pipe.hset(
            job_key,
            mapping={
                "state": JobState.RUNNING.value,
                "attempt_count": str(new_attempt),
                "updated_at": str(now),
                "lease_token": lease_token,
                "lease_expires_at": str(lease_expires_at),
            },
        )
        pipe.sadd(running_key, job.job_id)
        pipe.sadd(self.running_jobs_key, job.job_id)
        pipe.execute()

        claimed_job = self.get_job(job.job_id)
        return claimed_job, lease_token

    def claim_job(
        self,
        job_id: str,
        now: float | None = None,
        *,
        max_concurrency_per_repo: int | None = None,
    ) -> tuple[ReviewJob | None, str | None]:
        """Claim a specific job for worker execution (reconciles ARQ push model)."""
        current_time = time.time() if now is None else now
        job = self.get_job(job_id)
        if job is None:
            return None, "not_found"
        if job.state == JobState.CANCELLED:
            return None, "cancelled"
        if job.state == JobState.COMPLETED:
            return None, "already_completed"

        effective_cap = (
            max_concurrency_per_repo
            if max_concurrency_per_repo is not None
            else (
                getattr(self.budget_config, "max_concurrent_reviews_per_repo", None)
                if self.budget_config is not None
                else None
            )
        )

        # In ARQ push model, ARQ worker pops the job; we claim logical ownership and slot
        return self._attempt_claim(
            job,
            current_time,
            max_concurrency_per_repo=effective_cap,
            from_queue=False,
        )

    def mark_completed(
        self,
        job_id: str,
        now: float | None = None,
        *,
        lease_token: str | None = None,
    ) -> ReviewJob:
        """Mark job successfully completed, releasing repository slot and verifying lease token."""
        current_time = time.time() if now is None else now
        job_key = self._job_key(job_id)
        data = self.client.hgetall(job_key)
        if not data:
            raise KeyError(f"Job {job_id} not found")

        # Stale worker protection: verify lease token if provided
        if lease_token is not None:
            stored_token = self._to_str(data.get(b"lease_token") or data.get("lease_token"))
            if stored_token != lease_token:
                raise StaleLeaseError(f"Stale lease mutation rejected for job {job_id}")

        job = self._deserialize_job(data)
        running_key = self._repo_running_key(job.repository_id)

        pipe = self.client.pipeline(transaction=True)
        pipe.hset(
            job_key,
            mapping={
                "state": JobState.COMPLETED.value,
                "updated_at": str(current_time),
            },
        )
        pipe.srem(running_key, job_id)
        pipe.srem(self.running_jobs_key, job_id)
        pipe.zrem(self.queue_name, job_id)
        pipe.execute()

        completed = self.get_job(job_id)
        if completed is None:
            raise KeyError(f"Job {job_id} not found")
        return completed

    def mark_failed(
        self,
        job_id: str,
        error: str,
        now: float | None = None,
        *,
        lease_token: str | None = None,
    ) -> ReviewJob:
        """Handle failure: apply exponential backoff retry in arq:queue or transition to dead-letter."""
        current_time = time.time() if now is None else now
        job_key = self._job_key(job_id)
        data = self.client.hgetall(job_key)
        if not data:
            raise KeyError(f"Job {job_id} not found")

        # Stale worker protection: verify lease token if provided
        if lease_token is not None:
            stored_token = self._to_str(data.get(b"lease_token") or data.get("lease_token"))
            if stored_token != lease_token:
                raise StaleLeaseError(f"Stale lease mutation rejected for job {job_id}")

        job = self._deserialize_job(data)
        running_key = self._repo_running_key(job.repository_id)

        pipe = self.client.pipeline(transaction=True)
        pipe.srem(running_key, job_id)
        pipe.srem(self.running_jobs_key, job_id)

        if job.attempt_count >= job.max_retries:
            next_state = JobState.DEAD_LETTER
            next_run = current_time
            pipe.hset(
                job_key,
                mapping={
                    "state": next_state.value,
                    "last_error": error,
                    "next_run_at": str(next_run),
                    "updated_at": str(current_time),
                },
            )
            pipe.sadd(self.dead_letter_key, job_id)
            pipe.zrem(self.queue_name, job_id)
        else:
            next_state = JobState.QUEUED
            delay = job.backoff_base_seconds * (2 ** (job.attempt_count - 1))
            next_run = current_time + delay
            pipe.hset(
                job_key,
                mapping={
                    "state": next_state.value,
                    "last_error": error,
                    "next_run_at": str(next_run),
                    "updated_at": str(current_time),
                },
            )
            # Re-enqueue in arq:queue with score = next_run in ms
            score_ms = int(next_run * 1000)
            pipe.zadd(self.queue_name, {job_id: score_ms})

        pipe.execute()
        updated = self.get_job(job_id)
        if updated is None:
            raise KeyError(f"Job {job_id} not found")
        return updated

    def cancel_job(
        self,
        job_id: str,
        reason: str = "",
        now: float | None = None,
        *,
        lease_token: str | None = None,
    ) -> ReviewJob:
        """Cancel a job, recording reason, removing from queue, and signaling ARQ abort.

        ARQ Cancellation (Concern 2):
        Logical JobState.CANCELLED remains authoritative; ARQ abort via abort_jobs_ss
        remains an operational optimization to terminate active compute immediately.
        """
        current_time = time.time() if now is None else now
        job_key = self._job_key(job_id)
        data = self.client.hgetall(job_key)
        if not data:
            raise KeyError(f"Job {job_id} not found")

        # Stale worker protection if token passed
        if lease_token is not None:
            stored_token = self._to_str(data.get(b"lease_token") or data.get("lease_token"))
            if stored_token != lease_token:
                raise StaleLeaseError(f"Stale lease mutation rejected for job {job_id}")

        job = self._deserialize_job(data)
        running_key = self._repo_running_key(job.repository_id)

        pipe = self.client.pipeline(transaction=True)
        pipe.hset(
            job_key,
            mapping={
                "state": JobState.CANCELLED.value,
                "last_error": reason or "Cancelled by operator",
                "updated_at": str(current_time),
                "lease_token": "",
                "lease_expires_at": "0.0",
            },
        )
        pipe.srem(running_key, job_id)
        pipe.srem(self.running_jobs_key, job_id)
        pipe.zrem(self.queue_name, job_id)
        # Signal ARQ abort set as operational optimization
        pipe.zadd(abort_jobs_ss, {job_id: int(current_time * 1000)})
        pipe.execute()

        cancelled = self.get_job(job_id)
        if cancelled is None:
            raise KeyError(f"Job {job_id} not found")
        return cancelled

    def get_job(self, job_id: str) -> ReviewJob | None:
        """Retrieve a review job by its durable ID."""
        data = self.client.hgetall(self._job_key(job_id))
        if not data:
            return None
        return self._deserialize_job(data)

    def get_job_by_delivery(self, delivery_id: str) -> ReviewJob | None:
        """Retrieve a review job by its correlated delivery ID."""
        delivery_key = self._delivery_key(delivery_id)
        job_id_val = self.client.get(delivery_key)
        if not job_id_val:
            return None
        return self.get_job(self._to_str(job_id_val))

    def get_job_payload(self, job_id: str) -> dict[str, Any]:
        """Retrieve stored snapshot payload json for reconstruction."""
        job_key = self._job_key(job_id)
        payload_val = self.client.hget(job_key, "payload_json") or self.client.hget(job_key, b"payload_json")
        if not payload_val:
            return {}
        return json.loads(self._to_str(payload_val))

    def recover_zombie_jobs(self, now: float | None = None) -> list[ReviewJob]:
        """Recover RUNNING jobs whose leases expired without completion using the indexed running set."""
        current_time = time.time() if now is None else now
        recovered: list[ReviewJob] = []

        # Use indexed set review:jobs:running (O(K) where K is running jobs, avoiding full keyspace scan)
        running_job_ids = self.client.smembers(self.running_jobs_key)
        for raw_job_id in running_job_ids:
            job_id = self._to_str(raw_job_id)
            job_key = self._job_key(job_id)
            data = self.client.hgetall(job_key)
            if not data:
                self.client.srem(self.running_jobs_key, job_id)
                continue

            state_str = self._to_str(data.get(b"state") or data.get("state"))
            if state_str != JobState.RUNNING.value:
                self.client.srem(self.running_jobs_key, job_id)
                continue

            lease_expires_at = self._to_float(data.get(b"lease_expires_at") or data.get("lease_expires_at"))
            if current_time > lease_expires_at:
                job = self._deserialize_job(data)
                running_key = self._repo_running_key(job.repository_id)

                pipe = self.client.pipeline(transaction=True)
                pipe.srem(running_key, job.job_id)
                pipe.srem(self.running_jobs_key, job.job_id)

                if job.attempt_count >= job.max_retries:
                    pipe.hset(
                        job_key,
                        mapping={
                            "state": JobState.DEAD_LETTER.value,
                            "last_error": "Lease expired: worker heartbeat lost (max retries exhausted)",
                            "updated_at": str(current_time),
                            "lease_token": "",
                            "lease_expires_at": "0.0",
                        },
                    )
                    pipe.sadd(self.dead_letter_key, job.job_id)
                else:
                    delay = job.backoff_base_seconds * (2 ** (job.attempt_count - 1))
                    next_run = current_time + delay
                    pipe.hset(
                        job_key,
                        mapping={
                            "state": JobState.QUEUED.value,
                            "last_error": "Lease expired: worker heartbeat lost, rescheduled",
                            "next_run_at": str(next_run),
                            "updated_at": str(current_time),
                            "lease_token": "",
                            "lease_expires_at": "0.0",
                        },
                    )
                    pipe.zadd(self.queue_name, {job.job_id: int(next_run * 1000)})

                pipe.execute()
                updated = self.get_job(job.job_id)
                if updated:
                    recovered.append(updated)

        return recovered

    def close(self) -> None:
        """Close underlying Redis connection if owned."""
        if self._owns_client and hasattr(self.client, "close"):
            self.client.close()


async def review_job_task(ctx: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Real ARQ worker task function executing a claimed review job with lease token propagation.

    Retry Authority Coordination (Concern 1):
    1. ReviewJob.attempt_count, max_retries, and backoff_base_seconds remain the sole
       application retry authority.
    2. AutonomousReviewWorker.process_claimed_job() coordinates with RedisJobQueue.mark_failed().
    3. If retries remain (JobState.QUEUED):
       review_job_task raises arq.worker.Retry(defer=delay), matching the exact exponential
       backoff schedule computed by the application authority, without ARQ applying generic retries.
    4. If retries are exhausted (JobState.DEAD_LETTER):
       review_job_task returns a terminal status, allowing ARQ to cleanly finalize the task.
    5. If a worker process abruptly dies (crash/SIGKILL), ARQ's pessimistic locking
       (in_progress TTL expiration) guarantees safe re-execution without losing the job.

    ARQ Operational Cancellation (Concern 2):
    - When an ARQ worker with allow_abort_jobs=True aborts a task or when cancelled,
      operational cancellation is caught and logged cleanly.
    """
    queue: DurableQueueProtocol = ctx["queue"]
    worker: Any = ctx["worker"]
    current_time = time.time()

    if hasattr(queue, "claim_job"):
        job, lease_token = queue.claim_job(job_id, now=current_time)
    else:
        job = queue.get_job(job_id)
        lease_token = getattr(job, "lease_token", None) if job else None

    if job is None:
        if lease_token == "repo_at_capacity":
            from arq.worker import Retry
            raise Retry(defer=1.0)
        return {"status": "skipped", "reason": lease_token, "job_id": job_id}

    try:
        state = await worker.process_claimed_job(job, lease_token=lease_token, now=current_time)
        if state.is_cancelled or state.terminal_status in (JobState.CANCELLED.value, "cancelled"):
            return {
                "status": "cancelled",
                "job_id": job.job_id,
                "terminal_status": state.terminal_status,
            }

        if state.terminal_status in (JobState.FAILED.value, "failed"):
            updated_job = queue.get_job(job.job_id)
            if updated_job is not None and updated_job.state == JobState.QUEUED:
                delay = max(0.05, updated_job.next_run_at - current_time)
                orig_score = ctx.get("score")
                if orig_score is not None and hasattr(queue, "client") and hasattr(queue.client, "zadd"):
                    queue.client.zadd(getattr(queue, "queue_name", default_queue_name), {job.job_id: orig_score})
                from arq.worker import Retry
                raise Retry(defer=delay)
            elif updated_job is not None and updated_job.state == JobState.DEAD_LETTER:
                return {
                    "status": "dead_letter",
                    "job_id": job.job_id,
                    "terminal_status": state.terminal_status,
                }
            return {
                "status": "failed",
                "job_id": job.job_id,
                "terminal_status": state.terminal_status,
            }

        return {
            "status": "completed",
            "job_id": job.job_id,
            "terminal_status": state.terminal_status,
        }
    except StaleLeaseError as exc:
        logger.warning("Stale lease rejected for job %s (token: %s): %s", job_id, lease_token, exc)
        return {"status": "stale_lease_rejected", "job_id": job_id, "error": str(exc)}
    except asyncio.CancelledError:
        logger.info("ARQ task for job %s received operational abort/cancellation", job_id)
        return {"status": "cancelled", "job_id": job_id}
    except Exception as exc:
        # process_claimed_job already invoked mark_failed() enforcing ReviewJob authority
        updated_job = queue.get_job(job_id)
        if updated_job is not None and updated_job.state == JobState.QUEUED:
            delay = max(0.05, updated_job.next_run_at - current_time)
            logger.info(
                "Job %s failed attempt %d/%d; application retry authority deferring for %0.2fs",
                job_id,
                updated_job.attempt_count,
                updated_job.max_retries,
                delay,
            )
            # Re-align ARQ queue score so zincrby sets exact next_run_at timestamp
            orig_score = ctx.get("score")
            if orig_score is not None and hasattr(queue, "client") and hasattr(queue.client, "zadd"):
                queue.client.zadd(getattr(queue, "queue_name", default_queue_name), {job_id: orig_score})
            from arq.worker import Retry
            raise Retry(defer=delay)
        elif updated_job is not None and updated_job.state == JobState.DEAD_LETTER:
            logger.warning(
                "Job %s exhausted retries (%d/%d); dead-lettered by application authority",
                job_id,
                updated_job.attempt_count,
                updated_job.max_retries,
            )
            return {"status": "dead_letter", "job_id": job_id, "error": str(exc)}
        else:
            return {"status": "failed", "job_id": job_id, "error": str(exc)}
