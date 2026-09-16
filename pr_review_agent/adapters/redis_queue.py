"""Redis / ARQ distributed execution queue adapter conforming to DurableQueueProtocol."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
import uuid

import redis
from arq.constants import abort_jobs_ss, default_queue_name

from pr_review_agent.intake import ReviewSnapshot
from pr_review_agent.orchestration import (
    DurableQueueProtocol,
    JobState,
    ReviewJob,
)

logger = logging.getLogger(__name__)


class StaleLeaseError(RuntimeError):
    """Raised when a worker attempts to mutate a job whose lease has expired or been reassigned."""


class RedisJobQueue(DurableQueueProtocol):
    """Redis / ARQ execution queue adapter satisfying DurableQueueProtocol.

    Architecture & Invariants:
    1. ARQ as Single Queue: 'arq:queue' (or configured queue_name) is the sole scheduling queue.
       Jobs are scored by eligibility timestamp in milliseconds.
    2. Durable Review State: Domain job state and snapshot provenance are persisted in 'review:job:{job_id}'.
       ARQ execution is strictly coordinated without replacing business truth (ReviewTruth / AuditSpine).
    3. Delivery Idempotency: 'review:delivery:{delivery_id}' ensures O(1) duplicate protection.
    4. Logical Attempt Authority: ReviewJob.attempt_count and max_retries govern retry limits and exponential backoff.
    5. Lease Ownership: Atomic lease_token prevents stale workers from completing expired/reclaimed jobs.
    6. Repository Concurrency (NFR-05): 'review:repo_running:{repo_id}' enforces max_concurrency_per_repo.
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

        if client is not None:
            self.client = client
        else:
            target_url = redis_url or "redis://127.0.0.1:6379/0"
            self.client = redis.Redis.from_url(target_url, decode_responses=False)

    def _job_key(self, job_id: str) -> str:
        return f"review:job:{job_id}"

    def _delivery_key(self, delivery_id: str) -> str:
        return f"review:delivery:{delivery_id}"

    def _repo_running_key(self, repo_id: str) -> str:
        return f"review:repo_running:{repo_id}"

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
        """Enqueue a review job atomically with durable identity correlated to the delivery."""
        current_time = time.time() if now is None else now
        job_id = f"job-{delivery_id}"
        delivery_key = self._delivery_key(delivery_id)
        job_key = self._job_key(job_id)

        # Idempotency check: if delivery already exists, return existing job
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

        # Pipeline atomic write: delivery index, job hash, and ARQ queue scheduling
        pipe = self.client.pipeline(transaction=True)
        pipe.set(delivery_key, job_id)
        pipe.hset(job_key, mapping=job_data)
        score_ms = int(current_time * 1000)
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
            claimed, _ = self._attempt_claim(cand_job, current_time)
            if claimed is not None:
                return claimed

        return None

    def _attempt_claim(
        self,
        job: ReviewJob,
        now: float,
    ) -> tuple[ReviewJob | None, str | None]:
        """Atomically pop from arq:queue, transition to RUNNING, and issue lease token."""
        job_key = self._job_key(job.job_id)
        running_key = self._repo_running_key(job.repository_id)

        # Atomic removal from scheduling queue to guarantee single claim
        removed = self.client.zrem(self.queue_name, job.job_id)
        if not removed:
            return None, "already_leased"

        new_attempt = job.attempt_count + 1
        lease_token = str(uuid.uuid4())
        lease_expires_at = now + job.deadline_seconds

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

        if effective_cap is not None:
            running_key = self._repo_running_key(job.repository_id)
            running_count = self.client.scard(running_key)
            if running_count >= effective_cap:
                return None, "repo_at_capacity"

        return self._attempt_claim(job, current_time)

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
            if stored_token and stored_token != lease_token:
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
            if stored_token and stored_token != lease_token:
                raise StaleLeaseError(f"Stale lease mutation rejected for job {job_id}")

        job = self._deserialize_job(data)
        running_key = self._repo_running_key(job.repository_id)

        pipe = self.client.pipeline(transaction=True)
        pipe.srem(running_key, job_id)

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
    ) -> ReviewJob:
        """Cancel a job, recording reason, removing from queue, and signaling ARQ abort."""
        current_time = time.time() if now is None else now
        job_key = self._job_key(job_id)
        data = self.client.hgetall(job_key)
        if not data:
            raise KeyError(f"Job {job_id} not found")

        job = self._deserialize_job(data)
        running_key = self._repo_running_key(job.repository_id)

        pipe = self.client.pipeline(transaction=True)
        pipe.hset(
            job_key,
            mapping={
                "state": JobState.CANCELLED.value,
                "last_error": reason or "Cancelled by operator",
                "updated_at": str(current_time),
            },
        )
        pipe.srem(running_key, job_id)
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

    def get_job_payload(self, job_id: str) -> dict[str, Any]:
        """Retrieve stored snapshot payload json for reconstruction."""
        job_key = self._job_key(job_id)
        payload_val = self.client.hget(job_key, "payload_json") or self.client.hget(job_key, b"payload_json")
        if not payload_val:
            return {}
        return json.loads(self._to_str(payload_val))

    def recover_zombie_jobs(self, now: float | None = None) -> list[ReviewJob]:
        """Recover RUNNING jobs whose leases expired without completion (worker crash/partition)."""
        current_time = time.time() if now is None else now
        recovered: list[ReviewJob] = []

        # Find delivery keys to locate active jobs
        delivery_keys = self.client.keys("review:delivery:*")
        for d_key in delivery_keys:
            job_id_val = self.client.get(d_key)
            if not job_id_val:
                continue
            job_id = self._to_str(job_id_val)
            job_key = self._job_key(job_id)
            data = self.client.hgetall(job_key)
            if not data:
                continue

            state_str = self._to_str(data.get(b"state") or data.get("state"))
            if state_str != JobState.RUNNING.value:
                continue

            lease_expires_at = self._to_float(data.get(b"lease_expires_at") or data.get("lease_expires_at"))
            if current_time > lease_expires_at:
                job = self._deserialize_job(data)
                running_key = self._repo_running_key(job.repository_id)
                self.client.srem(running_key, job.job_id)

                if job.attempt_count >= job.max_retries:
                    self.client.hset(
                        job_key,
                        mapping={
                            "state": JobState.DEAD_LETTER.value,
                            "last_error": "Lease expired: worker heartbeat lost (max retries exhausted)",
                            "updated_at": str(current_time),
                        },
                    )
                    self.client.sadd(self.dead_letter_key, job.job_id)
                else:
                    delay = job.backoff_base_seconds * (2 ** (job.attempt_count - 1))
                    next_run = current_time + delay
                    self.client.hset(
                        job_key,
                        mapping={
                            "state": JobState.QUEUED.value,
                            "last_error": "Lease expired: worker heartbeat lost, rescheduled",
                            "next_run_at": str(next_run),
                            "updated_at": str(current_time),
                        },
                    )
                    self.client.zadd(self.queue_name, {job.job_id: int(next_run * 1000)})

                updated = self.get_job(job.job_id)
                if updated:
                    recovered.append(updated)

        return recovered
