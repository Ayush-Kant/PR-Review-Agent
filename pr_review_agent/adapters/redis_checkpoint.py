"""Redis-backed LangGraph workflow checkpoint adapter for PR-Review-Agent.

This adapter implements the LangGraph `BaseCheckpointSaver[str]` protocol using Redis
as the persistence store for intermediate workflow execution state.

Architectural Boundary:
1. Execution Plane (ARQ/Worker):
   - Handled by `RedisJobQueue` in `pr_review_agent/adapters/redis_queue.py`.
   - Keys: `review:job:*`, `arq:queue`, `review:jobs:running`, etc.
2. Checkpoint Plane (LangGraph Workflow State):
   - Handled by `RedisCheckpointSaver` in this module.
   - Keys: `review:checkpoint:*` strictly isolated.
   - Thread Identity: thread_id is derived from ReviewJob.job_id (1:1 with delivery).
   - Recovery: Worker crash recovery resumes from the latest checkpoint without re-executing
     already completed nodes or duplicating specialist fan-out.
3. Business Truth Plane:
   - ReviewTruthStore, AuditSpine, and future Tiger Cloud data remain authoritative for
     findings, audit history, and publication decisions. Checkpoint data is NOT business truth.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
import logging
import random
import re
import time
from types import TracebackType
from typing import Any

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    JsonPlusSerializer,
    PendingWrite,
    RunnableConfig,
    SerializerProtocol,
    WRITES_IDX_MAP,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
import redis

from pr_review_agent.orchestration import (
    CHECKPOINT_FORBIDDEN_KEYS,
    validate_checkpoint_state_security,
)

logger = logging.getLogger(__name__)


class CheckpointStorageError(RuntimeError):
    """Raised when Redis fails during a checkpoint read or write operation."""


class CheckpointDeserializationError(ValueError):
    """Raised when checkpoint data in Redis is corrupted or malformed."""


def _get_default_serializer() -> JsonPlusSerializer:
    """Create a JsonPlusSerializer with PR-Review-Agent domain types allowed."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("pr_review_agent.orchestration", "SpecialistType"),
            ("pr_review_agent.orchestration", "CandidateFinding"),
            ("pr_review_agent.orchestration", "SpecialistOutput"),
            ("pr_review_agent.orchestration", "SpecialistCoverageSummary"),
            ("pr_review_agent.orchestration", "AuditEvent"),
            ("pr_review_agent.orchestration", "SpecialistStatus"),
            ("pr_review_agent.orchestration", "JobState"),
            ("pr_review_agent.intake", "ReviewSnapshot"),
        ]
    )


def _validate_data_security_recursive(data: Any) -> None:
    """Recursively validate that no forbidden secret-bearing keys exist in checkpoint state."""
    if isinstance(data, Mapping):
        validate_checkpoint_state_security(data)
        for v in data.values():
            _validate_data_security_recursive(v)
    elif isinstance(data, (list, tuple, set, frozenset)):
        for item in data:
            _validate_data_security_recursive(item)


class RedisCheckpointSaver(
    BaseCheckpointSaver[str],
    AbstractContextManager,
    AbstractAsyncContextManager,
):
    """Redis-backed checkpointer satisfying the LangGraph BaseCheckpointSaver contract.

    Stores checkpoints, channel blobs, and pending writes in Redis with strict thread isolation
    and namespacing under `review:checkpoint:*`.
    """

    def __init__(
        self,
        client: redis.Redis[Any] | None = None,
        *,
        redis_url: str | None = None,
        serde: SerializerProtocol | None = None,
        key_prefix: str = "review:checkpoint",
        ttl_seconds: int | None = None,
    ) -> None:
        default_serde = _get_default_serializer() if serde is None else serde
        super().__init__(serde=default_serde)

        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self._raw_url = redis_url or ""
        self._owns_client = client is None

        if client is not None:
            self.client = client
        else:
            target_url = redis_url or "redis://127.0.0.1:6379/0"
            self.client = redis.Redis.from_url(target_url, decode_responses=False)

    @property
    def redis_url(self) -> str:
        return self._raw_url

    def __repr__(self) -> str:
        masked_url = re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", self._raw_url) if self._raw_url else "''"
        return f"RedisCheckpointSaver(key_prefix={self.key_prefix!r}, redis_url={masked_url!r}, ttl_seconds={self.ttl_seconds})"

    def __enter__(self) -> RedisCheckpointSaver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._owns_client:
            self.close()
        return None

    async def __aenter__(self) -> RedisCheckpointSaver:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._owns_client:
            self.close()
        return None

    def close(self) -> None:
        """Close the underlying Redis client if owned."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception as exc:
            logger.warning(f"Error closing Redis client: {exc}")

    # --- Redis Key Helpers ---

    def _data_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{self.key_prefix}:data:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _index_key(self, thread_id: str, checkpoint_ns: str) -> str:
        return f"{self.key_prefix}:index:{thread_id}:{checkpoint_ns}"

    def _blob_key(self, thread_id: str, checkpoint_ns: str, channel: str, version: str | int | float) -> str:
        return f"{self.key_prefix}:blob:{thread_id}:{checkpoint_ns}:{channel}:{version}"

    def _writes_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{self.key_prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _thread_keys_key(self, thread_id: str) -> str:
        return f"{self.key_prefix}:thread_keys:{thread_id}"

    @property
    def _threads_set_key(self) -> str:
        return f"{self.key_prefix}:threads"

    def _to_str(self, val: Any) -> str:
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val) if val is not None else ""

    # --- Sync Interface ---

    def get_next_version(self, current: str | int | float | None, channel: None = None) -> str:
        """Generate monotonically increasing channel version identifiers."""
        if current is None:
            current_v = 0
        elif isinstance(current, (int, float)):
            current_v = int(current)
        else:
            try:
                current_v = int(str(current).split(".")[0])
            except (ValueError, IndexError):
                current_v = 0
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        """Load channel values for specified versions from Redis blobs."""
        result: dict[str, Any] = {}
        for k, ver in versions.items():
            b_key = self._blob_key(thread_id, checkpoint_ns, k, ver)
            try:
                b_data = self.client.hgetall(b_key)
            except Exception as exc:
                raise CheckpointStorageError(f"Failed to read blob for channel '{k}': {exc}") from exc

            if not b_data:
                raise CheckpointDeserializationError(
                    f"Missing channel blob for channel '{k}' version '{ver}' in thread '{thread_id}'"
                )

            b_type = self._to_str(b_data.get(b"type") or b_data.get("type"))
            if b_type == "empty":
                continue

            b_bytes = b_data.get(b"data") if b"data" in b_data else b_data.get("data", b"")
            if not isinstance(b_bytes, bytes):
                b_bytes = bytes(b_bytes or b"")

            try:
                result[k] = self.serde.loads_typed((b_type, b_bytes))
            except Exception as exc:
                raise CheckpointDeserializationError(f"Failed to deserialize blob for channel '{k}': {exc}") from exc

        return result

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a checkpoint tuple from Redis by config (thread_id and optional checkpoint_id)."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not thread_id:
            return None

        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if not checkpoint_id:
            # Query index for latest checkpoint
            try:
                latest_members = self.client.zrevrange(self._index_key(thread_id, checkpoint_ns), 0, 0)
            except Exception as exc:
                raise CheckpointStorageError(f"Failed to query checkpoint index for thread {thread_id}: {exc}") from exc

            if not latest_members:
                return None
            checkpoint_id = self._to_str(latest_members[0])

        data_key = self._data_key(thread_id, checkpoint_ns, checkpoint_id)
        try:
            data = self.client.hgetall(data_key)
        except Exception as exc:
            raise CheckpointStorageError(f"Failed to read checkpoint data for {checkpoint_id}: {exc}") from exc

        if not data:
            return None

        cp_type = self._to_str(data.get(b"cp_type") or data.get("cp_type"))
        cp_data = data.get(b"cp_data") if b"cp_data" in data else data.get("cp_data", b"")
        meta_type = self._to_str(data.get(b"meta_type") or data.get("meta_type"))
        meta_data = data.get(b"meta_data") if b"meta_data" in data else data.get("meta_data", b"")
        parent_id = self._to_str(data.get(b"parent") or data.get("parent")) or None

        if not isinstance(cp_data, bytes):
            cp_data = bytes(cp_data or b"")
        if not isinstance(meta_data, bytes):
            meta_data = bytes(meta_data or b"")

        try:
            checkpoint_ = self.serde.loads_typed((cp_type, cp_data))
            metadata = self.serde.loads_typed((meta_type, meta_data))
            if not isinstance(checkpoint_, dict):
                raise CheckpointDeserializationError(f"Expected dict for checkpoint, got {type(checkpoint_).__name__}")
            if not isinstance(metadata, dict):
                raise CheckpointDeserializationError(f"Expected dict for metadata, got {type(metadata).__name__}")
        except Exception as exc:
            if isinstance(exc, CheckpointDeserializationError):
                raise
            raise CheckpointDeserializationError(f"Failed to deserialize checkpoint {checkpoint_id}: {exc}") from exc

        # Load blobs
        channel_values = self._load_blobs(thread_id, checkpoint_ns, checkpoint_.get("channel_versions", {}))

        # Load pending writes
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)
        pending_writes: list[PendingWrite] = []
        try:
            writes_dict = self.client.hgetall(writes_key)
        except Exception as exc:
            raise CheckpointStorageError(f"Failed to read pending writes for {checkpoint_id}: {exc}") from exc

        if writes_dict:
            # Sort writes deterministically by field key
            sorted_fields = sorted(writes_dict.keys(), key=lambda f: self._to_str(f))
            for f in sorted_fields:
                w_bytes = writes_dict[f]
                try:
                    w_item = self.serde.loads_typed(("msgpack", w_bytes)) if hasattr(self.serde, "loads_typed") else None
                    if w_item is not None and isinstance(w_item, (list, tuple)) and len(w_item) == 5:
                        t_id, ch, val_t, val_b, t_path = w_item
                        deserialized_val = self.serde.loads_typed((val_t, val_b))
                        pending_writes.append((t_id, ch, deserialized_val))
                except Exception as exc:
                    raise CheckpointDeserializationError(f"Failed to deserialize pending write in {checkpoint_id}: {exc}") from exc

        parent_config: RunnableConfig | None = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint_,
                "channel_values": channel_values,
            },
            metadata=metadata,
            pending_writes=pending_writes,
            parent_config=parent_config,
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint and its channel blobs atomically to Redis."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # 1. Enforce Security Contract (W1-02 Part 4 & W1-04 Requirement 9)
        channel_vals = checkpoint.get("channel_values", {})
        _validate_data_security_recursive(channel_vals)
        _validate_data_security_recursive(metadata)

        c = checkpoint.copy()
        values: dict[str, Any] = c.pop("channel_values", {})  # type: ignore[misc]
        c_versions = dict(c.get("channel_versions", {}))
        c_versions.update(new_versions)
        c["channel_versions"] = c_versions

        # 2. Serialize checkpoint and metadata
        try:
            cp_type, cp_bytes = self.serde.dumps_typed(c)
            meta_type, meta_bytes = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        except Exception as exc:
            raise ValueError(f"Failed to serialize checkpoint {checkpoint_id}: {exc}") from exc

        parent_id = config["configurable"].get("checkpoint_id", "") or ""

        data_key = self._data_key(thread_id, checkpoint_ns, checkpoint_id)
        index_key = self._index_key(thread_id, checkpoint_ns)
        thread_keys_key = self._thread_keys_key(thread_id)

        # 3. Persist atomically via Redis pipeline
        try:
            pipe = self.client.pipeline(transaction=True)

            # Store channel blobs for newly updated versions
            for k, ver in new_versions.items():
                blob_key = self._blob_key(thread_id, checkpoint_ns, k, ver)
                if k in values:
                    b_type, b_bytes = self.serde.dumps_typed(values[k])
                else:
                    b_type, b_bytes = "empty", b""
                pipe.hset(blob_key, mapping={"type": b_type, "data": b_bytes})
                pipe.sadd(thread_keys_key, blob_key)
                if self.ttl_seconds is not None:
                    pipe.expire(blob_key, self.ttl_seconds)

            # Store main checkpoint mapping
            pipe.hset(
                data_key,
                mapping={
                    "cp_type": cp_type,
                    "cp_data": cp_bytes,
                    "meta_type": meta_type,
                    "meta_data": meta_bytes,
                    "parent": parent_id,
                },
            )
            pipe.sadd(thread_keys_key, data_key)
            if self.ttl_seconds is not None:
                pipe.expire(data_key, self.ttl_seconds)

            # Index checkpoint by timestamp score
            score = time.time()
            pipe.zadd(index_key, {checkpoint_id: score})
            pipe.sadd(thread_keys_key, index_key)
            if self.ttl_seconds is not None:
                pipe.expire(index_key, self.ttl_seconds)

            # Track thread existence
            pipe.sadd(self._threads_set_key, thread_id)
            pipe.sadd(thread_keys_key, thread_keys_key)

            pipe.execute()
        except Exception as exc:
            raise CheckpointStorageError(f"Redis pipeline write failed for checkpoint {checkpoint_id}: {exc}") from exc

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save pending writes to Redis for a specific checkpoint step."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)
        thread_keys_key = self._thread_keys_key(thread_id)

        try:
            pipe = self.client.pipeline(transaction=True)
            for idx, (c, v) in enumerate(writes):
                inner_idx = WRITES_IDX_MAP.get(c, idx)
                field = f"{task_id}:{inner_idx}"
                v_type, v_bytes = self.serde.dumps_typed(v)
                payload = self.serde.dumps_typed([task_id, c, v_type, v_bytes, task_path])[1]
                pipe.hset(writes_key, field, payload)

            pipe.sadd(thread_keys_key, writes_key)
            if self.ttl_seconds is not None:
                pipe.expire(writes_key, self.ttl_seconds)
            pipe.execute()
        except Exception as exc:
            raise CheckpointStorageError(f"Redis write failed for pending writes in {checkpoint_id}: {exc}") from exc

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from Redis ordered from newest to oldest."""
        if config and "configurable" in config and "thread_id" in config["configurable"]:
            thread_ids = [config["configurable"]["thread_id"]]
        else:
            try:
                members = self.client.smembers(self._threads_set_key)
                thread_ids = [self._to_str(m) for m in members]
            except Exception as exc:
                raise CheckpointStorageError(f"Failed to list threads: {exc}") from exc

        config_checkpoint_ns = config["configurable"].get("checkpoint_ns") if config else None
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        remaining_limit = limit

        for thread_id in thread_ids:
            checkpoint_ns = config_checkpoint_ns or ""
            index_key = self._index_key(thread_id, checkpoint_ns)
            try:
                cp_ids = [self._to_str(m) for m in self.client.zrevrange(index_key, 0, -1)]
            except Exception as exc:
                raise CheckpointStorageError(f"Failed to query index for thread {thread_id}: {exc}") from exc

            # Chronological filtering by `before`:
            # `zrevrange` returns checkpoint IDs ordered chronologically from newest to oldest.
            # `before` specifies a checkpoint configuration; only checkpoints created
            # strictly before that checkpoint in the chronological order should be returned.
            if before_checkpoint_id:
                if before_checkpoint_id in cp_ids:
                    before_idx = cp_ids.index(before_checkpoint_id)
                    cp_ids = cp_ids[before_idx + 1 :]
                else:
                    try:
                        before_score = (
                            self.client.zscore(index_key, before_checkpoint_id)
                            if hasattr(self.client, "zscore")
                            else None
                        )
                    except Exception:
                        before_score = None

                    if before_score is not None:
                        filtered = []
                        for m in cp_ids:
                            try:
                                s = self.client.zscore(index_key, m)
                            except Exception:
                                s = None
                            if s is not None and s < before_score:
                                filtered.append(m)
                        cp_ids = filtered
                    else:
                        cp_ids = []

            for cp_id in cp_ids:
                if config_checkpoint_id and cp_id != config_checkpoint_id:
                    continue

                tup = self.get_tuple({
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cp_id,
                    }
                })
                if tup is None:
                    continue

                if filter and not all(
                    query_value == tup.metadata.get(query_key)
                    for query_key, query_value in filter.items()
                ):
                    continue

                yield tup

                if remaining_limit is not None:
                    remaining_limit -= 1
                    if remaining_limit <= 0:
                        return

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints, blobs, writes, and index keys for a thread atomically."""
        thread_keys_key = self._thread_keys_key(thread_id)
        try:
            keys_to_del = [self._to_str(k) for k in self.client.smembers(thread_keys_key)]
            pipe = self.client.pipeline(transaction=True)
            if keys_to_del:
                pipe.delete(*keys_to_del)
            pipe.delete(thread_keys_key)
            pipe.srem(self._threads_set_key, thread_id)
            pipe.execute()
        except Exception as exc:
            raise CheckpointStorageError(f"Failed to delete thread {thread_id}: {exc}") from exc

    # --- Async Interface (Non-blocking Thread Offload) ---

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Asynchronous get_tuple offloaded to executor thread."""
        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronous put offloaded to executor thread."""
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronous put_writes offloaded to executor thread."""
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronous list of checkpoint tuples."""
        tuples = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in tuples:
            yield item

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronous delete_thread offloaded to executor thread."""
        await asyncio.to_thread(self.delete_thread, thread_id)
