"""Unit tests for the W1-08B Gate 1 Real Redis validation harness.

Tests offline execution, mock Redis behavior, secret redaction, Lua failure handling,
and deterministic cleanup.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import tempfile
import time
from typing import Any
import unittest

from scripts.validate_redis_gate1 import (
    CheckResult,
    RedisGate1Validator,
    ValidationReport,
    mask_redis_url,
    parse_redis_metadata,
)
from tests.test_redis_queue_adapter import InMemoryRedisClient


class OfflineTestRedisClient(InMemoryRedisClient):
    """In-memory Redis mock supporting all operations required by RedisGate1Validator."""

    def __init__(self, *, redis_version: str = "7.2.4-offline", simulate_bad_lua: bool = False) -> None:
        super().__init__()
        self.redis_version = redis_version
        self.simulate_bad_lua = simulate_bad_lua
        self._is_offline_mock = True

    def ping(self) -> bool:
        self._ensure_open()
        return True

    def info(self, section: str | None = None) -> dict[str, Any]:
        self._ensure_open()
        return {
            "redis_version": self.redis_version,
            "server": {"redis_version": self.redis_version},
        }

    def sismember(self, name: str, value: Any) -> bool:
        self._ensure_open()
        s = self._sets.get(str(name), set())
        s_val = str(value)
        return s_val in s

    def zrevrange(self, name: str, start: int = 0, end: int = -1) -> list[bytes]:
        self._ensure_open()
        s_name = str(name)
        z = self._zsets.get(s_name, {})
        sorted_items = sorted(z.items(), key=lambda x: x[1], reverse=True)
        members = [m.encode("utf-8") for m, _ in sorted_items]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    def zscore(self, name: str, member: Any) -> float | None:
        self._ensure_open()
        z = self._zsets.get(str(name), {})
        return z.get(str(member))

    def expire(self, name: str, time_sec: int) -> bool:
        self._ensure_open()
        return True

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        if self.simulate_bad_lua:
            raise RuntimeError("Simulated Redis Lua engine script failure")
        return super().eval(script, numkeys, *keys_and_args)


class TestRedisGate1Validator(unittest.TestCase):
    """Offline validation harness unit tests."""

    def setUp(self) -> None:
        self.client = OfflineTestRedisClient()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.evidence_path = Path(self.temp_dir.name) / "test_evidence.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_url_masking(self) -> None:
        """Verify complete credential redaction in Redis URLs."""
        masked1 = mask_redis_url("rediss://default:super_secret_token@redis.staging.internal:6379/1")
        self.assertEqual(masked1, "rediss://default:***@redis.staging.internal:6379/1")
        self.assertNotIn("super_secret_token", masked1)

        masked2 = mask_redis_url("redis://:pass123@127.0.0.1:6379/0")
        self.assertEqual(masked2, "redis://:***@127.0.0.1:6379/0")
        self.assertNotIn("pass123", masked2)

        masked3 = mask_redis_url("redis://localhost:6379/0")
        self.assertEqual(masked3, "redis://localhost:6379/0")

    def test_metadata_parsing(self) -> None:
        """Verify metadata extraction from Redis URLs."""
        meta = parse_redis_metadata("rediss://agent:my-secret@staging-redis:6379/2?ssl=true")
        self.assertEqual(meta["scheme"], "rediss")
        self.assertEqual(meta["host"], "staging-redis")
        self.assertEqual(meta["port"], 6379)
        self.assertEqual(meta["db"], 2)
        self.assertTrue(meta["tls"])
        self.assertTrue(meta["authenticated"])
        self.assertNotIn("my-secret", meta["masked_url"])

    def test_offline_validation_success(self) -> None:
        """Run full validation suite with offline client and verify all checks pass."""
        validator = RedisGate1Validator(
            client=self.client,
            allow_insecure=True,
            require_staging_isolation=False,
            evidence_path=self.evidence_path,
        )

        report = validator.validate_all()

        self.assertEqual(report.overall_status, "offline_verified")
        self.assertFalse(report.live_redis_executed)
        self.assertIsNone(report.failure_details)
        self.assertEqual(report.server_version, "7.2.4-offline")

        # Verify all 8 checks passed
        expected_checks = [
            "connectivity",
            "authentication",
            "lua_engine",
            "queue_lifecycle",
            "arq_serialization",
            "checkpoint",
            "secret_redaction",
            "cleanup",
        ]
        for name in expected_checks:
            self.assertIn(name, report.checks)
            self.assertEqual(report.checks[name]["status"], "passed", f"Check {name} failed: {report.checks[name]}")

        # Verify report was persisted to evidence path
        self.assertTrue(self.evidence_path.exists())
        saved_data = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_data["overall_status"], "offline_verified")
        self.assertIn("capabilities", saved_data)
        self.assertIn("environmental_context", saved_data)

    def test_capabilities_distinction(self) -> None:
        """Verify that offline execution distinguishes offline verification from live proof,

        and correctly reports deferred/unproven infrastructure capabilities.
        """
        validator = RedisGate1Validator(
            client=self.client,
            allow_insecure=True,
            require_staging_isolation=False,
            evidence_path=self.evidence_path,
        )
        report = validator.validate_all()
        self.assertIsNotNone(report.capabilities)
        caps = report.capabilities or {}

        # 5 functional capabilities verified offline
        for func_cap in (
            "redis_functional_semantics",
            "redis_native_lua_execution",
            "redis_checkpoint_semantics",
            "redis_arq_serialization_compatibility",
            "redis_secret_handling",
        ):
            self.assertIn(func_cap, caps)
            self.assertEqual(caps[func_cap]["status"], "OFFLINE_VERIFIED")
            self.assertEqual(caps[func_cap]["verdict"], "pass")
            # Offline mock must NEVER claim PROVEN live status
            self.assertNotEqual(caps[func_cap]["status"], "PROVEN")

        # 5 infrastructure capabilities deferred
        for infra_cap in (
            "redis_tls_transport",
            "redis_ha_failover",
            "managed_production_backups_recovery",
            "private_production_networking",
        ):
            self.assertIn(infra_cap, caps)
            self.assertEqual(caps[infra_cap]["status"], "NOT_PROVEN")
            self.assertEqual(caps[infra_cap]["verdict"], "deferred")

        self.assertEqual(caps["production_scale_capacity"]["status"], "NOT_BENCHMARKED")
        self.assertEqual(caps["production_scale_capacity"]["verdict"], "deferred")


    def test_lua_failure_handling(self) -> None:
        """Verify that Lua script failure is caught and reported rather than falling back silently."""
        bad_client = OfflineTestRedisClient(simulate_bad_lua=True)
        validator = RedisGate1Validator(
            client=bad_client,
            allow_insecure=True,
            require_staging_isolation=False,
            evidence_path=self.evidence_path,
        )

        report = validator.validate_all()
        self.assertEqual(report.overall_status, "failed")
        self.assertEqual(report.checks["lua_engine"]["status"], "failed")
        self.assertIn("Simulated Redis Lua engine script failure", report.checks["lua_engine"]["error"])

    def test_no_url_or_client_behavior(self) -> None:
        """Verify validator fails closed when no client or REDIS_URL is provided."""
        validator = RedisGate1Validator(redis_url="", client=None, evidence_path=self.evidence_path)
        report = validator.validate_all()
        self.assertEqual(report.overall_status, "not_executed")
        self.assertFalse(report.live_redis_executed)
        self.assertIn("No REDIS_URL provided", report.failure_details or "")

    def test_clean_deterministic_cleanup(self) -> None:
        """Verify cleanup removes all keys created by the harness without touching unrelated keys."""
        # Pre-seed an unrelated key and an active unrelated job with a valid future lease
        self.client.set("unrelated_production_key", "must_not_be_touched")
        self.client.hset(
            "review:job:unrelated_running_job",
            mapping={"state": "running", "lease_expires_at": str(time.time() + 3600.0)},
        )
        self.client.sadd("review:jobs:running", "unrelated_running_job")

        validator = RedisGate1Validator(
            client=self.client,
            allow_insecure=True,
            require_staging_isolation=False,
            evidence_path=self.evidence_path,
        )
        report = validator.validate_all()
        self.assertEqual(report.overall_status, "offline_verified")

        # Unrelated key must remain intact
        self.assertEqual(self.client.get("unrelated_production_key"), b"must_not_be_touched")
        self.assertTrue(self.client.sismember("review:jobs:running", "unrelated_running_job"))


if __name__ == "__main__":
    unittest.main()
