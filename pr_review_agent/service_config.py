"""Unified service configuration for server and worker processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from typing import Any

from pr_review_agent.security import (
    RuntimeSecretRegistry,
    SecretType,
    SecurityConfig,
)


@dataclass(frozen=True)
class ServiceConfig:
    """Environment-backed configuration shared by ASGI server and worker."""

    github_token: str
    webhook_secret: bytes
    model_provider: str
    model_name: str
    database_path: str
    host: str
    port: int
    authorized_repositories: tuple[str, ...]
    authorized_tenant: str
    publish_enabled: bool = False
    max_diff_bytes: int = 500_000
    api_key: str = ""
    queue_backend: str = "sqlite"
    redis_url: str = ""
    checkpoint_backend: str = "none"
    database_backend: str = "sqlite"
    tiger_database_url: str = ""

    def __repr__(self) -> str:
        """Prevent secrets from leaking into logs or representations."""
        masked_redis = "'***'" if self.redis_url else "''"
        masked_tiger = "'***'" if self.tiger_database_url else "''"
        return (
            f"ServiceConfig(host={self.host!r}, port={self.port}, "
            f"model_provider={self.model_provider!r}, model_name={self.model_name!r}, "
            f"database_path={self.database_path!r}, database_backend={self.database_backend!r}, "
            f"authorized_tenant={self.authorized_tenant!r}, "
            f"authorized_repositories={self.authorized_repositories!r}, "
            f"publish_enabled={self.publish_enabled}, queue_backend={self.queue_backend!r}, "
            f"checkpoint_backend={self.checkpoint_backend!r}, "
            f"redis_url={masked_redis}, tiger_database_url={masked_tiger}, "
            f"github_token='***', webhook_secret=b'***', api_key='***')"
        )


    def to_security_config(
        self,
        secret_registry: RuntimeSecretRegistry | None = None,
        env_provider: Callable[[str], str | None] | None = None,
    ) -> SecurityConfig:
        """Construct SecurityConfig boundary from service configuration."""
        registry = secret_registry or RuntimeSecretRegistry()
        if self.github_token:
            registry.register_secret(SecretType.GITHUB_TOKEN, self.github_token)
        if self.tiger_database_url:
            from urllib.parse import urlparse
            parsed = urlparse(self.tiger_database_url)
            if parsed.password:
                registry.register_secret(SecretType.TIGERDB_CREDENTIAL, parsed.password)

        env_dict = {
            "GITHUB_TOKEN": self.github_token,
        }
        if self.model_provider == "openai":
            key = self.api_key or ((env_provider or os.environ.get)("OPENAI_API_KEY") or "")
            env_dict["OPENAI_API_KEY"] = key
            if key:
                registry.register_secret(SecretType.OPENAI_API_KEY, key)
        elif self.model_provider == "groq":
            key = self.api_key or ((env_provider or os.environ.get)("GROQ_API_KEY") or "")
            env_dict["GROQ_API_KEY"] = key
            if key:
                registry.register_secret(SecretType.GROQ_API_KEY, key)

        provider_fn = env_provider or env_dict.get
        return SecurityConfig(
            authorized_tenant=self.authorized_tenant,
            authorized_repositories=list(self.authorized_repositories),
            secret_registry=registry,
            env_provider=provider_fn,
        )


def load_service_config(
    env: Mapping[str, str] | None = None,
    *,
    require_live_credentials: bool = True,
) -> ServiceConfig:
    """Load and validate configuration from environment.

    Fails closed if required live credentials or repository authorizations are missing.
    """
    env_map = env if env is not None else os.environ

    github_token = env_map.get("GITHUB_TOKEN", "").strip()
    raw_secret = env_map.get("GITHUB_WEBHOOK_SECRET", "").strip()
    webhook_secret = raw_secret.encode("utf-8")

    provider = env_map.get("MODEL_PROVIDER", "openai").strip().lower()
    if provider not in ("openai", "groq"):
        raise ValueError(f"Invalid MODEL_PROVIDER '{provider}'. Must be 'openai' or 'groq'.")

    default_model = "gpt-4o" if provider == "openai" else "llama-3.3-70b-versatile"
    model_name = env_map.get("MODEL_NAME", default_model).strip() or default_model

    database_path = env_map.get("DATABASE_PATH", "pr_review_agent.db").strip()
    host = env_map.get("HOST", "127.0.0.1").strip()
    port = int(env_map.get("PORT", "8000").strip())

    repo_raw = env_map.get("GITHUB_REPOSITORY", "").strip()
    authorized_repos = tuple(r.strip() for r in repo_raw.split(",") if r.strip())

    tenant = env_map.get("AUTHORIZED_TENANT", "").strip()
    if not tenant and authorized_repos:
        tenant = authorized_repos[0].split("/")[0] if "/" in authorized_repos[0] else authorized_repos[0]

    publish_enabled = env_map.get("PUBLISH_LIVE_REVIEW", "0").strip() in ("1", "true", "TRUE")
    max_diff_bytes = int(env_map.get("MAX_DIFF_BYTES", "500000").strip())

    queue_backend = env_map.get("QUEUE_BACKEND", "sqlite").strip().lower()
    if queue_backend not in ("sqlite", "redis"):
        raise ValueError(f"Invalid QUEUE_BACKEND '{queue_backend}'. Must be 'sqlite' or 'redis'.")
    checkpoint_backend = env_map.get("CHECKPOINT_BACKEND", "none").strip().lower()
    if checkpoint_backend not in ("none", "memory", "redis"):
        raise ValueError(f"Invalid CHECKPOINT_BACKEND '{checkpoint_backend}'. Must be 'none', 'memory', or 'redis'.")
    redis_url = env_map.get("REDIS_URL", "").strip()

    database_backend = env_map.get("DATABASE_BACKEND", "sqlite").strip().lower()
    if database_backend not in ("sqlite", "tiger"):
        raise ValueError(f"Invalid DATABASE_BACKEND '{database_backend}'. Must be 'sqlite' or 'tiger'.")
    tiger_database_url = (
        env_map.get("TIGER_DATABASE_URL", "")
        or env_map.get("TIGER_URL", "")
    ).strip()

    if require_live_credentials:
        if not github_token:
            raise ValueError("Missing required GITHUB_TOKEN in environment.")
        if not webhook_secret:
            raise ValueError("Missing required GITHUB_WEBHOOK_SECRET in environment.")
        if not authorized_repos:
            raise ValueError("Missing required GITHUB_REPOSITORY in environment.")
        if provider == "openai" and not env_map.get("OPENAI_API_KEY", "").strip():
            raise ValueError("Missing required OPENAI_API_KEY for provider 'openai'.")
        if provider == "groq" and not env_map.get("GROQ_API_KEY", "").strip():
            raise ValueError("Missing required GROQ_API_KEY for provider 'groq'.")
        if (queue_backend == "redis" or checkpoint_backend == "redis") and not redis_url:
            raise ValueError("Missing required REDIS_URL when QUEUE_BACKEND or CHECKPOINT_BACKEND is 'redis'.")
        if database_backend == "tiger" and not tiger_database_url:
            raise ValueError("Missing required TIGER_DATABASE_URL when DATABASE_BACKEND is 'tiger'.")

    api_key = (
        env_map.get("OPENAI_API_KEY", "").strip()
        if provider == "openai"
        else env_map.get("GROQ_API_KEY", "").strip()
    )

    return ServiceConfig(
        github_token=github_token,
        webhook_secret=webhook_secret,
        model_provider=provider,
        model_name=model_name,
        database_path=database_path,
        host=host,
        port=port,
        authorized_repositories=authorized_repos,
        authorized_tenant=tenant or "default",
        publish_enabled=publish_enabled,
        max_diff_bytes=max_diff_bytes,
        api_key=api_key,
        queue_backend=queue_backend,
        redis_url=redis_url,
        checkpoint_backend=checkpoint_backend,
        database_backend=database_backend,
        tiger_database_url=tiger_database_url,
    )
