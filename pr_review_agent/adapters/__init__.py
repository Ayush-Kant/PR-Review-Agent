"""Network adapters and live integration foundation for PR-Review-Agent."""

from pr_review_agent.adapters.github import DiffRetrievalResult, GitHubNetworkClient
from pr_review_agent.adapters.llm import LLMSpecialistAdapter, create_specialist_handlers
from pr_review_agent.adapters.redis_checkpoint import (
    CheckpointDeserializationError,
    CheckpointStorageError,
    RedisCheckpointSaver,
)
from pr_review_agent.adapters.redis_queue import RedisJobQueue, StaleLeaseError
from pr_review_agent.adapters.tiger_connection import (
    TigerConfig,
    TigerConfigurationError,
    TigerConnectionError,
    TigerConnectionManager,
    TigerError,
    TigerExtensionError,
    mask_database_url,
)
from pr_review_agent.adapters.tiger_migrations import (
    CapabilityReport,
    ExtensionInspector,
    Migration,
    MigrationError,
    MigrationRunner,
)
from pr_review_agent.adapters.tiger_stores import (
    ConcurrentTransitionError,
    ConflictingFindingError,
    ReviewRunContext,
    RunNotFoundError,
    TigerAuditSpine,
    TigerCodeMemoryStore,
    TigerEffectStore,
    TigerReviewTruthStore,
    TigerStoreError,
    TigerTruthMissingParentError,
)
from pr_review_agent.adapters.webhook_ingress import create_webhook_app

__all__ = [
    "CapabilityReport",
    "CheckpointDeserializationError",
    "CheckpointStorageError",
    "ConcurrentTransitionError",
    "ConflictingFindingError",
    "DiffRetrievalResult",
    "ExtensionInspector",
    "GitHubNetworkClient",
    "LLMSpecialistAdapter",
    "Migration",
    "MigrationError",
    "MigrationRunner",
    "RedisCheckpointSaver",
    "RedisJobQueue",
    "ReviewRunContext",
    "RunNotFoundError",
    "StaleLeaseError",
    "TigerAuditSpine",
    "TigerCodeMemoryStore",
    "TigerConfig",
    "TigerConfigurationError",
    "TigerConnectionError",
    "TigerConnectionManager",
    "TigerEffectStore",
    "TigerError",
    "TigerExtensionError",
    "TigerReviewTruthStore",
    "TigerStoreError",
    "TigerTruthMissingParentError",
    "create_specialist_handlers",
    "create_webhook_app",
    "mask_database_url",
]
