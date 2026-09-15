"""Network adapters and live integration foundation for PR-Review-Agent."""

from pr_review_agent.adapters.github import DiffRetrievalResult, GitHubNetworkClient
from pr_review_agent.adapters.llm import LLMSpecialistAdapter, create_specialist_handlers
from pr_review_agent.adapters.webhook_ingress import create_webhook_app

__all__ = [
    "DiffRetrievalResult",
    "GitHubNetworkClient",
    "LLMSpecialistAdapter",
    "create_specialist_handlers",
    "create_webhook_app",
]
