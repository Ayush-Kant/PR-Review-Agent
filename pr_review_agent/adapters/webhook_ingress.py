"""Thin ASGI HTTP webhook ingress adapter for GitHub pull-request events."""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from pr_review_agent.intake import WebhookIntake
from pr_review_agent.observability import AuditEvent, AuditSpine
from pr_review_agent.orchestration import DurableJobQueue


class WebhookIngressHandler:
    """Handler encapsulating webhook ingestion endpoints and queue bridging."""

    def __init__(
        self,
        webhook_intake: WebhookIntake,
        *,
        job_queue: DurableJobQueue | None = None,
        audit_spine: AuditSpine | None = None,
        policy_version: str = "v1",
        prompt_version: str = "v1",
        retrieval_index_version: str = "v1",
        model_configuration: Mapping[str, str] | None = None,
    ) -> None:
        self.webhook_intake = webhook_intake
        self.job_queue = job_queue
        self.audit_spine = audit_spine
        self.policy_version = policy_version
        self.prompt_version = prompt_version
        self.retrieval_index_version = retrieval_index_version
        self.model_configuration = dict(
            model_configuration or {"provider": "openai", "model": "gpt-4o"}
        )

    async def handle_github_webhook(self, request: Request) -> JSONResponse:
        """Receive raw webhook bytes, verify HMAC before parse, and bridge to queue."""
        # 1. Extract raw bytes directly without premature parsing
        raw_body = await request.body()
        headers = dict(request.headers)

        # 2. Delegate to WebhookIntake (which performs HMAC check BEFORE JSON parse)
        result = self.webhook_intake.accept(
            headers,
            raw_body,
            policy_version=self.policy_version,
            prompt_version=self.prompt_version,
            retrieval_index_version=self.retrieval_index_version,
            model_configuration=self.model_configuration,
        )

        if result.status == "rejected":
            # Fail closed immediately. Never reflect untrusted payload back to sender.
            return JSONResponse(
                {"status": "rejected", "error": "Invalid signature or delivery headers"},
                status_code=401,
            )

        if result.status == "ignored":
            return JSONResponse(
                {"status": "ignored", "delivery_id": result.delivery_id},
                status_code=200,
            )

        if result.status == "duplicate":
            return JSONResponse(
                {"status": "duplicate", "delivery_id": result.delivery_id},
                status_code=200,
            )

        # status == "accepted"
        if self.job_queue is None or result.snapshot is None:
            # Deterministic safe failure: an accepted delivery cannot be scheduled without a durable queue
            return JSONResponse(
                {
                    "status": "failed",
                    "error": "Durable job queue is not configured; cannot schedule review",
                    "delivery_id": result.delivery_id,
                },
                status_code=503,
            )

        try:
            job = self.job_queue.enqueue(result.snapshot, delivery_id=result.delivery_id)
            job_id = job.job_id
        except Exception as exc:
            return JSONResponse(
                {
                    "status": "failed",
                    "error": f"Failed to enqueue review job: {exc}",
                    "delivery_id": result.delivery_id,
                },
                status_code=503,
            )

        if self.audit_spine is not None and result.delivery_id:
            self.audit_spine.record_event(
                AuditEvent(
                    correlation_id=result.delivery_id,
                    event_name="webhook_ingress_accepted",
                    step="ingress",
                    timestamp=time.time(),
                    details={"delivery_id": result.delivery_id, "job_id": job_id},
                )
            )

        return JSONResponse(
            {
                "status": "accepted",
                "delivery_id": result.delivery_id,
                "job_id": job_id,
            },
            status_code=202,
        )

    async def handle_healthz(self, request: Request) -> JSONResponse:
        """Liveness and health check endpoint."""
        return JSONResponse(
            {"status": "healthy"},
            status_code=200,
        )


def create_webhook_app(
    webhook_intake: WebhookIntake,
    *,
    job_queue: DurableJobQueue | None = None,
    audit_spine: AuditSpine | None = None,
    policy_version: str = "v1",
    prompt_version: str = "v1",
    retrieval_index_version: str = "v1",
    model_configuration: Mapping[str, str] | None = None,
    require_job_queue: bool = True,
) -> Starlette:
    """Create a production-ready Starlette ASGI application for webhook ingress.

    Fails early if job_queue is omitted in production mode.
    """
    if require_job_queue and job_queue is None:
        raise ValueError("create_webhook_app requires a DurableJobQueue to ensure accepted reviews are scheduled.")

    handler = WebhookIngressHandler(
        webhook_intake,
        job_queue=job_queue,
        audit_spine=audit_spine,
        policy_version=policy_version,
        prompt_version=prompt_version,
        retrieval_index_version=retrieval_index_version,
        model_configuration=model_configuration,
    )

    routes = [
        Route("/webhooks/github", handler.handle_github_webhook, methods=["POST"]),
        Route("/healthz", handler.handle_healthz, methods=["GET"]),
    ]

    return Starlette(routes=routes)
