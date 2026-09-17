"""Runnable ASGI server entrypoint exposing webhook ingress and health check endpoints."""

from __future__ import annotations

import sqlite3
from typing import Any

from starlette.applications import Starlette
import uvicorn

from pr_review_agent.adapters.redis_queue import RedisJobQueue
from pr_review_agent.adapters.tiger_connection import (
    TigerConfig,
    TigerConfigurationError,
    TigerConnectionManager,
)
from pr_review_agent.adapters.tiger_stores import TigerAuditSpine
from pr_review_agent.adapters.webhook_ingress import create_webhook_app
from pr_review_agent.intake import WebhookIntake
from pr_review_agent.observability import AuditSpine
from pr_review_agent.orchestration import DurableJobQueue, DurableQueueProtocol
from pr_review_agent.service_config import ServiceConfig, load_service_config


def create_server_app(
    config: ServiceConfig,
    connection: sqlite3.Connection | None = None,
    *,
    job_queue: DurableQueueProtocol | None = None,
    audit_spine: AuditSpine | TigerAuditSpine | None = None,
    tiger_connection_manager: TigerConnectionManager | None = None,
) -> Starlette:
    """Instantiate the Starlette application with shared persistent storage and components."""
    conn = connection
    if conn is None:
        conn = sqlite3.connect(config.database_path, check_same_thread=False)

    intake = WebhookIntake(conn, config.webhook_secret)

    if job_queue is None:
        if getattr(config, "queue_backend", "sqlite") == "redis":
            if not getattr(config, "redis_url", ""):
                raise ValueError("REDIS_URL is required when QUEUE_BACKEND is 'redis'")
            job_queue = RedisJobQueue(redis_url=config.redis_url)
        else:
            job_queue = DurableJobQueue(conn)

    if audit_spine is None:
        if getattr(config, "database_backend", "sqlite") == "tiger":
            if tiger_connection_manager is None:
                if not getattr(config, "tiger_database_url", ""):
                    raise TigerConfigurationError("TIGER_DATABASE_URL is required when DATABASE_BACKEND is 'tiger'")
                tiger_cfg = TigerConfig.from_url(config.tiger_database_url)
                tiger_connection_manager = TigerConnectionManager(tiger_cfg)
            audit_spine = TigerAuditSpine(tiger_connection_manager)
        else:
            audit_spine = AuditSpine(conn)

    app = create_webhook_app(
        intake,
        job_queue=job_queue,
        audit_spine=audit_spine,
        model_configuration={"provider": config.model_provider, "model": config.model_name},
        require_job_queue=True,
    )
    # Store resolved runtime components on app.state for inspection / testing / lifecycle
    app.state.config = config
    app.state.intake = intake
    app.state.job_queue = job_queue
    app.state.audit_spine = audit_spine
    app.state.tiger_connection_manager = tiger_connection_manager
    return app


def run_server(
    config: ServiceConfig | None = None,
    connection: sqlite3.Connection | None = None,
    *,
    tiger_connection_manager: TigerConnectionManager | None = None,
) -> None:
    """Run the ASGI server via uvicorn."""
    cfg = config or load_service_config(require_live_credentials=True)
    app = create_server_app(cfg, connection=connection, tiger_connection_manager=tiger_connection_manager)
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    run_server()
