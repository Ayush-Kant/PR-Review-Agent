"""Cost and budget controls, provider pricing, concurrency management, and context bounding.

Implements:
- FR-19: Token/model/provider usage and cost accounting at run and component granularity,
         with configurable budget limits, concurrency limits, context-size limits,
         and graceful degradation rules.
- AC-13: Documented degraded, held, or skipped outcomes on budget exhaustion or context/concurrency caps
         without silently bypassing safeguards.
- NFR-04: Configurable provisional targets without invented numeric SLOs or fixed defaults.
- NFR-05: Per-repository budgets and concurrency controls preventing cross-repository starvation.
- NFR-12: Auditable explanation of skipped, degraded, or escalated reviews caused by budget policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import time
from typing import Any, Mapping, Sequence


class UsageSource(str, Enum):
    """Source of token usage reporting."""

    PROVIDER_REPORTED = "provider_reported"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CostAndBudgetConfig:
    """Configurable budget, concurrency, and context limits.

    All limits default to None (unconstrained) until explicitly set by operators.
    Per DECISION-67711afd and ASSUMPTION-b774604f, the system must not invent
    numeric defaults for SLOs, budgets, retention limits, or max PR/diff sizes.
    """

    max_tokens_per_run: int | None = None
    max_cost_usd_per_run: float | None = None
    monthly_budget_usd: float | None = None
    max_diff_bytes: int | None = None
    max_concurrent_reviews_per_repo: int | None = None
    soft_budget_ratio: float | None = None
    fail_closed_on_context_cap: bool = False


@dataclass(frozen=True)
class ModelPricingRate:
    """Explicitly configured pricing rate per 1,000 tokens."""

    provider: str
    model: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float


class ProviderPricingRegistry:
    """Provider-neutral pricing registry configured with explicit rates.

    Does not contain hardcoded or authoritative model prices.
    Returns None for unconfigured models/providers rather than fabricating a price.
    """

    def __init__(self, rates: Sequence[ModelPricingRate] | None = None) -> None:
        self._rates: dict[tuple[str, str], ModelPricingRate] = {}
        for r in (rates or ()):
            self.register_rate(r)

    def register_rate(self, rate: ModelPricingRate) -> None:
        self._rates[(rate.provider.strip().lower(), rate.model.strip().lower())] = rate

    def get_rate(self, provider: str, model: str) -> ModelPricingRate | None:
        return self._rates.get((provider.strip().lower(), model.strip().lower()))

    def calculate_cost(
        self,
        provider: str | None,
        model: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> float | None:
        """Deterministically calculate cost in USD if rate and token counts are known.

        Returns None if rate is unconfigured or token counts are unavailable.
        """
        if not provider or not model or prompt_tokens is None or completion_tokens is None:
            return None
        rate = self._rates.get((provider.strip().lower(), model.strip().lower()))
        if rate is None:
            return None
        input_cost = (prompt_tokens / 1000.0) * rate.input_cost_per_1k_tokens
        output_cost = (completion_tokens / 1000.0) * rate.output_cost_per_1k_tokens
        return round(input_cost + output_cost, 6)


@dataclass(frozen=True)
class ComponentUsage:
    """Token and cost accounting at component granularity (FR-19)."""

    component: str  # e.g., "specialist_security", "specialist_quality", "retrieval"
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    usage_source: UsageSource = UsageSource.UNAVAILABLE
    pricing_configured: bool = False


@dataclass(frozen=True)
class RunCostSummary:
    """Aggregate token and cost summary for an entire review run."""

    run_id: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    is_cost_complete: bool
    has_unavailable_tokens: bool = False
    components: tuple[ComponentUsage, ...] = ()


@dataclass(frozen=True)
class RepositorySpendSummary:
    """Repository spend summary distinguishing known spend from unknown/unavailable records."""

    repository_id: str
    known_cost_usd: float
    has_unknown_cost: bool
    unknown_cost_records_count: int
    total_records_count: int


class CostLedger:
    """Durable SQLite storage for component and run usage records (FR-19, NFR-06)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._init_schema()

    def _init_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS component_usage_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    repository_id TEXT NOT NULL,
                    component TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    usage_source TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_run_id
                ON component_usage_records (run_id)
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_repo_time
                ON component_usage_records (repository_id, timestamp)
                """
            )

    def record_usage(
        self,
        run_id: str,
        correlation_id: str,
        repository_id: str,
        usage: ComponentUsage,
        timestamp: float | None = None,
    ) -> None:
        """Persist a component usage record."""
        current_time = time.time() if timestamp is None else timestamp
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO component_usage_records (
                    run_id, correlation_id, repository_id, component,
                    provider, model, prompt_tokens, completion_tokens,
                    total_tokens, cost_usd, usage_source, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    correlation_id,
                    repository_id,
                    usage.component,
                    usage.provider,
                    usage.model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                    usage.cost_usd,
                    usage.usage_source.value,
                    current_time,
                ),
            )

    def get_run_cost_summary(self, run_id: str) -> RunCostSummary:
        """Query aggregate usage and cost for a run."""
        rows = self.connection.execute(
            """
            SELECT component, provider, model, prompt_tokens, completion_tokens,
                   total_tokens, cost_usd, usage_source
            FROM component_usage_records
            WHERE run_id = ?
            ORDER BY record_id ASC
            """,
            (run_id,),
        ).fetchall()

        components: list[ComponentUsage] = []
        tot_prompt = 0
        tot_completion = 0
        tot_tokens = 0
        tot_cost = 0.0
        is_complete = True
        has_unavail_tokens = False

        for row in rows:
            c_name, prov, mdl, p_tok, c_tok, t_tok, cost, u_src = row
            if p_tok is not None:
                tot_prompt += p_tok
            else:
                has_unavail_tokens = True
            if c_tok is not None:
                tot_completion += c_tok
            else:
                has_unavail_tokens = True
            if t_tok is not None:
                tot_tokens += t_tok
            else:
                has_unavail_tokens = True
            if cost is not None:
                tot_cost += cost
            else:
                is_complete = False
            if u_src == UsageSource.UNAVAILABLE.value:
                has_unavail_tokens = True

            components.append(
                ComponentUsage(
                    component=c_name,
                    provider=prov,
                    model=mdl,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                    cost_usd=cost,
                    usage_source=UsageSource(u_src),
                    pricing_configured=(cost is not None),
                )
            )

        return RunCostSummary(
            run_id=run_id,
            total_prompt_tokens=tot_prompt,
            total_completion_tokens=tot_completion,
            total_tokens=tot_tokens,
            total_cost_usd=round(tot_cost, 6),
            is_cost_complete=is_complete,
            has_unavailable_tokens=has_unavail_tokens,
            components=tuple(components),
        )

    def get_repository_spend_summary(
        self,
        repository_id: str,
        since_timestamp: float | None = None,
    ) -> RepositorySpendSummary:
        """Query repository spend distinguishing known cost from unknown/unavailable records."""
        query = "SELECT cost_usd, usage_source FROM component_usage_records WHERE repository_id = ?"
        params: list[Any] = [repository_id]
        if since_timestamp is not None:
            query += " AND timestamp >= ?"
            params.append(since_timestamp)

        rows = self.connection.execute(query, params).fetchall()
        known_cost = 0.0
        unknown_count = 0
        total_count = len(rows)

        for cost, u_src in rows:
            if cost is not None and u_src != UsageSource.UNAVAILABLE.value:
                known_cost += float(cost)
            else:
                unknown_count += 1

        return RepositorySpendSummary(
            repository_id=repository_id,
            known_cost_usd=round(known_cost, 6),
            has_unknown_cost=(unknown_count > 0),
            unknown_cost_records_count=unknown_count,
            total_records_count=total_count,
        )

    def get_repository_spend(
        self,
        repository_id: str,
        since_timestamp: float | None = None,
    ) -> float:
        """Calculate total known recorded spend in USD for a repository.

        Note: For hard budget admission decisions, use get_repository_spend_summary()
        to verify that cost accounting is complete and has no unknown/unavailable records.
        """
        summary = self.get_repository_spend_summary(repository_id, since_timestamp)
        return summary.known_cost_usd

    def get_component_breakdown(self, run_id: str) -> dict[str, ComponentUsage]:
        """Get component usage map keyed by component name."""
        summary = self.get_run_cost_summary(run_id)
        return {c.component: c for c in summary.components}


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of pre-execution budget and context evaluation."""

    allowed: bool
    status: str  # "admitted", "budget_exhausted", "unknown_budget_state", "context_cap_exceeded"
    reason: str | None = None
    is_degraded: bool = False
    original_diff_bytes: int = 0
    retained_diff_bytes: int = 0
    truncated_diff_bytes: int = 0
    degraded_diff_content: str | None = None


class BudgetEnforcer:
    """Evaluates admission, budget thresholds, and context limits against configuration."""

    def __init__(
        self,
        config: CostAndBudgetConfig,
        cost_ledger: CostLedger | None = None,
    ) -> None:
        self.config = config
        self.cost_ledger = cost_ledger

    def check_admission(
        self,
        repository_id: str,
        diff_content: str,
        now: float | None = None,
    ) -> AdmissionDecision:
        """Evaluate pre-execution admission against monthly budget and diff context cap."""
        original_bytes = len(diff_content.encode("utf-8"))

        # 1. Monthly budget check
        if self.config.monthly_budget_usd is not None:
            if self.cost_ledger is None:
                return AdmissionDecision(
                    allowed=False,
                    status="unknown_budget_state",
                    reason=(
                        f"Repository '{repository_id}' monthly budget is configured "
                        f"(${self.config.monthly_budget_usd:.4f}) but cost ledger is unavailable. "
                        f"Spend cannot be safely verified."
                    ),
                    original_diff_bytes=original_bytes,
                    retained_diff_bytes=original_bytes,
                )
            spend_summary = self.cost_ledger.get_repository_spend_summary(repository_id)
            if spend_summary.has_unknown_cost:
                # Required invariant: known cost remains known; unavailable cost remains unavailable;
                # the system must never silently treat unavailable cost as zero for a hard budget decision.
                return AdmissionDecision(
                    allowed=False,
                    status="unknown_budget_state",
                    reason=(
                        f"Repository '{repository_id}' has {spend_summary.unknown_cost_records_count} "
                        f"usage records with unknown/unavailable cost. Configured hard monthly budget "
                        f"(${self.config.monthly_budget_usd:.4f}) cannot be safely verified."
                    ),
                    original_diff_bytes=original_bytes,
                    retained_diff_bytes=original_bytes,
                )
            if spend_summary.known_cost_usd >= self.config.monthly_budget_usd:
                return AdmissionDecision(
                    allowed=False,
                    status="budget_exhausted",
                    reason=(
                        f"Repository '{repository_id}' spend (${spend_summary.known_cost_usd:.4f}) "
                        f"meets or exceeds monthly budget cap (${self.config.monthly_budget_usd:.4f})"
                    ),
                    original_diff_bytes=original_bytes,
                    retained_diff_bytes=original_bytes,
                )

        # 2. Context limit check
        if self.config.max_diff_bytes is not None and original_bytes > self.config.max_diff_bytes:
            if self.config.fail_closed_on_context_cap:
                return AdmissionDecision(
                    allowed=False,
                    status="context_cap_exceeded",
                    reason=(
                        f"Diff size ({original_bytes} bytes) exceeds configured cap "
                        f"({self.config.max_diff_bytes} bytes) with fail_closed_on_context_cap enabled"
                    ),
                    original_diff_bytes=original_bytes,
                    retained_diff_bytes=0,
                    truncated_diff_bytes=original_bytes,
                )

            # Apply deterministic bounded truncation strategy (prefix bounding to max_diff_bytes)
            # Guardrail: Do NOT invent semantic heuristics. Truncate strictly deterministically.
            encoded = diff_content.encode("utf-8")
            truncated_encoded = encoded[: self.config.max_diff_bytes]
            degraded_text = truncated_encoded.decode("utf-8", errors="ignore")
            retained_bytes = len(truncated_encoded)
            truncated_bytes = original_bytes - retained_bytes

            return AdmissionDecision(
                allowed=True,
                status="admitted",
                reason="Diff truncated to meet configured max_diff_bytes cap",
                is_degraded=True,
                original_diff_bytes=original_bytes,
                retained_diff_bytes=retained_bytes,
                truncated_diff_bytes=truncated_bytes,
                degraded_diff_content=degraded_text,
            )

        return AdmissionDecision(
            allowed=True,
            status="admitted",
            original_diff_bytes=original_bytes,
            retained_diff_bytes=original_bytes,
            truncated_diff_bytes=0,
            degraded_diff_content=diff_content,
        )

    def check_run_limits(
        self,
        total_tokens: int | None = None,
        total_cost_usd: float | None = None,
        *,
        summary: RunCostSummary | None = None,
    ) -> tuple[bool, str | None]:
        """Check if run-level limits (max_tokens_per_run, max_cost_usd_per_run) were exceeded.

        Required invariant: Missing/unavailable usage is never treated as zero to falsely satisfy a limit.
        """
        toks = summary.total_tokens if summary is not None else total_tokens
        cost = summary.total_cost_usd if summary is not None else total_cost_usd
        cost_complete = summary.is_cost_complete if summary is not None else (cost is not None)
        unavail_tokens = summary.has_unavailable_tokens if summary is not None else (toks is None)

        if self.config.max_tokens_per_run is not None:
            if unavail_tokens:
                return False, f"Cannot verify max_tokens_per_run ({self.config.max_tokens_per_run}): component token usage is unavailable"
            if toks is not None and toks > self.config.max_tokens_per_run:
                return False, f"Total run tokens ({toks}) exceeded limit ({self.config.max_tokens_per_run})"

        if self.config.max_cost_usd_per_run is not None:
            if not cost_complete:
                return False, f"Cannot verify max_cost_usd_per_run (${self.config.max_cost_usd_per_run:.4f}): component cost is incomplete or unavailable"
            if cost is not None and cost > self.config.max_cost_usd_per_run:
                return False, f"Total run cost (${cost:.4f}) exceeded limit (${self.config.max_cost_usd_per_run:.4f})"

        return True, None
