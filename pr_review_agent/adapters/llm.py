"""Concrete LLM network provider adapters behind the existing SpecialistHandler contract."""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable
import httpx

from pr_review_agent.cost_controls import ComponentUsage, ProviderPricingRegistry, UsageSource
from pr_review_agent.observability import AuditEvent, AuditSpine
from pr_review_agent.orchestration import (
    CandidateFinding,
    SpecialistHandler,
    SpecialistInput,
    SpecialistOutput,
    SpecialistStatus,
    SpecialistType,
)
from pr_review_agent.security import (
    ContentIsolationFramer,
    PromptInjectionDetector,
    PromptInjectionFinding,
    SecretLeakageScanner,
    SecretType,
    SecurityConfig,
)


SUPPORTED_PROVIDERS = frozenset({"openai", "groq"})

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
}


class LLMSpecialistAdapter:
    """Concrete network adapter providing SpecialistHandler implementations for external LLMs.

    Invariants:
    - Provider-neutral behind Callable[[SpecialistInput], SpecialistOutput].
    - Secrets resolved strictly via SecurityConfig immediately before client initialization.
    - Prompts constructed with ContentIsolationFramer (untrusted diffs/text remain passive data).
    - Prompt injection signals scanned via PromptInjectionDetector and emitted as auditable events.
    - All finding fields and error messages scanned for secret leakage.
    - Token/cost usage captured as ComponentUsage compatible with CostLedger.
    - Explicitly bounded retries for transient timeout/rate-limit/network failures.
    - Uses only the 4 canonical specialist roles (SECURITY, QUALITY, TESTS, DOCUMENTATION).
    """

    def __init__(
        self,
        provider: str,
        security_config: SecurityConfig,
        *,
        model: str | None = None,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.05,
        max_retry_backoff_seconds: float = 10.0,
        pricing_registry: ProviderPricingRegistry | None = None,
        secret_scanner: SecretLeakageScanner | None = None,
        audit_spine: AuditSpine | None = None,
        env_var_name: str | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        clean_provider = provider.strip().lower()
        if clean_provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. Must be one of: {sorted(SUPPORTED_PROVIDERS)}"
            )

        self.provider = clean_provider
        self.security_config = security_config
        self.model = model or DEFAULT_MODELS[self.provider]
        self.base_url = (base_url or DEFAULT_BASE_URLS[self.provider]).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.max_retry_backoff_seconds = max(self.retry_backoff_seconds, max_retry_backoff_seconds)
        self._sleeper = sleeper or time.sleep
        self.pricing_registry = pricing_registry or ProviderPricingRegistry()
        self.secret_scanner = secret_scanner or SecretLeakageScanner(security_config.secret_registry)
        self.audit_spine = audit_spine
        self.injection_detector = PromptInjectionDetector()
        self.last_injection_finding: PromptInjectionFinding | None = None

        # Resolve credential strictly via the approved client boundary
        if self.provider == "openai":
            self._api_key = self.security_config.resolve_for_client(
                boundary_target="openai_provider",
                secret_type=SecretType.OPENAI_API_KEY,
                env_var_name=env_var_name or "OPENAI_API_KEY",
            )
        else:
            self._api_key = self.security_config.resolve_for_client(
                boundary_target="groq_provider",
                secret_type=SecretType.GROQ_API_KEY,
                env_var_name=env_var_name or "GROQ_API_KEY",
            )

        self._client = httpx.Client(
            transport=transport,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"PR-Review-Agent-LLM/{self.provider}",
            },
        )

    def _compute_backoff(self, attempt: int, resp: httpx.Response | None = None) -> float:
        """Compute bounded backoff delay, respecting Retry-After on HTTP 429 when available."""
        if resp is not None and resp.status_code == 429:
            retry_after_hdr = resp.headers.get("Retry-After")
            if retry_after_hdr:
                try:
                    delay = float(retry_after_hdr.strip())
                    return max(0.0, min(self.max_retry_backoff_seconds, delay))
                except (ValueError, TypeError):
                    pass
        delay = self.retry_backoff_seconds * (2 ** attempt)
        return max(0.0, min(self.max_retry_backoff_seconds, delay))

    def _sanitize_error(self, message: str) -> str:
        """Sanitize error message to ensure no secrets or sensitive payload text leak."""
        if not message:
            return message
        if self.secret_scanner:
            return self.secret_scanner.sanitize_text(message)
        return message

    def _build_payload(self, spec_input: SpecialistInput) -> dict[str, Any]:
        """Construct the isolated prompt and request parameters."""
        specialist_role = spec_input.specialist_type.value

        # Content isolation: frame untrusted diff & evidence
        isolated_content = ContentIsolationFramer.build_isolated_prompt(
            specialist_role=specialist_role,
            untrusted_diff=spec_input.diff_content,
            untrusted_description="",
            retrieved_evidence=spec_input.retrieved_evidence,
        )

        specialist_guidelines = ""
        role_lower = specialist_role.lower()
        if spec_input.specialist_type == SpecialistType.SECURITY or role_lower == "security":
            specialist_guidelines = (
                "\nSecurity Evaluation Calibration:\n"
                "- Severity must reflect verified exploitability, exposure, and realistic threat impact.\n"
                "- Generic use of random, non-cryptographic hashing, or debug logging is NOT automatically a security vulnerability.\n"
                "- A security finding requires evidence that the behavior is security-sensitive in context.\n"
                "- For randomness specifically, classify it as a security issue only when the code path is used for something security-sensitive such as:\n"
                "  authentication secrets, password reset tokens, session identifiers, CSRF tokens, cryptographic material, authorization/security tokens, or other explicitly security-sensitive values.\n"
                "- If the code is clearly using randomness for a benign identifier, demo/test value, display value, sampling, non-security ID, etc., do not emit a high/medium security vulnerability finding merely because the API is non-cryptographic.\n"
                "- When the context is ambiguous, prefer omission or a lower-severity informational/quality observation rather than an unsupported security claim.\n"
                "- Never suppress a genuine security issue when surrounding code/evidence establishes security-sensitive use.\n"
            )
        elif spec_input.specialist_type == SpecialistType.QUALITY or role_lower == "quality":
            specialist_guidelines = (
                "\nQuality & Correctness Evaluation Calibration:\n"
                "- Severity must reflect actual runtime effect, crash potential, data corruption, and blast radius.\n"
                "- High severity is reserved for genuine, reproducible runtime crashes (e.g. unhandled ZeroDivisionError, null pointer dereference, syntax error, unhandled exception), memory exhaustion, or data loss.\n"
                "- Do NOT treat every code smell or antipattern as HIGH. Antipatterns such as mutable default arguments, poor variable naming, redundancy, or suboptimal complexity are medium or low severity.\n"
                "- Do NOT inflate severity merely because a defect sounds important. If a defect requires unusual or non-production conditions to trigger, classify it as medium or low.\n"
                "- When reporting an issue, ground the rationale in concrete runtime consequences observable in the code or diff.\n"
            )
        elif spec_input.specialist_type == SpecialistType.TESTS or role_lower == "tests":
            specialist_guidelines = (
                "\nTest Gap Evaluation Calibration:\n"
                "- Severity must reflect actual regression risk and the criticality of the untested logic.\n"
                "- Missing tests for standard functions, edge conditions, or new functionality are normally medium or low severity.\n"
                "- Low severity applies to ineffective assertions, smoke tests that only exercise execution without validating state, or minor edge cases.\n"
                "- Do NOT treat every missing test as HIGH. High severity for test gaps is strictly reserved for completely untested mission-critical safety or security authorization boundaries.\n"
                "- Focus on practical gaps that could allow undetected regressions in modified code.\n"
            )
        elif spec_input.specialist_type == SpecialistType.DOCUMENTATION or role_lower == "documentation":
            specialist_guidelines = (
                "\nDocumentation Evaluation Calibration:\n"
                "- Severity must reflect documentation impact on developers and consumers.\n"
                "- Documentation is non-executable and cannot directly crash running systems; NEVER classify documentation findings as HIGH severity.\n"
                "- Medium severity is reserved for material contradictions where the documentation explicitly misleads consumers about runtime behavior, parameter types, or return values.\n"
                "- Low or info severity applies to typographical errors, formatting issues, missing optional docstrings, or general clarity enhancements.\n"
                "- Do NOT treat minor wording differences as high-priority defects.\n"
            )

        system_prompt = (
            f"You are a specialist code review agent focusing exclusively on {specialist_role.upper()}.\n"
            f"Instructions:\n{spec_input.instructions}\n"
            f"{specialist_guidelines}\n"
            "Output Requirement:\n"
            "You MUST respond ONLY with a valid JSON object adhering to the following schema:\n"
            "{\n"
            '  "findings": [\n'
            "    {\n"
            '      "category": "string (e.g. security_vulnerability, defect, test_gap, doc_issue)",\n'
            '      "severity": "high" | "medium" | "low" | "info",\n'
            '      "confidence": float between 0.0 and 1.0,\n'
            '      "summary": "concise one-line summary",\n'
            '      "rationale": "detailed evidence-backed explanation",\n'
            '      "file_path": "path/to/file or null",\n'
            '      "line_range": [start_line, end_line] or null,\n'
            '      "evidence_refs": ["file#Lline"],\n'
            '      "remediation": "actionable fix or null"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "If no issues are found, return {\"findings\": []}."
        )

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": isolated_content},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

    @staticmethod
    def _extract_json_text(raw_text: str) -> str:
        """Extract valid JSON substring from potentially markdown-wrapped or commentary-laden output."""
        cleaned = (raw_text or "").strip()
        if not cleaned:
            return ""

        # 1. Direct JSON object
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned

        # 2. Extract from markdown code fences if present
        if "```" in cleaned:
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
            if fence_match:
                candidate = fence_match.group(1).strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    return candidate
                first_b = candidate.find("{")
                last_b = candidate.rfind("}")
                if first_b != -1 and last_b != -1 and last_b > first_b:
                    return candidate[first_b : last_b + 1]
                return candidate

        # 3. Extract outermost object braces { ... } from surrounding commentary
        first_b = cleaned.find("{")
        last_b = cleaned.rfind("}")
        if first_b != -1 and last_b != -1 and last_b > first_b:
            return cleaned[first_b : last_b + 1]

        return cleaned

    def _parse_findings(
        self,
        raw_text: str,
        spec_input: SpecialistInput,
    ) -> tuple[CandidateFinding, ...]:
        """Parse structured model response into CandidateFinding contracts.

        Enforces secret scanning across ALL externally generated fields:
        category, summary, rationale, file_path, remediation, evidence_refs.
        """
        cleaned_text = self._extract_json_text(raw_text)

        try:
            data = json.loads(cleaned_text)
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise ValueError(f"Malformed JSON from model: {err}") from err

        if not isinstance(data, dict) or "findings" not in data or not isinstance(data["findings"], list):
            raise ValueError("Model response missing required top-level 'findings' array")

        parsed: list[CandidateFinding] = []
        for idx, item in enumerate(data["findings"]):
            if not isinstance(item, dict):
                raise ValueError(f"Finding item {idx} in 'findings' array is not an object")

            summary = str(item.get("summary", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            if not summary or not rationale:
                raise ValueError(f"Finding item {idx} in 'findings' array missing required 'summary' or 'rationale'")

            category = str(item.get("category", spec_input.specialist_type.value))
            file_path = item.get("file_path")
            if file_path is not None:
                file_path = str(file_path).strip() or None

            remediation = item.get("remediation")
            if remediation is not None:
                remediation = str(remediation).strip() or None

            raw_ev = item.get("evidence_refs", [])
            evidence_refs = tuple(str(e) for e in raw_ev) if isinstance(raw_ev, (list, tuple)) else ()

            # Secret leakage scan across all externally generated fields
            if self.secret_scanner:
                category = self.secret_scanner.sanitize_text(category)
                summary = self.secret_scanner.sanitize_text(summary)
                rationale = self.secret_scanner.sanitize_text(rationale)
                if file_path is not None:
                    file_path = self.secret_scanner.sanitize_text(file_path)
                if remediation is not None:
                    remediation = self.secret_scanner.sanitize_text(remediation)
                if evidence_refs:
                    evidence_refs = tuple(self.secret_scanner.sanitize_text(ref) for ref in evidence_refs)

            severity = str(item.get("severity", "medium")).lower()
            if severity not in ("high", "medium", "low", "info"):
                severity = "medium"

            try:
                conf_val = float(item.get("confidence", 0.8))
                confidence = max(0.0, min(1.0, conf_val))
            except (ValueError, TypeError):
                confidence = 0.5

            line_range = None
            raw_lr = item.get("line_range")
            if isinstance(raw_lr, (list, tuple)) and len(raw_lr) == 2:
                try:
                    s_line, e_line = int(raw_lr[0]), int(raw_lr[1])
                    line_range = (s_line, e_line)
                except (ValueError, TypeError):
                    line_range = None

            stable_seed = f"{spec_input.correlation_id}:{spec_input.specialist_type.value}:{file_path}:{line_range}:{summary}"
            finding_id = f"cand-{hashlib.sha256(stable_seed.encode('utf-8')).hexdigest()[:12]}"

            finding = CandidateFinding(
                finding_id=finding_id,
                correlation_id=spec_input.correlation_id,
                specialist_type=spec_input.specialist_type,
                category=category,
                severity=severity,
                confidence=confidence,
                summary=summary,
                rationale=rationale,
                file_path=file_path,
                line_range=line_range,
                evidence_refs=evidence_refs,
                remediation=remediation,
            )
            parsed.append(finding)

        return tuple(parsed)

    def __call__(self, spec_input: SpecialistInput) -> SpecialistOutput:
        """Execute specialist evaluation synchronously via HTTP client with bounded retries."""
        start_time = time.time()
        endpoint = f"{self.base_url}/chat/completions"
        payload = self._build_payload(spec_input)

        if self.audit_spine is not None:
            self.audit_spine.record_event(
                AuditEvent(
                    correlation_id=spec_input.correlation_id,
                    event_name="specialist_started",
                    step=f"specialist_{spec_input.specialist_type.value}",
                    timestamp=time.time(),
                    details={
                        "specialist_type": spec_input.specialist_type.value,
                        "provider": self.provider,
                        "model": self.model,
                    },
                )
            )

        # Pre-scan diff for prompt injection indicators
        if spec_input.diff_content:
            inj_result = self.injection_detector.scan(spec_input.diff_content, source="diff_content")
            self.last_injection_finding = inj_result
            if inj_result.risk_level != "none" and self.audit_spine is not None:
                self.audit_spine.record_event(
                    AuditEvent(
                        correlation_id=spec_input.correlation_id,
                        event_name="prompt_injection_detected",
                        step=f"specialist_{spec_input.specialist_type.value}",
                        timestamp=time.time(),
                        details={
                            "source": inj_result.source,
                            "risk_level": inj_result.risk_level,
                            "indicators": list(inj_result.indicators),
                        },
                    )
                )

        last_error_message: str | None = None
        last_status = SpecialistStatus.FAILED

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(endpoint, json=payload)
                duration = round(time.time() - start_time, 3)

                if not resp.is_success:
                    safe_err_text = self._sanitize_error(resp.text[:300])
                    last_error_message = self._sanitize_error(
                        f"Provider {self.provider} error {resp.status_code}: {safe_err_text}"
                    )
                    last_status = SpecialistStatus.FAILED
                    # Only retry on transient rate-limit or 5xx server errors
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                        backoff = self._compute_backoff(attempt, resp)
                        if self.audit_spine is not None:
                            self.audit_spine.record_event(
                                AuditEvent(
                                    correlation_id=spec_input.correlation_id,
                                    event_name="specialist_retry",
                                    step=f"specialist_{spec_input.specialist_type.value}",
                                    timestamp=time.time(),
                                    details={
                                        "attempt": attempt + 1,
                                        "max_retries": self.max_retries,
                                        "provider": self.provider,
                                        "model": self.model,
                                        "status_code": resp.status_code,
                                        "error": last_error_message,
                                        "backoff_seconds": backoff,
                                    },
                                )
                            )
                        self._sleeper(backoff)
                        continue

                    if self.audit_spine is not None:
                        self.audit_spine.record_event(
                            AuditEvent(
                                correlation_id=spec_input.correlation_id,
                                event_name="specialist_failed",
                                step=f"specialist_{spec_input.specialist_type.value}",
                                timestamp=time.time(),
                                details={
                                    "specialist_type": spec_input.specialist_type.value,
                                    "provider": self.provider,
                                    "model": self.model,
                                    "status_code": resp.status_code,
                                    "error": last_error_message,
                                    "duration": duration,
                                },
                            )
                        )
                    return SpecialistOutput(
                        specialist_type=spec_input.specialist_type,
                        correlation_id=spec_input.correlation_id,
                        status=SpecialistStatus.FAILED,
                        error_message=last_error_message,
                        execution_duration=duration,
                    )

                resp_data = resp.json()
                choices = resp_data.get("choices", [])
                if not choices or not isinstance(choices, list):
                    last_error_message = f"Provider {self.provider} returned no choices"
                    last_status = SpecialistStatus.FAILED
                    if attempt < self.max_retries:
                        backoff = self._compute_backoff(attempt)
                        if self.audit_spine is not None:
                            self.audit_spine.record_event(
                                AuditEvent(
                                    correlation_id=spec_input.correlation_id,
                                    event_name="specialist_retry",
                                    step=f"specialist_{spec_input.specialist_type.value}",
                                    timestamp=time.time(),
                                    details={
                                        "attempt": attempt + 1,
                                        "max_retries": self.max_retries,
                                        "provider": self.provider,
                                        "model": self.model,
                                        "error": last_error_message,
                                        "backoff_seconds": backoff,
                                    },
                                )
                            )
                        self._sleeper(backoff)
                        continue

                    if self.audit_spine is not None:
                        self.audit_spine.record_event(
                            AuditEvent(
                                correlation_id=spec_input.correlation_id,
                                event_name="specialist_failed",
                                step=f"specialist_{spec_input.specialist_type.value}",
                                timestamp=time.time(),
                                details={
                                    "specialist_type": spec_input.specialist_type.value,
                                    "provider": self.provider,
                                    "model": self.model,
                                    "error": last_error_message,
                                    "duration": duration,
                                },
                            )
                        )
                    return SpecialistOutput(
                        specialist_type=spec_input.specialist_type,
                        correlation_id=spec_input.correlation_id,
                        status=SpecialistStatus.FAILED,
                        error_message=last_error_message,
                        execution_duration=duration,
                    )

                raw_content = choices[0].get("message", {}).get("content", "")

                # Token / usage extraction
                usage_data = resp_data.get("usage", {})
                prompt_tokens = usage_data.get("prompt_tokens")
                completion_tokens = usage_data.get("completion_tokens")
                total_tokens = usage_data.get("total_tokens")

                cost_usd = None
                pricing_configured = False
                if prompt_tokens is not None and completion_tokens is not None:
                    calc = self.pricing_registry.calculate_cost(
                        self.provider, self.model, prompt_tokens, completion_tokens
                    )
                    if calc is not None:
                        cost_usd = calc
                        pricing_configured = True

                comp_usage = ComponentUsage(
                    component=f"specialist_{spec_input.specialist_type.value}",
                    provider=self.provider,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    usage_source=UsageSource.PROVIDER_REPORTED if total_tokens is not None else UsageSource.UNAVAILABLE,
                    pricing_configured=pricing_configured,
                )

                try:
                    findings = self._parse_findings(raw_content, spec_input)
                except ValueError as parse_err:
                    safe_err_text = self._sanitize_error(str(parse_err))
                    last_error_message = safe_err_text
                    last_status = SpecialistStatus.FAILED
                    if attempt < self.max_retries:
                        backoff = self._compute_backoff(attempt)
                        if self.audit_spine is not None:
                            self.audit_spine.record_event(
                                AuditEvent(
                                    correlation_id=spec_input.correlation_id,
                                    event_name="specialist_retry",
                                    step=f"specialist_{spec_input.specialist_type.value}",
                                    timestamp=time.time(),
                                    details={
                                        "attempt": attempt + 1,
                                        "max_retries": self.max_retries,
                                        "provider": self.provider,
                                        "model": self.model,
                                        "error": last_error_message,
                                        "backoff_seconds": backoff,
                                    },
                                )
                            )
                        self._sleeper(backoff)
                        continue

                    if self.audit_spine is not None:
                        self.audit_spine.record_event(
                            AuditEvent(
                                correlation_id=spec_input.correlation_id,
                                event_name="specialist_failed",
                                step=f"specialist_{spec_input.specialist_type.value}",
                                timestamp=time.time(),
                                details={
                                    "specialist_type": spec_input.specialist_type.value,
                                    "provider": self.provider,
                                    "model": self.model,
                                    "error": last_error_message,
                                    "duration": duration,
                                },
                            )
                        )
                    return SpecialistOutput(
                        specialist_type=spec_input.specialist_type,
                        correlation_id=spec_input.correlation_id,
                        status=SpecialistStatus.FAILED,
                        findings=(),
                        error_message=last_error_message,
                        execution_duration=duration,
                        usage=comp_usage,
                    )

                if self.audit_spine is not None:
                    self.audit_spine.record_event(
                        AuditEvent(
                            correlation_id=spec_input.correlation_id,
                            event_name="specialist_succeeded",
                            step=f"specialist_{spec_input.specialist_type.value}",
                            timestamp=time.time(),
                            details={
                                "specialist_type": spec_input.specialist_type.value,
                                "provider": self.provider,
                                "model": self.model,
                                "findings_count": len(findings),
                                "duration": duration,
                                "total_tokens": comp_usage.total_tokens,
                                "cost_usd": comp_usage.cost_usd,
                            },
                        )
                    )

                return SpecialistOutput(
                    specialist_type=spec_input.specialist_type,
                    correlation_id=spec_input.correlation_id,
                    status=SpecialistStatus.COMPLETED,
                    findings=findings,
                    execution_duration=duration,
                    usage=comp_usage,
                )

            except httpx.TimeoutException as exc:
                duration = round(time.time() - start_time, 3)
                safe_exc_text = self._sanitize_error(str(exc))
                last_error_message = self._sanitize_error(
                    f"Timeout after {self.timeout_seconds}s querying {self.provider}: {safe_exc_text}"
                )
                last_status = SpecialistStatus.TIMEOUT
                if attempt < self.max_retries:
                    backoff = self._compute_backoff(attempt)
                    if self.audit_spine is not None:
                        self.audit_spine.record_event(
                            AuditEvent(
                                correlation_id=spec_input.correlation_id,
                                event_name="specialist_retry",
                                step=f"specialist_{spec_input.specialist_type.value}",
                                timestamp=time.time(),
                                details={
                                    "attempt": attempt + 1,
                                    "max_retries": self.max_retries,
                                    "provider": self.provider,
                                    "model": self.model,
                                    "error": last_error_message,
                                    "backoff_seconds": backoff,
                                },
                            )
                        )
                    self._sleeper(backoff)
                    continue

                if self.audit_spine is not None:
                    self.audit_spine.record_event(
                        AuditEvent(
                            correlation_id=spec_input.correlation_id,
                            event_name="specialist_timeout",
                            step=f"specialist_{spec_input.specialist_type.value}",
                            timestamp=time.time(),
                            details={
                                "specialist_type": spec_input.specialist_type.value,
                                "provider": self.provider,
                                "model": self.model,
                                "error": last_error_message,
                                "duration": duration,
                            },
                        )
                    )
                return SpecialistOutput(
                    specialist_type=spec_input.specialist_type,
                    correlation_id=spec_input.correlation_id,
                    status=SpecialistStatus.TIMEOUT,
                    error_message=last_error_message,
                    execution_duration=duration,
                )
            except (httpx.NetworkError, httpx.TransportError) as exc:
                duration = round(time.time() - start_time, 3)
                safe_exc_text = self._sanitize_error(str(exc))
                last_error_message = self._sanitize_error(
                    f"Network communication failure querying {self.provider}: {safe_exc_text}"
                )
                last_status = SpecialistStatus.FAILED
                if attempt < self.max_retries:
                    backoff = self._compute_backoff(attempt)
                    if self.audit_spine is not None:
                        self.audit_spine.record_event(
                            AuditEvent(
                                correlation_id=spec_input.correlation_id,
                                event_name="specialist_retry",
                                step=f"specialist_{spec_input.specialist_type.value}",
                                timestamp=time.time(),
                                details={
                                    "attempt": attempt + 1,
                                    "max_retries": self.max_retries,
                                    "provider": self.provider,
                                    "model": self.model,
                                    "error": last_error_message,
                                    "backoff_seconds": backoff,
                                },
                            )
                        )
                    self._sleeper(backoff)
                    continue

                if self.audit_spine is not None:
                    self.audit_spine.record_event(
                        AuditEvent(
                            correlation_id=spec_input.correlation_id,
                            event_name="specialist_failed",
                            step=f"specialist_{spec_input.specialist_type.value}",
                            timestamp=time.time(),
                            details={
                                "specialist_type": spec_input.specialist_type.value,
                                "provider": self.provider,
                                "model": self.model,
                                "error": last_error_message,
                                "duration": duration,
                            },
                        )
                    )
                return SpecialistOutput(
                    specialist_type=spec_input.specialist_type,
                    correlation_id=spec_input.correlation_id,
                    status=SpecialistStatus.FAILED,
                    error_message=last_error_message,
                    execution_duration=duration,
                )
            except Exception as exc:
                duration = round(time.time() - start_time, 3)
                safe_exc_text = self._sanitize_error(str(exc))
                err_msg = self._sanitize_error(f"Error querying {self.provider}: {safe_exc_text}")
                if self.audit_spine is not None:
                    self.audit_spine.record_event(
                        AuditEvent(
                            correlation_id=spec_input.correlation_id,
                            event_name="specialist_failed",
                            step=f"specialist_{spec_input.specialist_type.value}",
                            timestamp=time.time(),
                            details={
                                "specialist_type": spec_input.specialist_type.value,
                                "provider": self.provider,
                                "model": self.model,
                                "error": err_msg,
                                "duration": duration,
                            },
                        )
                    )
                return SpecialistOutput(
                    specialist_type=spec_input.specialist_type,
                    correlation_id=spec_input.correlation_id,
                    status=SpecialistStatus.FAILED,
                    error_message=err_msg,
                    execution_duration=duration,
                )

        duration = round(time.time() - start_time, 3)
        err_msg = last_error_message or f"Retries exhausted for {self.provider}"
        if self.audit_spine is not None:
            self.audit_spine.record_event(
                AuditEvent(
                    correlation_id=spec_input.correlation_id,
                    event_name="specialist_failed" if last_status == SpecialistStatus.FAILED else "specialist_timeout",
                    step=f"specialist_{spec_input.specialist_type.value}",
                    timestamp=time.time(),
                    details={
                        "specialist_type": spec_input.specialist_type.value,
                        "provider": self.provider,
                        "model": self.model,
                        "error": err_msg,
                        "duration": duration,
                    },
                )
            )
        return SpecialistOutput(
            specialist_type=spec_input.specialist_type,
            correlation_id=spec_input.correlation_id,
            status=last_status,
            error_message=err_msg,
            execution_duration=duration,
        )

    def close(self) -> None:
        """Close underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> LLMSpecialistAdapter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def create_specialist_handlers(
    provider: str,
    security_config: SecurityConfig,
    *,
    model: str | None = None,
    transport: httpx.BaseTransport | None = None,
    pricing_registry: ProviderPricingRegistry | None = None,
    audit_spine: AuditSpine | None = None,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.05,
    max_retry_backoff_seconds: float = 10.0,
    sleeper: Callable[[float], None] | None = None,
) -> dict[SpecialistType, SpecialistHandler]:
    """Factory creating specialist handlers for all 4 canonical review roles."""
    adapter = LLMSpecialistAdapter(
        provider=provider,
        security_config=security_config,
        model=model,
        transport=transport,
        pricing_registry=pricing_registry,
        audit_spine=audit_spine,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        max_retry_backoff_seconds=max_retry_backoff_seconds,
        sleeper=sleeper,
    )
    return {
        SpecialistType.SECURITY: adapter,
        SpecialistType.QUALITY: adapter,
        SpecialistType.TESTS: adapter,
        SpecialistType.DOCUMENTATION: adapter,
    }
