# Product specification — PR-Review-Agent

> Status: draft — awaiting human approval. No product implementation may begin until approval is recorded through Genesis.

## Problem

Pull-request authors and reviewers need early, trustworthy feedback on changes that may introduce security vulnerabilities, correctness defects, inadequate tests, or misleading documentation. Existing automated review often produces generic, ungrounded comment floods that reviewers cannot trust. PR-Review-Agent must provide selective, repository-aware findings that a human can audit, accept, dispute, or use to improve future review quality.

## Users

- **Pull-request author:** of one of the user's own GitHub repositories; receives concise actionable feedback in the pull request and can respond or dispute a finding.
- **Repository maintainer/reviewer:** for one of the user's own GitHub repositories; decides whether findings may be published, resolves escalations, and supplies feedback on review quality.
- **Developer/engineering-team dashboard user:** views repository review status, findings, severity/confidence, history, failures, and relevant system-health information.
- **Platform/operator:** configures repository access, policy, budgets, retention, reliability controls, and observes system health.
- **Evaluation owner:** curates golden pull requests, measures quality and regressions, and promotes or rolls back review-policy changes.

## Scope and end-to-end outcome

For a supported GitHub pull-request event, the system validates and durably accepts the webhook, processes the associated repository change asynchronously, retrieves only relevant repository context, runs the applicable specialist reviews in parallel, aggregates grounded structured findings, applies deduplication and confidence/HITL policy, and creates a GitHub review containing only findings permitted by that policy. Every material action, input version, and decision is auditable.

## Functional requirements

- FR-01: **GitHub event intake.** For the user's configured GitHub repositories, receive pull-request opened, reopened, and synchronized/updated webhook deliveries; verify the GitHub HMAC signature before parsing or scheduling untrusted payload data; reject invalid requests without invoking review processing.
- FR-02: **Delivery idempotency.**** Persist the GitHub delivery ID and processing state atomically so duplicate deliveries neither create duplicate jobs nor publish duplicate reviews. A retry of an incomplete delivery resumes or safely replays according to its recorded state.
- FR-03: **Durable asynchronous dispatch.**** After validation, enqueue review work through Redis/ARQ (or an approved equivalent preserving the same semantics), with a durable job identity, bounded retries, exponential backoff, timeout/dead-letter outcome, and correlation to the delivery and pull request.
- FR-04: **Immutable review snapshot.**** Bind each review run to the repository identifier, pull-request number, head SHA, base SHA, changed-file set, policy version, prompt/template version, retrieval-index version, and model/provider configuration. Do not publish results from a run whose head SHA is no longer the current reviewed revision.
- FR-05: **Orchestration.**** Use a LangGraph-based orchestrator to execute the review lifecycle, track per-step state, tolerate partial specialist failure, enforce run deadlines and cancellation, and invoke aggregation only after required work completes or reaches a declared terminal state.
- FR-06: **Specialist separation.**** Provide independently scoped Security, Quality/Correctness, Tests, and Documentation specialists. Each specialist receives only its concern-specific instructions plus relevant diff and retrieved evidence; it must not make final publication decisions.
- FR-07: **Repository-aware hybrid retrieval.**** Retrieve relevant repository context using both lexical and semantic signals over versioned repository content. Rank and cap context by relevance; include paths, revisions, and excerpts/citations in the specialist input. Do not supply an unbounded repository dump to a model.
- FR-08: **Evidence-grounded findings.**** Require every candidate finding to conform to a versioned structured contract containing, at minimum: stable finding ID; category; severity; confidence; summary; rationale; changed-file location/range when applicable; supporting diff and repository evidence references; remediation; specialist identity; and review-run provenance. Findings without sufficient evidence must be suppressed or escalated, never presented as authoritative inline comments.
- FR-09: **Aggregation and deduplication.**** Aggregate specialist outputs; normalize locations and categories; detect duplicate or materially overlapping findings; preserve links to all contributing candidates; and produce one canonical finding with an explainable merge decision.
- FR-10: **Confidence and risk policy.** Calculate or assign calibrated confidence for canonical findings using declared, versioned policy inputs. Combine confidence with severity/risk and repository policy to decide publication, human approval, escalation, or suppression. High-impact or blocking findings must require human approval before publication; lower-risk findings may publish automatically only when configured confidence/severity policy permits it.
- FR-11: **Human-in-the-loop workflow.**** Provide a maintainer workflow to approve, reject, edit, dismiss, or dispute held findings, record the actor and rationale, and publish only the approved content. The workflow must present the evidence and provenance needed for an informed decision.
- FR-12: **GitHub review output.**** Create/update GitHub reviews and inline comments only for policy-permitted, still-current, deduplicated findings. Associate each output with the review run and finding ID, respect GitHub API limits, and avoid duplicate comments across redelivery, retry, or rerun.
- FR-13: **Durable code memory.**** Maintain versioned repository/code memory suitable for retrieval, including source revision, chunk/location metadata, embedding/index version, freshness status, and access controls. Re-index incrementally when repository content changes and make freshness visible to review policy.
- FR-14: **Review truth.**** Maintain a durable review-truth record for each finding through its lifecycle: candidate, merged, held, published, approved, rejected, disputed, resolved, and superseded states; human feedback; evidence references; and outcome timestamps. Truth records are append-only or versioned so prior decisions remain auditable.
- FR-15: **Event/audit spine.**** Emit time-ordered, correlation-ID-bearing events for webhook receipt/validation, queue transitions, orchestration steps, retrieval, model calls, finding state transitions, policy decisions, GitHub actions, retries, failures, and human actions. Events must permit reconstruction of a review run without retaining secrets.
- FR-16: **Evaluation and regression gates.**** Support a versioned golden pull-request dataset with expected findings and allowed tolerances. Evaluate model, prompt, retrieval, orchestration, and policy changes against development and holdout sets; block promotion when configured quality, safety, latency, or cost regression gates fail.
- FR-17: **Observability and operations.** Expose a repository-user-facing dashboard for developers and engineering teams, plus machine-readable telemetry, covering review status, findings, severity/confidence, history, failures, relevant health/operations information, review throughput, queue depth/age, processing duration, failure/retry rates, retrieval freshness, finding dispositions, reviewer feedback, model usage/cost, and audit links. Support correlation-based investigation of a single run.
- FR-18: **Security and prompt-injection defense.** Treat webhook payloads, repository text, PR text, issue references, and retrieved documents as untrusted data. Constrain model tools and data access by repository/policy; isolate instructions from content; detect and label suspected prompt injection; prevent secrets from prompts, logs, events, findings, and GitHub output; use environment/configuration-based secrets; and apply GitHub credentials limited to reading the repository/pull request and publishing reviews.
- FR-19: **Cost and budget controls.** Record token/model/provider usage and cost at run and component granularity. Enforce configurable repository budget limits, concurrency limits, context-size limits, and graceful degradation rules that preserve correctness and auditability. Exact numeric limits remain provisional until product evidence justifies them.
- FR-20: **Feedback, learning, and drift detection.**** Capture maintainer dispositions and disputes as labeled feedback linked to the originating finding and policy version. Detect material drift in finding quality, calibration, repository freshness, model behavior, or cost; require evaluation, independent review, explicit promotion, and rollback capability before feedback changes production policy.

## Non-functional requirements

- NFR-01: **Correctness before speed.** When freshness, evidence, idempotency, authorization, or policy state is uncertain, fail closed for publication and expose the review as held, incomplete, or failed with an actionable reason.
- NFR-02: **Security.**** All secrets are stored only in an approved secret manager; use TLS for external and service-to-service traffic; restrict GitHub credentials to the minimum repository and review permissions; and rotate/revoke credentials without code changes.
- NFR-03: **Reliability.**** The system must survive duplicate webhook delivery, worker restart, transient provider/API failure, and individual specialist failure without silent loss, duplicate outward effects, or loss of audit history. Terminal failures must be observable and recoverable by an operator.
- NFR-04: **Performance.** The product must document configurable provisional targets for webhook acknowledgement, queue delay, end-to-end review completion, GitHub publication, HITL escalation latency, supported PR size, budgets, and retention; numeric commitments must not be invented before they are justified. Measurable values for these targets must be defined before production launch. Automated publication is permitted only when the review completed within the policy’s freshness window for the reviewed head SHA.
- NFR-05: **Scalability.**** Queueing, orchestration, retrieval, and state stores must scale horizontally without relying on process-local delivery state; per-repository budgets and concurrency controls must prevent one repository from starving others.
- NFR-06: **Auditability.**** A privileged operator can reconstruct any review decision from immutable identifiers and retained evidence/provenance, including the responsible policy, model/prompt, retrieval index, specialists, human actors, and GitHub effects.
- NFR-07: **Privacy and data governance.**** Enforce repository tenancy boundaries; minimize stored content; encrypt retained sensitive data; make retention/deletion policy configurable; and never use private repository content or feedback for cross-tenant training or evaluation without explicit authorization.
- NFR-08: **Observability.**** Logs, metrics, traces, and audit events must be structured, correlated, redacted, access-controlled, and retained according to policy. Operational alerts must cover intake failures, queue aging, elevated errors, budget exhaustion, retrieval staleness, and publication failures.
- NFR-09: **Quality.**** Finding contracts, policy configuration, event schemas, and state transitions are versioned and backward-compatible or migrated explicitly. Changes affecting output quality require golden-set evaluation and regression gates before promotion.
- NFR-10: **Usability.**** GitHub feedback must be concise, evidence-linked, actionable, and explain its confidence/policy disposition where appropriate. The dashboard and HITL workflow must make held findings and failures understandable without inspecting raw logs.
- NFR-11: **Accessibility.**** The dashboard and human approval workflow must meet WCAG 2.2 AA for supported browsers.
- NFR-12: **Cost predictability.**** Operators can set hard and soft usage budgets and receive an auditable explanation of a skipped, degraded, or escalated review caused by budget policy.

## Constraints

### Invariants

- V1 is single-tenant and supports only the user's own GitHub repositories. GitHub is the sole V1 pull-request/source-control integration; GitHub webhook HMAC validation is mandatory.
- Redis/ARQ is the intended asynchronous queue and LangGraph is the intended orchestrator; any substitution needs a recorded architecture decision showing equivalent idempotency, durable-state, timeout, and audit semantics. Model/provider selection must remain configurable and provider-neutral where practical.
- The architecture uses specialist agents, not one undifferentiated reviewer prompt.
- Repository context is retrieved selectively through hybrid retrieval; unbounded repository context is prohibited.
- A finding may be published only after it is structured, evidence-grounded, deduplicated, confidence/risk evaluated, current for the reviewed SHA, and permitted by policy/HITL decision.
- Durable code memory, review truth, and time-ordered events are separate concerns with explicit schemas and identifiers; they are not incidental logs.
- The system must be selective and high-value; maximizing raw comment count is not a product goal.

## Non-goals

- Replacing human code review or automatically merging pull requests.
- Supporting non-GitHub source-control providers.
- Multi-tenant SaaS or organization-wide tenancy in v1.
- Autonomous remediation or code-writing commits.
- Publishing any finding with no repository/diff evidence or stale head-SHA binding.
- Automatically training or promoting production behavior directly from reviewer feedback.
- Building a broad generic chat assistant for repositories.

## Acceptance criteria

- AC-01: A valid configured GitHub pull-request webhook is HMAC-validated, assigned a durable delivery record and correlation ID, and results in exactly one review job for repeated deliveries with the same delivery ID.
- AC-02: An invalid or missing webhook signature produces no queue job, model/retrieval call, GitHub output, or leaked payload/secret in telemetry.
- AC-03: A review run records immutable snapshot provenance (repository, PR number, base/head SHAs, policy, prompt, retrieval-index, and model/provider versions) before specialist work starts.
- AC-04: For a representative PR, Security, Quality/Correctness, Tests, and Documentation specialist outputs are independently identifiable in the audit spine and each output is bound to a concern-specific scope and retrieved evidence.
- AC-05: A finding lacking an evidence reference to the reviewed diff or versioned repository content cannot be automatically published and is recorded as suppressed or held with a reason.
- AC-06: Two specialist candidates describing the same defect/location result in one canonical GitHub-facing finding, with a durable merge record linking both candidates.
- AC-07: A configured high-impact or blocking finding is held for human action and creates no GitHub inline comment until a maintainer approval is recorded; a lower-risk finding is automatically published only when configured confidence/severity policy permits it. Rejection/dispute is recorded with actor and rationale.
- AC-08: A permitted, approved finding generates one GitHub review/comment at the correct current diff location; redelivery, worker retry, and rerun do not duplicate it.
- AC-09: If the PR head SHA changes before publication, the stale run creates no GitHub output and is marked superseded or rerun against the new snapshot according to policy.
- AC-10: A worker restart, transient GitHub/model failure, and one specialist timeout each produce recoverable/terminal states, bounded retries, correlated audit events, and no duplicate outward action.
- AC-11: An operator can trace a published, held, suppressed, or failed finding from GitHub output (or its absence) through review truth, policy decision, specialist/retrieval evidence, model usage, queue events, and webhook delivery without accessing secrets.
- AC-12: A policy/model/prompt/retrieval change is prevented from production promotion when its configured golden-set holdout regression gate fails; the evaluation result identifies the version and failed metric.
- AC-13: Budget exhaustion or a configured context/concurrency cap produces a documented degraded, held, or skipped outcome with recorded cost/limit evidence and never silently bypasses required safeguards.
- AC-14: A suspected prompt-injection instruction in PR or repository content is treated as untrusted content, appears in the audit trail as a security signal, and cannot alter the orchestrator policy, tool permissions, or GitHub publication decision.
- AC-15: The repository-user-facing dashboard exposes review status, findings, severity/confidence, history, failures, relevant health/operations information, queue age, end-to-end duration, retry/failure rate, finding dispositions, retrieval freshness, and per-run cost, with correlation links to the audit spine.

## Risks

- **False positives and reviewer fatigue:** Require evidence, deduplication, confidence/risk policy, evaluation thresholds, and HITL escalation; measure accepted/rejected/disputed finding rates.
- **False negatives:** Use independently scoped specialists, curated golden PRs, feedback analysis, and regression gates; present the tool as assistive rather than complete assurance.
- **Prompt injection or malicious repository content:** Treat all repository/PR text as untrusted; constrain tools, separate instructions from content, redact secrets, and audit suspected injections.
- **Stale or insufficient context:** Bind runs to SHAs/index versions, surface retrieval freshness, cap and rank context, and fail closed or escalate when evidence is inadequate.
- **Duplicate or inconsistent outbound reviews:** Use delivery/job/output idempotency keys, review truth state, current-SHA checks, and GitHub action records.
- **Provider outage, latency, or cost spikes:** Use bounded retry/timeout, partial-failure state, queues/dead letters, rate/concurrency controls, budgets, and explicit degraded modes.
- **Sensitive code exposure and tenant leakage:** Enforce least privilege, tenant-scoped data/indexes, encryption, retention controls, redaction, and no cross-tenant learning without authorization.
- **Evaluation overfitting or unsafe learning:** Separate development/holdout datasets; require independent review, explicit promotion, and rollback for policy changes.

## Open questions

1. What evidence and review process will justify the initial numeric SLOs, budgets, retention limits, and maximum supported PR size before production launch?
2. Which hosting environment, data-residency requirements, and approved secret/observability services are permitted?
3. Which model providers are permitted initially, and what provider-selection, fallback, or fail-closed policy applies when no permitted provider is available?
4. Are draft PRs, forked PRs, and comment-triggered reruns intentionally excluded from v1, or should any be added to the supported event scope?

