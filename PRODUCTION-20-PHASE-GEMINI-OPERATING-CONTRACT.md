# PR-Review-Agent — 20-Phase Production Build Operating Contract

> **Branch:** `agent-v1`  
> **Baseline:** `4d59caa823b9de38e5766ca0885798a1b7bd4c54`  
> **Repository:** `Ayush-Kant/PR-Review-Agent`  
> **Purpose:** This document is the working contract for the transition from the validated V1 review engine to the full production-oriented 0–20 lifecycle described by the user-provided architecture design.

---

## 0. READ THIS FIRST

This file is not a replacement for Genesis, `SPEC.md`, repository code, tests, or approved Genesis decisions.

The authority order for implementation is:

1. **Genesis workflow/governance rules**
2. **Approved `SPEC.md` and Genesis decisions/invariants**
3. **Actual repository implementation and proven evidence**
4. **This document**
5. **The user-provided 20-phase architecture design as the target architecture**

The target architecture explains **where this project is going**. Genesis controls **what may be implemented next and how that work is proven**.

Do not silently turn any architecture target into an implementation requirement. Before implementation, the relevant target must be reconciled to the current Genesis SPEC/PLAN and converted into an approved bounded task.

---

# 1. CURRENT PROJECT BASELINE

At the beginning of this production transition, the repository is in a deliberately strong V1 state.

## 1.1 Genesis state

The current Genesis reconciliation reported:

- workflow: `new-product`
- state: `verify/ready`
- active task: none
- blockers: none
- approved SPEC: `SPEC-1`
- approved PLAN: all currently declared tasks complete
- `HEAD == origin/main`
- baseline commit: `4d59caa823b9de38e5766ca0885798a1b7bd4c54`
- worktree: clean apart from local development-only `artifacts/` / `scratch/` material not part of the product history

The current repository-backed reconciliation reports all 47 declared requirements satisfied, all 16 declared Genesis implementation tasks completed, and no missing Genesis evidence. The source report also records the binding invariants and decisions that govern the next stage.

## 1.2 Existing proven capabilities

The existing system already provides the core review engine and safety boundary:

- GitHub webhook intake
- HMAC validation
- delivery idempotency
- immutable review snapshotting
- current-head-SHA protection
- durable asynchronous dispatch
- LangGraph orchestration
- four specialist roles
  - Security
  - Quality
  - Tests
  - Docs
- repository-aware hybrid retrieval abstraction
- evidence-grounded structured findings
- finding aggregation and semantic deduplication
- calibrated severity and confidence policy
- HITL workflow
- Review Truth lifecycle
- current-SHA-safe GitHub review publication
- append-only audit/event spine
- operational telemetry/dashboard foundation
- prompt-injection detection and content isolation
- secret scanning/redaction boundaries
- cost and budget controls
- golden dataset evaluation
- development/holdout evaluation machinery
- feedback ledger
- drift detection
- promotion/rollback machinery
- real GitHub/provider network adapters
- live PR validation harness
- live real-model golden development validation

The latest validated milestone before this transition was the live-golden validation path with full regression at 227/227 tests, offline golden validation 5/5, and live development benchmark validation 5/5. Preserve this level of evidence discipline.

## 1.3 Binding safety decisions

### Correctness before speed

A finding may be published only when it is:

- structured
- traceable to repository and diff evidence
- deduplicated
- confidence-evaluated
- authorized by the applicable policy

When head SHA, retrieval freshness, evidence, delivery state, authorization, or policy outcome is uncertain, the system must fail closed.

### V1 publication boundary

High-impact or blocking findings require human approval. Lower-risk findings may auto-publish only under configured policy. The system must not autonomously modify code or merge pull requests.

### V1 persistence decision

The current approved V1 uses SQLite for durable queue/state semantics. Redis/ARQ is the target for future production multi-worker deployment. Do not bypass Genesis by introducing production infrastructure merely because this document mentions the target architecture.

### Provisional operational values

SLOs, budgets, retention limits, and maximum PR-size limits are configurable/provisional until justified by evidence.

---

# 2. WHY THIS BRANCH EXISTS

`agent-v1` is the production-evolution branch.

`main` remains the validated V1 baseline until a later, explicit promotion decision is made.

The intent is:

```text
main
  = validated V1 review engine

agent-v1
  = controlled evolution toward the full 0–20 production architecture
```

Do not destabilize `main` in order to accelerate `agent-v1`.

Every production-evolution change must preserve the validated behavior unless the active Genesis task explicitly changes the relevant contract.

---

# 3. THE TARGET PRODUCT

The intended final architecture is a governed, observable, evidence-grounded AI pull-request review platform.

The core runtime target is:

```text
GitHub
  ↓
FastAPI secure ingress
  ↓
Redis / ARQ durable worker queue
  ↓
LangGraph recoverable orchestration
  ↓
4 specialists in parallel
  ├── Security
  ├── Quality
  ├── Tests
  └── Docs
  ↓
hybrid repository retrieval
  ↓
structured findings
  ↓
aggregation + semantic deduplication
  ↓
severity / confidence / policy
  ↓
HITL when required
  ↓
GitHub publication
```

The durable production data target is:

```text
Tiger Cloud / Postgres-compatible durable spine

memory:
  code_chunks
  vector search
  full-text search

truth:
  review records
  finding records
  HITL state
  human feedback

time:
  agent events
  traces / decision events
  cost / latency data
  continuous aggregates
```

Supporting platform components are expected to include:

- Next.js product dashboard
- Redis queue and checkpoint infrastructure
- Docker sandboxing
- tool registry
- capability scopes
- OpenTelemetry
- immutable product audit trail
- prompt/model/policy registries
- prompt playground
- trace viewer
- evaluation gates
- AI CI/CD
- canary and rollback
- HITL queue/dispute/feedback UX
- continuous learning and drift detection

This target architecture must be built progressively, not as a rewrite.

---

# 4. THE 0–20 PHASE LIFECYCLE

The design document uses a lifecycle numbered `0` through `20`. That is 21 numbered phases; preserve the document's numbering rather than renumbering it.

Each phase is treated as a **capability contract**, not a promise to create a fixed directory structure.

## Phase 0 — Cognitive Design

### Target

Establish:

- why the system exists
- trigger
- output
- autonomy boundaries
- HITL boundaries
- expected failure modes

### Current state

**Strongly established.**

The current Genesis/SPEC foundation already encodes correctness-first, HITL, evidence grounding, fail-closed publication, reliability, security, cost and auditability.

### Future work

Maintain these as cross-cutting invariants. Do not add features that weaken them.

### Green gate

Every later capability can be traced to a requirement, decision, invariant, measurable operational need, or explicit target architecture item that has been reconciled into Genesis.

---

## Phase 1 — System Architecture

### Target

A modular-monolith architecture with clear inward-only dependency boundaries and explicit architectural decisions.

### Current state

**Foundation exists.**

Core abstractions and strong domain separation already exist, but the full target 23-module production surface is not assumed to exist merely because it appears in the architecture study.

### Future work

Incrementally align the repository boundaries with the target while preserving existing working abstractions.

Do not perform a cosmetic directory rewrite.

### Green gate

Architecture/dependency checks show that the system respects the approved dependency direction and every new outer component depends on stable inner contracts.

---

## Phase 2 — Frontend Engineering

### Target

A Next.js product dashboard covering:

- repository/review overview
- review status
- findings
- evidence
- HITL queue
- trace viewer
- economics
- operational state
- feedback

### Current state

**Incomplete relative to the final product.**

The backend already exposes enough durable review/telemetry truth to support this evolution.

### Future work

Build the UI around backend contracts, not around duplicated business logic.

The frontend must remain read-only unless an action is authorized through the backend.

### Green gate

A user can follow a real review end-to-end through the UI, inspect evidence, see status and costs, and safely perform authorized HITL actions.

---

## Phase 3 — Backend & API

### Target

FastAPI-based ingress and product APIs.

The webhook should:

1. validate HMAC
2. enforce delivery idempotency
3. persist required intake state
4. enqueue asynchronously
5. return quickly

### Current state

**Strongly implemented for the current V1 contract.**

### Future work

Expand the product API surface for:

- reviews
- HITL
- queue status
- economics
- traces
- feedback
- auth/RBAC

Do not move review logic into the HTTP layer.

### Green gate

Invalid webhook input cannot reach queue/model/retrieval/publication paths. Product API actions are authenticated, authorized and auditable.

---

## Phase 4 — Workflow Orchestration

### Target

LangGraph-based graph orchestration with parallel specialist fan-out and recoverable checkpointing.

### Current state

**Implemented and validated.**

### Future work

Move the checkpoint implementation toward Redis in a later production infrastructure task while keeping the workflow engine abstracted.

Never couple business logic directly to a specific orchestration engine when the existing abstraction can avoid it.

### Green gate

Worker interruption at workflow boundaries results in safe resume semantics without duplicate publication, lost Review Truth, or broken budget accounting.

---

## Phase 5 — LLM & Reasoning

### Target

- model routing
- structured outputs
- versioned prompts
- confidence
- rationale
- evidence grounding
- calibrated severity

### Current state

**Strongly implemented.**

### Future work

Formalize production registries for:

- prompts
- model configurations
- policies
- evaluation versions

Every review should be attributable to the exact versions used.

### Green gate

A production finding is reproducible to model/prompt/policy/retrieval configuration metadata, and malformed or unsafe model output remains fail-closed.

---

## Phase 6 — Memory Architecture

### Target

Hybrid repository retrieval backed by the production data spine:

- vector retrieval
- full-text retrieval
- repository/path filtering
- freshness tracking
- deterministic evidence mapping

The target production memory lane uses Tiger Cloud with pgvector/pgvectorscale/DiskANN plus full-text search.

### Current state

**Core retrieval abstraction is implemented and validated; production Tiger-backed memory is not yet the current V1 implementation.**

### Future work

Migrate behind the existing retrieval contract.

Do not rewrite evidence validation or specialist contracts merely to change the persistence backend.

### Green gate

Measured retrieval quality, freshness, latency and failure behavior demonstrate that the production backend satisfies the retrieval contract.

---

## Phase 7 — Tooling & Sandboxing

### Target

- tool registry
- capability scopes
- explicit agent permissions
- sandboxed execution
- Docker isolation

### Current state

**Not yet fully implemented as a production execution layer.**

### Future work

Introduce capabilities such as:

```text
READ_REPOSITORY
SEARCH_REPOSITORY
RUN_STATIC_ANALYZER
RUN_TESTS
READ_METADATA
```

Do not expose unrestricted host shell access to an agent.

### Green gate

Unauthorized capability attempts are denied and audited. Sandbox escape and host-access tests remain mandatory.

---

## Phase 8 — Multi-Agent Systems

### Target

Four independently attributable specialist roles plus deterministic aggregation.

### Current state

**Implemented and validated.**

### Future work

Extend specialization with:

- prompt registry
- specialist-specific evaluation
- specialist tool scopes
- model routing
- per-agent economics
- specialist drift metrics

### Green gate

A specialist may fail or degrade without the system silently treating it as successful coverage.

---

## Phase 9 — Evaluation

### Target

- golden dataset
- development split
- holdout split
- regression gates
- severity calibration
- quality metrics
- promotion protection

### Current state

**Strongly implemented.**

### Future work

Make evaluation a mandatory deployment gate for any change to:

- model
- prompt
- policy
- retrieval
- embeddings
- specialist logic

### Green gate

Candidate changes cannot advance when defined quality, security or cost gates fail.

---

## Phase 10 — Observability & Tracing

### Target

Distributed traces plus a durable product event spine carrying:

- span start/end
- LLM calls
- tool calls
- decisions
- escalation
- cost
- latency
- confidence
- outcome

### Current state

**Strong event/audit foundation already exists.**

### Future work

Integrate OpenTelemetry without replacing the product audit trail.

Treat:

```text
OTel = distributed technical tracing
Audit/Event Spine = product truth and evidence
```

### Green gate

A single review identifier can reconstruct the full execution path, with sensitive content appropriately redacted.

---

## Phase 11 — Security

### Target

- threat model
- prompt-injection defense
- RBAC
- least privilege
- secret masking
- immutable audit
- secure GitHub credentials

### Current state

**Strongly implemented at the core safety layer.**

### Future work

Productionize authorization, credential lifecycle, sandbox controls, frontend auth, GitHub App scopes and security monitoring.

### Green gate

Security tests cover hostile webhook data, malicious repository content, malicious PR content, retrieved content, model outputs and unauthorized tools.

---

## Phase 12 — Reliability

### Target

- retries
- bounded backoff
- timeouts
- circuit breakers
- dead-letter behavior
- idempotency
- restart recovery
- fault injection

### Current state

**Strongly implemented in the current V1 engine.**

### Future work

Extend fault testing to the new production infrastructure once introduced.

### Green gate

Failures degrade to safe held/degraded states rather than silently producing incorrect publication.

---

## Phase 13 — Infrastructure

### Target

Production deployment components including:

- FastAPI
- Redis/ARQ
- worker processes
- Tiger Cloud
- frontend deployment
- containers
- configuration and secrets
- health/readiness
- migration execution

### Current state

**Major production-evolution phase; current V1 deliberately avoids premature infrastructure expansion.**

### Future work

Introduce infrastructure only after the relevant Genesis decisions/tasks are approved.

### Green gate

The complete stack can be recreated from declared configuration/secrets with health/readiness and failure behavior proven.

---

## Phase 14 — Data Engineering

### Target

Production data ingestion and schema lifecycle, including:

- `code_chunks`
- `agent_events`
- review truth tables
- HITL tables
- continuous aggregates
- freshness/version metadata
- migrations

### Current state

**Core local data abstractions exist; the Tiger production data plane is not yet the V1 runtime.**

### Future work

Use staged migration:

1. infrastructure/schema
2. events
3. memory
4. dashboard/economics

Each stage must finish green before the next begins.

### Green gate

Schema, ingestion, freshness and query contracts are validated independently before workload migration.

---

## Phase 15 — Governance

### Target

Governed review operation with complete explainability/provenance:

- who/what acted
- why
- evidence
- model
- prompt
- policy
- decision
- publication
- human intervention

### Current state

**Core technical governance is already present through Review Truth, policy and audit.**

### Future work

Expose governance information operationally through APIs and UI.

### Green gate

Every externally visible decision is reconstructable and attributable.

---

## Phase 16 — Economics & Cost Control

### Target

- exact per-agent cost
- per-review cost
- token usage
- latency
- budget caps
- model routing guidance
- production economics dashboard

### Current state

**Strong accounting/control foundation exists.**

### Future work

Move aggregate reporting to the production event/data spine and expose it in the dashboard.

### Green gate

Budget controls act before unnecessary spend, while reported economics remain reconciled to raw event evidence.

---

## Phase 17 — Developer Experience

### Target

- prompt playground
- trace viewer
- replay
- finding/evidence inspection
- configuration visibility

### Current state

**Pending relative to target.**

### Future work

Build read-only replay/inspection first. Any mutation must use authenticated backend paths.

### Green gate

A developer can inspect why a review happened and replay the review context without mutating production state.

---

## Phase 18 — CI/CD for AI

### Target

Changes to prompts/models/policies should flow through:

```text
change
 ↓
unit/integration tests
 ↓
golden development evaluation
 ↓
holdout evaluation
 ↓
security/quality/cost gates
 ↓
promotion decision
 ↓
canary
 ↓
production
```

### Current state

**Large evaluation/promotion foundation already exists.**

### Future work

Integrate it into repository CI/CD and operational release workflows.

### Green gate

No unsafe model/prompt/policy change can bypass its configured evaluation/promotion path.

---

## Phase 19 — Human-in-the-Loop

### Target

- approval queue
- escalation
- assignment
- dispute
- feedback
- durable decisions

### Current state

**Core workflow logic is implemented; full product UX remains.**

### Future work

Build the production HITL experience over the existing Review Truth and policy contracts.

### Green gate

Every human action is authorized, versioned, durable and auditable.

---

## Phase 20 — Continuous Learning

### Target

```text
production reviews
 ↓
feedback
 ↓
drift detection
 ↓
candidate change
 ↓
golden + holdout evaluation
 ↓
promotion gate
 ↓
canary
 ↓
rollback/promotion
```

### Current state

**Core feedback, drift and promotion mechanisms already exist.**

### Future work

Connect those mechanisms to production telemetry and the AI CI/CD lifecycle.

### Green gate

Feedback cannot directly rewrite policy/model behavior without sufficient evidence and a successful evaluation/promotion path.

---

# 5. PRODUCTION EVOLUTION WAVES

The phases are numbered for the architecture lifecycle, but implementation must follow dependencies.

The project should evolve in controlled waves:

## Wave 0 — Genesis / target reconciliation

Purpose:

- reconcile the 0–20 target architecture with current SPEC/PLAN/decisions
- explicitly record conflicts
- determine which production capabilities require Genesis changes
- create bounded tasks only after reconciliation

No code migration should begin until the relevant Genesis state is explicit.

## Wave 1 — Production data + asynchronous infrastructure foundation

High-level target:

- production queue boundary
- Redis/ARQ path
- production workflow checkpoint boundary
- Tiger data-plane foundation
- events/memory/truth migration seams

This wave must preserve existing contracts while introducing production adapters.

## Wave 2 — Production memory + event spine + economics

High-level target:

- Tiger-backed memory
- hybrid retrieval
- event ingestion
- OpenTelemetry linkage
- continuous aggregates
- production cost reporting

## Wave 3 — Tooling, capabilities, sandbox and hardening

High-level target:

- tool registry
- capability scopes
- sandbox
- production RBAC
- security hardening
- fault injection against external infrastructure

## Wave 4 — Product UI, governance and HITL

High-level target:

- Next.js dashboard
- review UX
- evidence UX
- HITL queue
- dispute/feedback
- trace viewer
- economics
- governance views

## Wave 5 — AI CI/CD and continuous learning

High-level target:

- prompt/model/policy registry
- automated evaluation gates
- canary release path
- promotion/rollback
- drift monitoring
- safe feedback loop

**Detailed implementation of each wave must be designed and approved one wave at a time. Do not preemptively create implementation tasks for all waves.**

---

# 6. NON-NEGOTIABLE DEVELOPMENT RULES FOR GEMINI

Gemini is the implementation agent. It is not the authority for project governance.

## Rule 1 — Genesis first

Before changing code:

```text
genesis workflow status .
genesis status .
genesis spec status .
genesis plan status .
genesis brief .
genesis brief . --stage implement
git status
git rev-parse HEAD
git rev-parse origin/agent-v1
```

If an active task is not present, do not invent one.

## Rule 2 — Inspect before editing

Always inspect:

- relevant modules
- tests
- existing abstractions
- Genesis decisions/invariants
- current task scope

Prefer adapting a working abstraction over creating a parallel implementation.

## Rule 3 — One bounded task at a time

Every implementation task must state:

- outcome
- scope
- files/modules likely to change
- linked requirements
- linked decisions/invariants
- explicit non-goals
- executable tests
- gate
- rollback/safety considerations

## Rule 4 — Never silently redesign

If implementation reveals that the target requires a different architecture than the active task permits:

**STOP.**

Report:

```text
STATE
EVIDENCE
CONFLICT
BLOCKER
NEXT ACTION
```

Do not silently update the architecture.

## Rule 5 — Preserve validated behavior

Existing passing behavior is evidence.

Never:

- delete tests merely to make a change pass
- weaken security checks
- weaken fail-closed paths
- bypass evidence validation
- bypass current-SHA checks
- bypass publication idempotency
- remove auditability
- turn degraded specialist outcomes into success
- allow content to become trusted merely because an LLM produced it

## Rule 6 — New infrastructure goes behind contracts

Prefer seams such as:

```text
DurableJobQueue
WorkflowEngine
CodeMemoryStore / retrieval interface
Event/Audit interface
GitHub client interface
LLM/provider interface
```

When changing an implementation, keep the stable contract unless the task explicitly changes the contract.

## Rule 7 — Tests are part of the implementation

A code change is incomplete until the appropriate tests and gates are green.

Use:

```text
focused test
full regression
Genesis gate
independent review
```

where required by task risk.

## Rule 8 — Real infrastructure requires real failure testing

A new Redis/Tiger/GitHub/provider integration is not considered production-ready because the happy path works.

Test:

- timeout
- outage
- retry
- duplicate
- restart
- stale state
- partial failure
- unauthorized access
- malformed data

## Rule 9 — No secret leakage

Do not put credentials, tokens, connection strings or private payloads into:

- source
- tests
- logs
- audit events
- prompts
- findings
- screenshots
- commits

## Rule 10 — No autonomous scope expansion

Do not add unrelated dependencies, databases, services, dashboards, abstractions or refactors merely because they may be useful later.

Use the active task boundary.

---

# 7. REQUIRED END-OF-TASK PROTOCOL

When an implementation task is complete, Gemini must report:

```text
IMPLEMENTATION SUMMARY

CHANGED FILES

WHY EACH CHANGE WAS NECESSARY

TESTS RUN

FOCUSED GATE

FULL REGRESSION

GENESIS GATE

INDEPENDENT REVIEW

GENESIS TASK STATE

GIT STATUS

git diff --check

git diff --stat

DEVIATIONS FROM TASK

OPEN RISKS

EXACT NEXT ACTION
```

The report must distinguish:

- what was proposed
- what was actually implemented
- what is committed
- what is on GitHub

Do not describe unimplemented future work as complete.

---

# 8. COMMIT / PUSH POLICY

After the task has passed its required gates and approval:

1. complete the Genesis task
2. checkpoint Genesis
3. commit the implementation
4. push the commit to `agent-v1`
5. verify the pushed commit on GitHub
6. report exact SHA

Never create the impression that code is safely integrated merely because local tests pass.

---

# 9. BRANCH POLICY

### `main`

Validated baseline / production-reference V1.

Do not directly evolve main during the production transition unless explicitly authorized.

### `agent-v1`

Production evolution branch.

All 20-phase development should initially land here.

### Future release branches

Create only when the release/release-management task explicitly calls for them.

---

# 10. CURRENT ARCHITECTURAL MIGRATION PRINCIPLE

The most important engineering strategy for the transition is:

> **Preserve domain logic; evolve infrastructure through adapters.**

Examples:

```text
Current queue implementation
       ↓
DurableJobQueue contract
       ↓
production Redis/ARQ adapter
```

```text
Current code memory implementation
       ↓
retrieval / memory contract
       ↓
production Tiger adapter
```

```text
Current audit/event implementation
       ↓
event contract
       ↓
production Timescale/Tiger sink
       +
OTel tracing
```

```text
Current orchestration
       ↓
WorkflowEngine contract
       ↓
LangGraph + Redis checkpoint implementation
```

The purpose is to minimize migration risk and preserve the evidence already earned.

---

# 11. WHAT NOT TO DO

Do not:

- rewrite the entire repository into the 23-module tree just for visual conformity
- replace SQLite with Redis/Tiger in one giant commit
- introduce multiple production services without a measured need
- adopt Temporal just because it is more scalable
- replace proven retrieval logic without benchmark evidence
- build a UI that duplicates backend business logic
- introduce autonomous code modification
- auto-merge pull requests
- allow an LLM output to bypass deterministic validation
- let feedback directly alter production policy
- skip holdout evaluation
- skip security review for new tool/sandbox capabilities
- call a feature “production ready” because the happy path works

The target architecture is a destination. The migration path must remain evidence-driven.

---

# 12. DEFINITION OF PRODUCTION-READY

The project is not considered fully production-ready until the production stack has evidence for all of the following categories.

## Ingress

- authentic webhook verification
- delivery deduplication
- replay protection
- fast acknowledgement
- safe invalid-input handling

## Queue

- durable job semantics
- retry/backoff
- dead-letter handling
- cancellation
- restart recovery
- concurrency controls

## Orchestration

- parallel specialist execution
- bounded timeouts
- recoverable checkpoints
- deterministic join semantics
- explicit degradation

## Reasoning

- structured outputs
- evidence grounding
- confidence
- calibrated severity
- reproducible model/prompt/policy metadata

## Memory

- repository-aware indexing
- freshness
- hybrid retrieval
- retrieval-quality evidence
- safe degradation

## Security

- prompt-injection isolation
- secret protection
- credential segregation
- authorization
- least privilege
- sandbox controls

## Reliability

- provider fault injection
- database fault behavior
- queue fault behavior
- GitHub fault behavior
- worker restart behavior
- stale-SHA behavior

## Observability

- distributed tracing
- durable product events
- audit
- metrics
- alerts
- redaction
- reconstruction by review ID

## Evaluation

- golden development set
- holdout set
- regression gates
- severity calibration
- critical finding evaluation
- cost evaluation
- deployment gating

## HITL

- approval
- escalation
- dispute
- assignment
- feedback
- durable decision history

## Economics

- exact cost attribution
- budget enforcement
- aggregate reporting
- cost-aware routing

## CI/CD for AI

- prompt versioning
- model versioning
- policy versioning
- automated evaluation
- canary
- rollback

## Learning

- feedback thresholds
- drift detection
- safe candidate generation
- evaluation before promotion
- rollback

## Product

- dashboard
- finding/evidence UX
- HITL UX
- trace viewer
- economics
- developer replay

---

# 13. THE MASTER GEMINI INSTRUCTION

When starting work from this document, Gemini should use the following operating instruction.

```text
You are the implementation agent for PR-Review-Agent.

Repository:
https://github.com/Ayush-Kant/PR-Review-Agent

Working branch:
agent-v1

Read this file first:
PRODUCTION-20-PHASE-GEMINI-OPERATING-CONTRACT.md

Then read:
.genesis/KICKOFF.md
.genesis/project.json
.genesis/PLAN.md
SPEC.md

Genesis is the authoritative governance system.
The repository is the source of implementation truth.
The 20-phase architecture document is the long-term target architecture.
This operating contract explains how to move toward that target safely.

Current validated baseline:
4d59caa823b9de38e5766ca0885798a1b7bd4c54

Do not assume that a target phase is immediately implementable.
First reconcile it with Genesis.

Before every implementation task:
1. run Genesis status/brief commands
2. inspect the active task
3. inspect relevant decisions/invariants
4. inspect current implementation
5. inspect current tests
6. identify the smallest bounded change

Do not create implementation work without an approved active Genesis task.
Do not silently create new requirements.
Do not silently redesign the architecture.
Do not rewrite working modules unnecessarily.
Do not weaken existing safety behavior.

When a production component replaces a local/test component, prefer an adapter behind the existing interface.

For every task:
- state the exact outcome
- state scope
- state non-goals
- link requirements
- link decisions/invariants
- implement the smallest correct change
- add/update tests
- run focused gate
- run full regression when required
- run Genesis gate
- request/perform independent review when risk requires it
- checkpoint Genesis before stopping
- commit/push only after approval

If you discover a conflict:
STOP.
Report state, evidence, conflict, blocker and next action.
Do not silently resolve the architecture yourself.

At task completion report:
IMPLEMENTATION SUMMARY
CHANGED FILES
TESTS
GATES
REVIEW
GENESIS STATE
GIT STATE
DEVIATIONS
RISKS
EXACT NEXT ACTION

The system must always prefer a slower safe/degraded outcome over an incorrect or unauthorized publication.

Do not claim production readiness until the relevant behavior has been tested against real dependencies and failure conditions.
```

---

# 14. HOW THIS DOCUMENT SHOULD EVOLVE

This document should remain a stable operating contract.

Do not turn it into a running task log.

When architecture or governance changes materially:

- create/update the relevant Genesis decision
- update SPEC/PLAN where required
- update this contract only to reflect the approved change
- do not use this file to bypass Genesis

Task-level detail belongs in Genesis tasks and task evidence.

Wave-specific implementation plans will be created separately, **one wave at a time**, after the corresponding Genesis reconciliation.

---

# 15. FINAL PRINCIPLE

The objective is not to maximize the number of technologies in the stack.

The objective is to turn the currently validated review engine into a production system where:

```text
Every important action is bounded.
Every important decision is explainable.
Every important state transition is durable.
Every external action is authorized.
Every model change is evaluated.
Every failure has a defined safe outcome.
Every production claim has evidence.
```

That is the standard for the `agent-v1` transition.
