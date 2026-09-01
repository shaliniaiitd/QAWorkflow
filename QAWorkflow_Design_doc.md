# QAWorkflow — Multi-Agent AI Test Automation System
### PRD + Technical Design Doc
**Owner:** Shalini Agarwal | **Status:** Draft v2 | **Last updated:** 2026-09-01

---

## 1. Background

QAWorkflow began as a single LangGraph pipeline: retrieve context → analyze user story → generate BDD test cases → validate → review → report. It already has working input/output guardrails, observability (LangSmith-ready), a vector-memory layer (ChromaDB), and a checkpointer skeleton for multi-turn state.


 The goal: from *generating* test artifacts to *closing the loop* — generate → execute → self-heal → report — with human oversight at the points where automation shouldn't act unilaterally.

## 2. Goals

- Demonstrate a genuine multi-agent architecture (hybrid supervisor + swarm), not a single linear chain
- Close the loop from test generation to execution to reporting, without a human manually translating cases into scripts
- Build defensible self-healing that a human can audit and approve, not a silent black box
- Produce an architecture that generalizes: same core loop supports scripted execution now and scriptless/live execution later, without a redesign
- Produce documentation artifacts (this doc, diagrams) suitable for a GitHub portfolio and for a Test Architect-track conversation

## 3. Non-goals (explicitly out of scope for now)

- Live/scriptless execution (Phase 2)
- Jira/Xray integration, in either direction (Phase 3)
- Production-grade auth/security hardening of the approval mechanism (CLI stub is fine for now)
- Multi-user concurrent runs

## 4. Scope & Phasing

| Phase | Scope |
|---|---|
| **Phase 1** (this doc, build now) | Supervisor + swarm refactor, TestCaseGenAgent, TestCaseReviewAgent, ExecutionAgent (scripted), HealingAgent, ReportAgent, HITL at 3 gates, **CLI-based approval only** |
| **Phase 2** | Replace ExecutionAgent's script-generate-then-run with live/scriptless tool-calling execution (Playwright MCP, step-by-step against current DOM); redefine HealingAgent's trigger conditions for live failures; configurable execution mode (cache for smoke/regression, live for sanity/exploratory); **HITL wired into app.py; LangGraph Studio integration; mcp_server.py updated to support HITL over MCP** |
| **Phase 3** | Jira/Xray integration — pull user stories in as workflow input; publish final report back to Jira/Xray on completion |

## 5. Success Criteria

- A user story goes in; a reviewed, tagged, executed, (self-healed where applicable), human-approved report comes out — with zero manual script-writing
- Each of the 3 HITL gates correctly pauses the graph and resumes from the same state after approval (proven via checkpointer, not a restart)
- Architecture and routing decisions are documented well enough that they can be explained and defended.

---

## 6. Architecture

### 6.1 Agent Roster & Pattern per Edge

| Agent | Role | Edge type in/out |
|---|---|---|
| **Supervisor** | Entry point. Owns `screen_input` guardrail, run metadata (run_id, timestamps), initial dispatch | Supervisor-routed (in), hands off (out) |
| **TestCaseGenAgent** | Wraps USAnalyzer → BDDGen → schema-validate/retry loop | Swarm (peer handoff to/from Review) |
| **TestCaseReviewAgent** | Qualitative review (coverage/clarity) + assigns `test_type` tag (smoke/regression/sanity/exploratory) | Swarm — hands off directly back to TestCaseGenAgent on fail, or forward to Execution on pass |
| **ExecutionAgent** | Phase 1: generates a script from BDD case and runs it. Phase 2: live tool-calling execution | Swarm handoff to Healing (on failure) or Report (on pass) |
| **HealingAgent** | Diagnoses execution failures, proposes a fix, **requests HITL approval**, re-runs via Execution | Swarm; loops back to ExecutionAgent |
| **ReportAgent** | Aggregates results + healing actions into final report, **requests HITL approval before publishing** | Swarm → END |

**Why hybrid, not pure supervisor or pure swarm:** Supervisor is reserved for genuine dispatch decisions (which flow starts, holding shared run metadata, input validation). Every other transition is "this agent's own output determines the next agent" — a peer decision, not a routing decision — so it's modeled as swarm. Routing every hop through a central node would add cost (state round-trip, possibly an extra LLM call) for a decision that doesn't need central arbitration.

**How swarm handoffs are decided:** given the local model in use (see Section 12 — Risks), handoff decisions are **code-based, not LLM-tool-call-based**. Each agent's LLM call produces content (a BDD case, a review verdict, a healing proposal); a Python function inspects that output (pass/fail flags, schema validity) and decides the next hop — the model itself never emits a routing tool call (e.g. `transfer_to_bdd_gen`). This is still genuinely swarm — a peer handoff, no central dispatcher arbitrating — the decision mechanism is just deterministic rather than model-driven, which is the more reliable choice at this model scale. This mirrors the existing `validate_bdd` → `route_after_validate_bdd` pattern already in the codebase, extended to the new agents.

**Safety nets on swarm:** `recursion_limit` set at graph level; a `max_hops` counter carried in shared state as a second guard against runaway peer-handoff loops (e.g., Review ↔ Gen cycling indefinitely).

### 6.2 System Architecture Diagram

```mermaid
flowchart TD
    A[User Story Input] --> S[Supervisor<br/>screen_input guardrail, run metadata]
    S -->|blocked| BR[Blocked Report]
    S -->|clean| TCG[TestCaseGenAgent<br/>USAnalyzer + BDDGen + schema retry]
    TCG <-->|swarm handoff| TCR[TestCaseReviewAgent<br/>coverage/clarity + test_type tag]
    TCR -->|approved| EX[ExecutionAgent<br/>Phase1: script-gen + run]
    EX -->|pass| RPT[ReportAgent]
    EX -->|fail| HITL1{{HITL Gate 2:<br/>Approve execution run?}}
    HITL1 -.->|pre-run approval| EX
    EX -->|fail| HEAL[HealingAgent<br/>diagnose + propose fix]
    HEAL --> HITL2{{HITL Gate 1:<br/>Approve healing fix?}}
    HITL2 -->|approved| EX
    HITL2 -->|rejected| RPT
    RPT --> HITL3{{HITL Gate 3:<br/>Approve report publish?}}
    HITL3 -->|approved| DONE[Final Report Saved]
    HITL3 -->|rejected| REVISE[Back to Supervisor for revision]

    style S fill:#e1f0ff
    style HITL1 fill:#fff3cd
    style HITL2 fill:#fff3cd
    style HITL3 fill:#fff3cd
```

### 6.3 State Transition Diagram

Captures the actual node-level routing, including existing retry loops carried over from the current `workflow.py`.

```mermaid
stateDiagram-v2
    [*] --> load_static_memory
    load_static_memory --> screen_input
    screen_input --> blocked_report: input_blocked
    screen_input --> retrieve_memory: clean
    blocked_report --> save_report
    retrieve_memory --> analyze_story
    analyze_story --> write_dynamic_memory
    write_dynamic_memory --> generate_bdd
    generate_bdd --> validate_bdd
    validate_bdd --> generate_bdd: schema invalid, retries left
    validate_bdd --> review_bdd: valid OR retries exhausted
    review_bdd --> generate_bdd: reviewer rejects (swarm handoff)
    review_bdd --> execution_agent: reviewer approves + test_type tagged
    execution_agent --> healing_gate: execution fails
    execution_agent --> report_agent: execution passes
    healing_gate --> HITL_approve_fix
    HITL_approve_fix --> execution_agent: approved, re-run
    HITL_approve_fix --> report_agent: rejected, report as-is
    report_agent --> HITL_approve_publish
    HITL_approve_publish --> save_report: approved
    HITL_approve_publish --> supervisor_revise: rejected
    save_report --> [*]
```

### 6.4 HITL Sequence Diagram — Phase 1 (CLI)

All 3 gates use the same underlying mechanism: LangGraph's `interrupt()` inside a node, resumed via `Command(resume=...)` against the same `thread_id`, backed by the existing checkpointer (the graph already has `MemorySaver` wired for multi-turn runs — HITL reuses that, doesn't need a new mechanism). Since the workflow is run from the CLI in Phase 1, approval is surfaced on the CLI too — there's no separate UI yet for it to live in.

```mermaid
sequenceDiagram
    participant Graph as LangGraph Runtime
    participant Node as Agent Node (Gen/Exec/Report)
    participant CP as Checkpointer
    participant User as Human (CLI)

    Node->>Graph: interrupt(payload: what needs approval)
    Graph->>CP: persist state at interrupt point
    Graph-->>User: surface prompt (CLI stdout for Phase 1)
    User->>Graph: input (approve / reject / edit)
    Graph->>CP: load persisted state for thread_id
    Graph->>Node: Command(resume=user_decision)
    Node->>Node: branch on decision, continue or short-circuit
```

**The 3 gates, concretely:**

1. **Before ExecutionAgent runs tests** — pause after TestCaseReviewAgent approves, before any script is generated/run. Human confirms it's safe to execute against the target app.
2. **Before HealingAgent applies a fix** — pause after HealingAgent proposes a fix, before it patches and re-runs. This is the highest-value gate from an SDET standpoint: an auto-applied fix on a *regression* suite could be silently masking a real bug rather than a flaky selector — a human call is warranted, not automation.
3. **Before final report is published** — pause after ReportAgent assembles the report, before `save_report` (and, in Phase 3, before it's pushed to Jira/Xray). Human gets final sign-off before anything leaves the system.

---

## 7. State Schema (extension to existing `QAState`)

```python
class QAState(TypedDict, total=False):
    # ...existing fields (user_story, static_memory, dynamic_memory,
    # analysis, bdd_cases, review_notes, final_report, retrieved_context,
    # output_path, input_blocked, block_reason, bdd_valid, retry_count)

    # New — Phase 1
    test_type: str              # smoke | regression | sanity | exploratory
    execution_result: str       # pass | fail
    execution_log: str          # captured output/errors from script run
    healing_proposed: str       # description of proposed fix
    healing_applied: bool
    hitl_gate: str              # which gate is currently pending, if any
    hitl_decision: str          # approve | reject | edit
    run_id: str                 # stamped once by Supervisor
    hops: int                   # swarm loop safety counter
```

## 8. Test Case Bucketing (feeds Phase 2/3, designed now)

Assigned by TestCaseReviewAgent alongside its qualitative review pass — rule-based first, LLM as tiebreaker only when history is ambiguous:

| Type | Rule | Execution mode (Phase 2/3) |
|---|---|---|
| Smoke | Critical-path flow, run every build | Cache |
| Regression | Previously-covered flow, repeated to catch drift | Cache (cache-miss = signal, not just "heal and move on") |
| Sanity | Narrow, ad hoc, verifies a specific fix | Live |
| Exploratory | No fixed path, first-time coverage | Live (nothing to cache) |

---

## 9. Phase 2 Preview — Scriptless Execution, and What HITL Needs to Become

### 9.1 Scriptless / On-the-Fly Execution

ExecutionAgent's job changes from *generate script → run script* to *interpret each BDD step live against the current DOM* (Playwright MCP tool calls, no intermediate script file). Resilience shifts from "fix broken generated code after the fact" to "live interpretation adapts in the moment" — HealingAgent's trigger conditions get redefined here (cache-miss-on-replay vs. genuine live-execution failure are different failure modes). Configurable execution mode from Section 8 goes live in this phase.

### 9.2 HITL in app.py

Phase 1's CLI `input()` prompts are replaced with an approval UI surfaced through `app.py` (FastAPI backend) and the React/Vite frontend — the interrupt payload (what needs approving, at which gate) is returned to the frontend instead of printed to stdout, and the resume decision comes back as an API call instead of a keypress. The underlying `interrupt()`/`Command(resume=...)`/checkpointer mechanism from Section 6.4 doesn't change — only what triggers the resume changes.

### 9.3 LangGraph Studio

Worth being precise about what this is and isn't, since it's easy to conflate with LangSmith: **LangSmith is observability** — a passive trace viewer, including where a run paused at an interrupt, but not an interactive approve/resume UI. **LangGraph Studio is a separate tool** that gives a live, visual UI for a running graph — you can see a paused interrupt and resume it with a decision directly in the Studio UI, no custom frontend code needed for that interaction. It requires a `langgraph.json` manifest and running via `langgraph dev` — WealthDesk already has this set up; QAWorkflow does not yet. Adding Studio support in Phase 2 means:
- Add `langgraph.json` pointing at the compiled graph
- Confirm the graph runs under `langgraph dev`
- HITL gates become approvable/rejectable directly from the Studio UI, as an alternative to the app.py-based approval flow — useful for debugging/demoing without needing the full frontend running

This is additive, not a replacement for 9.2 — app.py is the "real" user-facing approval path; Studio is the "developer inspecting/demoing a specific run" path.

### 9.4 HITL over MCP (mcp_server.py)

This needs a structurally different approach from the other two, because MCP tool calls are stateless request/response — a tool invocation can't block indefinitely waiting on a human decision the way a CLI script or a long-lived app.py session can. The pattern:

- The existing "run workflow" MCP tool, when it hits an `interrupt()`, **returns** the interrupt payload + `thread_id` as its tool result, instead of blocking
- A **new, second MCP tool** (e.g. `resume_workflow`) is added, taking `thread_id` + the human's decision, and calls `Command(resume=...)` against the checkpointer to continue the run
- In MCP Inspector, a human operator would: call the run tool → see the interrupt payload returned → call `resume_workflow` with their decision → repeat until the run completes

This is a genuine addition to `mcp_server.py` (a new tool, not a tweak to the existing one), scoped for Phase 2 alongside app.py and Studio work — Phase 1 stays CLI-only, no MCP-level HITL yet.

## 10. Phase 3 Preview — Jira/Xray Integration

- **Inbound:** pull user stories from Jira as workflow input (replaces manual `user_story` string entry)
- **Outbound:** publish ReportAgent's final report back to Jira/Xray after HITL Gate 3 approval — field mapping TBD when this phase starts, informed by ReportAgent's finalized output shape from Phase 1

---

## 11. Open Assumptions (flag if wrong)

- HITL Phase 1 interaction = CLI `input()` prompt only; app.py, LangGraph Studio, and MCP-level HITL are all Phase 2
- Rejecting the report at Gate 3 routes back to Supervisor for revision rather than terminating the run
- `test_type` tagging logic starts rule-based; LLM tiebreaker only for ambiguous history

---

## 12. Risks

### 12.1 Model capacity (`qwen2.5-coder:0.5b`)

The local model backing every node is small (0.5B parameters), and that has concrete architectural consequences beyond "answers might be lower quality":

- **Tool-calling / structured-output reliability.** A 0.5B model does not reliably produce consistent structured output (a clean JSON schema, or a well-formed tool call) on the first attempt — the existing `validate_bdd` retry loop is already evidence of this in practice, since the model doesn't reliably emit valid Gherkin structure on the first try. This risk compounds if the model is asked to make a *routing* decision via tool-calling: a malformed handoff doesn't just produce one bad test case, it can send the graph somewhere structurally wrong. **Mitigation:** Section 6.1's code-based swarm handoff decision — the model produces content, code decides the next hop, never the reverse.
- **Reasoning depth for judgment-heavy nodes.** HealingAgent's job ("diagnose why this failed, propose a fix") is qualitatively harder than schema-shaped generation (writing a BDD case). A 0.5B model may produce shallow or generic diagnoses where a larger model would reason more concretely about the actual failure. **Mitigation:** extend the existing guardrail pattern — validate `healing_proposed` output shape and specificity before acting on it, same retry-on-failure treatment `validate_bdd` already gets. If output quality proves insufficient once built and tested, consider a mixed-model setup: keep `qwen2.5-coder:0.5b` for cheap, schema-shaped generation (BDD writing), but point HealingAgent's diagnosis step at a larger local model via Ollama (e.g. a 7B variant) — Ollama makes per-call model swaps cheap, so this is a config change, not a redesign.
- **Long-context degradation.** Nodes that combine static memory + dynamic memory + retrieved context + user story into one prompt (`analyze_story`, `generate_bdd`) risk quality drop-off as that combined prompt grows, more so at 0.5B than at larger scale. **Mitigation:** watch prompt length as new agents are added; keep each agent's prompt scoped to only the state fields it actually needs (already partially enforced via `PROMPT_FIELDS`).
- **Validate before building around it.** Before writing HealingAgent's retry/guardrail logic, worth manually running a handful of representative diagnosis prompts against the 0.5B model to see whether this is a real problem to design around or a hypothetical one — empirical check is cheaper than architecting defensively against an untested assumption.
