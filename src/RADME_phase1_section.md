## Phase 1: Multi-Agent Workflow (Supervisor + Swarm + HITL)

`src/workflow_phase1.py` refactors the original linear pipeline into a hybrid
supervisor + swarm architecture: Supervisor (entry, guardrail, run metadata) →
TestCaseGenAgent → TestCaseReviewAgent (classifies smoke/regression/sanity/
exploratory) → ExecutionAgent (generates + runs a pytest-playwright script) →
HealingAgent (rule-based failure diagnosis) → ReportAgent, with three
human-in-the-loop approval gates (pre-execution, pre-healing, pre-publish).
Full design rationale, diagrams, and phasing: see `QAWorkflow_Design_doc.md`.

### Batch mode (default) — CI/CD-style test generation

Classifies and generates (or reuses cached) pytest-playwright scripts for
**all** user stories under `user_stories/`, without interactive HITL —
this is the "test generation" stage of a pipeline, not a full execute/heal
run.

```bash
# Process all user stories (default)
python -m src.workflow_phase1

# Process just one story by id (matches the `id:` field in its frontmatter)
python -m src.workflow_phase1 --story projects_1

# Regenerate user stories from data/portfolio_content.json + re-seed the
# vector DB first, then batch-process
python -m src.workflow_phase1 --seed
```

**Caching rule:** smoke and regression test types reuse an existing script
(`tests/test_<story_id>.py`) if one's already been generated for that story;
sanity and exploratory always regenerate fresh. All types are saved locally
regardless, for audit purposes. Ends with a summary table (story id, type,
status, script path, cached or not).

### Demo mode — full interactive pipeline, one story

Runs the complete loop for a single story, including test execution via
pytest, rule-based healing on failure, and all three HITL approval gates.

```bash
# Interactive (prompts Y/N at each of the 3 HITL gates)
python -m src.workflow_phase1 --demo

# Auto-approve every gate (useful for a hands-off end-to-end check)
python -m src.workflow_phase1 --demo --auto-approve

# Provide your own story instead of the default
python -m src.workflow_phase1 --demo --user-story "As a user, I want to..."
```

**Note:** `project_memory.json` feeds context into every generation prompt —
keep it describing the *actual* app under test. A mismatch here (discovered
during Phase 1 build: a leftover "Auth Portal" placeholder while stories were
actually about the portfolio site) silently biases every downstream artifact,
including HealingAgent's diagnoses, without raising any error.
