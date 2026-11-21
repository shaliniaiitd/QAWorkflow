# QA Workflow — RAG + MCP

A small, real RAG pipeline: portfolio site (shaliniaiitd.github.io)
content -> generated user stories -> chunked, embedded, and indexed in a
vector DB with section metadata -> retrieval you can inspect and evaluate
-> all of it also exposed as MCP tools. 


## Pipeline, in order

```
data/portfolio_content.json     -- extracted from website         |
        v
scripts/generate_user_stories.py   --> user_stories/<section>/story_N.md
        |                               (LLM-generated, grounded in real content,
        |                                YAML frontmatter: id, section, source)
        v
seed_vector_db.py                  --> reads user_stories/**/*.md dynamically,
        |                               chunks each (src/utils/chunking.py),
        |                               embeds + upserts (src/utils/vector_store.py)
        v
   Chroma vector DB (chroma_db/, local, persistent)
        |
        +--> check_retrieval.py    (inspect what gets retrieved, with scores)
        +--> rag_eval.py           (Hit@k across a fixed test set)
        +--> src/workflow.py's retrieve_memory node (feeds analyze_story)
```


## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:0.5b      # chat model (src/workflow.py)
ollama pull nomic-embed-text        # embedding model (src/utils/vector_store.py)
```

## Running tests

```bash
python -m pytest -q
```

The default suite excludes generated scripts under `tests/phase1/` and the
Phase 2 live-site experiments under `tests/phase2/`. Generated scripts are run
artifacts, and Phase 2 checks depend on current third-party site behavior; run
those explicitly when validating a target deployment.

## Running it, step by step

```bash
# 1. Generate stories from your real portfolio content
python scripts/generate_user_stories.py
python scripts/generate_user_stories.py --section projects --count 3   # one section, more stories

# 2. Seed (or re-seed -- it's an upsert, always safe) the vector DB
python seed_vector_db.py
python seed_vector_db.py --chunk-size 40 --overlap 8   # try different chunking

# 3. See what retrieval actually finds
python check_retrieval.py "large scale data validation experience"
python check_retrieval.py "generative AI certifications" --top-k 5
python check_retrieval.py   # runs a few built-in demo queries

# 4. Check retrieval quality with a real (if small) eval
python rag_eval.py
python rag_eval.py --top-k 5
```
![alt text](<Screenshot 2026-07-05 224008-1.png>)

## Testing the MCP server

```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```
Sample run
![alt text](image.png)
## Registering with Claude Desktop

```json
{
  "mcpServers": {
    "qa-workflow": {
      "command": "python",
      "args": ["/absolute/path/to/project/mcp_server.py"]
    }
  }
}
```

## Registering with Claude Code

```bash
claude mcp add qa-workflow -- python /absolute/path/to/project/mcp_server.py
```
Tested on claude code
run the full QA workflow for the user story: As a user, I want to update my email address so I can receive notifications.

## Running observability

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`, fill in your LangSmith key (optional — works without it)
3. Run the workflow at least once: `python -m src.workflow`
4. See local latency baselines: `python observability_report.py`
![alt text](image-1.png)
   Expected output: a table of node names with call count, mean/min/max/p95 latency, avg tokens
5. (Optional, needs .env keys) Open smith.langchain.com — under your project you'll see one trace per workflow run, expandable into each 
node call
![alt text](image-6.png)
## Running guardrails

Requires `ollama pull llama-guard3` first.

- Via MCP tool: `check_guardrails` (pass `user_story` and/or `bdd_cases`)
- Directly: `python -c "from src.utils.guardrails import screen_user_story; print(screen_user_story('your text'))"`
- Expected: `GuardrailResult(passed=True, reason='')` for clean input, `passed=False` with a reason string if blocked
- The workflow applies these automatically — a blocked user_story short-circuits to a "Blocked" report; invalid BDD output triggers up to 2 silent regeneration attempts before proceeding anyway

## Hook-based guardrails and trace redaction

The project uses hook-driven sanitization instead of scattering PII handling across workflow nodes.

- `src/utils/guardrails.py` contains `redact_pii()` for masking emails, phone-like values, and Aadhaar-like identifiers.
- `src/utils/observability.py` wires that sanitizer into the LangSmith `@traceable` decorator through `process_inputs` and `process_outputs`.
- persisted artifacts such as saved reports and pytest execution logs also pass through `redact_pii()` before writing to disk.

This keeps the workflow logic focused on orchestration while trace/export and persistence stay safe by construction.

## Reusable skills

The repository now also treats reusable generation patterns as skills instead of one-off prompts.

- `skills/README.md` summarizes the skill catalog and hook model.
- `skills/dom_grounding_prompt.md` contains the DOM-grounded Playwright generation skill.
- this skill is used by the Phase 2a workflow path in `src/qa_workflow.py` via `scan_target_page()` plus the grounded prompt builder.

The key idea is simple: when a pattern is reusable across multiple flows, keep it as a skill and let the workflow compose it rather than hard-coding prompt text directly in each node.

## DeepEval roadmap

The next evaluation layer is DeepEval for retrieval and LLM quality checks.

- RAG evaluation: retrieval relevance, answerability, grounding, and rank stability across story queries
- output evaluation: schema compliance, unsupported-selector detection, and factual grounding against the page or retrieved content
- regression checks: fail fast when generated scripts invent selector names or unsupported Playwright APIs

Planned artifacts:

- `eval/deepeval_eval_plan.md` for the evaluation strategy
- future eval scripts under `eval/` to score retrieval and LLM outputs consistently

## MyYesm GenAI evaluation pilot

`eval/myyesm/` contains a separate, offline-first pilot suite for evaluating
an assistant grounded in MyYesm course pages. It includes a curated source
snapshot, scenarios for Q&A, extraction, ambiguity, abstention, prompt
injection, fairness, and summarization, plus a trace-scoring runner that
produces per-run JSON/Markdown artifacts. The runner consumes system traces;
it does not call a model or LLM judge. See
[`eval/myyesm/README.md`](eval/myyesm/README.md) for the trace contract and
commands.

## Streamlit UI
```bash
streamlit run app.py
```
check at  http://localhost:8501

Run Workflow
![alt text](image-3.png)

Check Guardrails
- check input guardrail(regex)
- check output guardrail(schema)
- check with LlamaGuard (model based)
![alt text](image-5.png)

Observability
![alt text](image-4.png)

## Phase 1 & 2a: Multi-Agent Workflow (Supervisor + Swarm + HITL)

`src/qa_workflow.py` refactors the original linear pipeline into a hybrid
supervisor + swarm architecture: Supervisor (entry, guardrail, run metadata) →
TestCaseGenAgent → TestCaseReviewAgent (classifies smoke/regression/sanity/
exploratory) → ExecutionAgent (generates + runs a pytest-playwright script) →
HealingAgent (rule-based failure diagnosis) → ReportAgent, with three
human-in-the-loop approval gates (pre-execution, pre-healing, pre-publish).
Full design rationale, diagrams, and phasing: see `QAWorkflow_Design_doc.md`.

The workflow also now distinguishes between two reusable patterns:

- hook patterns: trace export sanitization and persistence redaction
- prompt skills: DOM-grounded generation and other reusable LLM prompt contracts

Scripts and results are kept **separated by phase** (`tests/phase1/`,
`tests/phase2/`, `outputs/test_results/phase1/`, `outputs/test_results/phase2/`)
so the same story's before/after generations sit side by side, directly
comparable — see design doc Section 9.1.2 for the actual comparison.

### Batch mode — CI/CD-style test generation

Classifies and generates (or reuses cached) pytest-playwright scripts for
**all** user stories under `user_stories/`, without interactive HITL —
this is the "test generation" stage of a pipeline, not a full execute/heal
run.

```bash
# Process one story using Phase 1 ungrounded codegen
python -m src.qa_workflow --story projects_1

# Process just one story by id (matches the `id:` field in its frontmatter)
python -m src.qa_workflow --story projects_1

# Regenerate user stories from data/portfolio_content.json + re-seed the
# vector DB first, then batch-process
python -m src.qa_workflow --seed

# Phase 2a: scan the real target page first, then use DOM-grounded codegen
# instead of letting the LLM guess selectors/nav text/class names
python -m src.qa_workflow --phase2
python -m src.qa_workflow --phase2 --story projects_1
```

Running `python -m src.qa_workflow` without batch flags starts the interactive
single-story demo with the three human approval gates. Use `--story`, `--seed`,
or `--phase2` to select batch mode.

**Caching rule:** smoke and regression test types reuse an existing script
(`tests/<phase>/test_<story_id>.py`) if one's already been generated for that
story; sanity and exploratory always regenerate fresh. All types are saved
locally regardless, for audit purposes. A cached file is only reused if it
actually has real content — an empty/broken file is treated as a cache-miss
and regenerated. Ends with a summary table (story id, type, status, script
path, cached or not, pass/fail).

### Demo mode — full interactive pipeline, one story

Runs the complete loop for a single story, including test execution via
pytest, rule-based healing on failure, and all three HITL approval gates.
Demo mode currently always uses Phase 1's ungrounded codegen.

```bash
# Interactive (prompts Y/N at each of the 3 HITL gates)
python -m src.qa_workflow --demo

# Auto-approve every gate (useful for a hands-off end-to-end check)
python -m src.qa_workflow --demo --auto-approve

# Provide your own story instead of the default
python -m src.qa_workflow --demo --user-story "As a user, I want to..."
```

**Note:** `project_memory.json` feeds context into every generation prompt —
keep it describing the *actual* app under test. A mismatch here (discovered
during Phase 1 build: a leftover "Auth Portal" placeholder while stories were
actually about the portfolio site) silently biases every downstream artifact,
including HealingAgent's diagnoses, without raising any error.

