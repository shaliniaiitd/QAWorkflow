# AGENTS

## Project overview

This repo is a QA workflow for a portfolio site and retrieval-driven test generation. The main workflow entry is `src/qa_workflow.py`; retrieval is backed by Chroma and story chunks from `user_stories/`; prompt and logic guardrails live under `src/utils/`.

Primary docs:
- [README.md](README.md)
- [QAWorkflow_Design_doc.md](QAWorkflow_Design_doc.md)
- [src/qa_workflow.py](src/qa_workflow.py)

## Working conventions

- Prefer project-root commands using the active virtual environment.
- Keep workflow artifacts separated by phase: `tests/phase1/`, `tests/phase2/`, and `outputs/test_results/<phase>/`.
- Do not silently overwrite a passing generated script or run log; preserve fresh run output and phase comparison artifacts.
- Treat PII redaction and trace sanitization as hook-based concerns, not ad hoc per-node logic.
- Treat DOM-grounded generation as a reusable skill pattern, not a one-off prompt snippet.

## Useful commands

```bash
# install deps
pip install -r requirements.txt

# run unit tests
python -m pytest -q

# run the batch workflow for one story
python -m src.qa_workflow --story projects_1

# run the DOM-grounded phase 2 flow
python -m src.qa_workflow --phase2 --story projects_1

# seed vector DB before retrieval evals
python seed_vector_db.py

# retrieval baseline eval
python rag_eval.py
```

## Environment and pitfalls

- Ollama is required for LLM + embedding calls.
- The project expects a local Chroma vector DB and seeded stories under `user_stories/`.
- Phase 2 generation depends on a real page scan and is more stable when grounded in the live DOM.
- Some generated Playwright tests are intentionally flaky or assumption-heavy if they are not grounded; prefer validating against the live site and the repo’s design docs before changing behavior.

## Evaluation guidance

- Retrieval and generation checks live under `eval/`.
- Start with the golden dataset and rubric report before extending LLM-as-judge checks.
- Keep evaluation output separate from production workflow artifacts so regressions remain visible.

## When modifying code

- Reuse existing abstractions in `src/utils/` before adding new custom helpers.
- Prefer small, explicit fixes that keep the original workflow contract intact.
- Update or add tests where behavior changes materially.
- Keep agent-generated documentation concise and link to source docs instead of duplicating them.
