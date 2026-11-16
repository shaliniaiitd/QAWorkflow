# Skills and hook patterns

This project keeps reusable, repeatable behaviors in two layers:

1. Hook-based guardrails and observability
2. Reusable prompt skills for common generation tasks

## Current skill catalog

- `pii-redaction.skill.md` — redact personal identifiers before trace export or persistence
- `qa-test-generation-checklist.skill.md` — generation checklist for Playwright tests and BDD handoff
- `review-rubric-validation.skill.md` — validation rubric for output quality and guardrail checks
- `dom_grounding_prompt.md` — canonical DOM-grounding prompt pattern for live-page generation
- `dom-grounded-generation.prompt.md` — agent-facing invocation of the same practical pattern

## Hook-based guardrails and observability

The current implementation uses a hook pattern rather than hard-coding sanitization inside every node:

- `src/utils/guardrails.py` defines `redact_pii()` for masking personal data patterns.
- `src/utils/observability.py` wraps `@traceable` with `process_inputs` and `process_outputs` so LangSmith traces are sanitized before export.
- persisted artifacts such as saved reports and pytest logs call `redact_pii()` before writing to disk.

This keeps the rule in one place while preserving the workflow logic itself.

## Reusable prompt skills

The project now treats DOM-grounded generation as a reusable skill rather than an ad hoc one-off prompt fragment.

### `dom-grounded-test-generation`

Applies when a workflow needs to generate Playwright tests for a real, known page. It should:

- inspect the real DOM first
- ground selectors and nav text in the live page
- reject guessed selectors and nonexistent route assumptions
- use safe, copy-pasteable helper patterns for external-link status checks
- prefer Playwright APIs that are known to exist and are verifiable

The canonical prompt template is stored in `skills/dom_grounding_prompt.md`.

## Planned eval stack

The next layer is evaluation: DeepEval for RAG and LLM quality checks. The working plan now lives under `eval/deepeval_eval_plan.md`.

- Retrieval quality: answerability, chunk relevance, grounding, and recall for story and portfolio retrieval
- LLM output quality: schema adherence, prompt follow-through, and factual grounding against the scanned page or retrieved content
- regression guardrails: fail tests when the model invents selectors, unsupported APIs, or unverified content

This is the natural next step after the hook-based guardrails and DOM grounding skill pattern.
