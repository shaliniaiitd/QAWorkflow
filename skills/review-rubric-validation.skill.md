# Skill: review rubric and validation rules

## Purpose

Validate generated output before it is accepted as production-quality work.

## Validation rubric

- schema validity: Gherkin or output structure must match the expected contract
- syntactic validity: generated Python should compile
- grounding: output must reflect actual observed page or retrieval facts
- focus: no invented selectors, routes, or unsupported APIs
- traceability: evidence should map back to retrieved content or a live page scan

## Suggested checks

- Ensure `Given/When/Then` structure is present for BDD output.
- Reject empty, truncated, or test-free scripts.
- Score retrieval against expected sections and query intent.
- Reject hallucinated content when the underlying evidence is missing.

## Reference

- [src/utils/guardrails.py](src/utils/guardrails.py)
- [eval/run_deepeval.py](eval/run_deepeval.py)
- [rag_eval.py](rag_eval.py)
