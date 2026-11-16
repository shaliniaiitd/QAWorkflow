---
mode: agent
description: Run the project’s retrieval and generation evaluation workflow with a golden dataset and rubric.
---

# Eval quality

Use this when the task is to assess retrieval quality, response grounding, or generated output validity for the QA workflow.

## Rules
- Start from the golden dataset in [eval/golden_dataset.json](eval/golden_dataset.json).
- Prefer the project baseline in [rag_eval.py](rag_eval.py) before adding new heuristics.
- Keep evaluation output separate from workflow artifacts so result drift remains visible.
- Score retrieval, answer relevance, keyword coverage, and grounding separately.
- Call out hallucination risk and unsupported selectors as failure modes during generation review.

## Project references
- [eval/run_deepeval.py](eval/run_deepeval.py)
- [eval/deepeval_eval_plan.md](eval/deepeval_eval_plan.md)
- [AGENTS.md](AGENTS.md)

## Output contract
- Summarize the rubric with pass/fail metrics.
- Highlight the failing examples and the reason they failed.
- Recommend the smallest next fix for retrieval or generation quality.
