# DeepEval evaluation plan

This plan is intentionally grounded in the project’s actual repository structure and content. The workflow already has deterministic retrieval evaluation in `rag_eval.py`, but it is not yet an LLM-as-judge layer.

## Current state

- `rag_eval.py` evaluates retrieval quality using Hit@k against expected sections.
- There is no end-to-end DeepEval judge flow yet.
- A proper judge layer should start from a curated golden dataset, then add generation scoring on top of the retrieval baseline.

## Why a golden dataset is required

A golden dataset gives the evaluation layer a stable truth set:

- which query belongs to which portfolio section
- what the expected answer semantics are
- which retrieval chunks are relevant
- which output patterns should pass or fail

Without this, LLM-as-judge scoring becomes noisy and difficult to interpret.

## Golden dataset design

The dataset in `eval/golden_dataset.json` contains eight example queries mapped to the expected portfolio sections and expected keyword sets. These entries are intentionally fabricated but grounded in the real content structure that the project already uses.

## Evaluation metrics

Use these metrics in order:

1. Retrieval Hit@k
   - Did the expected section appear in the top-k results?
2. Section precision
   - Was the answer grounded in the expected portfolio area?
3. Keyword coverage
   - Does the answer cover the expected semantic terms?
4. Hallucination risk
   - Does the answer invent facts not present in the golden answer or source section?
5. Answer relevancy
   - Is the response aligned to the user’s actual intent?

## Recommended reporting format

A final report should include:

- total examples
- retrieval pass rate
- mean answer relevance
- mean grounding score
- failure examples with query, retrieved sections, and reason

## Suggested next step

Once `deepeval` is installed, the runner in `eval/run_deepeval.py` can be extended to call the exact DeepEval judge metrics and emit a final markdown report from the same dataset.
