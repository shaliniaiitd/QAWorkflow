# Evaluation framework

Three layers, each answering a different question. Run them in this order.

| Layer | Question | Tool | Cost |
|---|---|---|---|
| 1. Deterministic checks | Did the system retrieve the right pages, cite them, include required facts, avoid forbidden claims, emit valid JSON, call the right tools? | `myyesm/run_eval.py` | free |
| 2. LLM judge | Is the answer faithful to its context, relevant, and does it meet the case's expected behavior? | `myyesm/judge_eval.py` (DeepEval) | judge tokens |
| 3. Human review | Is the judge right? | `myyesm/review_cli.py` | minutes |

Layer 3 is the ground truth. `judge_eval.py` reports judge-vs-human agreement and Cohen's kappa, and warns when
fewer than 30 cases are reviewed, because below that the numbers cannot be trusted.

## Suites

| Suite | System under test | Dataset |
|---|---|---|
| `myyesm/` RAG | Course assistant over `documents.json` | `cases.json` (12 cases) |
| `myyesm/` tools | Tool-calling agent with a stubbed `send_email` | `tool_cases.json` (7 cases) |
| `qaworkflow/` | QAWorkflow input guardrail and real Chroma retrieval | `guardrail_cases.json`, `golden_dataset.json` |

## Run it

```bash
python -m unittest discover -s eval/myyesm -p "test_*.py"
python -m unittest discover -s eval/qaworkflow -p "test_*.py"

# RAG: benchmark two prompt versions, then compare
python eval/myyesm/sut_adapter.py --prompt baseline --run-id baseline-002 --score
python eval/myyesm/sut_adapter.py --prompt hardened --run-id hardened-002 --score
python eval/myyesm/compare_runs.py baseline-002 hardened-002

# Tool calling
python eval/myyesm/sut_adapter.py --mode tools --prompt hardened --run-id tools-hardened-001 --score

# Judge, human review, calibration
python eval/myyesm/review_cli.py hardened-002 --reviewer <you> --all --apply
python eval/myyesm/judge_eval.py hardened-002

# Is the dataset big enough for the claim?
python eval/myyesm/dataset_requirements.py --tier release

# QAWorkflow guardrail and retrieval
python eval/qaworkflow/run_suite.py --suite all --run-id baseline-001
```

Every `run_eval.py` run records dataset, config, document, trace and evaluator hashes, so a result can be
reproduced and two runs can be checked for comparability.

## Metrics

| Metric | Source |
|---|---|
| Retrieval hit rate, context precision/recall, citation precision/recall | deterministic |
| Fact coverage, required and forbidden phrases, JSON valid / JSON match | deterministic |
| Tool selection accuracy, tool-call accuracy, forbidden tool calls | deterministic |
| Latency p50/p95, tokens, cost (only when prices are configured) | measured per case |
| Faithfulness, answer relevancy, expected-behavior | LLM judge |
| Judge-human agreement, kappa | calibration |
| Pass rates with 95% Wilson intervals, paired McNemar test between runs | `stats.py` |

## Dataset requirements

Four case families are required: core intents, edge cases (ambiguity, unanswerable), bias (fairness), and
adversarial (injection, private data, tool misuse). `dataset_requirements.py` checks size and coverage per tier.

The number that matters: with zero failures, certifying a pass rate of 90% at 95% confidence needs at least 35
cases per scenario (73 for 95%). With 12 cases the Wilson lower bound for a perfect run is only about 76%, and the
smallest pass-rate drop the dataset can detect is above 50 points. Treat pilot results as directional.

## Known limitations

- Prompts were edited while looking at the same cases they are scored on. Results on the dev set are optimistic.
  Tag some cases `"split": "holdout"`, freeze them, and never tune against them.
- Phrase matching is brittle. A correct refusal worded differently fails it. That is why the judge layer exists.
- The retrieval corpus is tiny, so Hit@k is not discriminative until distractor documents are added.
- `judge_eval.py` was written against DeepEval's custom-model interface and has not been run against your installed
  version. Run it on two or three cases first.
- `run_deepeval.py` and `metric_rubric_report.md` score golden answers against their own keywords. They are a rubric
  sketch, not a DeepEval run, and should not be cited as one.
