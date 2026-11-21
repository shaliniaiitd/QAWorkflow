# MyYesm GenAI evaluation pilot

This package is a first evaluation suite for an assistant grounded in the
public MyYesm course pages. The curated content snapshot is in
`documents.json`; `cases.json` covers factual Q&A, comparison, extraction,
ambiguity, abstention, direct and indirect prompt injection, sensitive-data
requests, fairness, and summarization.

The runner scores **traces produced by a system under test**. It does not
crawl the live site, call an LLM, or claim to be an LLM judge. The system
adapter should produce one JSON object per case in JSONL format:

```json
{"case_id":"ba_duration","answer":"The course is 85 hours (8-9 weeks).","retrieved_source_ids":["business_analyst"],"citations":["business_analyst"],"latency_ms":350,"input_tokens":120,"output_tokens":25,"cost_usd":0.0,"model_id":"local-model-v1","prompt_version":"qa-v1","retriever_version":"chroma-v1","tool_calls":[]}
```

`retrieved_source_ids` and `citations` use the `source_id` values in
`documents.json`. Record model, prompt, and retriever versions when available.
For `indirect_injection`, pass the case's `untrusted_context` to the system
under test and record its `untrusted_context_id` in `untrusted_context_ids`.
`human_review` may be added for cases marked for review:

```json
{"case_id":"direct_injection","answer":"...","retrieved_source_ids":["business_analyst"],"citations":["business_analyst"],"human_review":{"verdict":"pass","reviewer_id":"reviewer-1"}}
```

Run the evaluator:

```bash
python eval/myyesm/run_eval.py --run-file path/to/system-traces.jsonl
python eval/myyesm/run_eval.py --run-file path/to/system-traces.jsonl --run-id model-a-prompt-v2
```

Each run writes `trace.jsonl`, `report.json`, and `report.md` beneath
`eval/myyesm/runs/<run-id>/`. Saved trace strings are passed through the
repository's PII redactor. Use synthetic/test data and inspect artifacts
before sharing them.

The thresholds in `config.json` are **pilot gates**, not statistically
validated release criteria. The dataset has only twelve scenarios; expand and
stratify it before drawing broad conclusions. Fact and phrase matching are
deterministic proxies, not semantic faithfulness checks. Adversarial and
fairness cases require human review. Add a calibrated LLM judge only after
the system adapter and human-scored baseline are in place.

Run the offline unit tests:

```bash
python -m unittest discover -s eval/myyesm -p 'test_*.py'
```
