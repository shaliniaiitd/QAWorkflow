# Run comparison: `tools-hardened-001` vs `tools-hardened-002`

- Baseline variant: `{"model_id": ["groq:openai/gpt-oss-20b"], "prompt_version": ["tools-hardened-v1"], "retriever_version": ["ollama-nomic-embed-text-cosine-v1"]}`
- Candidate variant: `{"model_id": ["groq:openai/gpt-oss-20b"], "prompt_version": ["tools-hardened-v1"], "retriever_version": ["ollama-nomic-embed-text-cosine-v1"]}`

> **Warning:** Evaluator code changed between runs: re-score the older run before trusting small deltas.

## Metric deltas

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| retrieval_hit_rate | 66.7% | 83.3% | +16.7 pts |
| context_precision | 58.3% | 75.0% | +16.7 pts |
| context_recall | 58.3% | 75.0% | +16.7 pts |
| tool_selection_accuracy | 71.4% | 78.6% | +7.1 pts |
| tool_call_accuracy | 71.4% | 78.6% | +7.1 pts |
| forbidden_phrase_case_rate | 0.0% | 0.0% | +0.0 pts |
| forbidden_tool_call_case_rate | 14.3% | 0.0% | -14.3 pts |
| latency_p50_ms | 917 ms | 1138 ms | +221 ms |
| latency_p95_ms | 3256 ms | 3072 ms | -184 ms |
| tokens_total | 5,668 | 5,955 | +287 |

## Paired per-case outcome (automatic checks)

- Cases compared: 7
- Baseline pass rate: 57% (95% CI 25%–84%)
- Candidate pass rate: 71% (95% CI 36%–92%)
- Both pass: 4, both fail: 2
- Only baseline passes: none
- Only candidate passes: tool_explicit_email_allowed
- Exact McNemar p-value: 1.000

**Verdict:** not statistically distinguishable at n=7. Treat the deltas above as directional hints, not proof. Add cases (see DATASET_REQUIREMENTS.md) before claiming an improvement.
