# Final metric rubric report

## Summary

- Total examples: 8
- Retrieval Hit@5: 62.50%
- Mean keyword coverage: 85.00%
- Mean answer relevancy: 100.00%
- Mean grounding score: 90.62%
- Mean total score: 79.75%
- Passed checks: 5/8

## Rubric

| Metric | Threshold | Score | Result |
|---|---:|---:|---|
| Retrieval Hit@5 | >= 0.80 | 62.50% | FAIL |
| Keyword coverage | >= 0.50 | 85.00% | PASS |
| Answer relevancy | >= 0.50 | 100.00% | PASS |
| Grounding score | >= 0.60 | 90.62% | PASS |
| Overall pass rate | >= 0.75 | 62.50% | FAIL |

## Per-example results

| Example | Section | Retrieval hit | Keyword coverage | Total score | Status |
|---|---|---:|---:|---:|---|
| education_1 | education | Yes | 60% | 86% | PASS |
| projects_1 | projects | No | 40% | 39% | FAIL |
| skills_1 | skills | Yes | 100% | 100% | PASS |
| frameworks_1 | frameworks_delivered | Yes | 80% | 93% | PASS |
| training_1 | training_certifications | Yes | 100% | 100% | PASS |
| recognition_1 | recognition | Yes | 100% | 100% | PASS |
| work_experience_1 | work_experience | No | 100% | 60% | FAIL |
| projects_2 | projects | No | 100% | 60% | FAIL |

## Notes

- This rubric is a practical LLM-judge proxy before a stricter DeepEval integration is added.
- The golden dataset is intentionally small and grounded in the portfolio sections already used by the project.
- Retrieval quality is the first gate; generation quality is scored only when the expected section is discoverable.
