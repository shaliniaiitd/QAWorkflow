# Dataset readiness — release tier: NOT READY

_Supports a pass-rate claim with 95% confidence even when every case passes._

- Total cases: 12
- Smallest pass-rate drop from 90% this dataset can detect (80% power, 95% confidence): 53% points

| Check | Status | Detail |
|---|---|---|
| scenario size: ambiguous_query (low risk) | **gap** | 1 case(s), need 22 |
| scenario size: comparison (low risk) | **gap** | 1 case(s), need 22 |
| scenario size: direct_prompt_injection (high risk) | **gap** | 1 case(s), need 73 |
| scenario size: factual_qa (low risk) | **gap** | 3 case(s), need 22 |
| scenario size: fairness (medium risk) | **gap** | 1 case(s), need 35 |
| scenario size: indirect_prompt_injection (high risk) | **gap** | 1 case(s), need 73 |
| scenario size: sensitive_data_disclosure (high risk) | **gap** | 1 case(s), need 73 |
| scenario size: structured_extraction (low risk) | **gap** | 1 case(s), need 22 |
| scenario size: summarization (low risk) | **gap** | 1 case(s), need 22 |
| scenario size: unanswerable (medium risk) | **gap** | 1 case(s), need 35 |
| family coverage: edge | ok | 2 of 12 (17%), need >= 15% and >= 1 case(s) |
| family coverage: bias | **gap** | 1 of 12 (8%), need >= 10% and >= 1 case(s) |
| family coverage: adversarial | ok | 3 of 12 (25%), need >= 25% and >= 1 case(s) |
| corpus size for retrieval metrics | **gap** | 4 document(s) with top_k=2, need >= 20; 1 never expected in any case |
| held-out split | **gap** | 0 of 12 case(s) tagged split=holdout, need >= 30% |
| human review on high-risk cases | ok | all high-risk cases have manual_review |
