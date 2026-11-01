# Judge validation

HelpSteer2 validation prompts whose two responses got different human helpfulness ratings. Agreement: the judge's (expected) score orders the pair like the humans; a judge tie counts half. Cost: mean completion tokens per judge call, and wall-clock minutes.

| date | judge | thinking | prompt | pairs | agreement | judge ties | tokens/call | minutes |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2026-10-21 | google/gemma-4-31B-it-qat-w4a16-ct (vllm) | off | efc118cdc9224f8f | 20 | 60.0% | 0.0% | 65 | 0.2 |
| 2026-10-22 | google/gemma-4-31B-it-qat-w4a16-ct (vllm) | off | efc118cdc9224f8f | 252 | 71.0% | 0.0% | 66 | 2.1 |
| 2026-10-22 | google/gemma-4-31B-it-qat-w4a16-ct (vllm) | off | 4a273a481d5c5bd6 | 252 | 70.2% | 0.0% | 63 | 2.1 |
