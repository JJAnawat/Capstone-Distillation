# Teacher output analysis

Inputs: data/pilot/helpsteer2_open_off.jsonl, data/pilot/helpsteer2_open_on.jsonl, data/pilot/no_robots_open_off.jsonl, data/pilot/no_robots_open_on.jsonl, data/pilot/ultrafeedback_open_off.jsonl, data/pilot/ultrafeedback_open_on.jsonl

## helpsteer2 / mode=open

| metric | thinking on | thinking off |
|---|---:|---:|
| rows | 10 | 10 |
| reasoning tokens median | 1389 | 0 |
| reasoning tokens p90 | 4683 | 0 |
| reasoning tokens p95 | 5045 | 0 |
| reasoning tokens max | 5045 | 0 |
| answer tokens median | 628 | 678 |
| answer tokens p90 | 947 | 1097 |
| answer tokens p95 | 1844 | 2647 |
| answer tokens max | 1844 | 2647 |
| finish_reason == length | 0% | 0% |
| empty reasoning | 0% | 100% |
| unclosed think tag | 0% | 0% |
| total cost (USD) | 0.0065 | 0.0015 |

Matched prompts: 10. Median answer tokens: on 628, off 678. Median total completion tokens: on 2269, off 678.

## no_robots / mode=open

| metric | thinking on | thinking off |
|---|---:|---:|
| rows | 10 | 10 |
| reasoning tokens median | 1768 | 0 |
| reasoning tokens p90 | 3272 | 0 |
| reasoning tokens p95 | 7163 | 0 |
| reasoning tokens max | 7163 | 0 |
| answer tokens median | 229 | 234 |
| answer tokens p90 | 616 | 446 |
| answer tokens p95 | 1334 | 614 |
| answer tokens max | 1334 | 614 |
| finish_reason == length | 0% | 0% |
| empty reasoning | 0% | 100% |
| unclosed think tag | 0% | 0% |
| total cost (USD) | 0.0045 | 0.0007 |

Matched prompts: 10. Median answer tokens: on 229, off 234. Median total completion tokens: on 1982, off 234.

## ultrafeedback / mode=open

| metric | thinking on | thinking off |
|---|---:|---:|
| rows | 10 | 10 |
| reasoning tokens median | 1542 | 0 |
| reasoning tokens p90 | 3978 | 0 |
| reasoning tokens p95 | 7414 | 0 |
| reasoning tokens max | 7414 | 0 |
| answer tokens median | 162 | 183 |
| answer tokens p90 | 415 | 536 |
| answer tokens p95 | 976 | 1410 |
| answer tokens max | 976 | 1410 |
| finish_reason == length | 0% | 0% |
| empty reasoning | 0% | 100% |
| unclosed think tag | 0% | 0% |
| total cost (USD) | 0.0041 | 0.0007 |

Matched prompts: 10. Median answer tokens: on 162, off 183. Median total completion tokens: on 1555, off 183.
