# Dataset status (2026-11-08 17:35 UTC)

Final training rows (`*.final.jsonl` in `data/full` and `data/full_th`). Tokens are the student's (`student_tokens.total`: input + rationale + answer + 32 template tokens; the context is 2,048).

| set | rows | prompts | rows/prompt | prompts dropped | tokens | tokens median / p90 / max | > 1,024 tokens | median input / rationale / answer | median steps | complexity 1/2/3/4/5/unrated | rule+judge / judge | judge mean / ≥ 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| helpsteer2_open (English) | 2,322 | 2,322 | 1.00 | 124 | 1,277,501 | 440 / 1073 / 2041 | 11.8% | 27 / 151 / 194 | 5 | 1020 / 282 / 682 / 261 / 56 / 21 | 0 / 2,322 | 8.94 / 99% |
| logiqa_closed (English) | 4,664 | 1,946 | 2.40 | 554 | 3,026,075 | 619 / 922 / 2002 | 5.1% | 208 / 155 / 212 | 6 | 29 / 2035 / 2248 / 43 / 0 / 309 | 4,664 / 0 | 7.56 / 45% |
| no_robots_open (English) | 2,296 | 2,296 | 1.00 | 149 | 1,163,131 | 440 / 877 / 2020 | 6.3% | 90 / 133 / 102 | 4 | 1207 / 525 / 517 / 32 / 2 / 13 | 0 / 2,296 | 8.92 / 99% |
| ultrafeedback_closed (English) | 846 | 458 | 1.85 | 37 | 407,029 | 386 / 857 / 1766 | 6.3% | 192 / 108 / 41 | 5 | 324 / 455 / 50 / 0 / 0 / 17 | 708 / 138 | 8.08 / 84% |
| ultrafeedback_open (English) | 2,304 | 2,304 | 1.00 | 149 | 1,224,264 | 402 / 1154 / 2048 | 12.4% | 72 / 123 / 57 | 4 | 1256 / 529 / 353 / 131 / 24 / 11 | 0 / 2,304 | 8.87 / 98% |
| helpsteer2_th_open (Thai) | 1,161 | 1,161 | 1.00 | 110 | 516,825 | 351 / 823 / 2008 | 5.9% | 26 / 114 / 160 | 5 | 502 / 135 / 347 / 138 / 38 / 1 | 0 / 1,161 | 8.85 / 98% |
| logiqa_th_closed (Thai) | 2,736 | 1,011 | 2.71 | 189 | 1,161,839 | 404 / 564 / 1147 | 0.2% | 163 / 124 / 72 | 6 | 9 / 1317 / 1342 / 40 / 0 / 28 | 2,736 / 0 | 8.23 / 85% |
| no_robots_th_open (Thai) | 1,052 | 1,052 | 1.00 | 210 | 417,338 | 341 / 667 / 2042 | 2.9% | 35 / 103 / 128 | 4 | 566 / 217 / 244 / 17 / 8 / 0 | 0 / 1,052 | 8.73 / 94% |
| ultrafeedback_th_open (Thai) | 1,005 | 1,005 | 1.00 | 113 | 396,690 | 287 / 806 / 1883 | 5.4% | 27 / 89 / 67 | 4 | 532 / 211 / 172 / 71 / 17 / 2 | 0 / 1,005 | 8.82 / 96% |
| **English** | 12,432 | 9,326 | 1.33 | 1,013 | 7,098,000 | 512 / 959 / 2048 | 8.0% | 161 / 143 / 136 | 5 | 3836 / 3826 / 3850 / 467 / 82 / 371 | 5,372 / 7,060 | 8.35 / 77% |
| **Thai** | 5,954 | 4,229 | 1.41 | 622 | 2,492,692 | 382 / 644 / 2042 | 2.7% | 120 / 115 / 89 | 5 | 1609 / 1880 / 2105 / 266 / 63 / 31 | 2,736 / 3,218 | 8.54 / 91% |
| **all** | 18,386 | 13,555 | 1.36 | 1,635 | 9,590,692 | 453 / 891 / 2048 | 6.3% | 144 / 133 / 115 | 5 | 5445 / 5706 / 5955 / 733 / 145 / 402 | 8,108 / 10,278 | 8.41 / 82% |

English : Thai = 12,432 : 5,954 = 1 : 0.48 (target about 1 : 0.5).

## Why select dropped prompts

| set | dropped: best score < 7 | dropped: no correct answer |
|---|---:|---:|
| helpsteer2_open (English) | 124 | 0 |
| logiqa_closed (English) | 304 | 250 |
| no_robots_open (English) | 149 | 0 |
| ultrafeedback_closed (English) | 37 | 0 |
| ultrafeedback_open (English) | 149 | 0 |
| helpsteer2_th_open (Thai) | 110 | 0 |
| logiqa_th_closed (Thai) | 23 | 166 |
| no_robots_th_open (Thai) | 210 | 0 |
| ultrafeedback_th_open (Thai) | 113 | 0 |

## Closed sets

| set | prompts kept | prompts without a correct answer | correct rate |
|---|---:|---:|---:|
| logiqa_closed (English) | 1,946 | 250 | 78% |
| ultrafeedback_closed (English) | 458 | 0 | 65% |
| logiqa_th_closed (Thai) | 1,011 | 166 | 84% |
