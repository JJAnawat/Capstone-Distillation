# Teacher throughput

Variant B (thinking on), no_robots prompts. Output tokens per second over a whole generate run, including the slow tail. `base_c256` is the pilot run.

| setting | samples | tok/s | minutes | vs pilot |
|---|---:|---:|---:|---:|
| base_c256 (pilot) | 800 | 1,802 | 13.7 | 1.00x |
| mtp_c256 | 800 | 2,014 | 11.7 | 1.12x |
