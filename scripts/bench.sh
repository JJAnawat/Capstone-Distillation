#!/usr/bin/env bash
# Teacher throughput under different server settings, on the pilot's no_robots prompts (LIMIT prompts x N samples,
# variants A and B). The pilot itself is the baseline: base server at concurrency 256, same prompts and seeds.
# Run on the GPU VM after the pilot, inside tmux:
#   tmux new -s bench 'bash scripts/bench.sh 2>&1 | tee -ai logs/bench.log'
# Writes data/bench/<setting>_{A,B_raw}.jsonl and the comparison table reports/bench.md.
set -euo pipefail
LIMIT=${LIMIT:-100}
N=${N:-8}
OUT=${OUT:-data/bench}
PILOT_OUT=${PILOT_OUT:-data/pilot2}
source scripts/lib.sh
mkdir -p "$OUT"

run() {  # SETTING EXTRA_VLLM_ARGS CONCURRENCY
  echo "== $1 (concurrency $3, extra: ${2:-none})"
  start_server teacher "$2"
  distill generate --dataset no_robots --only open --limit "$LIMIT" --n "$N" --mode cot --thinking off \
    --concurrency "$3" --out "$OUT/$1_A.jsonl"
  distill generate --dataset no_robots --only open --limit "$LIMIT" --n "$N" --mode open --thinking on \
    --concurrency "$3" --out "$OUT/$1_B_raw.jsonl"
  stop_server
}
run base_c128 "" 128
run async_c256 "--async-scheduling" 256
run mtp_c256 "--speculative-config '{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}'" 256

python - "$OUT" "$PILOT_OUT" <<'EOF' | tee reports/bench.md
import json, pathlib, sys
bench, pilot = map(pathlib.Path, sys.argv[1:])
print("# Teacher throughput\n\nOutput tokens per second over a whole generate run (including the slow tail), "
      "no_robots prompts. `base_c256` is the pilot run.\n")
print("| setting | variant | samples | tok/s | minutes |\n|---|---|---:|---:|---:|")
files = [("base_c256", f) for f in sorted(pilot.glob("no_robots_open_*.jsonl.runs.jsonl"))]
files += [("_".join(f.name.split("_")[:2]), f) for f in sorted(bench.glob("*.runs.jsonl"))]  # e.g. async_c256
for setting, f in files:
    runs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = [r for r in runs if r["stage"] == "generate"]
    if runs:
        r = runs[-1]
        variant = "A (thinking off)" if r["mode"] == "cot" else "B (thinking on)"
        print(f"| {setting} | {variant} | {r['done']} | {r['tok_per_s']:,.0f} | {r['seconds'] / 60:,.1f} |")
EOF
