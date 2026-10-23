#!/usr/bin/env bash
# Teacher throughput under different server settings, on the pilot's no_robots prompts (LIMIT prompts x N samples,
# variant B: thinking on). The pilot itself is the baseline: base server at concurrency 256, same prompts and seeds.
# Run on the GPU VM inside tmux:
#   tmux new -s bench 'bash scripts/bench.sh 2>&1 | tee -ai logs/bench.log'
# Writes data/bench/<setting>_B_raw.jsonl and the comparison table reports/bench.md. A setting whose server does not
# start, or whose run fails, is skipped; the table is written either way.
set -euo pipefail
LIMIT=${LIMIT:-100}
N=${N:-8}
OUT=${OUT:-data/bench}
PILOT_OUT=${PILOT_OUT:-data/pilot2}
source scripts/lib.sh
mkdir -p "$OUT"

run() {  # SETTING EXTRA_VLLM_ARGS CONCURRENCY. Called in a || list, where set -e is off: failures return explicitly.
  echo "== $1 (concurrency $3, extra: ${2:-none})"
  start_server teacher "$2" || { stop_server; return 1; }
  distill generate --dataset no_robots --only open --limit "$LIMIT" --n "$N" --mode open --thinking on \
    --concurrency "$3" --out "$OUT/$1_B_raw.jsonl" || { stop_server; return 1; }
  stop_server
}
skipped() { echo "== $1 skipped: its server or run failed (see logs/vllm_teacher.log)" >&2; }
run base_c128 "" 128 || skipped base_c128
run async_c256 "--async-scheduling" 256 || skipped async_c256
# vLLM maps "mtp" to the model's own multi-token-prediction head (qwen3_5_mtp for Qwen3.5).
run mtp_c256 "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'" 256 || skipped mtp_c256

python - "$OUT" "$PILOT_OUT" <<'EOF' | tee reports/bench.md
import json, pathlib, sys
bench, pilot = map(pathlib.Path, sys.argv[1:])

def biggest_run(path):  # the run that generated the most samples (a resumed run only does the rest)
    if not path.exists():
        return None
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = [r for r in runs if r["stage"] == "generate" and r["done"]]
    return max(runs, key=lambda r: r["done"]) if runs else None

print("# Teacher throughput\n\nVariant B (thinking on), no_robots prompts. Output tokens per second over a whole "
      "generate run, including the slow tail. `base_c256` is the pilot run.\n")
print("| setting | samples | tok/s | minutes | vs pilot |\n|---|---:|---:|---:|---:|")
base = biggest_run(pilot / "no_robots_open_B_raw.jsonl.runs.jsonl")
rows = [("base_c256 (pilot)", base)]
rows += [("_".join(f.name.split("_")[:2]), biggest_run(f)) for f in sorted(bench.glob("*_B_raw.jsonl.runs.jsonl"))]
for setting, r in rows:
    if r:
        speed = f"{r['tok_per_s'] / base['tok_per_s']:.2f}x" if base else "—"
        print(f"| {setting} | {r['done']} | {r['tok_per_s']:,.0f} | {r['seconds'] / 60:,.1f} | {speed} |")
EOF
