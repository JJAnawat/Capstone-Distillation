#!/usr/bin/env bash
# The whole pipeline for a set of prompts: teacher phase (rate, generate, compress), filter, judge phase, select,
# report. Re-running resumes: every stage skips work that is already in its output file.
# Settings come from the environment (the run_*.sh wrappers set them):
#   BACKEND   vllm (start/stop local servers; GPU VM) | openrouter (laptop smoke test)   [vllm]
#   LIMIT     prompts per dataset and subset, 0 = all                                      [100]
#   N         samples per prompt                                                          [8]
#   VARIANTS  "A B": A = thinking off + <reasoning> tags, B = thinking on + compress      ["A B"]
#   DATASETS  datasets to run                                     ["no_robots helpsteer2 ultrafeedback"]
#   VALIDATE  HelpSteer2 pairs for judge validation (once per OUT), 0 = skip              [300]
#   JUDGE_ALL_CLOSED  1 = judge rule-failing closed samples too (judge-vs-rules check)     [0]
#   OUT, REPORT  output directory and report path                       [data/pilot2, reports/pilot2.md]
# Stop with Ctrl-C: the server is shut down and the next run resumes where this one stopped.
set -euo pipefail

BACKEND=${BACKEND:-vllm}
LIMIT=${LIMIT:-100}
N=${N:-8}
VARIANTS=${VARIANTS:-"A B"}
DATASETS=${DATASETS:-"no_robots helpsteer2 ultrafeedback"}
VALIDATE=${VALIDATE:-300}
JUDGE_ALL_CLOSED=${JUDGE_ALL_CLOSED:-0}
OUT=${OUT:-data/pilot2}
REPORT=${REPORT:-reports/pilot2.md}

source scripts/lib.sh
mkdir -p "$OUT"
BK=(--backend "$BACKEND")
LIM=()
if [ "$LIMIT" -gt 0 ]; then LIM=(--limit "$LIMIT"); fi
start() { if [ "$BACKEND" = vllm ]; then start_server "$1"; fi; }  # OpenRouter needs no local server
stop() { stop_server; }
has() { [[ " $1 " == *" $2 "* ]]; }

# Each set is dataset:subset. UltraFeedback's closed (known-answer) prompts form their own set.
SETS=()
for ds in $DATASETS; do SETS+=("$ds:open"); done
if has "$DATASETS" ultrafeedback; then SETS+=("ultrafeedback:closed"); fi

echo "== teacher phase ($BACKEND)"
start teacher
for s in "${SETS[@]}"; do
  ds=${s%%:*}; only=${s##*:}; name="${ds}_${only}"
  distill rate --dataset "$ds" --only "$only" "${LIM[@]}" "${BK[@]}"
  if has "$VARIANTS" A; then
    distill generate --dataset "$ds" --only "$only" "${LIM[@]}" --n "$N" --mode cot --thinking off "${BK[@]}" \
      --out "$OUT/${name}_A.jsonl"
  fi
  if has "$VARIANTS" B; then
    distill generate --dataset "$ds" --only "$only" "${LIM[@]}" --n "$N" --mode open --thinking on "${BK[@]}" \
      --out "$OUT/${name}_B_raw.jsonl"
    distill compress "$OUT/${name}_B_raw.jsonl" "${BK[@]}" --out "$OUT/${name}_B.jsonl"
  fi
done
stop

echo "== filter"
KEPT=(); KEPT_OPEN=(); KEPT_CLOSED=()
for s in "${SETS[@]}"; do
  name="${s%%:*}_${s##*:}"
  for v in $VARIANTS; do
    f="$OUT/${name}_${v}.kept.jsonl"
    distill filter "$OUT/${name}_${v}.jsonl" --out "$f"
    KEPT+=("$f")
    if [ "${s##*:}" = closed ]; then KEPT_CLOSED+=("$f"); else KEPT_OPEN+=("$f"); fi
  done
done

echo "== judge phase ($BACKEND)"
start judge
if [ "$VALIDATE" -gt 0 ] && [ ! -e "$OUT/.judge_validated" ]; then
  distill judge --validate "$VALIDATE" "${BK[@]}" && touch "$OUT/.judge_validated"
fi
if [ ${#KEPT_OPEN[@]} -gt 0 ]; then distill judge "${KEPT_OPEN[@]}" "${BK[@]}"; fi
if [ ${#KEPT_CLOSED[@]} -gt 0 ]; then
  ALL=(); if [ "$JUDGE_ALL_CLOSED" = 1 ]; then ALL=(--all); fi
  distill judge "${KEPT_CLOSED[@]}" "${ALL[@]}" "${BK[@]}"
fi
stop

echo "== select + report"
for f in "${KEPT[@]}"; do distill select "$f" --out "${f%.kept.jsonl}.final.jsonl"; done
distill analyze "${KEPT[@]}" --out "$REPORT"
echo "== done: final data in $OUT/*.final.jsonl, report in $REPORT"
