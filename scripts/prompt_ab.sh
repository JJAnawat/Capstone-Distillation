#!/usr/bin/env bash
# Teacher prompt test for the Thai sets on the GPU VM (~20 min): the same LIMIT prompts of each Thai set go through
# variant B with the English prompts (arm en: open_v2 / open_closed_v2, compress_v3) and with the Thai ones (arm th:
# *_th1). Then the filter, one judge session for both arms, and a report per arm: reports/prompt_ab_{en,th}.md.
# Run by `PROMPT_AB=1 bash scripts/run_thai.sh` after the Thai prep, or on its own once data/prepared/*_th.jsonl exist.
set -euo pipefail
OUT=${OUT:-data/prompt_ab}
LIMIT=${LIMIT:-13}
SETS=${SETS:-"no_robots_th helpsteer2_th ultrafeedback_th logiqa_th"}

source scripts/lib.sh
mkdir -p "$OUT"
echo "== $(date -u +%H:%M) prompt test, teacher phase: $SETS ($LIMIT prompts each)"
start_server teacher
for ds in $SETS; do
  only=$(distill subsets "$ds" | tail -n 1)  # each Thai set has one kind: open, or closed (LogiQA)
  with_server teacher distill rate --dataset "$ds" --only "$only" --limit "$LIMIT"
  for arm in en th; do
    PV=(); if [ "$arm" = th ]; then PV=(--prompt-version th1); fi
    f="$OUT/${ds}_${only}_$arm"
    with_server teacher distill generate --dataset "$ds" --only "$only" --limit "$LIMIT" --mode open --thinking on \
      "${PV[@]}" --out "${f}_raw.jsonl"
    with_server teacher distill compress "${f}_raw.jsonl" "${PV[@]}" --out "$f.jsonl"
  done
done
stop_server
for f in "$OUT"/*_en.jsonl "$OUT"/*_th.jsonl; do distill filter "$f" --out "${f%.jsonl}.kept.jsonl"; done
echo "== $(date -u +%H:%M) prompt test, judge phase"
start_server judge
with_server judge distill judge "$OUT"/*_en.kept.jsonl "$OUT"/*_th.kept.jsonl
stop_server
distill analyze "$OUT"/*_en.kept.jsonl --out reports/prompt_ab_en.md
distill analyze "$OUT"/*_th.kept.jsonl --out reports/prompt_ab_th.md
echo "== done: compare reports/prompt_ab_en.md with reports/prompt_ab_th.md"
