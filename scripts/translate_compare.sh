#!/usr/bin/env bash
# Translator comparison on the GPU VM (~30 min): the first translate.compare_limit prompts of each Thai set, translated
# by Qwen (teacher), Typhoon Translate and Gemma (judge). The judge then checks every translation, embeddings score
# them, and the grading sheet goes to reports/translate_compare.xlsx (its key: reports/translate_compare_key.json).
#   tmux new -s compare 'bash scripts/translate_compare.sh 2>&1 | tee -ai logs/compare.log'
# After grading the sheet by hand: distill compare-report reports/translate_compare.xlsx
# Re-running resumes: every stage skips what is already in its output.
set -euo pipefail
OUT=${OUT:-data/translate_compare}
SETS=${SETS:-"no_robots_th helpsteer2_th ultrafeedback_th logiqa_th"}
TRANSLATORS=${TRANSLATORS:-"teacher typhoon judge"}  # the judge last: its server stays up for the check
SHEET=${SHEET:-reports/translate_compare.xlsx}

source scripts/lib.sh
mkdir -p "$OUT"
LIMIT=$(config_value "c['translate']['compare_limit']")
FILES=()
for role in $TRANSLATORS; do
  echo "== $(date -u +%H:%M) translate with the $role: $SETS ($LIMIT prompts each)"
  start_server "$role"
  for ds in $SETS; do
    with_server "$role" distill translate --dataset "$ds" --translator "$role" --limit "$LIMIT" --no-reference \
      --out "$OUT/$ds.$role.jsonl"
    FILES+=("$OUT/$ds.$role.jsonl")
  done
  if [ "$role" != judge ]; then stop_server; fi
done
if [ -z "$SERVER" ]; then start_server judge; fi
echo "== $(date -u +%H:%M) check every translation with the judge"
with_server judge distill check-translation "${FILES[@]}"
stop_server
distill similarity "${FILES[@]}"
distill compare-sheet "${FILES[@]}" --out "$SHEET"
distill push "$OUT" || echo "warning: the Hub backup failed; the translations are still in $OUT" >&2
echo "== done: grade $SHEET (meaning and fluency, 1-5), then: distill compare-report $SHEET"
