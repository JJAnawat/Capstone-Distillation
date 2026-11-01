#!/usr/bin/env bash
# The second translator comparison on the GPU VM (~20 min): PER_SET questions per Thai set that the run has not
# distilled yet, with the translations Gemma already made (no GPU), against the sentence-level translators in
# TRANSLATORS (in-process transformers, one model on the GPU at a time). The judge then checks the new translations,
# and the grading sheet goes to reports/translate_compare2.xlsx (its key: reports/translate_compare2_key.json).
#   tmux new -s compare2 'bash scripts/translate_compare2.sh 2>&1 | tee -ai logs/compare2.log'
# After grading the sheet by hand: distill compare-report reports/translate_compare2.xlsx --out reports/translate_compare2.md
set -euo pipefail
OUT=${OUT:-data/translate_compare2}
PER_SET=${PER_SET:-10}
TRANSLATORS=${TRANSLATORS:-"nllb madlad seedx"}
SHEET=${SHEET:-reports/translate_compare2.xlsx}

source scripts/lib.sh
if [ ! -s "$OUT/logiqa_th.judge.jsonl" ]; then distill compare-pick --per-set "$PER_SET" --out-dir "$OUT"; fi
NEW=()
for role in $TRANSLATORS; do
  echo "== $(date -u +%H:%M) translate the pool with $role"
  distill compare-translate --role "$role" --dir "$OUT"
  NEW+=("$OUT"/*_th."$role".jsonl)
done
echo "== $(date -u +%H:%M) check the new translations with the judge"
start_server judge
with_server judge distill check-translation --role judge "${NEW[@]}"
stop_server
distill compare-sheet "$OUT"/*_th.judge.jsonl "${NEW[@]}" --out "$SHEET"
distill push "$OUT" || echo "warning: the Hub backup failed; the translations are still in $OUT" >&2
echo "== done: grade $SHEET (meaning and fluency, 1-5), then: distill compare-report $SHEET --out reports/translate_compare2.md"
