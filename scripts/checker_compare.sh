#!/usr/bin/env bash
# The checker comparison on the GPU VM (~25 min, mostly model loads): other checkers score the translator comparison's
# graded translations (G1: their agreement with the hand grades) and a sample of night 1's translations, PER_SET per
# Thai set (where they and the judge disagree). One model on the GPU at a time:
#   metricx    MetricX-24 XXL, in-process (vLLM's venv)
#   cometkiwi  CometKiwi XL, in-process from .venv-comet (unbabel-comet pins its own numpy and torch)
#   typhoon25  Typhoon 2.5 behind vLLM, quantized to FP8 while loading; if it does not start, the rest goes on
# Then G1's report gains the new checkers, and the disagreement pool goes to a grading sheet.
#   tmux new -s checkers 'bash scripts/checker_compare.sh 2>&1 | tee -ai logs/checkers.log'
# After grading the sheet by hand: distill checker-report reports/checker_compare.xlsx --out reports/checker_compare.md
set -euo pipefail
OUT=${OUT:-data/checker_compare}
SAMPLE=${SAMPLE:-$OUT/sample}
PER_SET=${PER_SET:-300}
SHEET=${SHEET:-reports/checker_compare.xlsx}
G1_SHEET=${G1_SHEET:-reports/translate_compare.xlsx}
CHECKERS=${CHECKERS:-"metricx cometkiwi typhoon25"}

source scripts/lib.sh
if [ ! -s "$SAMPLE/logiqa_th.jsonl" ]; then distill checker-sample --per-set "$PER_SET" --out-dir "$SAMPLE"; fi
FILES=(data/translate_compare/*_th.teacher.jsonl data/translate_compare/*_th.judge.jsonl
       data/translate_compare/*_th.typhoon.jsonl "$SAMPLE"/*_th.jsonl)
for role in $CHECKERS; do
  echo "== $(date -u +%H:%M) check ${#FILES[@]} translation files with the $role"
  case $role in
    metricx) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True distill check-translation --role metricx "${FILES[@]}" ;;
    cometkiwi) .venv-comet/bin/distill check-translation --role cometkiwi "${FILES[@]}" ;;
    *)
      if start_server "$role"; then
        with_server "$role" distill check-translation --role "$role" "${FILES[@]}" ||
          echo "warning: the $role checks did not finish; the comparison goes on with what it has" >&2
      else
        echo "warning: the $role server did not start; the comparison goes on without it" >&2
      fi
      stop_server ;;
  esac
done
distill compare-report "$G1_SHEET" --out reports/translate_compare.md
distill checker-pick --roles $CHECKERS --from "$SAMPLE" --out-dir "$OUT"
distill compare-sheet "$OUT"/*_th.jsonl --out "$SHEET"
for dir in data/translate_compare "$OUT"; do
  distill push "$dir" || echo "warning: the Hub backup of $dir failed; the files are still on the VM" >&2
done
echo "== done: grade $SHEET (meaning and fluency, 1-5), then: distill checker-report $SHEET --out reports/checker_compare.md"
