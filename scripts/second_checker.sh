#!/usr/bin/env bash
# Second-family checker test on the GPU VM (~10 min): a checker from outside the Gemma family (ROLE, default the
# `checker` role in config.yaml) scores the translator comparison's 156 translations. The report then compares it with
# the judge (Gemma) against the hand grades: agreement, thresholds, and each checker's lean on its own family's work.
#   tmux new -s checker 'bash scripts/second_checker.sh 2>&1 | tee -ai logs/checker.log'
# Needs the graded sheet next to its key (reports/translate_compare.xlsx, reports/translate_compare_key.json).
# Afterwards set translate.checker and translate.min_meaning from reports/translate_compare.md.
set -euo pipefail
ROLE=${ROLE:-checker}
SHEET=${SHEET:-reports/translate_compare.xlsx}

source scripts/lib.sh
if [ ! -f "$SHEET" ]; then echo "no graded sheet at $SHEET" >&2; exit 1; fi
FILES=(data/translate_compare/*_th.teacher.jsonl data/translate_compare/*_th.judge.jsonl
       data/translate_compare/*_th.typhoon.jsonl)
echo "== $(date -u +%H:%M) check the comparison's ${#FILES[@]} translation files with the $ROLE"
start_server "$ROLE"
with_server "$ROLE" distill check-translation --role "$ROLE" "${FILES[@]}"
stop_server
distill compare-report "$SHEET" --out reports/translate_compare.md
distill push data/translate_compare || echo "warning: the Hub backup failed; the checks are still in data/translate_compare" >&2
echo "== done: reports/translate_compare.md compares the checkers"
