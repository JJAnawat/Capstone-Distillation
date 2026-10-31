#!/usr/bin/env bash
# The Thai sets on the GPU VM.
# 1. Prep, for each set without data/prepared/<set>.jsonl: translate its new questions with TRANSLATOR (default:
#    config translate.translator), the judge checks them, embeddings score them, build-translated writes the prepared
#    file, and data/translate is backed up to the Hub.
# 2. PROMPT_AB=1: the teacher prompt test (scripts/prompt_ab.sh), then stop so the arm can be chosen.
#    Otherwise: distill every Thai set into data/full_th with pipeline.sh (variant B, set by set).
# Every step resumes, so the same command continues in the next GPU window.
#   tmux new -s thai 'PROMPT_AB=1 bash scripts/run_thai.sh 2>&1 | tee -ai logs/thai.log'
#   tmux new -s thai 'bash scripts/run_thai.sh 2>&1 | tee -ai logs/thai.log'
set -euo pipefail
SETS=${SETS:-"no_robots_th helpsteer2_th ultrafeedback_th logiqa_th"}
PROMPT_AB=${PROMPT_AB:-0}

source scripts/lib.sh
TRANSLATOR=${TRANSLATOR:-$(config_value "c['translate']['translator'] or ''")}

prep() {
  local todo=() ds
  for ds in $SETS; do if [ ! -s "data/prepared/$ds.jsonl" ]; then todo+=("$ds"); fi; done
  if [ ${#todo[@]} -eq 0 ]; then echo "== Thai prep: every set is built already"; return; fi
  if [ -z "$TRANSLATOR" ]; then echo "set translate.translator in config.yaml (or TRANSLATOR)" >&2; exit 1; fi
  echo "== $(date -u +%H:%M) Thai prep: translate with the $TRANSLATOR: ${todo[*]}"
  start_server "$TRANSLATOR"
  for ds in "${todo[@]}"; do with_server "$TRANSLATOR" distill translate --dataset "$ds" --translator "$TRANSLATOR"; done
  if [ "$TRANSLATOR" != judge ]; then stop_server; start_server judge; fi
  echo "== $(date -u +%H:%M) Thai prep: check the translations"
  for ds in "${todo[@]}"; do with_server judge distill check-translation "data/translate/$ds.jsonl"; done
  stop_server
  for ds in "${todo[@]}"; do
    distill similarity "data/translate/$ds.jsonl"
    distill build-translated --dataset "$ds"
  done
  distill push data/translate || echo "warning: the Hub backup failed; the translations are still in data/translate" >&2
}

prep
if [ "$PROMPT_AB" = 1 ]; then
  SETS="$SETS" bash scripts/prompt_ab.sh
  echo "== choose the arm from reports/prompt_ab_{en,th}.md (Thai: prompt_versions on the *_th sets in config.yaml)," \
       "then run this script without PROMPT_AB"
  exit 0
fi
DATASETS="$SETS" LIMIT=0 BY_SET=1 VARIANTS=B VALIDATE=0 OUT=${OUT:-data/full_th} REPORT=${REPORT:-reports/full_th.md} \
  exec bash scripts/pipeline.sh
