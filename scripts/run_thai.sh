#!/usr/bin/env bash
# The Thai sets on the GPU VM, in runs that continue each other.
# 1. Prep, once, for each set without data/prepared/<set>.jsonl: translate its new questions with the translator
#    (config translate.translator), check them with the checker (translate.checker), build the prepared file, and back
#    up data/translate to the Hub. Prep runs once so the prompt pool stays fixed, which keeps step 2's runs nested.
# 2. Distill LIMIT prompts per Thai set into data/full_th (variant B, the English teacher prompts). A later run with
#    a larger LIMIT only adds prompts (the sampler is nested); LIMIT=0 takes all the rest. Each run ends with final
#    rows for every prompt done so far, backed up to the Hub.
#    BY_SET: a run with a LIMIT, sized to a night, loads each model once (BY_SET=0); an unlimited run goes set by set
#    (BY_SET=1), so a cut still leaves finished sets.
# Every step resumes, so a run cut by the stopper continues on the next run.
#   tmux new -s thai 'LIMIT=500 bash scripts/run_thai.sh 2>&1 | tee -ai logs/thai.log'
set -euo pipefail
SETS=${SETS:-"no_robots_th helpsteer2_th ultrafeedback_th logiqa_th"}
LIMIT=${LIMIT:-0}
if [ "$LIMIT" -gt 0 ]; then BY_SET=${BY_SET:-0}; else BY_SET=${BY_SET:-1}; fi

source scripts/lib.sh
TRANSLATOR=${TRANSLATOR:-$(config_value "c['translate']['translator'] or ''")}
CHECKER=${CHECKER:-$(config_value "c['translate'].get('checker') or 'judge'")}

prep() {
  local todo=() ds
  for ds in $SETS; do if [ ! -s "data/prepared/$ds.jsonl" ]; then todo+=("$ds"); fi; done
  if [ ${#todo[@]} -eq 0 ]; then echo "== Thai prep: every set is built already"; return; fi
  if [ -z "$TRANSLATOR" ]; then echo "set translate.translator in config.yaml (or TRANSLATOR)" >&2; exit 1; fi
  echo "== $(date -u +%H:%M) Thai prep: translate with the $TRANSLATOR: ${todo[*]}"
  start_server "$TRANSLATOR"
  for ds in "${todo[@]}"; do with_server "$TRANSLATOR" distill translate --dataset "$ds" --translator "$TRANSLATOR"; done
  if [ "$TRANSLATOR" != "$CHECKER" ]; then stop_server; start_server "$CHECKER"; fi
  echo "== $(date -u +%H:%M) Thai prep: check the translations with the $CHECKER"
  for ds in "${todo[@]}"; do
    with_server "$CHECKER" distill check-translation --role "$CHECKER" "data/translate/$ds.jsonl"
  done
  stop_server
  for ds in "${todo[@]}"; do distill build-translated --dataset "$ds"; done
  distill push data/translate || echo "warning: the Hub backup failed; the translations are still in data/translate" >&2
}

prep
echo "== $(date -u +%H:%M) distill: LIMIT=$LIMIT prompts per set (0 = all), BY_SET=$BY_SET"
DATASETS="$SETS" LIMIT="$LIMIT" BY_SET="$BY_SET" VARIANTS=B VALIDATE=0 OUT=${OUT:-data/full_th} \
  REPORT=${REPORT:-reports/full_th.md} exec bash scripts/pipeline.sh
