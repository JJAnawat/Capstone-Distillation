#!/usr/bin/env bash
# The whole pipeline for a set of prompts: teacher phase (rate, generate, compress), filter, judge phase, select,
# report. Re-running resumes: every stage skips work that is already in its output file.
# Settings come from the environment (run_thai.sh sets them for the Thai sets). The English full run was
#   DATASETS="no_robots helpsteer2 ultrafeedback" LIMIT=2500 LIMIT_CLOSED=500 BY_SET=1 bash scripts/pipeline.sh
# and LogiQA on its own with DATASETS=logiqa LIMIT_CLOSED=2500.
#   LIMIT     prompts per dataset and subset, 0 = all                                      [0]
#   LIMIT_CLOSED  prompts in each closed set, e.g. 500 for UltraFeedback's noisy one     [LIMIT]
#   N         samples per prompt for every prompt; empty = by complexity (config)      [config]
#   VARIANTS  A = thinking off + <reasoning> tags (the pilot's comparison), B = thinking on + compress   [B]
#   DATASETS  datasets to run; logiqa runs on its own (LIMIT_CLOSED)  ["no_robots helpsteer2 ultrafeedback"]
#   VALIDATE  HelpSteer2 pairs for judge validation (once per OUT), 0 = skip              [0]
#   JUDGE_ALL_CLOSED  1 = judge rule-failing closed samples too (judge-vs-rules check)     [0]
#   VALIDATE_PROMPTS  judge rubric versions to validate                          [config version]
#   BY_SET    1 = run each set through teacher, judge and select before the next (time-boxed runs)  [0]
#   REDO_LANGUAGE  1 = compress first drops translated rows whose rationale is in the wrong language, to redo them [0]
#   OUT, REPORT  output directory and report path                       [data/full, reports/full.md]
# Stop with Ctrl-C: the server is shut down and the next run resumes where this one stopped.
# After each finished set, `distill push` backs up $OUT to the private Hub repo in config hub.repo (if set).
# The report covers every set in $OUT, also those finished in an earlier session (a resume may name fewer DATASETS).
set -euo pipefail

LIMIT=${LIMIT:-0}
LIMIT_CLOSED=${LIMIT_CLOSED:-$LIMIT}  # the noisy known-answer set can be smaller
N=${N:-}  # empty: samples per prompt by complexity (config generate.n_by_complexity)
VARIANTS=${VARIANTS:-B}
DATASETS=${DATASETS:-"no_robots helpsteer2 ultrafeedback"}
VALIDATE=${VALIDATE:-0}
JUDGE_ALL_CLOSED=${JUDGE_ALL_CLOSED:-0}
BY_SET=${BY_SET:-0}
REDO_LANGUAGE=${REDO_LANGUAGE:-0}
OUT=${OUT:-data/full}
REPORT=${REPORT:-reports/full.md}

source scripts/lib.sh
mkdir -p "$OUT"
limit_args() {  # --limit for a set: LIMIT, or LIMIT_CLOSED for the closed set (0 = all)
  local l=$LIMIT; if [ "$1" = closed ]; then l=$LIMIT_CLOSED; fi
  if [ "$l" -gt 0 ]; then echo "--limit $l"; fi
}
NARG=(); if [ -n "$N" ]; then NARG=(--n "$N"); fi
has() { [[ " $1 " == *" $2 "* ]]; }

# Each set is dataset:subset, one per kind of prompt the dataset has: open, and closed (a known answer). UltraFeedback
# has both, LogiQA only closed. The assignment (not a for list) lets set -e stop on a failure.
SETS=()
for ds in $DATASETS; do
  subs=$(distill subsets "$ds" | tail -n 1)
  for sub in $subs; do SETS+=("$ds:$sub"); done
done

teacher_stage() {  # SET...: rate, generate and compress with one teacher server
  echo "== teacher phase: $*"
  start_server teacher
  for s in "$@"; do
    ds=${s%%:*}; only=${s##*:}; name="${ds}_${only}"
    read -r -a LIM <<< "$(limit_args "$only")"
    with_server teacher distill rate --dataset "$ds" --only "$only" "${LIM[@]}"
    if has "$VARIANTS" A; then
      with_server teacher distill generate --dataset "$ds" --only "$only" "${LIM[@]}" "${NARG[@]}" \
        --mode cot --thinking off --out "$OUT/${name}_A.jsonl"
    fi
    if has "$VARIANTS" B; then
      with_server teacher distill generate --dataset "$ds" --only "$only" "${LIM[@]}" "${NARG[@]}" \
        --mode open --thinking on --out "$OUT/${name}_B_raw.jsonl"
      REDO=(); if [ "$REDO_LANGUAGE" = 1 ]; then REDO=(--redo-wrong-language); fi
      with_server teacher distill compress "$OUT/${name}_B_raw.jsonl" "${REDO[@]}" --out "$OUT/${name}_B.jsonl"
    fi
  done
  stop_server
}

ALL_KEPT=()
filter_stage() {  # SET...: filter, and remember the kept files of this round for judging and selection
  echo "== filter: $*"
  KEPT=(); KEPT_OPEN=(); KEPT_CLOSED=()
  for s in "$@"; do
    name="${s%%:*}_${s##*:}"
    for v in $VARIANTS; do
      f="$OUT/${name}_${v}.kept.jsonl"
      distill filter "$OUT/${name}_${v}.jsonl" --out "$f"
      KEPT+=("$f"); ALL_KEPT+=("$f")
      if [ "${s##*:}" = closed ]; then KEPT_CLOSED+=("$f"); else KEPT_OPEN+=("$f"); fi
    done
  done
}

judge_stage() {  # judge this round's kept files with one judge server
  echo "== judge phase"
  start_server judge
  for pv in ${VALIDATE_PROMPTS:-config}; do  # each rubric version is validated once per OUT
    if [ "$VALIDATE" -gt 0 ] && [ ! -e "$OUT/.judge_validated_$pv" ]; then
      PV=(); if [ "$pv" != config ]; then PV=(--prompt-version "$pv"); fi
      with_server judge distill judge --validate "$VALIDATE" "${PV[@]}" && touch "$OUT/.judge_validated_$pv"
    fi
  done
  if [ ${#KEPT_OPEN[@]} -gt 0 ]; then with_server judge distill judge "${KEPT_OPEN[@]}"; fi
  if [ ${#KEPT_CLOSED[@]} -gt 0 ]; then
    ALL=(); if [ "$JUDGE_ALL_CLOSED" = 1 ]; then ALL=(--all); fi
    with_server judge distill judge "${KEPT_CLOSED[@]}" "${ALL[@]}"
  fi
  stop_server
}

select_stage() {
  echo "== select"
  for f in "${KEPT[@]}"; do distill select "$f" --out "${f%.kept.jsonl}.final.jsonl"; done
}

backup() {  # a failed upload never stops the run: the data is still in $OUT
  distill push "$OUT" || echo "warning: the Hub backup failed; the data is still in $OUT" >&2
}

if [ "$BY_SET" = 1 ]; then  # finish each set completely before the next: a run cut short still leaves final data
  for s in "${SETS[@]}"; do teacher_stage "$s"; filter_stage "$s"; judge_stage; select_stage; backup; done
else                        # one teacher phase and one judge phase for everything: fewest model swaps
  teacher_stage "${SETS[@]}"; filter_stage "${SETS[@]}"; judge_stage; select_stage; backup
fi
for v in $VARIANTS; do  # add the sets finished in earlier sessions
  for f in "$OUT"/*_"$v".kept.jsonl; do if [ -e "$f" ] && ! has "${ALL_KEPT[*]}" "$f"; then ALL_KEPT+=("$f"); fi; done
done
distill analyze "${ALL_KEPT[@]}" --out "$REPORT"
echo "== done: final data in $OUT/*.final.jsonl, report in $REPORT"
