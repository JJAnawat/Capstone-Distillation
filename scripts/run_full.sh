#!/usr/bin/env bash
# Full run on the GPU VM over every prompt, with the variant chosen from the pilot:
#   tmux new -s full 'VARIANTS=A bash scripts/run_full.sh 2>&1 | tee -ai logs/full.log'
BACKEND=vllm LIMIT=${LIMIT:-0} VARIANTS=${VARIANTS:?set VARIANTS to A or B (the pilot winner)} VALIDATE=${VALIDATE:-0} \
  OUT=${OUT:-data/full} REPORT=${REPORT:-reports/full.md} exec bash scripts/pipeline.sh
