#!/usr/bin/env bash
# CoT pilot on the GPU VM: both variants, 100 prompts per dataset and subset, 8 samples each.
# Run inside tmux from the repo root so it survives SSH disconnects:
#   tmux new -s pilot 'bash scripts/run_pilot.sh 2>&1 | tee -a logs/pilot.log'
# Smoke-test the VM first with:
#   LIMIT=5 VALIDATE=20 OUT=data/smoke_vm REPORT=reports/smoke_vm.md bash scripts/run_pilot.sh
# Rule-failing closed samples are judged too (JUDGE_ALL_CLOSED=1), to measure how well the judge spots wrong answers.
# Both judge rubrics are validated (VALIDATE_PROMPTS); samples are judged with the config's version.
BACKEND=vllm LIMIT=${LIMIT:-100} JUDGE_ALL_CLOSED=${JUDGE_ALL_CLOSED:-1} VALIDATE_PROMPTS=${VALIDATE_PROMPTS:-"v1 v2"} \
  OUT=${OUT:-data/pilot2} REPORT=${REPORT:-reports/pilot2.md} exec bash scripts/pipeline.sh
