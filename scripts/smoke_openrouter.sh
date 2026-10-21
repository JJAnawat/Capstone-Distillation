#!/usr/bin/env bash
# Laptop smoke test of the whole pipeline through OpenRouter (Qwen3.5-9B teacher, Gemma 4 31B judge); costs cents.
# Needs OPENROUTER_API_KEY in .env. Run from the repo root in Git Bash: bash scripts/smoke_openrouter.sh
BACKEND=openrouter LIMIT=${LIMIT:-3} N=${N:-2} VALIDATE=${VALIDATE:-10} OUT=${OUT:-data/smoke} \
  REPORT=${REPORT:-reports/smoke.md} exec bash scripts/pipeline.sh
