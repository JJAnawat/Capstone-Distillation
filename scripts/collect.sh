#!/usr/bin/env bash
# Copy the small artifacts of a run into reports/NAME/, so `git add reports && git commit && git push` brings them
# back to the laptop. The run's Markdown report is already under reports/.
# Usage: bash scripts/collect.sh NAME OUT_DIR [--data]
#   --data  also pack the kept + judge files into data/NAME_data.tar.gz (for scp; generated data is not committed)
set -uo pipefail
NAME=${1:?usage: collect.sh NAME OUT_DIR [--data]}
OUT=${2:?usage: collect.sh NAME OUT_DIR [--data]}
DEST="reports/$NAME"
mkdir -p "$DEST"

shopt -s nullglob
for f in "$OUT"/*.runs.jsonl "$OUT"/*.rejections.json "$OUT"/*.select_stats.json data/prepared/*.runs.jsonl \
         reports/judge_validation.md reports/vm_check.txt; do
  if [ -e "$f" ]; then cp "$f" "$DEST/"; fi
done
for f in "$OUT"/*.errors.jsonl; do head -n 50 "$f" > "$DEST/$(basename "$f")"; done

# Server facts and stats: KV cache size, max concurrency, throughput / running / KV usage / prefix-cache lines,
# speculative-decoding acceptance, errors. The "===" lines mark each server start.
for log in logs/vllm_*.log; do
  grep -E "^===|KV cache size|Maximum concurrency|throughput|cache hit rate|[Aa]cceptance|ERROR|Error|Traceback" "$log" \
    | tail -n 300 > "$DEST/$(basename "$log" .log)_highlights.txt"
done

# The end of each run log (logs/smoke.log, logs/pilot.log, ...): stage summaries and any errors.
for log in logs/*.log; do
  case "$(basename "$log")" in vllm_*) ;; *) tail -n 200 "$log" > "$DEST/$(basename "$log" .log)_tail.txt" ;; esac
done

# Complexity ratings per dataset and level (the ratings themselves stay on the VM).
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
"$PY" - "$DEST/complexity_counts.json" <<'EOF'
import collections, json, pathlib, sys
counts = {}
for f in sorted(pathlib.Path("data/prepared").glob("*.complexity.jsonl")):
    c = collections.Counter(json.loads(line)["complexity"] for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
    counts[f.name.removesuffix(".complexity.jsonl")] = {str(k): v for k, v in sorted(c.items(), key=lambda x: (x[0] is None, x[0] or 0))}
pathlib.Path(sys.argv[1]).write_text(json.dumps(counts, indent=2), encoding="utf-8")
EOF

if [ "${3:-}" = --data ]; then
  tar czf "data/${NAME}_data.tar.gz" "$OUT"/*.kept.jsonl "$OUT"/*.judge.jsonl
  echo "data: data/${NAME}_data.tar.gz ($(du -h "data/${NAME}_data.tar.gz" | cut -f1)); copy it with scp"
fi
echo "collected into $DEST:"; ls -1 "$DEST"
echo "next: git add reports && git commit -m 'Results: $NAME' && git push"
