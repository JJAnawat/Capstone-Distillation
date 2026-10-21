# Shared by pipeline.sh and bench.sh (source it from the repo root): activate the venv and start/stop a local vLLM
# server. Ctrl-C or any exit stops the server, so the next run can start cleanly and resume.
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
mkdir -p logs

SERVER=""
start_server() {  # ROLE [EXTRA_VLLM_ARGS]: start `vllm serve` and wait until it serves its model (fails fast if it dies)
  echo "=== $(date -u +%FT%TZ) start $1 ${2:-}" >> "logs/vllm_$1.log"
  VLLM_EXTRA_ARGS="${2:-}" distill serve "$1" >> "logs/vllm_$1.log" 2>&1 &
  SERVER=$!
  distill wait "$1" --pid "$SERVER"
}
stop_server() {  # shut the server down and wait until the GPU memory is free for the next model
  if [ -z "$SERVER" ]; then return; fi
  kill "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  SERVER=""
  for _ in $(seq 60); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then return; fi
    sleep 2
  done
  echo "warning: GPU memory still in use after stopping the server" >&2
}
trap stop_server EXIT
trap 'exit 130' INT TERM  # turn Ctrl-C into a normal exit so the EXIT trap stops the server
