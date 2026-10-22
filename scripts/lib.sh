# Shared by pipeline.sh and bench.sh (source it from the repo root): load machine settings, activate the venv and
# start/stop a local vLLM server. Ctrl-C or any exit stops the server, so the next run can start cleanly and resume.
# .vm_env (untracked) holds machine-specific exports such as HF_HOME; ~/.bashrc is often skipped by tmux and scripts.
if [ -f .vm_env ]; then source .vm_env; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
mkdir -p logs

SERVER=""
start_server() {  # ROLE [EXTRA_VLLM_ARGS]: start `vllm serve` and wait until it serves its model (fails fast if it dies)
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
  if [ -n "$busy" ]; then  # e.g. a server left over from an earlier run
    echo "The GPU is already in use. Stop these processes first (for an old server: pkill -f 'vllm serve'):" >&2
    echo "$busy" >&2
    return 1
  fi
  echo "=== $(date -u +%FT%TZ) start $1 ${2:-}" >> "logs/vllm_$1.log"
  DISTILL_SERVE_ARGS="${2:-}" distill serve "$1" >> "logs/vllm_$1.log" 2>&1 &
  SERVER=$!
  if ! distill wait "$1" --pid "$SERVER"; then
    echo "--- the $1 server did not start; last lines of logs/vllm_$1.log:" >&2
    tail -n 40 "logs/vllm_$1.log" >&2
    return 1
  fi
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
trap 'exit 130' INT TERM HUP  # Ctrl-C, kill or a closed tmux window: exit normally so the EXIT trap stops the server
