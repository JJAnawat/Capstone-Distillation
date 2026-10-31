# Shared by pipeline.sh and bench.sh (source it from the repo root): load machine settings, activate the venv and
# start/stop a local vLLM server. Ctrl-C or any exit stops the server, so the next run can start cleanly and resume.
# .vm_env (untracked) holds machine-specific exports such as HF_HOME; ~/.bashrc is often skipped by tmux and scripts.
if [ -f .vm_env ]; then source .vm_env; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
mkdir -p logs

SERVER=""
own_gpu_pids() {  # this user's processes on the GPU; other users' (the GPU can be shared) are not ours to wait for
  local pid
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
    if [ "$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$(id -u)" ]; then echo "$pid"; fi
  done
}
start_server() {  # ROLE [EXTRA_VLLM_ARGS]: start `vllm serve` and wait until it serves its model (fails fast if it dies)
  local mine others
  mine=$(own_gpu_pids)
  if [ -n "$mine" ]; then  # e.g. a server left over from an earlier run
    echo "Our processes already use the GPU; stop them first (for an old server: pkill -f 'vllm serve'): $mine" >&2
    return 1
  fi
  others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
  if [ -n "$others" ]; then  # vLLM itself refuses to start if the free memory is short of --gpu-memory-utilization
    echo "note: other users' processes on the GPU: $(tr '\n' ' ' <<< "$others")" >&2
  fi
  echo "=== $(date -u +%FT%TZ) start $1 ${2:-}" >> "logs/vllm_$1.log"
  local mark; mark=$(wc -l < "logs/vllm_$1.log")
  DISTILL_SERVE_ARGS="${2:-}" distill serve "$1" >> "logs/vllm_$1.log" 2>&1 &
  SERVER=$!
  if ! distill wait "$1" --pid "$SERVER"; then
    echo "--- the $1 server did not start; last lines of logs/vllm_$1.log:" >&2
    tail -n 40 "logs/vllm_$1.log" >&2
    return 1
  fi
  tail -n +"$mark" "logs/vllm_$1.log" | grep "^\[serve\] other processes" || true  # started smaller on a shared GPU
}
stop_server() {  # shut the server down and wait until its processes have released the GPU for the next model
  if [ -z "$SERVER" ]; then return; fi
  kill "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  SERVER=""
  for _ in $(seq 60); do
    if [ -z "$(own_gpu_pids)" ]; then return; fi
    sleep 2
  done
  echo "warning: our processes still hold the GPU after stopping the server: $(own_gpu_pids)" >&2
}
SERVER_DOWN=75   # distill's exit code when its server stopped answering (server.SERVER_DOWN_EXIT)
with_server() {  # ROLE CMD...: run CMD; if the server died under it, restart the server and resume CMD (twice at most)
  local role=$1 restarts=0 rc; shift
  while true; do
    rc=0; "$@" || rc=$?
    if [ "$rc" -ne "$SERVER_DOWN" ] || [ "${BACKEND:-vllm}" != vllm ] || [ "$restarts" -ge 2 ]; then return "$rc"; fi
    restarts=$((restarts + 1))
    echo "== $(date -u +%H:%M) the $role server died; restarting it ($restarts of 2) and resuming" >&2
    stop_server
    start_server "$role"
  done
}
config_value() {  # PYTHON_EXPR over the config `c`, e.g. config_value "c['translate']['compare_limit']"
  python -c "import yaml; c = yaml.safe_load(open('${DISTILL_CONFIG:-config.yaml}', encoding='utf-8')); print($1)"
}
trap stop_server EXIT
trap 'exit 130' INT TERM HUP  # Ctrl-C, kill or a closed tmux window: exit normally so the EXIT trap stops the server
