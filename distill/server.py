"""HTTP plumbing shared by every stage: endpoints, request bodies, retries, a thread-pool runner, and starting or
waiting for a local vLLM server."""
import json
import math
import os
import random
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}
# Sampling values that change nothing. They are left out on OpenRouter, where require_parameters would otherwise route
# only to providers that accept them; a local vLLM server always gets every field.
NEUTRAL = {"min_p": 0.0, "repetition_penalty": 1.0}


def endpoint(cfg, role, backend=None):
    """Settings for `role` (teacher|judge) on `backend` (vllm|openrouter), with `backend` and `api_base` filled in."""
    backend = backend or cfg["backend"]
    ep = {**cfg[role][backend], "backend": backend}
    if backend == "openrouter":
        ep["api_base"] = cfg["openrouter_api_base"]
    return ep


def server_root(api_base):
    """http://host:port of an OpenAI-style base URL; vLLM serves /health and /tokenize outside /v1."""
    return api_base.rstrip("/").removesuffix("/v1")


def make_client(ep, concurrency=None):
    concurrency = concurrency or ep["concurrency"]
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"} if ep["backend"] == "openrouter" else {}
    # httpx allows 100 connections by default, which would leave most of a large thread pool waiting.
    return httpx.Client(base_url=ep["api_base"], headers=headers, timeout=httpx.Timeout(ep["timeout_s"], connect=30),
                        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency))


def build_body(ep, messages, sampling, thinking, seed, max_tokens, **extra):
    """Chat-completions body. vLLM takes the thinking switch as a chat-template argument; OpenRouter has `reasoning`."""
    if ep["backend"] != "vllm":
        sampling = {k: v for k, v in sampling.items() if NEUTRAL.get(k, object()) != v}
    body = {"model": ep["model"], "messages": messages, "max_tokens": max_tokens, "seed": seed, **sampling, **extra}
    if ep["backend"] == "vllm":
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    else:
        body["reasoning"] = {"enabled": thinking}
        body["provider"] = ep["provider"]
    return body


def call_api(client, body, max_retries):
    """POST with exponential backoff on rate limits, server errors, timeouts and in-body errors."""
    for attempt in range(max_retries + 1):
        try:
            r = client.post("/chat/completions", json=body)
            if r.status_code in RETRYABLE:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}: {r.text[:500]}")  # not retryable
            j = r.json()
            if "error" in j or not j.get("choices") or j["choices"][0].get("error"):
                raise RuntimeError(f"error in body: {str(j.get('error') or j.get('choices'))[:300]}")
            return j
        except (httpx.TransportError, RuntimeError, json.JSONDecodeError):
            if attempt == max_retries:
                raise
            time.sleep(min(60, 2 ** attempt) + random.random())


class Progress:
    """Status line every `every_s` seconds (done/total, tokens/s, ETA); `finish` appends a run summary to
    <out>.runs.jsonl so throughput can be read back after the run."""

    def __init__(self, stage, total, out, every_s=60):
        self.stage, self.total, self.out, self.every_s = stage, total, out, every_s
        self.start = self.last = time.time()
        self.done = self.failed = self.tokens = 0

    def update(self, tokens=0, failed=False):
        self.failed += failed
        self.done += not failed
        self.tokens += tokens
        if time.time() - self.last >= self.every_s:
            self.last = time.time()
            elapsed, seen = self.last - self.start, self.done + self.failed
            print(f"[{self.stage}] {seen}/{self.total} ({self.failed} failed), {self.tokens / elapsed:,.0f} tok/s, "
                  f"ETA {(self.total - seen) * elapsed / seen / 60:,.0f} min", flush=True)

    def finish(self, **info):
        elapsed = max(time.time() - self.start, 1e-9)
        summary = {"stage": self.stage, **info, "done": self.done, "failed": self.failed,
                   "completion_tokens": self.tokens, "seconds": round(elapsed, 1),
                   "tok_per_s": round(self.tokens / elapsed, 1),
                   "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with open(f"{self.out}.runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        return summary


def run_pool(stage, work, todo, out, concurrency, key, **info):
    """Run `work(item) -> (record, completion_tokens) | None` over `todo` on a thread pool. Each record is appended to
    `out` as soon as it finishes, so an interrupted run loses only requests in flight. Failures go to
    <out>.errors.jsonl, tagged with `key(item)`, and are retried on the next run. Returns the run summary.
    On Ctrl-C the queued items are dropped and the interrupt propagates at once; `cli.main` then exits without joining
    the threads still waiting on a reply (the server may itself be shutting down and never answer)."""
    prog = Progress(stage, len(todo), out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ex = ThreadPoolExecutor(concurrency)
    try:
        with open(out, "a", encoding="utf-8") as fout:
            futures = {ex.submit(work, item): item for item in todo}
            for fut in as_completed(futures):
                if fut.exception():
                    with open(f"{out}.errors.jsonl", "a", encoding="utf-8") as ferr:
                        ferr.write(json.dumps({**key(futures[fut]), "error": str(fut.exception())[:500]}) + "\n")
                    prog.update(failed=True)
                elif (result := fut.result()) is not None:
                    rec, tokens = result
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    prog.update(tokens)
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return prog.finish(**info)


def serve_command(cfg, role):
    """`vllm serve` for `role` with the flags from config, plus any in $DISTILL_SERVE_ARGS (used by scripts/bench.sh)."""
    ep = endpoint(cfg, role, "vllm")
    return ["vllm", "serve", ep["model"], *map(str, ep["serve_args"]),
            *shlex.split(os.environ.get("DISTILL_SERVE_ARGS", ""))]


def serve_env(cfg, role):
    """Environment variables the server needs, from config (`serve_env`); they override the caller's environment."""
    return {k: str(v) for k, v in (endpoint(cfg, role, "vllm").get("serve_env") or {}).items()}


UTIL_FLAG = "--gpu-memory-utilization"


def requested_utilization(cmd):
    """The last --gpu-memory-utilization in `cmd`, as `--flag=x` or `--flag x` (vLLM's default 0.92 if absent)."""
    util = 0.92
    for i, a in enumerate(cmd):
        if a.startswith(UTIL_FLAG + "="):
            util = float(a.split("=", 1)[1])
        elif a == UTIL_FLAG and i + 1 < len(cmd):
            util = float(cmd[i + 1])
    return util


def with_utilization(cmd, util):
    """`cmd` with every --gpu-memory-utilization replaced by one set to `util`."""
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] == UTIL_FLAG:
            i += 2
            continue
        if not cmd[i].startswith(UTIL_FLAG + "="):
            out.append(cmd[i])
        i += 1
    return out + [f"{UTIL_FLAG}={util}"]


def gpu_memory_mib():
    """(free, total) MiB of the GPU from nvidia-smi, or None where there is none (the laptop)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    free, total = map(float, out.splitlines()[0].split(","))
    return free, total


def fit_gpu(cfg, role, cmd, memory=gpu_memory_mib, sleep=time.sleep):
    """`cmd` with --gpu-memory-utilization lowered so the server fits beside other users' processes on a shared GPU,
    leaving `gpu_share.headroom_gib` free for them. Only the KV cache shrinks, so fewer requests run at once. Waits up
    to `gpu_share.wait_s` while not even the role's `min_gpu_gib` fits, then raises."""
    share, min_gib = cfg.get("gpu_share"), endpoint(cfg, role, "vllm").get("min_gpu_gib")
    if not share or not min_gib:
        return cmd
    requested, deadline = requested_utilization(cmd), time.time() + share["wait_s"]
    while (mem := memory()) is not None:
        free, total = mem
        usable = free / 1024 - share["headroom_gib"]
        if usable >= min_gib:
            util = min(requested, math.floor(usable * 1024 / total * 100) / 100)
            if util == requested:
                return cmd
            print(f"[serve] other processes use {(total - free) / 1024:.1f} GiB of the GPU: {UTIL_FLAG} {util} "
                  f"instead of {requested} (a smaller KV cache)", flush=True)
            return with_utilization(cmd, util)
        need = min_gib + share["headroom_gib"]
        if time.time() >= deadline:
            raise RuntimeError(f"only {free / 1024:.1f} GiB of GPU memory free after waiting {share['wait_s']}s; "
                               f"the {role} needs {need:.1f} GiB (min_gpu_gib + headroom)")
        print(f"[serve] waiting for GPU memory: {free / 1024:.1f} GiB free, the {role} needs {need:.1f} GiB", flush=True)
        sleep(share["poll_s"])
    return cmd


def serve(cfg, role):
    """Replace this process with `vllm serve` for `role` (GPU VM only), sized to the free GPU memory (`fit_gpu`)."""
    cmd, env = fit_gpu(cfg, role, serve_command(cfg, role)), serve_env(cfg, role)
    print("[serve]", " ".join(f"{k}={v}" for k, v in env.items()), " ".join(cmd), flush=True)
    os.environ.update(env)
    os.execvp(cmd[0], cmd)


def alive(pid):
    """Whether process `pid` is still running (Linux; a zombie counts as dead)."""
    if os.name == "nt":  # os.kill would terminate the process on Windows; the servers only run on Linux
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z")


def wait_ready(cfg, role, client=None, poll_s=5, pid=None):
    """Block until the local vLLM server answers /health and serves `role`'s model. Raise after startup_timeout_s,
    or as soon as the server process `pid` exits (a failed start would otherwise only show after the timeout)."""
    ep = endpoint(cfg, role, "vllm")
    client = client or httpx.Client(timeout=10)
    root = server_root(ep["api_base"])
    deadline = time.time() + ep["startup_timeout_s"]
    while True:
        if pid and not alive(pid):
            raise RuntimeError(f"the {role} server (pid {pid}) exited during startup; see logs/vllm_{role}.log")
        try:
            if client.get(f"{root}/health").status_code == 200:
                served = [m["id"] for m in client.get(f"{ep['api_base'].rstrip('/')}/models").json()["data"]]
                if ep["model"] not in served:
                    raise RuntimeError(f"{root} serves {served}, not {ep['model']}: is another server still running?")
                print(f"[wait] {role} server ready: {ep['model']}", flush=True)
                return
        except httpx.TransportError:
            pass
        if time.time() > deadline:
            raise TimeoutError(f"{role} server at {root} not ready after {ep['startup_timeout_s']}s")
        time.sleep(poll_s)
