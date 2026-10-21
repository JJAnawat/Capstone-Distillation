"""HTTP plumbing shared by every stage: endpoints, request bodies, retries, a thread-pool runner, and starting or
waiting for a local vLLM server."""
import json
import os
import random
import shlex
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
    <out>.errors.jsonl, tagged with `key(item)`, and are retried on the next run. Returns the run summary."""
    prog = Progress(stage, len(todo), out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(concurrency) as ex:
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
    return prog.finish(**info)


def serve_command(cfg, role):
    """`vllm serve` for `role` with the flags from config, plus any in $VLLM_EXTRA_ARGS (used by scripts/bench.sh)."""
    ep = endpoint(cfg, role, "vllm")
    return ["vllm", "serve", ep["model"], *map(str, ep["serve_args"]), *shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))]


def serve(cfg, role):
    """Replace this process with `vllm serve` for `role` (GPU VM only)."""
    cmd = serve_command(cfg, role)
    print("[serve]", " ".join(cmd), flush=True)
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
