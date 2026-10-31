"""Query the teacher N times per prompt and append one JSONL record per sample."""
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from distill import data, server, verify


def load_template(name, version):
    text = Path(f"prompts/{name}_{version}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")  # same hash on Windows
    system, user = text.split("### user\n", 1)
    system = system.removeprefix("### system\n").strip()
    return system, user.strip(), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_templates(mode, version):
    """(template for open prompts, template for closed prompts); the second is `{mode}_closed_{version}` if it exists."""
    base = load_template(mode, version)
    if Path(f"prompts/{mode}_closed_{version}.txt").exists():
        return base, load_template(f"{mode}_closed", version)
    return base, base


def prompt_version_for(cfg, dataset, name, default):
    """The dataset's own template version (`prompt_versions` in its config, e.g. th1 for a Thai set), else `default`."""
    return ((cfg["datasets"].get(dataset) or {}).get("prompt_versions") or {}).get(name, default)


def step_limit(cfg, level):
    """(max steps, max words) for a rationale, from the prompt's complexity rating (DeReason's step ranges)."""
    c = cfg["cot"]
    steps = data.by_complexity(c["max_steps_by_complexity"], level, c["default_max_steps"])
    return steps, steps * c["words_per_step"]


def sample_seed(seed, rid, k):
    """Seed of sample k of a prompt: distinct for every (prompt, sample) and reproducible."""
    return int(hashlib.sha256(f"{seed}:{rid}:{k}".encode()).hexdigest()[:8], 16)


def parse_response(j, thinking=True, cot=False):
    """Split a chat completion into reasoning, rationale (cot mode) and answer, plus token counts."""
    choice = j["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""  # OpenRouter / vLLM
    if not thinking and not content.strip():  # vLLM's reasoning parser can file a thinking-off reply as reasoning
        content, reasoning = reasoning, ""
    think_unclosed = False
    # Fallback for providers that leave <think>...</think> inline in content.
    m = re.search(r"<think>(.*?)</think>", content, re.S)
    if m:
        reasoning = reasoning or m.group(1)
        content = content[:m.start()] + content[m.end():]
    elif "</think>" in content:  # opening tag was part of the chat template
        head, content = content.split("</think>", 1)
        reasoning = reasoning or head
    elif "<think>" in content:
        head, tail = content.split("<think>", 1)
        reasoning, content, think_unclosed = reasoning or tail, head, True
    rationale = ""
    if cot and (t := re.search(r"<reasoning>(.*?)</reasoning>", content, re.S)):
        rationale, content = verify.format_steps(t.group(1)), content[:t.start()] + content[t.end():]
    usage = j.get("usage") or {}
    completion = usage.get("completion_tokens", 0)
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return {"reasoning": reasoning.strip(), "rationale": rationale, "answer": content.strip(),
            "finish_reason": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens", 0),
            "reasoning_tokens": reasoning_tokens, "answer_tokens": completion - reasoning_tokens,
            "think_unclosed": think_unclosed, "provider": j.get("provider"), "cost": usage.get("cost")}


def count_tokens(client, ep, text):
    """Token count from the vLLM server's /tokenize endpoint (vLLM's usage has no reasoning-token count)."""
    r = client.post(f"{server.server_root(ep['api_base'])}/tokenize",
                    json={"model": ep["model"], "prompt": text, "add_special_tokens": False})
    r.raise_for_status()
    return r.json()["count"]


def read_done_keys(path):
    """(id, sample_idx) of the records already in `path`; records from before multi-sampling count as sample 0."""
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(j["id"], j.get("sample_idx", 0)) for j in map(json.loads, filter(str.strip, f))}


def run(cfg, dataset, mode="open", thinking=True, limit=None, seed=None, max_tokens=None, out=None, n=None,
        concurrency=None, max_cost=None, include_reference=False, model=None, prompt_version=None, only=None,
        backend=None, client=None):
    if mode == "cot" and thinking:
        raise ValueError("--mode cot asks for a short <reasoning> section and needs --thinking off; "
                         "thinking-on output gets its short rationale from `distill compress`")
    g = cfg["generate"]
    ep = server.endpoint(cfg, "teacher", backend)
    ep["model"] = model or ep["model"]
    seed = g["seed"] if seed is None else seed
    max_tokens = max_tokens or (cfg["cot"]["max_tokens"] if mode == "cot" else g["max_tokens"])
    prompt_version = prompt_version or prompt_version_for(cfg, dataset, mode, g["prompt_versions"][mode])
    sampling = cfg["sampling"]["thinking_on" if thinking else "thinking_off"]
    open_tmpl, closed_tmpl = load_templates(mode, prompt_version)

    rows = data.select_rows(cfg, dataset, mode, only, limit, seed)
    levels = data.load_complexity(dataset)  # rated before generation; drives N and the step limit
    n_for = (lambda r: n) if n else (lambda r: data.by_complexity(g["n_by_complexity"], levels.get(r["id"]), g["n"]))
    done = read_done_keys(out)
    # Grouped by prompt, so a prompt's samples run together and share the server's prefix cache.
    todo = [(r, k) for r in rows for k in range(n_for(r)) if (r["id"], k) not in done]
    planned = sum(map(n_for, rows))
    print(f"[generate] {dataset} mode={mode} thinking={thinking} n={n or 'by complexity'}: {len(rows)} prompts, "
          f"{planned} samples, {planned - len(todo)} already done, {len(todo)} to run")
    state = {"spent": 0.0, "ok": 0, "failed": 0}
    if not todo:
        return state
    concurrency = concurrency or ep["concurrency"]
    client = client or server.make_client(ep, concurrency)
    price = ep.get("price_per_mtok")
    lock, stop = threading.Lock(), threading.Event()

    def work(item):
        r, k = item
        if stop.is_set():
            return None
        system, user, prompt_hash = closed_tmpl if r.get("closed") else open_tmpl
        max_steps, max_words = step_limit(cfg, levels.get(r["id"]))
        kw = {"prompt": r["input"], "reference": r["reference"], "max_steps": max_steps, "max_words": max_words}
        messages = [{"role": "system", "content": system.format(**kw)}, {"role": "user", "content": user.format(**kw)}]
        s = sample_seed(seed, r["id"], k)
        body = server.build_body(ep, messages, sampling, thinking, s, max_tokens)
        p = parse_response(server.call_api(client, body, g["max_retries"]), thinking, cot=mode == "cot")
        if ep["backend"] == "vllm":
            p["provider"] = "local-vllm"
            if p["reasoning"] and not p["reasoning_tokens"]:
                p["reasoning_tokens"] = count_tokens(client, ep, p["reasoning"])
                p["answer_tokens"] = max(0, p["answer_tokens"] - p["reasoning_tokens"])
        if p["cost"] is None:
            p["cost"] = (p["prompt_tokens"] * price["prompt"]
                         + (p["reasoning_tokens"] + p["answer_tokens"]) * price["completion"]) / 1e6 if price else 0.0
        rec = {"id": r["id"], "sample_idx": k, "source_dataset": dataset, "input": r["input"],
               **{f: r[f] for f in ("lang", "input_en", "translation") if r.get(f)},  # Thai sets: carried to final rows
               "closed": bool(r.get("closed")), **({"gold": r["gold"]} if r.get("closed") else {}), **p,
               **({"max_steps": max_steps} if mode == "cot" else {}),
               "cot_method": "prompted" if mode == "cot" else None, "model": ep["model"], "mode": mode,
               "thinking": thinking, "prompt_version": prompt_version, "prompt_hash": prompt_hash, **sampling,
               "seed": s, "max_tokens": max_tokens,
               "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if include_reference:
            rec["reference_response"] = r["reference"]
        with lock:
            state["spent"] += p["cost"]
            if max_cost and state["spent"] >= max_cost and not stop.is_set():
                print(f"[generate] budget ${max_cost} reached; finishing in-flight requests and stopping")
                stop.set()
        return rec, p["reasoning_tokens"] + p["answer_tokens"]

    summary = server.run_pool("generate", work, todo, out, concurrency,
                              lambda item: {"id": item[0]["id"], "sample_idx": item[1]}, dataset=dataset, mode=mode,
                              thinking=thinking, n=n or "by complexity", backend=ep["backend"], model=ep["model"])
    state["ok"], state["failed"] = summary["done"], summary["failed"]
    print(f"[generate] done: {state['ok']} written, {state['failed']} failed, spent ${state['spent']:.4f}, "
          f"{summary['tok_per_s']:,.0f} tok/s")
    return state
