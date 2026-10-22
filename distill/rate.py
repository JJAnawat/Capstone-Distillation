"""Rate each question's reasoning complexity 1-5 with the "Reasoning Complexity Rating Prompt" of DeReason
(Hu et al. 2026, arXiv:2603.11193, Appendix A.1). `select` uses the ratings to order the data from easy to hard."""
import re

from distill import data, server
from distill.generate import load_template, read_done_keys

RATING = re.compile(r"ReasoningRequired\W*([1-5])")


def parse_rating(text):
    """(complexity 1-5, or None when the reply has no rating; the analysis written before the rating)."""
    matches = list(RATING.finditer(text))
    if not matches:
        return None, text.strip()
    return int(matches[-1].group(1)), text[:matches[-1].start()].strip()


def run(cfg, dataset, only=None, limit=None, seed=None, concurrency=None, backend=None, client=None, out=None):
    rc = cfg["rate"]
    ep = server.endpoint(cfg, "teacher", backend)
    seed = cfg["generate"]["seed"] if seed is None else seed
    system, user, prompt_hash = load_template("rate", rc["prompt_version"])
    out = out or data.complexity_path(dataset)
    rows = data.select_rows(cfg, dataset, "open", only, limit, seed)
    done = {rid for rid, _ in read_done_keys(out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[rate] {dataset}: {len(rows)} prompts, {len(rows) - len(todo)} already rated, {len(todo)} to run")
    if not todo:
        return {}
    concurrency = concurrency or ep["concurrency"]
    client = client or server.make_client(ep, concurrency)

    def work(r):
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user.format(question=r["input"])})
        body = server.build_body(ep, messages, {"temperature": rc["temperature"]}, False, seed, rc["max_tokens"])
        j = server.call_api(client, body, cfg["generate"]["max_retries"])
        msg = j["choices"][0]["message"]
        complexity, analysis = parse_rating(msg.get("content") or msg.get("reasoning_content") or "")
        rec = {"id": r["id"], "complexity": complexity, "analysis": analysis, "model": ep["model"],
               "prompt_hash": prompt_hash}
        return rec, (j.get("usage") or {}).get("completion_tokens", 0)

    return server.run_pool("rate", work, todo, out, concurrency, lambda r: {"id": r["id"]}, dataset=dataset,
                           backend=ep["backend"], model=ep["model"])
