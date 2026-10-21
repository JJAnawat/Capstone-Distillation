"""Variant B of the CoT pilot: rewrite each thinking-on trace into a short numbered rationale reaching the same answer."""
from distill import server, verify
from distill.generate import load_template, parse_response, read_done_keys
from distill.report import read_jsonl


def eligible(r):
    """Only finished samples with an answer and a complete trace are worth compressing."""
    return (r.get("finish_reason") == "stop" and bool((r.get("answer") or "").strip())
            and bool((r.get("reasoning") or "").strip()) and not r.get("think_unclosed"))


def run(cfg, in_path, out, concurrency=None, backend=None, client=None):
    cc = cfg["compress"]
    ep = server.endpoint(cfg, "teacher", backend)
    system, user, prompt_hash = load_template("compress", cc["prompt_version"])
    sampling = cfg["sampling"]["thinking_off"]
    done = read_done_keys(out)
    todo = [r for r in read_jsonl(in_path) if (r["id"], r.get("sample_idx", 0)) not in done]
    print(f"[compress] {in_path}: {len(done)} already done, {len(todo)} to run "
          f"({sum(map(eligible, todo))} need the model)")
    if not todo:
        return {}
    concurrency = concurrency or ep["concurrency"]
    client = client or server.make_client(ep, concurrency)
    meta = {"cot_method": "compressed", "compress_model": ep["model"], "compress_prompt_hash": prompt_hash}

    def work(r):
        if not eligible(r):  # kept with an empty rationale, so the filter records why the sample was dropped
            return {**r, "rationale": "", **meta}, 0
        kw = {"prompt": r["input"], "reasoning": r["reasoning"], "answer": r["answer"]}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user.format(**kw)}]
        body = server.build_body(ep, messages, sampling, False, r.get("seed", 0), cc["max_tokens"])
        p = parse_response(server.call_api(client, body, cfg["generate"]["max_retries"]), thinking=False)
        rationale = verify.format_steps(p["answer"]) if p["finish_reason"] == "stop" else ""
        return {**r, "rationale": rationale, **meta}, p["answer_tokens"]

    return server.run_pool("compress", work, todo, out, concurrency,
                           lambda r: {"id": r["id"], "sample_idx": r.get("sample_idx", 0)}, input=str(in_path),
                           backend=ep["backend"], model=ep["model"])
