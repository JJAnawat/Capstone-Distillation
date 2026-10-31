"""Variant B of the CoT pilot: rewrite each thinking-on trace into a short numbered rationale reaching the same answer."""
from distill import data, server, verify
from distill.generate import load_template, parse_response, prompt_version_for, read_done_keys, step_limit
from distill.report import read_jsonl


def eligible(r):
    """Only finished samples with an answer and a complete trace are worth compressing."""
    return (r.get("finish_reason") == "stop" and bool((r.get("answer") or "").strip())
            and bool((r.get("reasoning") or "").strip()) and not r.get("think_unclosed"))


def run(cfg, in_path, out, concurrency=None, backend=None, client=None, prompt_version=None):
    cc = cfg["compress"]
    ep = server.endpoint(cfg, "teacher", backend)
    sampling = cfg["sampling"]["thinking_off"]
    done = read_done_keys(out)
    todo = [r for r in read_jsonl(in_path) if (r["id"], r.get("sample_idx", 0)) not in done]
    print(f"[compress] {in_path}: {len(done)} already done, {len(todo)} to run "
          f"({sum(map(eligible, todo))} need the model)")
    if not todo:
        return {}
    concurrency = concurrency or ep["concurrency"]
    client = client or server.make_client(ep, concurrency)
    datasets = {r["source_dataset"] for r in todo}
    levels = {ds: data.load_complexity(ds) for ds in datasets}
    templates = {ds: load_template("compress", prompt_version
                                   or prompt_version_for(cfg, ds, "compress", cc["prompt_version"])) for ds in datasets}

    def work(r):
        max_steps, max_words = step_limit(cfg, levels[r["source_dataset"]].get(r["id"]))
        system, user, prompt_hash = templates[r["source_dataset"]]
        meta = {"cot_method": "compressed", "compress_model": ep["model"], "compress_prompt_hash": prompt_hash}
        if not eligible(r):  # kept with an empty rationale, so the filter records why the sample was dropped
            return {**r, "rationale": "", "max_steps": max_steps, **meta}, 0
        kw = {"prompt": r["input"], "reasoning": r["reasoning"], "answer": r["answer"], "max_steps": max_steps,
              "max_words": max_words}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user.format(**kw)}]
        body = server.build_body(ep, messages, sampling, False, r.get("seed", 0), cc["max_tokens"])
        p = parse_response(server.call_api(client, body, cfg["generate"]["max_retries"]), thinking=False)
        rationale = verify.format_steps(p["answer"]) if p["finish_reason"] == "stop" else ""
        return {**r, "rationale": rationale, "max_steps": max_steps, **meta}, p["answer_tokens"]

    return server.run_pool("compress", work, todo, out, concurrency,
                           lambda r: {"id": r["id"], "sample_idx": r.get("sample_idx", 0)}, input=str(in_path),
                           backend=ep["backend"], model=ep["model"])
