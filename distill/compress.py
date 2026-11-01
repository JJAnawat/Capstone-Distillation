"""Variant B of the CoT pilot: rewrite each thinking-on trace into a short numbered rationale reaching the same answer.
Translated rows (`lang`) get a template that names the target language: Qwen thinks in English, and "the language of
the question" alone let a fifth of the Thai rationales come out in English (Thai night 1)."""
import os
from pathlib import Path

from distill import data, server, verify
from distill.generate import load_template, parse_response, read_done_keys, step_limit
from distill.report import detect_language, expected_language, read_jsonl, write_jsonl


def eligible(r):
    """Only finished samples with an answer and a complete trace are worth compressing."""
    return (r.get("finish_reason") == "stop" and bool((r.get("answer") or "").strip())
            and bool((r.get("reasoning") or "").strip()) and not r.get("think_unclosed"))


def wrong_language(r, min_chars):
    """Whether a translated row's rationale is in another language than its target (English steps for Thai)."""
    if not r.get("lang") or not (r.get("rationale") or "").strip():
        return False
    found = detect_language(r["rationale"], min_chars)
    return found is not None and found != expected_language(r, min_chars)


def drop_wrong_language(cfg, out):
    """Remove the records of `out` whose rationale is in the wrong language, so the next run compresses them again.
    The file is rewritten through a temporary file, so no key ever appears twice. Returns how many were removed."""
    if not Path(out).exists():
        return 0
    rows = read_jsonl(out)
    keep = [r for r in rows if not wrong_language(r, cfg["filter"]["language"]["min_chars"])]
    if len(keep) < len(rows):
        write_jsonl(f"{out}.tmp", keep)
        os.replace(f"{out}.tmp", out)
    return len(rows) - len(keep)


def run(cfg, in_path, out, concurrency=None, backend=None, client=None, redo_wrong_language=False):
    cc = cfg["compress"]
    ep = server.endpoint(cfg, "teacher", backend)
    templates = {False: load_template("compress", cc["prompt_version"]),
                 True: load_template("compress", cc.get("translated_prompt_version") or cc["prompt_version"])}
    language = (cfg.get("translate") or {}).get("target", "")
    sampling = cfg["sampling"]["thinking_off"]
    if redo_wrong_language:
        print(f"[compress] {out}: {drop_wrong_language(cfg, out)} rationales in the wrong language will be redone")
    done = read_done_keys(out)
    todo = [r for r in read_jsonl(in_path) if (r["id"], r.get("sample_idx", 0)) not in done]
    print(f"[compress] {in_path}: {len(done)} already done, {len(todo)} to run "
          f"({sum(map(eligible, todo))} need the model)")
    if not todo:
        return {}
    concurrency = concurrency or ep["concurrency"]
    client = client or server.make_client(ep, concurrency)
    levels = {ds: data.load_complexity(ds) for ds in {r["source_dataset"] for r in todo}}

    def work(r):
        max_steps, max_words = step_limit(cfg, levels[r["source_dataset"]].get(r["id"]))
        system, user, prompt_hash = templates[bool(r.get("lang"))]
        meta = {"cot_method": "compressed", "compress_model": ep["model"], "compress_prompt_hash": prompt_hash}
        if not eligible(r):  # kept with an empty rationale, so the filter records why the sample was dropped
            return {**r, "rationale": "", "max_steps": max_steps, **meta}, 0
        kw = {"prompt": r["input"], "reasoning": r["reasoning"], "answer": r["answer"], "max_steps": max_steps,
              "max_words": max_words, "language": language}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user.format(**kw)}]
        body = server.build_body(ep, messages, sampling, False, r.get("seed", 0), cc["max_tokens"])
        p = parse_response(server.call_api(client, body, cfg["generate"]["max_retries"]), thinking=False)
        rationale = verify.format_steps(p["answer"]) if p["finish_reason"] == "stop" else ""
        return {**r, "rationale": rationale, "max_steps": max_steps, **meta}, p["answer_tokens"]

    return server.run_pool("compress", work, todo, out, concurrency,
                           lambda r: {"id": r["id"], "sample_idx": r.get("sample_idx", 0)}, input=str(in_path),
                           backend=ep["backend"], model=ep["model"])
