"""Grade each sample's rationale and answer with the judge model (pointwise rubric, scale 1-9), and check the judge
against human preferences on HelpSteer2."""
import json
import math
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from distill import data, server, verify
from distill.generate import load_template, read_done_keys
from distill.report import iter_jsonl, judge_path

# Scores are single digits, so the score is one token and its alternatives' probabilities give an expected score.
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["critique", "score"],
          "properties": {"critique": {"type": "string"}, "score": {"type": "integer", "enum": list(range(1, 10))}}}
DIGITS = {str(d) for d in range(1, 10)}


def judge_config(cfg, thinking=None, prompt_version=None):
    """The judge settings, with thinking or the rubric version overridden when asked; thinking needs a larger token
    budget."""
    jc = dict(cfg["judge"])
    if thinking is not None:
        jc["thinking"] = thinking
    if prompt_version:
        jc["prompt_version"] = prompt_version
    if jc["thinking"]:
        jc["max_tokens"] = jc["max_tokens_thinking"]
    return jc


def candidates(rows, all_samples=False):
    """Samples worth judging: all samples of an open prompt; for a closed prompt only those whose final answer passes
    the rule check, or all of them when none passes (the gold may be right without being exactly matchable).
    `all_samples` judges every sample, to measure how well the judge separates right from wrong answers."""
    if all_samples:
        return list(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["id"]].append(r)
    out = []
    for group in groups.values():
        passing = [r for r in group if r.get("closed") and verify.is_correct(r["answer"], r.get("gold") or [])]
        out += passing or group
    return out


def reference_for(prompt_row, use_reference):
    """Gold answers if the dataset has them, else the dataset's own response when it is rated highly enough."""
    if not use_reference or not prompt_row:
        return None
    if prompt_row.get("gold"):
        return "\n".join(f"- {g}" for g in prompt_row["gold"])
    return prompt_row["reference"] if prompt_row.get("hint_ok") else None


def judge_messages(system, user, prompt, reference, rationale, answer):
    """Shared parts first (rubric, question, reference) and the graded response last, so the server computes the
    prefix once for all samples of a prompt."""
    ref = f"[Reference answer (may be imperfect)]\n{reference}\n\n" if reference else ""
    response = (f"[Reasoning]\n{rationale}\n\n" if rationale else "") + f"[Answer]\n{answer}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(prompt=prompt, reference_block=ref, response=response)}]


def judge_body(ep, jc, messages):
    extra = {"response_format": {"type": "json_schema", "json_schema": {"name": "grade", "strict": True,
                                                                        "schema": SCHEMA}}}
    if ep.get("logprobs"):
        extra |= {"logprobs": True, "top_logprobs": 10}
    return server.build_body(ep, messages, {"temperature": jc["temperature"]}, jc["thinking"], 0, jc["max_tokens"],
                             **extra)


def expected_score(logprobs, score):
    """Probability-weighted mean of the digit alternatives at the score token; `score` itself without logprobs."""
    for t in reversed((logprobs or {}).get("content") or []):
        if t["token"].strip() == str(score):
            probs = defaultdict(float)
            for alt in t.get("top_logprobs") or []:
                if alt["token"].strip() in DIGITS:
                    probs[int(alt["token"].strip())] += math.exp(alt["logprob"])
            total = sum(probs.values())
            return sum(d * p for d, p in probs.items()) / total if total else float(score)
    return float(score)


def parse_grade(j):
    """(score, expected score, critique); raises ValueError when the reply holds no valid score."""
    choice = j["choices"][0]
    content = re.sub(r"<\|channel>.*?<channel\|>", "", choice["message"].get("content") or "", flags=re.S)
    m = re.search(r"\{.*\}", content, re.S)
    grade = json.loads(m.group(0)) if m else {}
    score = grade.get("score")
    if not isinstance(score, int) or not 1 <= score <= 9:
        raise ValueError(f"no valid score in judge reply: {content[:200]!r}")
    return score, expected_score(choice.get("logprobs"), score), str(grade.get("critique", ""))


def run(cfg, files, concurrency=None, backend=None, client=None, all_samples=False, thinking=None,
        prompt_version=None):
    jc = judge_config(cfg, thinking, prompt_version)
    ep = server.endpoint(cfg, "judge", backend)
    system, user, prompt_hash = load_template("judge", jc["prompt_version"])
    concurrency = concurrency or ep["concurrency"]
    prompts = {}  # dataset -> {id: prepared row}, for gold answers and references
    for path in files:
        # The judge never sees the raw thinking, and dropping it keeps long-trace files small in memory.
        rows = [{k: v for k, v in r.items() if k != "reasoning"} for r in iter_jsonl(path)]
        for ds in {r["source_dataset"] for r in rows} - prompts.keys():
            prompts[ds] = {p["id"]: p for p in data.load_prompts(cfg, ds)}
        out = judge_path(path)
        done = read_done_keys(out)
        todo = [r for r in candidates(rows, all_samples) if (r["id"], r.get("sample_idx", 0)) not in done]
        print(f"[judge] {path}: {len(done)} already judged, {len(todo)} to run")
        if not todo:
            continue
        client = client or server.make_client(ep, concurrency)

        def work(r):
            ref = reference_for(prompts[r["source_dataset"]].get(r["id"]), jc["use_reference"])
            messages = judge_messages(system, user, r["input"], ref, r.get("rationale"), r["answer"])
            j = server.call_api(client, judge_body(ep, jc, messages), jc["max_retries"])
            score, expected, critique = parse_grade(j)
            rec = {"id": r["id"], "sample_idx": r.get("sample_idx", 0), "score": score,
                   "expected_score": round(expected, 4), "critique": critique, "reference_used": ref is not None,
                   "judge_model": ep["model"], "judge_prompt_hash": prompt_hash, "judge_thinking": jc["thinking"]}
            return rec, (j.get("usage") or {}).get("completion_tokens", 0)

        server.run_pool("judge", work, todo, out, concurrency,
                        lambda r: {"id": r["id"], "sample_idx": r.get("sample_idx", 0)}, input=str(path),
                        backend=ep["backend"], model=ep["model"])


def agreement(results):
    """Share of (judge_a, judge_b, human_a, human_b) pairs where the judge prefers the response humans rated higher;
    a judge tie counts half."""
    return sum(0.5 if ja == jb else float((ja > jb) == (ha > hb)) for ja, jb, ha, hb in results) / len(results)


def validate(cfg, n_pairs, out="reports/judge_validation.md", concurrency=None, backend=None, client=None, seed=0,
             thinking=None, prompt_version=None):
    """Judge both responses of HelpSteer2 validation prompts whose human helpfulness ratings differ (no reference,
    no rationale) and append the agreement with the humans, and what it cost, to `out`. The validation split is never
    generated from."""
    from datasets import load_dataset

    jc = judge_config(cfg, thinking, prompt_version)
    ep = server.endpoint(cfg, "judge", backend)
    system, user, prompt_hash = load_template("judge", jc["prompt_version"])
    by_prompt = defaultdict(list)
    for r in load_dataset(cfg["datasets"]["helpsteer2"]["hf_id"], split="validation"):
        if "<extra_id_1>" not in r["prompt"]:
            by_prompt[r["prompt"]].append(r)
    pairs = [(p, rs[0], rs[1]) for p, rs in by_prompt.items()
             if len(rs) >= 2 and rs[0]["helpfulness"] != rs[1]["helpfulness"]]
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:n_pairs]
    client = client or server.make_client(ep, concurrency)

    tokens = []

    def grade(pair):
        prompt, a, b = pair
        try:
            replies = [server.call_api(client, judge_body(ep, jc, judge_messages(
                system, user, prompt, None, None, r["response"])), jc["max_retries"]) for r in (a, b)]
            sa, sb = (parse_grade(j)[1] for j in replies)
        except Exception as e:  # a failed pair is left out rather than aborting the validation
            print(f"[judge] validation pair skipped: {str(e)[:200]}")
            return None
        tokens.extend((j.get("usage") or {}).get("completion_tokens", 0) for j in replies)
        return sa, sb, a["helpfulness"], b["helpfulness"]

    start = time.time()
    ex = ThreadPoolExecutor(concurrency or ep["concurrency"])
    try:
        results = [r for r in ex.map(grade, pairs) if r]
    finally:  # all done, or Ctrl-C: drop the queued pairs instead of waiting for them (see server.run_pool)
        ex.shutdown(wait=False, cancel_futures=True)
    if not results:
        raise RuntimeError("no validation pair could be judged")
    ties = sum(ja == jb for ja, jb, _, _ in results) / len(results)
    row = (f"| {datetime.now(timezone.utc):%Y-%m-%d} | {ep['model']} ({ep['backend']}) | "
           f"{'on' if jc['thinking'] else 'off'} | {prompt_hash} | {len(results)} | {agreement(results):.1%} | "
           f"{ties:.1%} | {sum(tokens) / len(tokens):,.0f} | {(time.time() - start) / 60:,.1f} |")
    path = Path(out)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Judge validation\n\nHelpSteer2 validation prompts whose two responses got different human "
                        "helpfulness ratings. Agreement: the judge's (expected) score orders the pair like the humans; "
                        "a judge tie counts half. Cost: mean completion tokens per judge call, and wall-clock minutes."
                        "\n\n| date | judge | thinking | prompt | pairs | agreement | judge ties | tokens/call "
                        "| minutes |\n|---|---|---|---|---:|---:|---:|---:|---:|\n", encoding="utf-8")
    with open(path, "a", encoding="utf-8") as f:
        f.write(row + "\n")
    print(f"[judge] validation: {row}")
    return agreement(results)
