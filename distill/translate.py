"""Thai sets: translate questions the English run never used, check each translation (rule flags and a checker
model), and build the Thai prepared file that the rest of the pipeline distills. The in-process translators and metric
checkers are in `inproc`; the translator and checker comparisons are in `compare`."""
import json
import re
from collections import Counter
from pathlib import Path

from distill import data, judge, report, server, verify
from distill.generate import load_template, parse_response, read_done_keys, sample_seed

TRANSLATE_DIR = Path("data/translate")
SCALE = {str(d) for d in range(1, 6)}
CHECK_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["english_only", "critique", "fluency", "meaning"],  # meaning last: see `check`
                "properties": {"english_only": {"type": "boolean"}, "critique": {"type": "string"},
                               "fluency": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                               "meaning": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}}
REFUSAL = re.compile(r"\b(?:I'm sorry|I am sorry|I can(?:not|'t)|I won't|I will not)\b|ขออภัย|ไม่สามารถ(?:แปล|ช่วย)", re.I)
URL = re.compile(r"https?://\S+")
CODE = re.compile(r"```.*?```", re.S)
OPTION = re.compile(r"^([A-H])\.\s*(.+?)\s*$", re.M)
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def stem(path):
    return str(path).removesuffix(".jsonl")


def by_id(path):
    """{id: record} of a JSONL file, or {} if it does not exist."""
    return {r["id"]: r for r in report.iter_jsonl(path)} if Path(path).exists() else {}


# ---------------------------------------------------------------- translate

def source_rows(cfg, dataset, limit=None):
    """(rows, pool size, used ids) for a Thai set: a seeded stratified sample of its source dataset without the ids in
    the source's ratings. Every question the English run used was rated first, so these are all new. The sample is
    nested: a smaller limit (the comparison's) gives the first rows of a larger one (the full run's)."""
    src = data.translate_source(cfg, dataset)
    if not src:
        raise ValueError(f"{dataset} has no `translate:` block in the config")
    used = data.load_complexity(src["from"])
    if not used:
        raise FileNotFoundError(
            f"{data.complexity_path(src['from'])} is missing or empty, so the questions the English run used cannot be "
            f"excluded. Restore it: hf download {cfg['hub']['repo']} --repo-type dataset --local-dir . "
            f"--include 'data/prepared/*'")
    limit = limit or src["limit"]
    pool = data.select_rows(cfg, src["from"], "open", src["only"], exclude=set(used))
    if len(pool) < limit:
        raise ValueError(f"{dataset}: only {len(pool)} unused {src['from']} prompts left for a limit of {limit}")
    return data.sample(pool, limit, cfg["generate"]["seed"]), len(pool), len(used)


def record(cfg, dataset, r, meta, text, finish, reference, tokens):
    """A translation record in the schema every later step reads, with its rule flags."""
    rec = {"id": r["id"], "dataset": dataset, "source_dataset": r["source_dataset"], "input_en": r["input"],
           "input": text, "finish_reason": finish, "reference_en": r.get("reference"), "reference": reference,
           "strata": r.get("strata"), "closed": bool(r.get("closed")), "gold": r.get("gold") or [],
           "hint_ok": bool(r.get("hint_ok")), "translator": meta, "tokens": tokens}
    rec["rule_flags"] = rule_flags(rec, cfg)
    return rec


def run(cfg, dataset, translator=None, limit=None, out=None, reference=True, concurrency=None, client=None,
        dry_run=False, ids=None):
    """Translate the Thai set's source questions with the `translator` role's model, and their reference answers when
    the judge would use them (`hint_ok`). Each record carries its rule flags. A role with an `hf:` block (NLLB, MADLAD,
    Seed-X) runs in-process (`inproc.run_hf`); the others go to their vLLM server. `ids` restricts the run to those
    source questions (the translator comparison)."""
    tc = cfg["translate"]
    rows, pool, used = source_rows(cfg, dataset, limit)
    print(f"[translate] {dataset}: {used} prompts used in English excluded, {pool} left, {len(rows)} sampled")
    if dry_run:
        return {}
    if ids is not None:
        wanted = set(ids)
        if missing := wanted - {r["id"] for r in rows}:
            raise ValueError(f"{dataset}: {len(missing)} ids are not in the set's sample, e.g. {sorted(missing)[0]}")
        rows = [r for r in rows if r["id"] in wanted]
    translator = translator or tc["translator"]
    if not translator:
        raise ValueError("no translator: pass --translator or set translate.translator (the comparison's winner)")
    out = out or TRANSLATE_DIR / f"{dataset}.jsonl"
    done = {rid for rid, _ in read_done_keys(out)}
    todo = [r for r in rows if r["id"] not in done]
    if "hf" in cfg[translator]:
        from distill.inproc import run_hf
        return run_hf(cfg, dataset, translator, todo, out, reference)
    ep = server.endpoint(cfg, translator)
    system, user, prompt_hash = load_template("translate_typhoon" if translator == "typhoon" else "translate",
                                              tc["prompt_version"])
    print(f"[translate] {dataset} with the {translator} ({ep['model']}): {len(rows) - len(todo)} already done, "
          f"{len(todo)} to run")
    if not todo:
        return {}
    concurrency = concurrency or ep["concurrency"]
    client = client or server.make_client(ep, concurrency)
    meta = {"model": ep["model"], "role": translator, "prompt_version": tc["prompt_version"],
            "prompt_hash": prompt_hash, "sampling": tc["sampling"]}

    def ask(text, seed):
        kw = {"text": text, "target": tc["target"]}
        messages = ([{"role": "system", "content": system.format(**kw)}] if system else []) + [
            {"role": "user", "content": user.format(**kw)}]
        body = server.build_body(ep, messages, tc["sampling"], False, seed, tc["max_tokens"])
        p = parse_response(server.call_api(client, body, tc["max_retries"]), thinking=False)
        return {**p, "answer": judge.strip_thought(p["answer"]).strip()}  # Gemma as the translator

    def work(r):
        seed = sample_seed(cfg["generate"]["seed"], r["id"], 0)
        p = ask(r["input"], seed)
        ref = ask(r["reference"], seed) if reference and r.get("hint_ok") and r.get("reference") else None
        rec = record(cfg, dataset, r, meta, p["answer"], p["finish_reason"],
                     ref["answer"] if ref and ref["finish_reason"] == "stop" else None,
                     p["answer_tokens"] + (ref["answer_tokens"] if ref else 0))
        return rec, rec["tokens"]

    return server.run_pool("translate", work, todo, str(out), concurrency, lambda r: {"id": r["id"]},
                           dataset=dataset, translator=translator, model=ep["model"])


def numbers(text):
    """Digit sequences, with Thai digits read as Arabic ones and thousands separators dropped."""
    return Counter(re.findall(r"\d+", re.sub(r"(?<=\d),(?=\d{3})", "", text.translate(THAI_DIGITS))))


def urls(text):
    return {u.rstrip(".,;:!?)]}'\"") for u in URL.findall(text)}


def rule_flags(rec, cfg):
    """Cheap checks of a translated prompt; any flag drops it in `build`. The comparison report shows how well each
    flag matches the user's grades."""
    rules, rep = cfg["translate"]["rules"], cfg["filter"]["repetition"]
    en, th = rec["input_en"], rec["input"] or ""
    flags = []
    if rec.get("finish_reason") != "stop":
        flags.append("not_stop")
    if verify.thai_share(URL.sub(" ", CODE.sub(" ", th))) < rules["min_thai_share"]:
        flags.append("not_thai")
    if REFUSAL.search(th) and not REFUSAL.search(en):
        flags.append("refusal")
    if not rules["min_length_ratio"] <= len(th) / max(1, len(en)) <= rules["max_length_ratio"]:
        flags.append("length_ratio")
    if numbers(en) - numbers(th):
        flags.append("numbers_changed")
    if CODE.findall(en) != CODE.findall(th):
        flags.append("code_changed")
    if urls(en) - urls(th):
        flags.append("urls_changed")
    if [k for k, _ in OPTION.findall(en)] != [k for k, _ in OPTION.findall(th)]:
        flags.append("options_missing")
    if any(re.search(p, en, re.I) for p in cfg["filter"]["language_skip_patterns"]):
        flags.append("english_task_pattern")
    ratio = lambda t: report.repetition_ratio(t, rep["n"], rep["min_ngrams"])
    if ratio(th) > rep["max_ratio"] >= ratio(en):
        flags.append("repetition")
    return flags


# ---------------------------------------------------------------- check

def parse_check(j):
    """(english_only, fluency, meaning, expected meaning, critique); raises ValueError without both scores."""
    choice = j["choices"][0]
    c = judge.reply_json(choice)
    fluency, meaning = c.get("fluency"), c.get("meaning")
    if not all(isinstance(s, int) and 1 <= s <= 5 for s in (fluency, meaning)):
        raise ValueError(f"no valid scores in check reply: {(choice['message'].get('content') or '')[:200]!r}")
    return (bool(c.get("english_only")), fluency, meaning,
            judge.expected_score(choice.get("logprobs"), meaning, SCALE), str(c.get("critique", "")))


def check_path(path, role):
    """Where `role`'s checks of a translation file go: <stem>.check.jsonl for the judge, <stem>.check_<role>.jsonl for
    another checker, so several checkers can score the same translations."""
    return f"{stem(path)}.check.jsonl" if role == "judge" else f"{stem(path)}.check_{role}.jsonl"


def quality(chk):
    """A check's score, higher is better: a metric checker's `quality`, or a model checker's expected meaning."""
    return None if not chk else chk.get("quality", chk.get("meaning_expected"))


def check(cfg, path, role=None, concurrency=None, client=None):
    """A checker model (`role`, default translate.checker) reads the English and Thai prompt: is the task bound to
    English, and how fluent and faithful (1-5) is the translation? `meaning` comes last in the JSON, so its token is
    the last digit and its logprobs give `meaning_expected`, a continuous score for the threshold. A role with a
    `metric:` block (MetricX, CometKiwi) runs in-process instead (`inproc.metric_check`)."""
    tc = cfg["translate"]
    role = role or tc.get("checker") or "judge"
    if "metric" in cfg[role]:
        from distill.inproc import metric_check
        return metric_check(cfg, path, role)
    jc = {**judge.judge_config(cfg), "max_tokens": cfg["translate"]["check_max_tokens"]}
    ep = server.endpoint(cfg, role)
    system, user, prompt_hash = load_template("translation_check", tc["check_prompt_version"])
    out = check_path(path, role)
    done = {rid for rid, _ in read_done_keys(out)}
    todo = [r for r in report.iter_jsonl(path) if r["id"] not in done and (r.get("input") or "").strip()]
    print(f"[check] {path} by the {role} ({ep['model']}): {len(done)} already checked, {len(todo)} to run")
    if not todo:
        return {}
    concurrency = concurrency or ep["concurrency"]
    client = client or server.make_client(ep, concurrency)

    def work(r):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user.format(source=r["input_en"], translation=r["input"])}]
        j = server.call_api(client, judge.judge_body(ep, jc, messages, CHECK_SCHEMA, "translation_check"),
                            jc["max_retries"])
        english_only, fluency, meaning, expected, critique = parse_check(j)
        rec = {"id": r["id"], "english_only": english_only, "fluency": fluency, "meaning": meaning,
               "meaning_expected": round(expected, 4), "critique": critique, "judge_model": ep["model"],
               "check_prompt_hash": prompt_hash}
        return rec, (j.get("usage") or {}).get("completion_tokens", 0)

    return server.run_pool("check", work, todo, out, concurrency, lambda r: {"id": r["id"]}, input=str(path),
                           role=role, model=ep["model"])


# ---------------------------------------------------------------- build

def thai_gold(rec):
    """Gold answers of a translated prompt. An option letter keeps its letter, and its option text is read from the
    translated `A.`-`D.` lines; other golds stay as they are (the rule check then falls back to the judge)."""
    gold = rec.get("gold") or []
    if not gold or not re.fullmatch(r"[A-H]", gold[0]):
        return gold
    options = dict(OPTION.findall(rec["input"]))
    k = gold[0]
    return [k, f"{k}. {options[k]}", options[k]] if k in options else [k]


def drop_reason(rec, chk, sim, tc):
    """Why a translation is not used, or None: a rule flag, no check, a task bound to English, or a score under the
    threshold the comparison set (unset thresholds are not applied)."""
    if rec["rule_flags"]:
        return f"rule: {rec['rule_flags'][0]}"
    if not chk:
        return "unchecked"
    if chk["english_only"]:
        return "english_only"
    if tc["min_meaning"] is not None and chk["meaning_expected"] < tc["min_meaning"]:
        return "meaning"
    if tc["min_similarity"] is not None and (sim is None or sim["similarity"] < tc["min_similarity"]):
        return "similarity"
    return None


def build(cfg, dataset, path=None):
    """Write data/prepared/<dataset>.jsonl from the translations that pass every check (translate.checker's), in the
    prepared-row schema plus `input_en`, `lang` and `translation`, so every later stage reads it like any other
    dataset. Rows are sorted by id: the seeded sample of each run, and so the continuation of a larger LIMIT, then
    does not depend on the order in which the translations finished.
    A gate (translate.gates: a checker role and the `quality` a translation needs, missing counting as failed) was
    added after night 1: a row that fails it keeps its place with `translation.gate_failed`, and the stages skip it
    after sampling (`data.select_rows`, `selection.run`). Dropping it here would reshuffle the seeded sample."""
    tc = cfg["translate"]
    path = path or TRANSLATE_DIR / f"{dataset}.jsonl"
    checks = by_id(check_path(path, tc.get("checker") or "judge"))
    sims = by_id(f"{stem(path)}.similarity.jsonl")
    gates = [(g["role"], g["min_quality"], by_id(check_path(path, g["role"]))) for g in tc.get("gates") or []]
    stats, rows = Counter(), []
    for rec in report.iter_jsonl(path):
        chk, sim = checks.get(rec["id"]), sims.get(rec["id"])
        stats["translated"] += 1
        if reason := drop_reason(rec, chk, sim, tc):
            stats[f"dropped: {reason}"] += 1
            continue
        failed = [role for role, low, gc in gates if (q := quality(gc.get(rec["id"]))) is None or q < low]
        stats.update(f"gated: {role}" for role in failed[:1])
        rows.append({"id": rec["id"], "input": rec["input"], "strata": rec.get("strata"), "multi_turn": False,
                     "reference": rec.get("reference"), "ratings": {}, "gold": thai_gold(rec),
                     "source_dataset": dataset, "hint_ok": rec.get("hint_ok", False) and bool(rec.get("reference")),
                     "closed": rec.get("closed", False), "lang": "th", "input_en": rec["input_en"],
                     "translation": {"model": rec["translator"]["model"], "checker": chk.get("judge_model"),
                                     "meaning": chk["meaning_expected"], "fluency": chk["fluency"],
                                     "similarity": sim and sim["similarity"],
                                     **({"gate_failed": failed} if failed else {})}})
    stats["kept"] = len(rows)
    data.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    report.write_jsonl(data.PREPARED_DIR / f"{dataset}.jsonl", sorted(rows, key=lambda r: r["id"]))
    stats_path = data.PREPARED_DIR / "stats.json"
    all_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    all_stats[dataset] = dict(stats)
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"[build] {dataset}: {dict(stats)}")
    return dict(stats)

