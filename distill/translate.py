"""Thai sets: translate questions the English run never used, check each translation (rules, the judge, embedding
similarity), and build the Thai prepared file that the rest of the pipeline distills. Also the translator comparison:
a grading sheet for the user, and a report that picks the translator and calibrates the automatic checks."""
import json
import math
import random
import re
from collections import Counter, defaultdict
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


def run(cfg, dataset, translator=None, limit=None, out=None, reference=True, concurrency=None, backend=None,
        client=None, dry_run=False):
    """Translate the Thai set's source questions with the `translator` role's model (teacher, judge or typhoon), and
    their reference answers when the judge would use them (`hint_ok`). Each record carries its rule flags."""
    tc = cfg["translate"]
    rows, pool, used = source_rows(cfg, dataset, limit)
    print(f"[translate] {dataset}: {used} prompts used in English excluded, {pool} left, {len(rows)} sampled")
    if dry_run:
        return {}
    translator = translator or tc["translator"]
    if not translator:
        raise ValueError("no translator: pass --translator or set translate.translator (the comparison's winner)")
    backend = backend or cfg["backend"]
    if backend not in cfg[translator]:
        raise ValueError(f"the {translator} has no {backend} endpoint in the config")
    ep = server.endpoint(cfg, translator, backend)
    out = out or TRANSLATE_DIR / f"{dataset}.jsonl"
    system, user, prompt_hash = load_template("translate_typhoon" if translator == "typhoon" else "translate",
                                              tc["prompt_version"])
    done = {rid for rid, _ in read_done_keys(out)}
    todo = [r for r in rows if r["id"] not in done]
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
        return parse_response(server.call_api(client, body, tc["max_retries"]), thinking=False)

    def work(r):
        seed = sample_seed(cfg["generate"]["seed"], r["id"], 0)
        p = ask(r["input"], seed)
        ref = ask(r["reference"], seed) if reference and r.get("hint_ok") and r.get("reference") else None
        rec = {"id": r["id"], "dataset": dataset, "source_dataset": r["source_dataset"], "input_en": r["input"],
               "input": p["answer"], "finish_reason": p["finish_reason"],
               "reference_en": r.get("reference"),
               "reference": ref["answer"] if ref and ref["finish_reason"] == "stop" else None,
               "strata": r.get("strata"), "closed": bool(r.get("closed")), "gold": r.get("gold") or [],
               "hint_ok": bool(r.get("hint_ok")), "translator": meta,
               "tokens": p["answer_tokens"] + (ref["answer_tokens"] if ref else 0)}
        rec["rule_flags"] = rule_flags(rec, cfg)
        return rec, rec["tokens"]

    return server.run_pool("translate", work, todo, str(out), concurrency, lambda r: {"id": r["id"]},
                           dataset=dataset, translator=translator, backend=ep["backend"], model=ep["model"])


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


# ---------------------------------------------------------------- checks

def parse_check(j):
    """(english_only, fluency, meaning, expected meaning, critique); raises ValueError without both scores."""
    choice = j["choices"][0]
    c = judge.reply_json(choice)
    fluency, meaning = c.get("fluency"), c.get("meaning")
    if not all(isinstance(s, int) and 1 <= s <= 5 for s in (fluency, meaning)):
        raise ValueError(f"no valid scores in check reply: {(choice['message'].get('content') or '')[:200]!r}")
    return (bool(c.get("english_only")), fluency, meaning,
            judge.expected_score(choice.get("logprobs"), meaning, SCALE), str(c.get("critique", "")))


def check(cfg, path, concurrency=None, backend=None, client=None):
    """The judge reads the English and Thai prompt: is the task bound to English, and how fluent and faithful (1-5) is
    the translation? `meaning` comes last in the JSON, so its token is the last digit and its logprobs give
    `meaning_expected`, a continuous score for the threshold. Writes <stem>.check.jsonl."""
    tc = cfg["translate"]
    jc = {**judge.judge_config(cfg), "max_tokens": cfg["translate"]["check_max_tokens"]}
    ep = server.endpoint(cfg, "judge", backend)
    system, user, prompt_hash = load_template("translation_check", tc["check_prompt_version"])
    out = f"{stem(path)}.check.jsonl"
    done = {rid for rid, _ in read_done_keys(out)}
    todo = [r for r in report.iter_jsonl(path) if r["id"] not in done and (r.get("input") or "").strip()]
    print(f"[check] {path}: {len(done)} already checked, {len(todo)} to run")
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
                           backend=ep["backend"], model=ep["model"])


def cosine(a, b):
    a, b = list(map(float, a)), list(map(float, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def similarity(cfg, path, encode=None, batch=32):
    """Cosine similarity of the English and Thai prompt in a multilingual embedding space (BGE-M3 by default), on
    CPU. Writes <stem>.similarity.jsonl."""
    out = f"{stem(path)}.similarity.jsonl"
    done = {rid for rid, _ in read_done_keys(out)}
    todo = [r for r in report.iter_jsonl(path) if r["id"] not in done and (r.get("input") or "").strip()]
    print(f"[similarity] {path}: {len(done)} already scored, {len(todo)} to run")
    if not todo:
        return
    name = cfg["translate"]["embed_model"]
    if encode is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name, device="cpu")
        model.max_seq_length = 1024  # both texts are cut the same way; long prompts keep their opening
        encode = lambda texts: model.encode(texts, batch_size=8, normalize_embeddings=True)
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            rows = todo[i:i + batch]
            pairs = zip(rows, encode([r["input_en"] for r in rows]), encode([r["input"] for r in rows]))
            for r, a, b in pairs:
                f.write(json.dumps({"id": r["id"], "similarity": round(cosine(a, b), 4), "embed_model": name}) + "\n")
            f.flush()


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
    """Write data/prepared/<dataset>.jsonl from the translations that pass every check, in the prepared-row schema plus
    `input_en`, `lang` and `translation`, so every later stage reads it like any other dataset."""
    tc = cfg["translate"]
    path = path or TRANSLATE_DIR / f"{dataset}.jsonl"
    checks, sims = by_id(f"{stem(path)}.check.jsonl"), by_id(f"{stem(path)}.similarity.jsonl")
    stats, rows = Counter(), []
    for rec in report.iter_jsonl(path):
        chk, sim = checks.get(rec["id"]), sims.get(rec["id"])
        stats["translated"] += 1
        if reason := drop_reason(rec, chk, sim, tc):
            stats[f"dropped: {reason}"] += 1
            continue
        rows.append({"id": rec["id"], "input": rec["input"], "strata": rec.get("strata"), "multi_turn": False,
                     "reference": rec.get("reference"), "ratings": {}, "gold": thai_gold(rec),
                     "source_dataset": dataset, "hint_ok": rec.get("hint_ok", False) and bool(rec.get("reference")),
                     "closed": rec.get("closed", False), "lang": "th", "input_en": rec["input_en"],
                     "translation": {"model": rec["translator"]["model"], "meaning": chk["meaning_expected"],
                                     "fluency": chk["fluency"], "similarity": sim and sim["similarity"]}})
    stats["kept"] = len(rows)
    data.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    report.write_jsonl(data.PREPARED_DIR / f"{dataset}.jsonl", rows)
    stats_path = data.PREPARED_DIR / "stats.json"
    all_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    all_stats[dataset] = dict(stats)
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"[build] {dataset}: {dict(stats)}")
    return dict(stats)


# ---------------------------------------------------------------- translator comparison

GUIDE = [
    ("Grade each Thai translation of the English prompt. Model names are hidden; three translations of a prompt sit "
     "next to each other in random order.", ""),
    ("meaning (1-5)", "Does the Thai ask for exactly the same thing?"),
    ("5", "The same request: every instruction, constraint, number, name, negation and answer option kept, "
          "nothing added."),
    ("4", "One small slip that does not change what is asked."),
    ("3", "A noticeable change: a constraint or detail lost, weakened or added."),
    ("2", "A different request in part: a negation flipped, a number or option changed, or a key part missing."),
    ("1", "A different request, a refusal, an answer instead of a translation, or not translated."),
    ("fluency (1-5)", "Does it read like Thai a native speaker would write?"),
    ("5", "Natural Thai."), ("4", "Natural, with a slightly unusual phrase."),
    ("3", "Understandable but clearly translated."), ("2", "Awkward, hard to read in places."),
    ("1", "Not readable Thai."),
    ("note", "Optional: anything worth knowing, e.g. the task only makes sense in English."),
]
HEADER = ["row", "set", "English prompt", "Thai translation", "meaning (1-5)", "fluency (1-5)", "note"]


def cell(text, limit=32000):
    """Text an Excel cell accepts: no control characters, at most `limit` characters (Excel's cap is 32,767)."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    text = ILLEGAL_CHARACTERS_RE.sub("", text or "")
    return text if len(text) <= limit else text[:limit] + " …[cut]"


def compare_sheet(paths, out, seed=0):
    """The grading sheet for the translator comparison: one row per translation, the translations of a prompt next to
    each other, prompts and translators in a seeded random order. Model names are not in the sheet; the key
    (row -> translator, id, file) goes to <out stem>_key.json."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    groups = defaultdict(list)
    for path in paths:
        for rec in report.iter_jsonl(path):
            groups[(rec["dataset"], rec["id"])].append((str(path), rec))
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    wb = Workbook()
    ws = wb.active
    ws.title = "grades"
    ws.append(HEADER)
    key = {}
    for k in keys:
        group = groups[k]
        rng.shuffle(group)
        for path, rec in group:
            row = ws.max_row
            ws.append([row, rec["source_dataset"], cell(rec["input_en"]), cell(rec["input"]), None, None, None])
            key[row] = {"id": rec["id"], "dataset": rec["dataset"], "translator": rec["translator"]["role"],
                        "model": rec["translator"]["model"], "path": path}
    for col, width in zip("ABCDEFG", (6, 14, 70, 70, 13, 13, 30)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "A2"
    grades = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    ws.add_data_validation(grades)
    grades.add(f"E2:F{ws.max_row}")
    guide = wb.create_sheet("how to grade")
    for line in GUIDE:
        guide.append(list(line))
    guide.column_dimensions["A"].width, guide.column_dimensions["B"].width = 16, 110
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    key_path = f"{str(out).removesuffix('.xlsx')}_key.json"
    Path(key_path).write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"[compare] {len(key)} translations of {len(keys)} prompts -> {out} (key: {key_path})")
    return key


def read_grades(xlsx):
    """{row: (meaning, fluency, note)} for the graded rows of the sheet."""
    from openpyxl import load_workbook
    ws = load_workbook(xlsx, read_only=True)["grades"]
    grades = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        n, meaning, fluency, note = row[0], row[4], row[5], row[6]
        if n is not None and meaning not in (None, "") and fluency not in (None, ""):
            grades[int(n)] = (int(meaning), int(fluency), note or "")
    return grades


def ranks(xs):
    """Ranks from 1, ties sharing their mean rank."""
    order, r, i = sorted(range(len(xs)), key=xs.__getitem__), [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            r[k] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    norm = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / norm if norm else None


def fmt(x, spec=".2f"):
    return "—" if x is None else format(x, spec)


def compare_report(xlsx, out="reports/translate_compare.md"):
    """Read the user's grades and write the comparison report: each translator's grades, how well each automatic
    signal (the judge's meaning and fluency, embedding similarity, rule flags) agrees with them, and the thresholds."""
    key = {int(k): v for k, v in json.loads(Path(f"{str(xlsx).removesuffix('.xlsx')}_key.json")
                                          .read_text(encoding="utf-8")).items()}
    grades = read_grades(xlsx)
    files = {}
    for p in {v["path"] for v in key.values()}:
        files[p] = (by_id(p), by_id(f"{stem(p)}.check.jsonl"), by_id(f"{stem(p)}.similarity.jsonl"))
    rows = []
    for n, k in key.items():
        rec, chk, sim = (f.get(k["id"]) for f in files[k["path"]])
        meaning, fluency, note = grades.get(n, (None, None, ""))
        rows.append({"row": n, **k, "meaning": meaning, "fluency": fluency, "note": note, "rec": rec,
                     "chk": chk, "sim": (sim or {}).get("similarity"), "flags": (rec or {}).get("rule_flags", [])})
    graded = [r for r in rows if r["meaning"] is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    lines = ["# Translator comparison", "",
             f"{len(graded)} of {len(rows)} translations graded by hand (meaning and fluency, 1-5). "
             "Bad = meaning ≤ 3.", "", "## Translators", "",
             "| translator | model | graded | meaning | fluency | meaning ≤ 2 | rule-flagged | judge: English-only "
             "| judge meaning − user |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in sorted({r["translator"] for r in rows}):
        g = [r for r in graded if r["translator"] == t]
        a = [r for r in rows if r["translator"] == t]
        diff = mean([r["chk"]["meaning_expected"] - r["meaning"] for r in g if r["chk"]])
        lines.append(f"| {t} | {a[0]['model']} | {len(g)} | {fmt(mean([r['meaning'] for r in g]))} | "
                     f"{fmt(mean([r['fluency'] for r in g]))} | {report.share(g, lambda r: r['meaning'] <= 2) if g else '—'} | "
                     f"{report.share(a, lambda r: bool(r['flags']))} | "
                     f"{report.share(a, lambda r: bool((r['chk'] or {}).get('english_only')))} | {fmt(diff, '+.2f')} |")
    lines += ["", "## Automatic signals vs the grades", "",
              "Spearman: rank correlation with the user's grade. AUC: how often a good translation (meaning ≥ 4) "
              "scores above a bad one.", "", "| signal | vs | rows | Spearman | AUC good > bad |", "|---|---|---:|---:|---:|"]
    signals = [("judge meaning (expected)", "meaning", lambda r: r["chk"] and r["chk"]["meaning_expected"]),
               ("judge fluency", "fluency", lambda r: r["chk"] and r["chk"]["fluency"]),
               ("embedding similarity", "meaning", lambda r: r["sim"])]
    for name, vs, f in signals:
        pts = [(f(r), r[vs], r["meaning"]) for r in graded if f(r) is not None]
        good, bad = [s for s, _, m in pts if m >= 4], [s for s, _, m in pts if m <= 3]
        rho = spearman([s for s, _, _ in pts], [g for _, g, _ in pts]) if len(pts) > 2 else None
        lines.append(f"| {name} | {vs} | {len(pts)} | {fmt(rho)} | "
                     f"{fmt(report.auc(good, bad) if good and bad else None)} |")
    lines += ["", "## Thresholds", "", "Share of bad translations rejected and of good ones kept at each threshold.",
              "", "| signal | threshold | bad rejected | good kept |", "|---|---:|---:|---:|"]
    for name, f, grid in (("judge meaning (expected)", lambda r: r["chk"] and r["chk"]["meaning_expected"],
                           (3.0, 3.5, 4.0, 4.5, 4.8)),
                          ("embedding similarity", lambda r: r["sim"], (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9))):
        pts = [(f(r), r["meaning"]) for r in graded if f(r) is not None]
        good, bad = [s for s, m in pts if m >= 4], [s for s, m in pts if m <= 3]
        for t in grid:
            lines.append(f"| {name} | {t} | {fmt(mean([s < t for s in bad]), '.0%')} | "
                         f"{fmt(mean([s >= t for s in good]), '.0%')} |")
    flags = Counter(f for r in rows for f in r["flags"])
    lines += ["", "## Rule flags", "", "| flag | translations | mean user meaning of the flagged |", "|---|---:|---:|"]
    lines += [f"| {f} | {n} | {fmt(mean([r['meaning'] for r in graded if f in r['flags']]))} |"
              for f, n in flags.most_common()] or ["| none | 0 | — |"]
    english = sorted({(r["dataset"], r["id"], r["rec"]["input_en"][:150].replace("\n", " ").replace("|", "/"))
                      for r in rows if (r["chk"] or {}).get("english_only") and r["rec"]})
    lines += ["", "## Flagged by the judge as bound to English", "", "| set | id | prompt |", "|---|---|---|"]
    lines += [f"| {d} | {i} | {t} |" for d, i, t in english] or ["| — | — | none |"]
    notes = [r for r in graded if r["note"]]
    if notes:
        lines += ["", "## Grader notes", "", "| row | translator | meaning | note |", "|---:|---|---:|---|"]
        lines += [f"| {r['row']} | {r['translator']} | {r['meaning']} | {str(r['note']).replace('|', '/')} |"
                  for r in notes]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare] report -> {out}")
    return rows
