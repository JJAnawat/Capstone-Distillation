"""The translator and checker comparisons, kept for the project report: the pools to grade (`compare_pick`,
`checker_sample`, `checker_pick`), the blind grading sheet, the reports that read the grades back (`compare_report`,
`checker_report`), and the embedding similarity that the first comparison ruled out."""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from distill import data, report, translate, verify
from distill.generate import read_done_keys
from distill.translate import by_id, check_path, quality, stem

GUIDE = [
    ("Grade each Thai translation of the English prompt. Model names are hidden; when a prompt has several "
     "translations, they sit next to each other in random order.", ""),
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


# ---------------------------------------------------------------- embedding similarity (ruled out by the first comparison)

def cosine(a, b):
    a, b = list(map(float, a)), list(map(float, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def similarity(cfg, path, encode=None, batch=32):
    """Cosine similarity of the English and Thai prompt in a multilingual embedding space (BGE-M3 by default), on the
    GPU when there is one (the scripts run it with no server up), else on CPU. Writes <stem>.similarity.jsonl."""
    out = f"{stem(path)}.similarity.jsonl"
    done = {rid for rid, _ in read_done_keys(out)}
    todo = [r for r in report.iter_jsonl(path) if r["id"] not in done and (r.get("input") or "").strip()]
    print(f"[similarity] {path}: {len(done)} already scored, {len(todo)} to run")
    if not todo:
        return
    name = cfg["translate"]["embed_model"]
    if encode is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)  # picks the GPU if there is one
        model.max_seq_length = 1024  # both texts are cut the same way; long prompts keep their opening
        encode = lambda texts: model.encode(texts, batch_size=8, normalize_embeddings=True)
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            rows = todo[i:i + batch]
            pairs = zip(rows, encode([r["input_en"] for r in rows]), encode([r["input"] for r in rows]))
            for r, a, b in pairs:
                f.write(json.dumps({"id": r["id"], "similarity": round(cosine(a, b), 4), "embed_model": name}) + "\n")
            f.flush()


# ---------------------------------------------------------------- translator comparisons

def compare_pick(cfg, per_set, out_dir, exclude_run="data/full_th", seed=0):
    """The pool of a later translator comparison: for each Thai set, a seeded stratified sample of `per_set` of the
    translator's records (data/translate/<set>.jsonl, failed translations included) whose questions the run has not
    distilled yet (no samples in `exclude_run`). They and their checks are copied to <out_dir>/<set>.judge.jsonl and
    <set>.judge.check.jsonl, so the existing translations need no GPU."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in (n for n in cfg["datasets"] if data.translate_source(cfg, n)):
        path = translate.TRANSLATE_DIR / f"{ds}.jsonl"
        if not path.exists():
            print(f"[compare] {ds}: no translations in {path}; skipped")
            continue
        done = {r["id"] for f in Path(exclude_run).glob(f"{ds}_*_B_raw.jsonl") for r in report.iter_jsonl(f)}
        recs = [r for r in report.iter_jsonl(path) if r["id"] not in done]
        picked, checks = data.sample(recs, per_set, seed), by_id(check_path(path, "judge"))
        report.write_jsonl(out_dir / f"{ds}.judge.jsonl", picked)
        report.write_jsonl(out_dir / f"{ds}.judge.check.jsonl", [checks[r["id"]] for r in picked if r["id"] in checks])
        print(f"[compare] {ds}: {len(picked)} of {len(recs)} undistilled translations picked ({len(done)} distilled)")


def compare_translate(cfg, role, out_dir):
    """Translate the questions of every <set>.judge.jsonl in `out_dir` with `role`, into <set>.<role>.jsonl. An `hf:`
    role's model loads once for all the sets."""
    for pick in sorted(Path(out_dir).glob("*.judge.jsonl")):
        ds = pick.name.removesuffix(".judge.jsonl")
        translate.run(cfg, ds, role, ids=[r["id"] for r in report.iter_jsonl(pick)], reference=False,
            out=Path(out_dir) / f"{ds}.{role}.jsonl")


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
    Path(key_path(out)).write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"[compare] {len(key)} translations of {len(keys)} prompts -> {out} (key: {key_path(out)})")
    return key


def grade(value):
    """A 1-5 grade from a sheet cell, or None if it is empty. Raises ValueError for anything else."""
    if value is None or not str(value).strip():
        return None
    g = float(str(value).strip())  # Excel may store 4 as 4.0, or as text
    if g != int(g) or not 1 <= g <= 5:
        raise ValueError(value)
    return int(g)


def read_grades(xlsx):
    """({row: (meaning, fluency, note)}, [(row, column, value)] of unreadable grades). The sheet may be graded only in
    part: rows without a grade or a note are left out, and either grade of a row may be None."""
    from openpyxl import load_workbook
    ws = load_workbook(xlsx, read_only=True)["grades"]
    grades, bad = {}, []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        scores = []
        for column, value in (("meaning", row[4]), ("fluency", row[5])):
            try:
                scores.append(grade(value))
            except ValueError:
                bad.append((int(row[0]), column, value))
                scores.append(None)
        if any(s is not None for s in scores) or row[6]:
            grades[int(row[0])] = (*scores, row[6] or "")
    return grades, bad


def key_path(xlsx):
    return f"{str(xlsx).removesuffix('.xlsx')}_key.json"


def load_graded(xlsx):
    """(key, grades, unreadable) of a graded sheet: its key (row -> translator, id, file) and `read_grades`."""
    key = {int(k): v for k, v in json.loads(Path(key_path(xlsx)).read_text(encoding="utf-8")).items()}
    return (key, *read_grades(xlsx))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


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


def checks_of(path):
    """{role: {id: check}} for every checker that scored a translation file (see `check_path`)."""
    name = Path(stem(path)).name
    found = {"judge": by_id(check_path(path, "judge"))}
    for p in sorted(Path(path).parent.glob(f"{name}.check_*.jsonl")):
        if not p.name.endswith((".runs.jsonl", ".errors.jsonl")):
            found[p.name.removeprefix(f"{name}.check_").removesuffix(".jsonl")] = by_id(p)
    return {role: checks for role, checks in found.items() if checks}


def compare_report(xlsx, out="reports/translate_compare.md"):
    """Read the user's grades and write the comparison report: each translator's grades, how well each checker's
    meaning and fluency scores (and embedding similarity, and the rule flags) agree with them, each checker's lean per
    translator (does it favour its own family?), and the thresholds.
    - Only graded cells count. Translators are compared on the prompts whose translations are all graded for meaning,
      so a sheet graded part of the way stays fair; the signals use every graded translation.
    - A translation with no Thai letters at all is an untranslated copy and counts as meaning 1, as the grading
      guide's scale says, whatever it was graded.
    - A metric checker (MetricX, CometKiwi) has a `quality` score on its own scale and no fluency: its thresholds are
      where it keeps 98% to 85% of the good translations, and the lean table leaves it out."""
    key, grades, bad = load_graded(xlsx)
    files = {p: (by_id(p), checks_of(p), by_id(f"{stem(p)}.similarity.jsonl")) for p in {v["path"] for v in key.values()}}
    roles = sorted({role for _, checks, _ in files.values() for role in checks}, key=lambda r: (r != "judge", r))
    models = {role: next(c.get("judge_model") or role for _, checks, _ in files.values() if role in checks
                         for c in checks[role].values()) for role in roles}
    rated = [role for role in roles if any(c.get("meaning") is not None for _, checks, _ in files.values()
                                           for c in checks.get(role, {}).values())]  # 1-5 model checkers
    rows, recounted = [], 0
    for n, k in key.items():
        recs, checks, sims = files[k["path"]]
        rec = recs.get(k["id"])
        meaning, fluency, note = grades.get(n, (None, None, ""))
        if meaning not in (None, 1) and rec and verify.thai_share(rec["input"]) == 0:
            meaning, recounted = 1, recounted + 1
        rows.append({"row": n, **k, "meaning": meaning, "fluency": fluency, "note": note, "rec": rec,
                     "chk": {role: checks.get(role, {}).get(k["id"]) for role in roles},
                     "sim": (sims.get(k["id"]) or {}).get("similarity"), "flags": (rec or {}).get("rule_flags", [])})
    graded = [r for r in rows if r["meaning"] is not None]
    fluent = [r for r in rows if r["fluency"] is not None]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["id"])].append(r)
    complete = {g for g, rs in groups.items() if all(r["meaning"] is not None for r in rs)}
    fair = [r for r in rows if (r["dataset"], r["id"]) in complete]
    translators = sorted({r["translator"] for r in rows})
    meaning_of = lambda role: lambda r: quality(r["chk"][role])
    fluency_of = lambda role: lambda r: (r["chk"][role] or {}).get("fluency")
    signal_name = lambda role: f"{role} meaning (expected)" if role in rated else f"{role} quality"
    lines = ["# Translator comparison", "",
             f"Graded by hand: {len(graded)} of {len(rows)} translations for meaning, {len(fluent)} for fluency "
             f"(1-5). Translators are compared on the {len(complete)} prompts whose translations are all graded for "
             "meaning; rule flags cover all their translations. The signals use every graded translation. "
             "Bad = meaning ≤ 3.", "",
             f"Untranslated copies (no Thai letters) count as meaning 1, as the grading guide's scale says: "
             f"{recounted} grades changed.", "",
             "Checkers: " + ", ".join(f"`{role}` ({models[role]})" for role in roles) + ".", ""]
    if len(complete) < 15:
        lines += [f"Only {len(complete)} complete prompts: read the differences between translators as rough.", ""]
    if bad:
        lines += ["Unreadable grades, left out: " + ", ".join(f"row {n} {c} {v!r}" for n, c, v in bad) + ".", ""]
    lines += ["## Translators", "", "| translator | model | prompts | meaning | fluency | meaning ≤ 2 | rule-flagged |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for t in translators:
        f, a = [r for r in fair if r["translator"] == t], [r for r in rows if r["translator"] == t]
        lines.append(f"| {t} | {a[0]['model']} | {len(f)} | {fmt(mean([r['meaning'] for r in f]))} | "
                     f"{fmt(mean([r['fluency'] for r in f if r['fluency'] is not None]))} | "
                     f"{report.share(f, lambda r: r['meaning'] <= 2) if f else '—'} | "
                     f"{report.share(a, lambda r: bool(r['flags']))} |")
    lines += ["", "## Each checker's lean, by translator", "",
              "Mean of the checker's meaning score minus the user's grade (+ = more lenient than the user), and the "
              "share of translations it calls bound to English. A checker that favours its own family leans further "
              "up on that family's translations.", "",
              "| translator | " + " | ".join(f"{role}: meaning − user | {role}: English-only" for role in rated) + " |",
              "|---|" + "---:|---:|" * len(rated)]
    for t in translators:
        g, a = [r for r in graded if r["translator"] == t], [r for r in rows if r["translator"] == t]
        cells = [f"{fmt(mean([meaning_of(role)(r) - r['meaning'] for r in g if meaning_of(role)(r) is not None]), '+.2f')}"
                 f" | {report.share(a, lambda r: bool((r['chk'][role] or {}).get('english_only')))}" for role in rated]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Automatic signals vs the grades", "",
              "Spearman: rank correlation with the user's grade. AUC: how often a good translation (grade ≥ 4) "
              "scores above a bad one (≤ 3).", "", "| signal | vs | rows | Spearman | AUC good > bad |",
              "|---|---|---:|---:|---:|"]
    signals = [(signal_name(role), "meaning", meaning_of(role)) for role in roles]
    signals += [(f"{role} fluency", "fluency", fluency_of(role)) for role in rated]
    for name, vs, f in signals + [("embedding similarity", "meaning", lambda r: r["sim"])]:
        pts = [(f(r), r[vs]) for r in (graded if vs == "meaning" else fluent) if f(r) is not None]
        good, low = [s for s, g in pts if g >= 4], [s for s, g in pts if g <= 3]
        rho = spearman([s for s, _ in pts], [g for _, g in pts]) if len(pts) > 2 else None
        lines.append(f"| {name} | {vs} | {len(pts)} | {fmt(rho)} | "
                     f"{fmt(report.auc(good, low) if good and low else None)} |")
    lines += ["", "## Thresholds", "", "Share of bad translations rejected and of good ones kept at each threshold.",
              "", "| signal | threshold | bad rejected | good kept |", "|---|---:|---:|---:|"]
    grids = [(signal_name(role), meaning_of(role), (3.0, 3.5, 4.0, 4.5, 4.8) if role in rated else None)
             for role in roles]
    for name, f, grid in grids + [("embedding similarity", lambda r: r["sim"], (0.6, 0.7, 0.8, 0.9))]:
        pts = [(f(r), r["meaning"]) for r in graded if f(r) is not None]
        good, low = [s for s, m in pts if m >= 4], [s for s, m in pts if m <= 3]
        labels = [str(t) for t in grid or ()]
        if grid is None:  # a metric's own scale: the scores that keep 98% ... 85% of the good translations
            grid = sorted({sorted(good)[int((1 - keep) * len(good))] for keep in (0.98, 0.95, 0.93, 0.9, 0.85)}
                          if good else set())
            labels = [fmt(t, ".3f") for t in grid]
        for t, label in zip(grid, labels):
            lines.append(f"| {name} | {label} | {fmt(mean([s < t for s in low]), '.0%')} | "
                         f"{fmt(mean([s >= t for s in good]), '.0%')} |")
    flags = Counter(f for r in rows for f in r["flags"])
    lines += ["", "## Rule flags", "", "| flag | translations | mean user meaning of the flagged |", "|---|---:|---:|"]
    lines += [f"| {f} | {n} | {fmt(mean([r['meaning'] for r in graded if f in r['flags']]))} |"
              for f, n in flags.most_common()] or ["| none | 0 | — |"]
    english = defaultdict(set)
    for r in rows:
        for role in roles:
            if (r["chk"][role] or {}).get("english_only") and r["rec"]:
                english[(r["dataset"], r["id"], r["rec"]["input_en"][:150].replace("\n", " ").replace("|", "/"))].add(role)
    lines += ["", "## Called bound to English by a checker", "", "| set | id | by | prompt |", "|---|---|---|---|"]
    lines += [f"| {d} | {i} | {', '.join(sorted(by))} | {t} |" for (d, i, t), by in sorted(english.items())] or [
        "| — | — | — | none |"]
    notes = [r for r in rows if r["note"]]
    if notes:
        lines += ["", "## Grader notes", "", "| row | translator | meaning | note |", "|---:|---|---:|---|"]
        lines += [f"| {r['row']} | {r['translator']} | {'—' if r['meaning'] is None else r['meaning']} | "
                  f"{str(r['note']).replace('|', '/')} |" for r in notes]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare] report -> {out}")
    return rows


# ---------------------------------------------------------------- checker comparison

def gate_candidate(rec, judge_check):
    """Whether a gate would act on a translation: no rule flag, checked by the judge and not bound to English."""
    return not rec["rule_flags"] and bool(judge_check) and not judge_check["english_only"]


def checker_sample(cfg, per_set, out_dir, seed=0):
    """A seeded stratified sample of `per_set` translations per Thai set that a gate would act on (`gate_candidate`),
    copied with the judge's checks to <out_dir>/<set>.jsonl and <set>.check.jsonl. The candidate checkers score these
    instead of every translation, which saves GPU time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in (n for n in cfg["datasets"] if data.translate_source(cfg, n)):
        path = translate.TRANSLATE_DIR / f"{ds}.jsonl"
        if not path.exists():
            continue
        checks = by_id(check_path(path, "judge"))
        recs = [r for r in report.iter_jsonl(path) if gate_candidate(r, checks.get(r["id"]))]
        picked = data.sample(recs, per_set, seed)
        report.write_jsonl(out_dir / f"{ds}.jsonl", picked)
        report.write_jsonl(out_dir / f"{ds}.check.jsonl", [checks[r["id"]] for r in picked])
        print(f"[checkers] {ds}: {len(picked)} of {len(recs)} translations sampled into {out_dir}")


def checker_pick(cfg, out_dir, roles, per_group=15, random_n=20, seed=0, src_dir=None):
    """The checker comparison's grading pool, from the translations in `src_dir` (<set>.jsonl with the checks next to
    them; default data/translate, or the sample `checker_sample` wrote).
    - The universe: translations a gate would act on (`gate_candidate`), scored by every candidate (a candidate with
      no checks at all, e.g. a model that failed to load, is left out).
    - The judge rejects a translation under translate.min_meaning. Each candidate rejects its lowest-scored
      translations, as many as the judge rejects (fewer if scores tie at the threshold), so the gates are comparable;
      its threshold is the first score kept, and as a gate it rejects the scores below it.
    - Seeded draws: `per_group` from each "only <candidate> rejects" group, `per_group` from "only judge rejects", and
      `random_n` from "all pass", the baseline. A row already drawn counts for every group it belongs to.
    The records go to <out_dir>/<set>.jsonl (for `compare_sheet`), and the groups, thresholds and why each row was drawn
    to <out_dir>/picks.json, which the sheet does not show."""
    tc, out_dir, src_dir = cfg["translate"], Path(out_dir), Path(src_dir or translate.TRANSLATE_DIR)
    recs, found, sets = {}, {}, []
    for ds in (n for n in cfg["datasets"] if data.translate_source(cfg, n)):
        path = src_dir / f"{ds}.jsonl"
        if not path.exists():
            continue
        sets.append(ds)
        checks = {role: by_id(check_path(path, role)) for role in ("judge", *roles)}
        for rec in report.iter_jsonl(path):
            if gate_candidate(rec, checks["judge"].get(rec["id"])):
                recs[(ds, rec["id"])] = rec
                found[(ds, rec["id"])] = {role: c[rec["id"]] for role, c in checks.items() if rec["id"] in c}
    missing = [role for role in roles if not any(role in f for f in found.values())]
    roles = [role for role in roles if role not in missing]
    universe = sorted(k for k, f in found.items() if all(role in f for role in roles))
    judge_rejects = {k for k in universe if quality(found[k]["judge"]) < tc["min_meaning"]}
    n, rejects, thresholds = len(judge_rejects), {}, {}
    for role in roles:  # rejected as a gate would: under the threshold, so ties at it are kept
        ranked = sorted(universe, key=lambda k: (quality(found[k][role]), k))
        t = thresholds[role] = quality(found[ranked[n]][role]) if n < len(ranked) else None
        rejects[role] = {k for k in ranked[:n] if t is None or quality(found[k][role]) < t}
    any_rejects = set().union(*rejects.values())
    groups = {f"only {role} rejects": rejects[role] - judge_rejects for role in roles}
    groups |= {"only judge rejects": judge_rejects - any_rejects, "all pass": set(universe) - judge_rejects - any_rejects}
    rng, picked = random.Random(seed), set()
    for name, members in groups.items():
        need = (random_n if name == "all pass" else per_group) - len(members & picked)
        pool = sorted(members - picked)
        picked |= set(rng.sample(pool, max(0, min(need, len(pool)))))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in sets:
        report.write_jsonl(out_dir / f"{ds}.jsonl", [recs[k] for k in sorted(picked) if k[0] == ds])
    summary = {"universe": len(universe), "min_meaning": tc["min_meaning"], "judge_rejects": n,
               "share": round(n / len(universe), 4) if universe else None, "thresholds": thresholds,
               "missing": missing, "sizes": {name: len(m) for name, m in groups.items()},
               "rejects": {role: len(rejects[role]) for role in roles},
               "overlap": {role: len(rejects[role] & judge_rejects) for role in roles},
               "english_only": {role: sum(bool(found[k][role].get("english_only")) for k in universe) for role in roles
                                if any("english_only" in found[k][role] for k in universe)},
               "rows": {f"{ds}/{rid}": {"groups": [name for name, m in groups.items() if (ds, rid) in m],
                                         "quality": {role: quality(found[(ds, rid)][role]) for role in ("judge", *roles)}}
                        for ds, rid in sorted(picked)}}
    (out_dir / "picks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[checkers] {len(picked)} of {len(universe)} translations picked into {out_dir}; the judge rejects {n}; "
          f"groups {summary['sizes']}" + (f"; no checks from {', '.join(missing)}" if missing else ""))
    return summary


def wilson(k, n, z=1.96):
    """95% Wilson interval of a share k / n, or (None, None) for no rows."""
    if not n:
        return None, None
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_greater(a, b, c, d):
    """One-sided Fisher exact p for [[a, b], [c, d]] (rows: groups; columns: bad, good): the chance of `a` or more bad
    in the first row if both rows had the same bad rate."""
    n1, bad, total = a + b, a + c, a + b + c + d
    return sum(math.comb(bad, x) * math.comb(total - bad, n1 - x)
               for x in range(a, min(bad, n1) + 1)) / math.comb(total, n1)


def checker_report(xlsx, out="reports/checker_compare.md", picks=None):
    """Read the user's grades of the checker pool and write the report: for each group, the share graded bad (meaning
    ≤ 3) with its 95% interval and the mean grades, and a one-sided Fisher test of each disagreement group against
    "all pass".
    Decision rule: a candidate becomes a second gate if at least half of the graded translations only it rejects are
    bad, and that share beats the baseline (p < 0.05); of several, the larger share wins. Otherwise the judge stays
    the only checker."""
    key, grades, unreadable = load_graded(xlsx)
    picks = json.loads(Path(picks or Path(next(iter(key.values()))["path"]).parent / "picks.json")
                       .read_text(encoding="utf-8"))
    graded = {f"{k['dataset']}/{k['id']}": grades[n] for n, k in key.items() if grades.get(n, (None,))[0] is not None}
    stats = {}
    for name in picks["sizes"]:
        g = [graded[k] for k, v in picks["rows"].items() if name in v["groups"] and k in graded]
        stats[name] = (sum(m <= 3 for m, _, _ in g), len(g), mean([m for m, _, _ in g]),
                       mean([f for _, f, _ in g if f is not None]))
    base_bad, base_n = stats["all pass"][:2]
    lines = ["# Checker comparison", "",
             f"The universe is {picks['universe']} of the translator's translations with no rule flag that the judge "
             f"does not call bound to English. The judge rejects {picks['judge_rejects']} ({fmt(picks['share'], '.1%')}) "
             f"under meaning {picks['min_meaning']}; each candidate rejects as many, its lowest-scored (fewer when "
             "scores tie at its threshold). Bad = meaning ≤ 3.",
             ""]
    if picks["missing"]:
        lines += [f"No checks from {', '.join(picks['missing'])}: left out.", ""]
    if unreadable:
        lines += ["Unreadable grades, left out: " + ", ".join(f"row {n} {c} {v!r}" for n, c, v in unreadable) + ".", ""]
    lines += ["## Candidates", "", "| candidate | threshold (quality) | rejects | also rejected by the judge | "
              "English-only calls |", "|---|---:|---:|---:|---:|"]
    for role, t in picks["thresholds"].items():
        lines.append(f"| {role} | {fmt(t, '.3f')} | {picks['rejects'][role]} | {picks['overlap'][role]} | "
                     f"{picks['english_only'].get(role, '—')} |")
    lines += ["", "## Groups", "", "| group | in the universe | graded | bad | share bad (95% interval) | mean meaning "
              "| mean fluency | vs all pass (Fisher p) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    pvals = {}
    for name, (bad, n, m, f) in stats.items():
        lo, hi = wilson(bad, n)
        if name != "all pass" and n and base_n:
            pvals[name] = fisher_greater(bad, n - bad, base_bad, base_n - base_bad)
        share = f"{fmt(bad / n if n else None, '.0%')} ({fmt(lo, '.0%')}–{fmt(hi, '.0%')})"
        lines.append(f"| {name} | {picks['sizes'][name]} | {n} | {bad} | {share} | {fmt(m)} | {fmt(f)} | "
                     f"{fmt(pvals.get(name), '.3f')} |")
    alone = {role: stats[f"only {role} rejects"][:2] for role in picks["thresholds"]}
    qualified = sorted(((bad / n, role) for role, (bad, n) in alone.items()
                        if n and bad / n >= 0.5 and pvals.get(f"only {role} rejects", 1) < 0.05), reverse=True)
    lines += ["", "## Decision", ""]
    if qualified:
        role = qualified[0][1]
        lines.append(f"Add `{role}` as a second gate: `translate.gates: [{{role: {role}, min_quality: "
                     f"{picks['thresholds'][role]}}}]`. It scored only a sample, so first `distill check-translation "
                     f"--role {role} data/translate/*_th.jsonl` (a translation it has not scored fails the gate), then "
                     "`distill build-translated --dataset <set>` for each Thai set. The judge stays the checker.")
    else:
        lines.append("No candidate qualifies: the judge stays the only checker.")
    judge_bad, judge_n = stats["only judge rejects"][:2]
    if judge_n and judge_bad / judge_n < 0.5:
        lines += ["", "Most translations only the judge rejects were graded good: the judge rejects good translations "
                      "too. Taking them back would reshuffle the seeded sample, so that is a follow-up."]
    notes = [(n, k, grades[n]) for n, k in key.items() if n in grades and grades[n][2]]
    if notes:
        lines += ["", "## Grader notes", "", "| row | groups | meaning | note |", "|---:|---|---:|---|"]
    for n, k, (meaning, _, note) in notes:
        groups = picks["rows"].get(f"{k['dataset']}/{k['id']}", {}).get("groups", [])
        lines.append(f"| {n} | {', '.join(groups)} | {'—' if meaning is None else meaning} | "
                     f"{str(note).replace('|', '/')} |")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[checkers] report -> {out}")
    return stats

