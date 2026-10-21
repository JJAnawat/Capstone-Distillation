"""Load prompts from the HF datasets, dedupe them, and draw seeded stratified samples."""
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

PREPARED_DIR = Path("data/prepared")


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def make_id(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


# Each loader yields dicts: input, strata, multi_turn, reference, ratings (of the reference), gold (known answers).

def _no_robots(ds, cfg):
    for r in ds:
        msgs = r["messages"]
        yield {"input": r[cfg["input_field"]], "strata": r["category"],
               "multi_turn": [m["role"] for m in msgs] != ["user", "assistant"],
               "reference": msgs[-1]["content"], "ratings": {}, "gold": []}


def _helpsteer2(ds, cfg):
    # Each prompt has ~2 rated responses; keep the best one as the reference.
    best = {}
    for r in ds:
        p = r[cfg["input_field"]]
        key = (r["helpfulness"], r["correctness"], r["coherence"])
        if p not in best or key > best[p][0]:
            best[p] = (key, r)
    for p, (_, r) in best.items():
        yield {"input": p, "strata": None, "multi_turn": "<extra_id_1>" in p, "reference": r["response"],
               "ratings": {k: r[k] for k in ("helpfulness", "correctness", "coherence", "complexity", "verbosity")},
               "gold": []}


def _ultrafeedback(ds, cfg):
    for r in ds:
        if not r["completions"]:
            continue
        c = max(r["completions"], key=lambda c: (c["fine-grained_score"] or 0, c["overall_score"] or 0))
        # correct_answers holds the literal string "None" for sources without a known answer.
        gold = [a.strip() for a in r["correct_answers"] or [] if a and a.strip() != "None"]
        yield {"input": r[cfg["input_field"]], "strata": r["source"], "multi_turn": False,
               "reference": c["response"],
               "ratings": {"fine_grained_score": c["fine-grained_score"] or 0, "overall_score": c["overall_score"] or 0},
               "gold": gold}


LOADERS = {"no_robots": _no_robots, "helpsteer2": _helpsteer2, "ultrafeedback": _ultrafeedback}


def is_closed(r, vcfg):
    """Closed-ended: every gold answer is short and comes from a source whose answers a rule can check."""
    return (r["strata"] in vcfg["sources"] and bool(r["gold"])
            and all(len(a.split()) <= vcfg["max_gold_words"] for a in r["gold"]))


def prepare(cfg):
    """Load every dataset in config order, dedupe within and across them, write data/prepared/*.jsonl."""
    from datasets import load_dataset

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    seen, stats = set(), {}
    for name, dcfg in cfg["datasets"].items():
        ds = load_dataset(dcfg["hf_id"], split=dcfg["split"])
        thresholds = dcfg.get("hint", {}).get("min_ratings", {})
        st = {"raw_rows": len(ds), "empty": 0, "multi_turn": 0, "duplicate": 0, "kept": 0, "hint_eligible": 0,
              "closed": 0}
        with open(PREPARED_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in LOADERS[name](ds, dcfg):
                if not r["input"].strip():
                    st["empty"] += 1
                    continue
                if r["multi_turn"]:
                    st["multi_turn"] += 1
                    continue
                rid = make_id(r["input"])
                if rid in seen:
                    st["duplicate"] += 1
                    continue
                seen.add(rid)
                r["id"], r["source_dataset"] = rid, name
                r["hint_ok"] = all(r["ratings"].get(k, 0) >= v for k, v in thresholds.items())
                r["closed"] = is_closed(r, cfg["verify"])
                st["kept"] += 1
                st["hint_eligible"] += r["hint_ok"]
                st["closed"] += r["closed"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = st
        print(f"[prepare] {name}: {st}")
    (PREPARED_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_prompts(cfg, name):
    path = PREPARED_DIR / f"{name}.jsonl"
    if not path.exists():
        prepare(cfg)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if rows and "closed" not in rows[0]:
        print(f"[data] {path} predates gold answers; treating every prompt as open. Re-run `distill prepare`.")
        for r in rows:
            r.update(gold=[], closed=False)
    return rows


def select_rows(cfg, dataset, mode="open", only=None, limit=None, seed=0):
    """The prompts a stage works on: hint-eligible ones in hint mode, optionally only closed or open ones, then a seeded
    stratified sample. rate and generate get the same subset from the same arguments."""
    rows = load_prompts(cfg, dataset)
    if mode == "hint":
        rows = [r for r in rows if r["hint_ok"]]
    if only:
        rows = [r for r in rows if bool(r.get("closed")) == (only == "closed")]
    if limit:
        rows = sample(rows, limit, seed)
    return rows


def sample(rows, limit, seed):
    """Seeded sample with equal allocation across strata (round-robin over shuffled groups)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["strata"]].append(r)
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out = []
    while len(out) < limit and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < limit:
                out.append(groups[k].pop())
    return out
