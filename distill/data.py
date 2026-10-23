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


def _json_objects(text):
    """Every JSON object in `text`, back to back (two LogiQA mirror rows hold two)."""
    dec, text, pos = json.JSONDecoder(), text.strip(), 0
    while pos < len(text):
        obj, pos = dec.raw_decode(text, pos)
        yield obj
        while pos < len(text) and text[pos].isspace():
            pos += 1


def _logiqa(ds, cfg):
    # The datatune/LogiQA2.0 mirror stores each row as a JSON string in `text`. It mixes English and Chinese
    # multiple-choice rows with NLI rows; only the 12,567 English multiple-choice rows have both `options` and `type`.
    for j in (j for r in ds for j in _json_objects(r[cfg["input_field"]])):
        if "options" not in j or "type" not in j:
            continue
        letters = "ABCDEFGH"[:len(j["options"])]
        options = "\n".join(f"{k}. {o.strip()}" for k, o in zip(letters, j["options"]))
        prompt = (f"{j['text'].strip()}\n\n{j['question'].strip()}\n\n{options}\n\n"
                  "Answer with the letter of the correct option.")
        k, option = letters[j["answer"]], j["options"][j["answer"]].strip()  # `answer` is a 0-based index
        yield {"input": prompt, "strata": next((t for t, on in j["type"].items() if on), None), "multi_turn": False,
               "reference": None, "ratings": {}, "gold": [k, f"{k}. {option}", option]}


LOADERS = {"no_robots": _no_robots, "helpsteer2": _helpsteer2, "ultrafeedback": _ultrafeedback, "logiqa": _logiqa}


def exact_gold(cfg):
    """Datasets whose gold answers are exact (`exact_gold: true`, e.g. LogiQA's option letter). There, a closed prompt
    with no correct sample is dropped rather than judged: the judge could only pick among wrong answers. Other closed
    prompts fall back to the judge, since a noisy gold may be right without matching exactly."""
    return {name for name, d in cfg["datasets"].items() if d.get("exact_gold")}


def is_closed(r, vcfg):
    """Closed-ended: every gold answer is short and comes from a source whose answers a rule can check."""
    return (r["strata"] in vcfg["sources"] and bool(r["gold"])
            and all(len(a.split()) <= vcfg["max_gold_words"] for a in r["gold"]))


def prepare(cfg, only=None):
    """Load every dataset in config order, dedupe within and across them, write data/prepared/<name>.jsonl. With
    `only`, build just that dataset: the earlier datasets' prepared files supply the ids to dedupe against and stay
    as they are, so their samples (and runs in progress) do not change. Meant for a dataset added at the end.
    A dataset with `closed: true` in its config is closed wherever it has gold answers."""
    from datasets import load_dataset

    names = list(cfg["datasets"])
    if only and only not in names:
        raise ValueError(f"unknown dataset {only}; config has {names}")
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = PREPARED_DIR / "stats.json"
    seen = set()
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if only and stats_path.exists() else {}
    for name, dcfg in cfg["datasets"].items():
        if only and name != only:
            if names.index(name) < names.index(only):
                with open(PREPARED_DIR / f"{name}.jsonl", encoding="utf-8") as f:
                    seen.update(json.loads(line)["id"] for line in f)
            continue
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
                r["closed"] = bool(r["gold"]) if dcfg.get("closed") else is_closed(r, cfg["verify"])
                st["kept"] += 1
                st["hint_eligible"] += r["hint_ok"]
                st["closed"] += r["closed"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = st
        print(f"[prepare] {name}: {st}")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_prompts(cfg, name):
    """The dataset's prepared prompts. A missing file is an error, not a silent full `prepare`: that rebuilds every
    dataset and downloads them all (it once did so by accident on the VM, onto the full root disk)."""
    path = PREPARED_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `distill prepare` (every dataset) or "
                                f"`distill prepare --dataset {name}` (one added at the end of the config)")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if rows and "closed" not in rows[0]:
        print(f"[data] {path} predates gold answers; treating every prompt as open. Re-run `distill prepare`.")
        for r in rows:
            r.update(gold=[], closed=False)
    return rows


def subsets(cfg, name):
    """Which kinds of prompts the dataset has, `open` and/or `closed`; pipeline.sh makes a set of each."""
    kinds = {bool(r.get("closed")) for r in load_prompts(cfg, name)}
    return [s for s, closed in (("open", False), ("closed", True)) if closed in kinds]


def complexity_path(dataset):
    """Where `rate` writes the dataset's complexity ratings (1-5)."""
    return PREPARED_DIR / f"{dataset}.complexity.jsonl"


def load_complexity(dataset):
    """{id: complexity} from the dataset's ratings, or {} if it has not been rated."""
    path = complexity_path(dataset)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {j["id"]: j["complexity"] for j in map(json.loads, filter(str.strip, f))}


def by_complexity(table, level, default):
    """A per-level setting such as `generate.n_by_complexity` (YAML keys may load as int or str)."""
    return table.get(level, table.get(str(level), default)) if level is not None else default


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
