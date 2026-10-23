"""Back up a run to a private Hugging Face dataset repo: a safety net for data that otherwise lives only on the VM.
Files keep their local paths, so `hf download REPO --repo-type dataset --local-dir .` puts them back where every
stage resumes from them. Unchanged files are not uploaded again. A dataset card (README.md, from
docs/dataset_card.md) names the subsets the Hub viewer and `load_dataset` show, with one split per set."""
import json
import os
import re
from collections import Counter
from pathlib import Path

import yaml

from distill.data import PREPARED_DIR

RUN_FILES = ["*.jsonl", "*.json"]                 # generations (with the raw thinking), kept, judge, final, run logs
RATINGS = ["*.complexity.jsonl", "*.runs.jsonl"]  # the prompt caches next to them are rebuilt by `prepare`
CARD_TEMPLATE = Path(__file__).resolve().parent.parent / "docs" / "dataset_card.md"
# <set>_<variant><stage>.jsonl, e.g. no_robots_open_B.final.jsonl or helpsteer2_open_B_raw.jsonl
RUN_FILE = re.compile(r"(?P<set>\w+?)_(?P<variant>[A-Z])(?P<stage>_raw|\.kept\.judge|\.kept\.rejected|\.kept|\.final)?"
                      r"\.jsonl")
SUBSETS = [  # (subset, file stage, contents); the raw and kept files are left out: `samples` holds the same rows
    ("final", ".final", "The training rows (fields below). Start here."),
    ("judge", ".kept.judge", "The judge's score, expected score and critique for every sample that passed the filter."),
    ("samples", "", "Every generated sample, with Qwen's raw thinking (`reasoning`) and the compressed `rationale`."),
    ("rejected", ".kept.rejected", "Samples the filter dropped, with `reject_reasons`."),
]


def push(cfg, out, repo=None, api=None):
    """Upload the run directory `out`, the complexity ratings and a fresh dataset card to `repo` (default: config
    hub.repo). Returns False when no repo is set. Only a private repo is accepted: the data is unreviewed and No
    Robots outputs are non-commercial."""
    repo = repo or (cfg.get("hub") or {}).get("repo")
    if not repo:
        print("[push] no hub.repo in the config; skipping the backup")
        return False
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise RuntimeError(f"{repo} is public; the backup only goes to a private repo")
    for folder, patterns in ((Path(out), RUN_FILES), (PREPARED_DIR, RATINGS)):
        if any(any(folder.glob(p)) for p in patterns):
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, path_in_repo=repo_path(folder),
                              allow_patterns=patterns, commit_message=f"backup {repo_path(folder)}")
    api.upload_file(path_or_fileobj=card(out, repo).encode("utf-8"), path_in_repo="README.md", repo_id=repo,
                    repo_type="dataset", commit_message="dataset card")
    print(f"[push] {out} and the ratings are backed up to https://huggingface.co/datasets/{repo}")
    return True


def repo_path(path):
    """`path` relative to the working directory, as the repo stores it."""
    rel = Path(os.path.relpath(path)).as_posix()
    if rel.startswith(".."):
        raise ValueError(f"{path} is outside the working directory")
    return rel


def card(out, repo):
    """The dataset card: metadata with one subset per file stage (one split per set), a status table, and one example
    row per set. Closed sets get their own subsets (`final_closed`, ...): their rows carry `gold`, and the splits of
    one subset must share a schema. It only names files that exist, so it stays valid while a run is in progress."""
    files = run_files(out)
    ratings = [(p.name.removesuffix(".complexity.jsonl"), p) for p in sorted(PREPARED_DIR.glob("*.complexity.jsonl"))]
    subsets = [(name + ("_closed" if closed else ""), text + (" Closed prompts: a known short answer in `gold`."
                                                              if closed else ""), files[(stage, closed)])
               for name, stage, text in SUBSETS for closed in (False, True) if files.get((stage, closed))]
    if ratings:
        subsets.append(("ratings", "Each prompt's complexity level (1–5) and the rater's analysis.", ratings))
    configs = [{"config_name": name, **({"default": True} if name == "final" else {}),
                "data_files": [{"split": s, "path": repo_path(p)} for s, p in pairs]} for name, _, pairs in subsets]
    meta = {"pretty_name": "Capstone distillation data", "language": ["en"], "license": "other",
            "license_name": "mixed-sources", "task_categories": ["text-generation"],
            "tags": ["distillation", "chain-of-thought", "synthetic", "reasoning"], "configs": configs}
    table = ["| subset | contents | splits |", "|---|---|---|"]
    table += [f"| `{name}` | {text} | {', '.join(s for s, _ in pairs)} |" for name, text, pairs in subsets]
    examples = [example_md(s + (" (closed)" if closed else ""), row) for closed in (False, True)
                for s, p in files.get((".final", closed), []) if (row := pick_example(p))]
    body = (CARD_TEMPLATE.read_text(encoding="utf-8").replace("<!-- repo -->", repo)
            .replace("<!-- status -->", status_table(files)).replace("<!-- subsets -->", "\n".join(table))
            .replace("<!-- examples -->", "\n\n".join(examples) or "No final rows yet."))
    return f"---\n{yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)}---\n\n{body}"


def run_files(out):
    """{(stage, closed): [(split, path)]} for the run files in `out`. A split is the dataset name (the set without
    `_open` / `_closed`), plus the variant when the directory holds more than one."""
    found = [(m, p) for p in sorted(Path(out).glob("*.jsonl")) if (m := RUN_FILE.fullmatch(p.name))]
    several = len({m["variant"] for m, _ in found}) > 1
    files = {}
    for m, p in found:
        dataset, _, subset = m["set"].rpartition("_")
        split = (dataset if subset in ("open", "closed") else m["set"]) + (f"_{m['variant']}" if several else "")
        files.setdefault((m["stage"] or "", subset == "closed"), []).append((split, p))
    return files


def count_lines(path):
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))


def status_table(files):
    """Samples, kept and final rows per set; a set without final rows is still in progress."""
    by_stage = {key: dict(pairs) for key, pairs in files.items()}
    rows = ["| set | samples | kept by the filter | final rows | final rows by complexity 1 / 2 / 3 / 4 / 5 / unrated |",
            "|---|---:|---:|---:|---|"]
    for closed, split in sorted({(closed, s) for (_, closed), pairs in files.items() for s, _ in pairs}):
        path = {stage: by_stage.get((stage, closed), {}).get(split) for stage in ("", "_raw", ".kept", ".final")}
        samples, kept, final = path[""] or path["_raw"], path[".kept"], path[".final"]
        if closed:
            split += " (closed)"
        levels = "—"
        if final:
            with open(final, encoding="utf-8") as f:
                counts = Counter(json.loads(line).get("complexity") for line in f)
            levels = " / ".join(f"{counts.get(level, 0):,}" for level in (1, 2, 3, 4, 5, None))
        rows.append(f"| {split} | {lines(samples)} | {lines(kept)} | {lines(final, 'in progress')} | {levels} |")
    return "\n".join(rows)


def lines(path, missing="—"):
    return f"{count_lines(path):,}" if path else missing


def pick_example(path, level=3):
    """The first row at complexity `level`, else the first row."""
    first = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("complexity") == level:
                return row
            first = first or row
    return first


def shorten(text, limit):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def example_md(split, row):
    parts = [f"### {split}: complexity {row.get('complexity')}, judge score {(row.get('selection') or {}).get('score')}"]
    for field, limit in (("input", 500), ("rationale", 1200), ("answer", 700)):
        parts.append(f"**{field}**\n\n````text\n{shorten(row.get(field), limit)}\n````")  # 4 backticks: rows may hold ```
    return "\n\n".join(parts)
