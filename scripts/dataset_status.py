"""Markdown status of the distilled dataset (every *.final.jsonl in data/full and data/full_th), aggregates only."""
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def pct(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else 0


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


sets = []
for run in ("data/full", "data/full_th"):
    for path in sorted(Path(run).glob("*.final.jsonl")):
        rows = load(path)
        stats = Path(str(path).removesuffix(".jsonl") + ".select_stats.json")
        sel = json.loads(stats.read_text(encoding="utf-8")) if stats.exists() else {}
        name = path.name.removesuffix("_B.final.jsonl")
        sets.append({"name": name, "lang": "Thai" if run.endswith("_th") else "English", "rows": rows, "sel": sel})


def line(name, rows, sel=None):
    tot = [r["student_tokens"]["total"] for r in rows]
    part = lambda k: statistics.median(r["student_tokens"][k] for r in rows)
    prompts = len({r["id"] for r in rows})
    lv = Counter(r.get("complexity") for r in rows)
    meth = Counter(r["selection"].get("method") for r in rows)
    scores = [r["selection"]["score"] for r in rows if r["selection"].get("score") is not None]
    dropped = sum(v for k, v in (sel or {}).items() if k.startswith("dropped"))
    return (f"| {name} | {len(rows):,} | {prompts:,} | {len(rows) / prompts:.2f} | {dropped:,} | {sum(tot):,} | "
            f"{statistics.median(tot):.0f} / {pct(tot, 90)} / {max(tot)} | "
            f"{100 * sum(t > 1024 for t in tot) / len(tot):.1f}% | "
            f"{part('input'):.0f} / {part('rationale'):.0f} / {part('answer'):.0f} | "
            f"{statistics.median(r['n_steps'] for r in rows):.0f} | "
            + " / ".join(str(lv[k]) for k in (1, 2, 3, 4, 5)) + f" / {lv[None]} | "
            f"{meth['rule+judge']:,} / {meth['judge']:,} | {statistics.mean(scores):.2f} / "
            f"{100 * sum(s >= 8 for s in scores) / len(scores):.0f}% |")


out = [f"# Dataset status ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)", "",
       "Final training rows (`*.final.jsonl` in `data/full` and `data/full_th`). Tokens are the student's "
       "(`student_tokens.total`: input + rationale + answer + 32 template tokens; the context is 2,048).", "",
       "| set | rows | prompts | rows/prompt | prompts dropped | tokens | tokens median / p90 / max | > 1,024 tokens "
       "| median input / rationale / answer | median steps | complexity 1/2/3/4/5/unrated | rule+judge / judge | "
       "judge mean / ≥ 8 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for s in sets:
    out.append(line(f"{s['name']} ({s['lang']})", s["rows"], s["sel"]))
for lang in ("English", "Thai"):
    group = [s for s in sets if s["lang"] == lang]
    out.append(line(f"**{lang}**", [r for s in group for r in s["rows"]],
                    Counter() + sum((Counter({k: v for k, v in s["sel"].items() if k.startswith("dropped")})
                                     for s in group), Counter())))
everything = [r for s in sets for r in s["rows"]]
out.append(line("**all**", everything, sum((Counter({k: v for k, v in s["sel"].items() if k.startswith("dropped")})
                                             for s in sets), Counter())))
en = sum(len(s["rows"]) for s in sets if s["lang"] == "English")
th = sum(len(s["rows"]) for s in sets if s["lang"] == "Thai")
out += ["", f"English : Thai = {en:,} : {th:,} = 1 : {th / en:.2f} (target about 1 : 0.5).", "",
        "## Why select dropped prompts", "", "| set | " + " | ".join(
            k := sorted({k for s in sets for k in s["sel"] if k.startswith("dropped")})) + " |",
        "|---|" + "---:|" * len(k)]
out += [f"| {s['name']} ({s['lang']}) | " + " | ".join(str(s["sel"].get(x, 0)) for x in k) + " |" for s in sets]
closed = [s for s in sets if "closed" in s["name"]]
out += ["", "## Closed sets", "", "| set | prompts kept | prompts without a correct answer | correct rate |",
        "|---|---:|---:|---:|"]
for s in closed:
    kept = s["sel"].get("kept: rule+judge", 0) + s["sel"].get("kept: judge", 0)
    none = s["sel"].get("dropped: no correct answer", 0)
    out.append(f"| {s['name']} ({s['lang']}) | {kept:,} | {none:,} | "
               f"{100 * s['sel'].get('kept: rule+judge', 0) / max(1, s['sel'].get('prompts', 1)):.0f}% |")
print("\n".join(out))
