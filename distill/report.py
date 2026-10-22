"""Analyze generated JSONL files and filter them with heuristics."""
import bisect
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from distill import verify
from distill.data import load_complexity

STUDENT_TEMPLATE_TOKENS = 32  # system:/ผู้ใช้:/ผู้ช่วย: markers, BOS/EOS and a short system prompt


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path):
    return list(iter_jsonl(path))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def judge_path(path):
    """Where `judge` writes the scores for a filtered file."""
    return f"{str(path).removesuffix('.jsonl')}.judge.jsonl"


@lru_cache(maxsize=1)
def _student_tokenizer(name):
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained(name)


def student_counter(tokenizer):
    """A function counting tokens of a text in the student's tokenizer."""
    tok = _student_tokenizer(tokenizer)
    return lambda text: len(tok.encode(text or "", add_special_tokens=False).ids)


def student_tokens(r, tokenizer):
    """Tokens a sample takes in the student's context: input, rationale and answer, plus chat-template markers."""
    text = "\n".join(t for t in (r.get("input"), r.get("rationale"), r.get("answer")) if t)
    return len(_student_tokenizer(tokenizer).encode(text, add_special_tokens=False).ids) + STUDENT_TEMPLATE_TOKENS


# ---------------------------------------------------------------- analyze

def pct(values, p):
    """Nearest-rank percentile."""
    if not values:
        return 0
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def share(rows, cond):
    return f"{100 * sum(map(cond, rows)) / len(rows):.0f}%"


def variant(r):
    """Column label: how a sample was generated, e.g. cot/off/prompted or open/on/compressed."""
    label = f"{r.get('mode', 'open')}/{'on' if r.get('thinking') else 'off'}"
    return f"{label}/{r['cot_method']}" if r.get("cot_method") else label


def by_prompt(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["id"]].append(r)
    return groups


def correct(r):
    return verify.is_correct(r.get("answer"), r.get("gold") or [])


def best_scores(groups):
    """Best expected judge score of every prompt that has judged samples."""
    return [max(r["_judge"]["expected_score"] for r in g if r.get("_judge")) for g in groups
            if any(r.get("_judge") for r in g)]


def group_stats(rows, min_score, student=None):
    prompts = by_prompt(rows)
    rt, at = [r.get("reasoning_tokens", 0) for r in rows], [r.get("answer_tokens", 0) for r in rows]
    rw = [len((r.get("rationale") or "").split()) for r in rows]
    stats = {
        "samples / prompts": f"{len(rows)} / {len(prompts)}",
        "raw reasoning tokens median / p90": f"{pct(rt, 50)} / {pct(rt, 90)}",
        "rationale words median / p90": f"{pct(rw, 50)} / {pct(rw, 90)}",
        "rationale steps median": pct([verify.count_steps(r.get("rationale")) for r in rows], 50),
        "answer tokens median / p90": f"{pct(at, 50)} / {pct(at, 90)}",
        "finish_reason == length": share(rows, lambda r: r.get("finish_reason") == "length"),
        "empty rationale": share(rows, lambda r: not (r.get("rationale") or "").strip()),
        "unclosed think tag": share(rows, lambda r: r.get("think_unclosed")),
        "total cost (USD)": f"{sum(r.get('cost') or 0 for r in rows):.4f}",
    }
    if student:
        st = [student_tokens(r, student) for r in rows]
        stats["student tokens median / p90"] = f"{pct(st, 50)} / {pct(st, 90)}"
        stats["fits student ≤1024 / ≤1536 / ≤2048"] = " / ".join(
            f"{100 * sum(t <= lim for t in st) / len(st):.0f}%" for lim in (1024, 1536, 2048))
    judged = [[r["_judge"] for r in g if r.get("_judge")] for g in prompts.values()]
    judged = [js for js in judged if js]
    if judged:
        scores = [j["score"] for js in judged for j in js]
        stats["judged samples"] = len(scores)
        stats["score histogram 1..9"] = " ".join(str(scores.count(s)) for s in range(1, 10))
        stats["mean score / mean best-of-N"] = (f"{mean(j['expected_score'] for js in judged for j in js):.2f} / "
                                               f"{mean(max(j['expected_score'] for j in js) for js in judged):.2f}")
        good = sum(max(j["score"] for j in js) >= min_score for js in judged)
        stats[f"prompts with a sample scoring ≥ {min_score}"] = f"{100 * good / len(prompts):.0f}%"
        # Logprobs arrived when the expected score differs from the integer score.
        stats["judge scores with logprob detail"] = share([j for js in judged for j in js],
                                                          lambda j: j["expected_score"] != j["score"])
    closed = [g for g in prompts.values() if g[0].get("closed")]
    if closed:
        hits = [[correct(r) for r in g] for g in closed]
        stats["closed pass@1 / pass@N"] = (f"{100 * mean(mean(h) for h in hits):.0f}% / "
                                          f"{100 * mean(any(h) for h in hits):.0f}%")
        stats.update(judge_vs_rules([r for g in closed for r in g], min_score))
    return stats


def auc(pos, neg):
    """P(a random pos score > a random neg score), ties counting half (Mann-Whitney U)."""
    neg = sorted(neg)
    wins = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p))
               for p in pos)
    return wins / (len(pos) * len(neg))


def judge_vs_rules(rows, min_score):
    """How well the judge separates rule-correct from rule-wrong closed samples; needs wrong samples judged too
    (`judge --all`)."""
    right = [r["_judge"] for r in rows if r.get("_judge") and correct(r)]
    wrong = [r["_judge"] for r in rows if r.get("_judge") and not correct(r)]
    if not right or not wrong:
        return {}
    separation = auc([j["expected_score"] for j in right], [j["expected_score"] for j in wrong])
    return {"judge vs rules: judged correct / wrong samples": f"{len(right)} / {len(wrong)}",
            "judge vs rules: mean score correct / wrong": (f"{mean(j['expected_score'] for j in right):.2f} / "
                                                          f"{mean(j['expected_score'] for j in wrong):.2f}"),
            "judge vs rules: AUC (correct scored above wrong)": f"{separation:.2f}",
            f"judge vs rules: wrong scored ≥ {min_score} (false accept)": share(wrong, lambda j: j["score"] >= min_score),
            f"judge vs rules: correct scored < {min_score}": share(right, lambda j: j["score"] < min_score)}


def kept_within(samples, min_score):
    """Whether `select` would keep anything from these samples: rule-correct ones first when a prompt is closed and
    any are correct, else all of them; at least one must be judged ≥ min_score."""
    passing = [r for r in samples if r.get("closed") and correct(r)]
    return any((r.get("_judge") or {}).get("score", 0) >= min_score for r in passing or samples)


def best_of_k_lines(name, rows, min_score):
    """What the first k samples of every prompt would give, for k up to N: how much extra samples are worth.
    Filtered-out samples count as missing, as they would in a run with N = k."""
    groups = list(by_prompt(rows).values())
    n = 1 + max(r.get("sample_idx", 0) for r in rows)
    ks = sorted({k for k in (1, 2, 4, 6, 8) if k < n} | {n})
    closed = groups[0][0].get("closed")
    lines = ["", f"Best-of-k ({name}; {len(groups)} prompts):", "",
             "| k | mean best score | yield (kept by select) |" + (" pass@k |" if closed else ""),
             "|---:|---:|---:|" + ("---:|" if closed else "")]
    for k in ks:
        firsts = [[r for r in g if r.get("sample_idx", 0) < k] for g in groups]
        best = best_scores([f for f in firsts if f])
        cells = [f"{mean(best):.2f}" if best else "–",
                 f"{100 * mean(kept_within(f, min_score) for f in firsts):.0f}%"]
        if closed:
            cells.append(f"{100 * mean(any(correct(r) for r in f) for f in firsts):.0f}%")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return lines


def matched_line(cols, min_score):
    """Key numbers of every column restricted to the prompts that all columns share (a fair A-vs-B comparison)."""
    common = set.intersection(*(set(by_prompt(rows)) for rows in cols.values()))
    parts = []
    for name, rows in sorted(cols.items()):
        groups = [g for rid, g in by_prompt(rows).items() if rid in common]
        best = best_scores(groups)
        part = f"{name}: mean best score {mean(best):.2f}" if best else f"{name}: not judged"
        if groups and groups[0][0].get("closed"):
            part += f", pass@N {100 * mean(any(correct(r) for r in g) for g in groups):.0f}%"
        parts.append(part)
    return f"Matched prompts present in every column: {len(common)}. " + "; ".join(parts) + "."


def level_lines(cols, levels, min_score):
    """Per complexity level and column: prompts, and the share with a sample scoring ≥ min_score (open prompts) or with
    a correct sample (closed prompts)."""
    names = sorted(cols)
    lines = ["", f"By complexity level (prompts, and share with a sample scoring ≥ {min_score} / pass@N):", "",
             "| complexity | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for level in (1, 2, 3, 4, 5, None):
        cells = []
        for name in names:
            groups = [g for rid, g in by_prompt(cols[name]).items() if levels.get(rid) == level]
            if not groups:
                cells.append("–")
            elif groups[0][0].get("closed"):
                cells.append(f"{len(groups)}, {100 * mean(any(correct(r) for r in g) for g in groups):.0f}%")
            elif any(r.get("_judge") for g in groups for r in g):
                good = mean(any((r.get("_judge") or {}).get("score", 0) >= min_score for r in g) for g in groups)
                cells.append(f"{len(groups)}, {100 * good:.0f}%")
            else:
                cells.append(str(len(groups)))
        lines.append(f"| {level or 'unrated'} | " + " | ".join(cells) + " |")
    return lines


def clip(text, n=600):
    text = text if len(text) <= n else text[:n] + "…"
    return html.escape(text).replace("\n", "<br>")


def example_lines(rows, k, seed=0):
    """Up to k prompts with their best-scored sample (any sample before judging), clipped for reading."""
    groups = list(by_prompt(rows).values())
    random.Random(seed).shuffle(groups)
    lines = []
    for g in groups[:k]:
        r = max(g, key=lambda s: (s.get("_judge") or {}).get("expected_score", 0))
        j = r.get("_judge") or {}
        critique = f"<br><br>**Judge:** {clip(j.get('critique', ''))}" if j else ""
        lines += [f"<details><summary>{r['id']} / sample {r.get('sample_idx', 0)} "
                  f"({'score ' + str(j['score']) if j else 'not judged'})</summary>", "",
                  f"**Prompt:** {clip(r['input'])}<br><br>**Rationale:** {clip(r.get('rationale') or '(none)')}<br><br>"
                  f"**Answer:** {clip(r.get('answer') or '')}{critique}", "</details>", ""]
    return lines


def throughput_lines(paths):
    """Run summaries (<out>.runs.jsonl) found next to the inputs."""
    runs = [(f.name.removesuffix(".runs.jsonl"), s) for d in sorted({Path(p).parent for p in paths})
            for f in sorted(d.glob("*.runs.jsonl")) for s in iter_jsonl(f)]
    if not runs:
        return []
    lines = ["## Throughput", "", "| output | stage | backend | done | failed | tok/s | minutes |",
             "|---|---|---|---:|---:|---:|---:|"]
    lines += [f"| {name} | {s['stage']} | {s.get('backend', '')} | {s['done']} | {s['failed']} | "
              f"{s['tok_per_s']:,.0f} | {s['seconds'] / 60:,.1f} |" for name, s in runs]
    return lines + [""]


def analyze(paths, out_path, min_score=7, student=None, examples=5):
    """Markdown report: one section per (dataset, open|closed), one column per generation variant. Judge scores are
    read from each input's sibling .judge.jsonl and complexity ratings from data/prepared/."""
    sections = defaultdict(lambda: defaultdict(list))  # (dataset, subset) -> variant -> rows
    for p in paths:
        jp = judge_path(p)
        scores = {(j["id"], j["sample_idx"]): j for j in iter_jsonl(jp)} if Path(jp).exists() else {}
        for r in iter_jsonl(p):
            r["_judge"] = scores.get((r["id"], r.get("sample_idx", 0)))
            sections[(r["source_dataset"], "closed" if r.get("closed") else "open")][variant(r)].append(r)
    if student:
        try:
            _student_tokenizer(student)
        except Exception as e:  # offline or package missing: report everything else
            print(f"[analyze] student tokenizer unavailable ({e}); skipping student-token metrics")
            student = None

    lines = ["# Teacher output analysis", "", f"Inputs: {', '.join(map(str, paths))}", ""]
    for (dataset, subset), cols in sorted(sections.items()):
        names = sorted(cols)
        stats = {c: group_stats(cols[c], min_score, student) for c in names}
        metrics = list(dict.fromkeys(m for c in names for m in stats[c]))
        lines += [f"## {dataset} / {subset}", "", "| metric | " + " | ".join(names) + " |",
                  "|---|" + "---:|" * len(names)]
        lines += [f"| {m} | " + " | ".join(str(stats[c].get(m, "–")) for c in names) + " |" for m in metrics]
        if len(names) > 1:
            lines += ["", matched_line(cols, min_score)]
        levels = load_complexity(dataset)
        if levels:
            lines += level_lines(cols, levels, min_score)
        for c in names:
            if any(r.get("_judge") for r in cols[c]):
                lines += best_of_k_lines(c, cols[c], min_score)
        for c in names if examples else []:
            lines += ["", f"### Examples: {c}", ""] + example_lines(cols[c], examples)
        lines.append("")
    lines += throughput_lines(paths)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[analyze] wrote {out_path}")


# ---------------------------------------------------------------- filter

@lru_cache(maxsize=1)
def _detector():
    from lingua import LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_all_languages().build()


def detect_language(text, min_chars):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code says nothing about natural language
    if len(text.strip()) < min_chars:
        return None
    return _detector().detect_language_of(text)


def repetition_ratio(text, n, min_ngrams):
    """1 - unique/total n-grams. Word n-grams, or character n-grams for scripts without spaces."""
    tokens = text.split()
    if len(tokens) < len(text) / 20:  # e.g. Chinese/Japanese/Thai
        tokens, n = list(text.replace(" ", "")), n * 3
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if len(grams) < min_ngrams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def rejection_reasons(r, fcfg, max_reasoning_tokens=None, max_rationale_words=None, student=None):
    """`student` is (tokenizer name, max tokens): reject samples that would not fit the student's context."""
    reasons = []
    answer, reasoning, rationale = r.get("answer") or "", r.get("reasoning") or "", r.get("rationale") or ""
    if not answer.strip():
        reasons.append("missing_answer")
    if r.get("think_unclosed") or re.search(r"</?think>", answer):
        reasons.append("malformed_think_tag")
    if r.get("finish_reason") != "stop":
        reasons.append("finish_not_stop")
    if answer.strip() and not any(re.search(p, r["input"], re.I) for p in fcfg["language_skip_patterns"]):
        lp = detect_language(r["input"], fcfg["language"]["min_chars"])
        la = detect_language(answer, fcfg["language"]["min_chars"])
        if lp and la and lp != la:
            reasons.append("language_mismatch")
    rep = fcfg["repetition"]
    # Raw thinking is not trained on once a rationale replaces it, so only legacy rows have it checked.
    texts = (answer, rationale) if r.get("cot_method") else (answer, reasoning)
    if max(repetition_ratio(t, rep["n"], rep["min_ngrams"]) for t in texts) > rep["max_ratio"]:
        reasons.append("repetition")
    if r.get("mode") == "hint":
        text = "\n".join((reasoning, rationale, answer)).lower()
        if any(p.lower() in text for p in fcfg["leakage_phrases"]):
            reasons.append("hint_leakage")
    if max_reasoning_tokens and r.get("reasoning_tokens", 0) > max_reasoning_tokens:
        reasons.append("reasoning_too_long")
    if r.get("cot_method") and not rationale.strip():
        reasons.append("missing_rationale")
    if max_rationale_words and len(rationale.split()) > max_rationale_words:
        reasons.append("rationale_too_long")
    if student and student_tokens(r, student[0]) > student[1]:
        reasons.append("over_student_budget")
    return reasons


def filter_file(in_path, out_path, fcfg, max_reasoning_tokens=None, max_rationale_words=None, student=None):
    """Streams rows into <out> (kept) and <out stem>.rejected.jsonl, so large files need not fit in memory."""
    counts, n_kept, n_rejected = Counter(), 0, 0
    stem = str(out_path).removesuffix(".jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fk, open(f"{stem}.rejected.jsonl", "w", encoding="utf-8") as fr:
        for r in iter_jsonl(in_path):
            reasons = rejection_reasons(r, fcfg, max_reasoning_tokens, max_rationale_words, student)
            if reasons:
                fr.write(json.dumps({**r, "reject_reasons": reasons}, ensure_ascii=False) + "\n")
                counts.update(reasons)
                n_rejected += 1
            else:
                fk.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_kept += 1
    summary = {"input": str(in_path), "kept": n_kept, "rejected": n_rejected, "by_reason": dict(counts)}
    Path(f"{stem}.rejections.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[filter] {summary}")
    return summary
