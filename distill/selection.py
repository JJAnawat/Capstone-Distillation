"""Choose the samples that become training data and write them in curriculum order (complexity 1 -> 5)."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from distill import verify
from distill.data import exact_gold, load_complexity
from distill.report import STUDENT_TEMPLATE_TOKENS, iter_jsonl, judge_path, student_counter, write_jsonl


def trigrams(text):
    words = verify.words(text.lower())
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)} or {tuple(words)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def pick(candidates, k, min_score, max_similarity):
    """Up to k candidates scoring at least min_score: best expected score first, the shorter sample on ties, skipping
    any whose rationale is near-identical to one already kept."""
    ranked = sorted((c for c in candidates if c["score"] is not None and c["score"] >= min_score),
                    key=lambda c: (-c["expected"], c["words"]))
    kept = []
    for c in ranked:
        if len(kept) < k and all(jaccard(c["grams"], o["grams"]) <= max_similarity for o in kept):
            kept.append(c)
    return kept


def curriculum(rows, seed):
    """Rows grouped by complexity level 1 -> 5 (unrated last), shuffled within each level."""
    rng, levels = random.Random(seed), defaultdict(list)
    for r in rows:
        levels[r.get("complexity")].append(r)
    out = []
    for level in sorted(levels, key=lambda x: (x is None, x or 0)):
        rng.shuffle(levels[level])
        out += levels[level]
    return out


SAMPLING = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")


def final_row(r, complexity, selection, count):
    """A training row: the sample, its curriculum and selection info, its size in the student's tokenizer, and how the
    teacher made it. Raw thinking and the teacher call's bookkeeping stay in the generation files."""
    sizes = {part: count(r.get(part) or "") for part in ("input", "rationale", "answer")}
    teacher = {"model": r.get("model"), "mode": r.get("mode"), "thinking": r.get("thinking"),
               "cot_method": r.get("cot_method"), "prompt_version": r.get("prompt_version"),
               "prompt_hash": r.get("prompt_hash"), "compress_prompt_hash": r.get("compress_prompt_hash"),
               "sampling": {k: r[k] for k in SAMPLING if k in r}, "seed": r.get("seed"),
               "max_tokens": r.get("max_tokens"),             # the teacher's generation cap, not the student's
               "thinking_tokens": r.get("reasoning_tokens", 0),  # raw thinking (variant B), not in this file
               "reply_tokens": r.get("answer_tokens", 0),      # the visible reply; for A it includes the rationale
               "timestamp": r.get("timestamp")}
    translated = {k: r[k] for k in ("lang", "input_en", "translation") if r.get(k)}  # Thai rows only
    return {"id": r["id"], "sample_idx": r.get("sample_idx", 0), "source_dataset": r["source_dataset"], **translated,
            "input": r["input"], "rationale": r.get("rationale") or "", "answer": r.get("answer") or "",
            "closed": bool(r.get("closed")), **({"gold": r["gold"]} if r.get("gold") else {}),
            "complexity": complexity, "max_steps": r.get("max_steps"), "n_steps": verify.count_steps(r.get("rationale")),
            "student_tokens": {**sizes, "total": sum(sizes.values()) + STUDENT_TEMPLATE_TOKENS},
            "selection": selection, "teacher": {k: v for k, v in teacher.items() if v is not None}}


def run(cfg, in_path, out, order=None, min_score=None, count=None):
    sc = cfg["select"]
    order = order or sc["order"]
    min_score = sc["min_score"] if min_score is None else min_score
    jp = judge_path(in_path)
    judged = {(j["id"], j["sample_idx"]): j for j in iter_jsonl(jp)} if Path(jp).exists() else {}

    groups = defaultdict(list)  # pass 1: a light view of every sample, so large files need not fit in memory
    for r in iter_jsonl(in_path):
        key = (r["id"], r.get("sample_idx", 0))
        j = judged.get(key, {})
        groups[r["id"]].append({
            "key": key, "dataset": r["source_dataset"], "score": j.get("score"),
            "expected": j.get("expected_score", j.get("score")),
            "closed": bool(r.get("closed")),
            "correct": bool(r.get("closed")) and verify.is_correct(r["answer"], r.get("gold") or []),
            "words": len(verify.words(f"{r.get('rationale') or ''} {r.get('answer') or ''}")),
            "grams": trigrams(r.get("rationale") or "")})

    winners, stats, exact = {}, Counter(), exact_gold(cfg)
    for group in groups.values():
        passing = [c for c in group if c["correct"]]
        if not passing and group[0]["closed"] and group[0]["dataset"] in exact:
            stats["dropped: no correct answer"] += 1  # an exact gold: the judge would only pick among wrong answers
            continue
        pool, k, method = (passing, sc["k_closed"], "rule+judge") if passing else (group, sc["k_open"], "judge")
        kept = pick(pool, k, min_score, sc["max_similarity"])
        if not kept:
            unjudged = all(c["score"] is None for c in pool)
            stats["dropped: no judged sample" if unjudged else f"dropped: best score < {min_score}"] += 1
            continue
        stats[f"kept: {method}"] += 1
        for c in kept:
            winners[c["key"]] = {"method": method, "score": c["score"], "expected_score": c["expected"],
                                 "n_candidates": len(group), "n_passing": len(passing)}

    levels = {ds: load_complexity(ds) for ds in {g[0]["dataset"] for g in groups.values()}}
    count = count or (student_counter(cfg["student"]["tokenizer"]) if winners else None)
    rows = []  # pass 2: the winners as training rows; raw thinking is for analysis, never a target
    for r in iter_jsonl(in_path):
        if sel := winners.get((r["id"], r.get("sample_idx", 0))):
            rows.append(final_row(r, levels[r["source_dataset"]].get(r["id"]), sel, count))
    rows = curriculum(rows, sc["seed"]) if order == "curriculum" else random.Random(sc["seed"]).sample(rows, len(rows))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    by_level = Counter(r["complexity"] for r in rows)
    summary = {"input": str(in_path), "prompts": len(groups), "rows_written": len(rows), "order": order, **stats,
               "rows_by_complexity": {str(lv): by_level[lv] for lv in sorted(by_level, key=lambda x: (x is None, x or 0))}}
    Path(f"{str(out).removesuffix('.jsonl')}.select_stats.json").write_text(json.dumps(summary, indent=2),
                                                                           encoding="utf-8")
    print(f"[select] {summary}")
    return summary
