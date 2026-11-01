"""Models run in-process with transformers instead of behind vLLM: the sentence-level translators of the second
translator comparison (NLLB, MADLAD, Seed-X; config `hf:`) and the metric checkers of the checker comparison (MetricX,
CometKiwi; config `metric:`). torch and transformers (or comet) are imported only when a model loads."""
import json
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from distill import report
from distill.generate import read_done_keys
from distill.translate import check_path, record

MT_PREFIX = re.compile(r"^(?:\s*(?:[A-H]\.|#{1,6}|[-*•]|\d+[.)])\s+|\s+)")  # kept as is by sentence-level models
SENTENCE = re.compile(r"(?<=[.!?])\s+")
SPLITS = ((re.compile(r"\n\s*\n"), "\n\n"), (re.compile(r"\n"), "\n"))  # paragraphs, then lines
METRICX_SCORE_ID = 250089  # <extra_id_10>: MetricX-24 reads its score from this logit


def in_halves(fn, idx):
    """fn(idx), retried as two halves (and so on) when the GPU runs out of memory."""
    import torch
    try:
        fn(idx)
    except torch.OutOfMemoryError:
        if len(idx) == 1:
            raise
        torch.cuda.empty_cache()
        in_halves(fn, idx[:len(idx) // 2])
        in_halves(fn, idx[len(idx) // 2:])


# ---------------------------------------------------------------- sentence-level translators (`hf:` roles)

def chunks(text, max_chars):
    """`text` cut at sentence ends into pieces of at most `max_chars`; a longer sentence is cut at spaces."""
    out, cur = [], ""
    for s in SENTENCE.split(text):
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                out, cur = out + [cur], ""
            out.append(s[:cut])
            s = s[cut:].lstrip()
        if cur and len(cur) + 1 + len(s) > max_chars:
            out, cur = out + [cur], s
        else:
            cur = f"{cur} {s}" if cur else s
    return out + [cur] if cur else out


def split_for_mt(text, max_chars=600):
    """A sentence-level model's view of a prompt, line by line: a line kept as is (blank, a code fence or inside a code
    block), or (prefix, pieces). The prefix (an option label, list marker, heading or indent) is kept, and the pieces,
    at most `max_chars` long, are what gets translated."""
    plan, code = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            code = not code
        if code or not line.strip() or line.strip().startswith("```"):
            plan.append(line)
            continue
        prefix = m.group(0) if (m := MT_PREFIX.match(line)) else ""
        plan.append((prefix, chunks(line[len(prefix):].strip(), max_chars)))
    return plan


def join_mt(plan, translated):
    """The translated prompt: each line rebuilt from its prefix and translated pieces."""
    return "\n".join(p if isinstance(p, str) else p[0] + " ".join(translated[c] for c in p[1]) for p in plan)


def translate_texts(texts, engine, max_chars):
    """[(translation, truncated)] for `texts` with a sentence-level `engine`: each text is split (`split_for_mt`), the
    distinct pieces are translated once, and each text is put back together."""
    plans = [split_for_mt(t, max_chars) for t in texts]
    pieces_of = lambda plan: [c for p in plan if not isinstance(p, str) for c in p[1]]
    pieces = list(dict.fromkeys(c for plan in plans for c in pieces_of(plan)))
    out, cut = engine(pieces) if pieces else ([], [])
    done = {c: (o, k) for c, o, k in zip(pieces, out, cut)}
    return [(join_mt(plan, {c: done[c][0] for c in pieces_of(plan)}), any(done[c][1] for c in pieces_of(plan)))
            for plan in plans]


def fix_special_tokens(tok, model_config):
    """A causal tokenizer shipped as a bare tokenizer.json (Seed-X) has no bos, eos or pad token under transformers 5:
    take bos and eos from the model config, pad with eos on the left, and return whether a prompt needs its bos token
    added by hand (the tokenizer does not add it)."""
    for name in ("bos", "eos"):
        token_id = getattr(model_config, f"{name}_token_id", None)
        if getattr(tok, f"{name}_token") is None and token_id is not None:
            setattr(tok, f"{name}_token", tok.convert_ids_to_tokens(token_id))
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    return bool(tok.bos_token) and tok("a")["input_ids"][:1] != [tok.bos_token_id]


def load_engine(h):
    """translate(pieces) -> (texts, truncated) for an `hf:` role, with transformers on the GPU. `seq2seq` models (NLLB,
    MADLAD) get a source language or a target-language prefix; `causal` ones (Seed-X) get each piece in the role's
    prompt, and only the continuation is decoded. Beam search; a piece is truncated when it reaches the token cap, and a
    batch that runs out of GPU memory is retried in halves."""
    import torch
    import transformers as tf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    causal = h.get("kind") == "causal"
    tok = tf.AutoTokenizer.from_pretrained(h["model"], **({"src_lang": h["src_lang"]} if "src_lang" in h else {}))
    add_bos = causal and fix_special_tokens(tok, tf.AutoConfig.from_pretrained(h["model"]))
    model_cls = tf.AutoModelForCausalLM if causal else tf.AutoModelForSeq2SeqLM
    model = model_cls.from_pretrained(h["model"], dtype=getattr(torch, h.get("dtype", "bfloat16"))).to(device).eval()
    extra = {"forced_bos_token_id": tok.convert_tokens_to_ids(h["tgt_lang"])} if "tgt_lang" in h else {}
    if causal:
        extra["pad_token_id"] = tok.pad_token_id
    wrap = ((lambda p: (tok.bos_token if add_bos else "") + h["prompt"].format(text=p)) if causal
            else (lambda p: h.get("prefix", "") + p))

    def translate(pieces):
        texts, cut = [""] * len(pieces), [False] * len(pieces)

        def batch(idx):
            enc = tok([wrap(pieces[i]) for i in idx], return_tensors="pt", padding=True).to(device)
            cap = min(h.get("max_new_tokens", 512), int(h.get("max_new_ratio", 2.5) * enc["input_ids"].shape[1]) + 16)
            with torch.no_grad():
                gen = model.generate(**enc, num_beams=h.get("num_beams", 1), do_sample=False, max_new_tokens=cap,
                                     **extra)
            if causal:
                gen = gen[:, enc["input_ids"].shape[1]:]
            for i, g in zip(idx, gen):
                if causal:  # the continuation ends at its first eos (pad is eos too); none means it hit the cap
                    stop = (g == tok.eos_token_id).nonzero()
                    g, cut[i] = (g[:stop[0, 0]], False) if stop.numel() else (g, True)
                else:
                    cut[i] = int((g != tok.pad_token_id).sum()) >= cap
                texts[i] = clean_mt(tok.decode(g, skip_special_tokens=True))
        order, size = sorted(range(len(pieces)), key=lambda i: len(pieces[i])), h.get("batch_size", 16)
        for b in range(0, len(order), size):
            in_halves(batch, order[b:b + size])
        return texts, cut
    return translate


def clean_mt(text):
    """A model's translation without a stray target-language tag (Seed-X's prompts end in `<th>`)."""
    return re.sub(r"\s*<th>\s*", " ", text).strip()


@lru_cache(maxsize=1)
def hf_engine(spec):
    """`load_engine` for an `hf:` role given as JSON, cached: a model loads once per process, whatever the sets."""
    return load_engine(json.loads(spec))


def run_hf(cfg, dataset, role, todo, out, reference=True):
    """Translate with an in-process `hf:` role: each prompt (and reference, when asked) is split into lines and
    sentences, the pieces are translated in batches, and the translation is put back together."""
    h = cfg[role]["hf"]
    print(f"[translate] {dataset} with the {role} ({h['model']}, transformers): {len(todo)} to run")
    if not todo:
        return {}
    start, engine = time.time(), hf_engine(json.dumps(h, sort_keys=True))
    refs = [r for r in todo if reference and r.get("hint_ok") and r.get("reference")]
    done = translate_texts([r["input"] for r in todo] + [r["reference"] for r in refs], engine, h.get("max_chars", 600))
    ref_of = dict(zip((r["id"] for r in refs), done[len(todo):]))
    meta = {"model": h["model"], "role": role, "engine": "transformers", "num_beams": h.get("num_beams", 1)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for r, (text, cut) in zip(todo, done):
            ref = ref_of.get(r["id"])
            rec = record(cfg, dataset, r, meta, text, "length" if cut else "stop",
                         ref[0] if ref and not ref[1] else None, 0)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[translate] {dataset}: {len(todo)} translations written in {(time.time() - start) / 60:.1f} min")
    return {"done": len(todo)}


# ---------------------------------------------------------------- metric checkers (`metric:` roles)

def aligned_pieces(en, th, fits, level=0):
    """(pieces, truncated): (English, Thai) pairs a metric model can read whole. The pair itself when it fits; else its
    paragraphs, paired in order (the translator keeps the formatting) and grouped while they fit, and a paragraph pair
    still too long goes down to its lines. When the counts differ the pair cannot be aligned at that level; with no
    level left the model reads it cut (truncated)."""
    if fits(en, th):
        return [(en, th)], False
    if level == len(SPLITS):
        return [(en, th)], True
    split, sep = SPLITS[level]
    a, b = ([p for p in split.split(t) if p.strip()] for t in (en, th))
    if len(a) != len(b) or len(a) < 2:
        return aligned_pieces(en, th, fits, level + 1)
    pieces, truncated, cur = [], False, None
    for pair in zip(a, b):
        joined = cur and (cur[0] + sep + pair[0], cur[1] + sep + pair[1])
        if joined and fits(*joined):
            cur = joined
            continue
        if cur:
            pieces.append(cur)
        cur = pair
        if not fits(*pair):
            sub, cut = aligned_pieces(*pair, fits, level + 1)
            pieces, truncated, cur = pieces + sub, truncated or cut, None
    return pieces + ([cur] if cur else []), truncated


def metricx_text(en, th):
    """MetricX-24's input in QE mode (no reference), as in its repo's predict.py."""
    return f"source: {en} candidate: {th}"


def load_metric(m):
    """(score, fits) for a `metric:` role: score(pairs) -> raw scores, fits(en, th) -> whether a pair fits the model's
    input whole.
    - `metricx`: MetricX-24's MT5ForRegression is MT5ForConditionalGeneration read at one decoder step from token 0:
      the logit of <extra_id_10>, clamped to 0-25 (an error score, lower is better). The input's final eos is dropped,
      as the repo does.
    - `comet`: CometKiwi through unbabel-comet (installed in .venv-comet, not vLLM's venv): 0-1, higher is better."""
    import torch
    gpu = torch.cuda.is_available()
    if m["kind"] == "comet":
        from comet import download_model, load_from_checkpoint
        model = load_from_checkpoint(download_model(m["model"]))
        size = lambda t: len(model.encoder.tokenizer(t, add_special_tokens=False)["input_ids"])
        fits = lambda en, th: size(en) + size(th) + 4 <= m["max_tokens"]  # <s> th </s></s> en </s>
        score = lambda pairs: list(model.predict([{"src": en, "mt": th} for en, th in pairs], progress_bar=False,
                                                 batch_size=m.get("batch_size", 16), gpus=int(gpu)).scores)
        return score, fits
    import transformers as tf
    device = "cuda" if gpu else "cpu"
    tok = tf.AutoTokenizer.from_pretrained(m["tokenizer"])
    model = tf.MT5ForConditionalGeneration.from_pretrained(m["model"], dtype=getattr(torch, m.get("dtype", "bfloat16")))
    model = model.to(device).eval()
    ids = lambda en, th: tok(metricx_text(en, th))["input_ids"][:-1]
    fits = lambda en, th: len(ids(en, th)) < m["max_tokens"]

    def score(pairs):
        enc = [ids(en, th)[:m["max_tokens"] - 1] for en, th in pairs]
        out = [0.0] * len(pairs)

        def batch(idx):
            width = max(len(enc[i]) for i in idx)
            input_ids = torch.tensor([enc[i] + [tok.pad_token_id] * (width - len(enc[i])) for i in idx], device=device)
            mask = (input_ids != tok.pad_token_id).long()
            start = torch.zeros(len(idx), 1, dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=mask, decoder_input_ids=start).logits
            for i, s in zip(idx, logits[:, 0, METRICX_SCORE_ID].clamp(0, 25).float().tolist()):
                out[i] = s
        order, size = sorted(range(len(pairs)), key=lambda i: len(enc[i])), m.get("batch_size", 8)
        for b in range(0, len(order), size):
            in_halves(batch, order[b:b + size])
        return out
    return score, fits


@lru_cache(maxsize=1)
def metric_engine(spec):
    """`load_metric` for a `metric:` role given as JSON, cached: a model loads once per process, whatever the files."""
    return load_metric(json.loads(spec))


def metric_check(cfg, path, role, block=512):
    """A metric checker scores each translation against its English, with no reference translation (QE). A pair too
    long for the model is read in aligned pieces (`aligned_pieces`), and its worst piece decides. `quality` is
    higher-is-better for every checker: MetricX's error negated, CometKiwi's score as is; `raw` is the model's own
    score. Written `block` translations at a time, so a cut run resumes by id."""
    m = cfg[role]["metric"]
    out = check_path(path, role)
    done = {rid for rid, _ in read_done_keys(out)}
    todo = [r for r in report.iter_jsonl(path) if r["id"] not in done and (r.get("input") or "").strip()]
    print(f"[check] {path} by the {role} ({m['model']}): {len(done)} already checked, {len(todo)} to run")
    if not todo:
        return {}
    start, (score, fits) = time.time(), metric_engine(json.dumps(m, sort_keys=True))
    error = m["kind"] == "metricx"
    stats = Counter()
    for b in range(0, len(todo), block):
        rows = todo[b:b + block]
        split = [aligned_pieces(r["input_en"], r["input"], fits) for r in rows]
        raw = iter(score([p for pieces, _ in split for p in pieces]))
        with open(out, "a", encoding="utf-8") as f:
            for r, (pieces, cut) in zip(rows, split):
                s = (max if error else min)(next(raw) for _ in pieces)
                stats.update(scored=1, in_pieces=len(pieces) > 1, truncated=cut)
                f.write(json.dumps({"id": r["id"], "quality": round(-s if error else s, 4), "raw": round(s, 4),
                                    "pieces": len(pieces), "truncated": cut, "judge_model": m["model"]}) + "\n")
    print(f"[check] {path}: {stats['scored']} scored in {(time.time() - start) / 60:.1f} min, "
          f"{stats['in_pieces']} read in pieces, {stats['truncated']} truncated")
    return dict(stats)

