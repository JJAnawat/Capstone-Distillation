"""Thai sets: translate questions the English run never used, check each translation (rules, the judge, embedding
similarity), and build the Thai prepared file that the rest of the pipeline distills. Also the translator comparison:
a grading sheet for the user, and a report that picks the translator and calibrates the automatic checks; and the
checker comparison: other checkers (translation-quality metrics, a Thai-tuned model) against the judge, graded on the
translations where they disagree."""
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
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
MT_PREFIX = re.compile(r"^(?:\s*(?:[A-H]\.|#{1,6}|[-*•]|\d+[.)])\s+|\s+)")  # kept as is by sentence-level models
SENTENCE = re.compile(r"(?<=[.!?])\s+")


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


def run(cfg, dataset, translator=None, limit=None, out=None, reference=True, concurrency=None, backend=None,
        client=None, dry_run=False, ids=None):
    """Translate the Thai set's source questions with the `translator` role's model, and their reference answers when
    the judge would use them (`hint_ok`). Each record carries its rule flags. A role with an `hf:` block (NLLB, MADLAD,
    Seed-X) runs in-process (`run_hf`); the others go to their vLLM or OpenRouter endpoint. `ids` restricts the run to
    those source questions (the translator comparison)."""
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
        return run_hf(cfg, dataset, translator, todo, out, reference)
    backend = backend or cfg["backend"]
    if backend not in cfg[translator]:
        raise ValueError(f"the {translator} has no {backend} endpoint in the config")
    ep = server.endpoint(cfg, translator, backend)
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
                           dataset=dataset, translator=translator, backend=ep["backend"], model=ep["model"])


# ---------------------------------------------------------------- sentence-level models (in-process)

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
    prompt, and only the continuation is decoded. Beam search; a piece is truncated when it reaches the token cap."""
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

    def batch(pieces, idx, texts, cut):
        """Translate pieces[idx] in one generate call; on running out of GPU memory, retry as two halves."""
        try:
            enc = tok([wrap(pieces[i]) for i in idx], return_tensors="pt", padding=True).to(device)
            cap = min(h.get("max_new_tokens", 512), int(h.get("max_new_ratio", 2.5) * enc["input_ids"].shape[1]) + 16)
            with torch.no_grad():
                gen = model.generate(**enc, num_beams=h.get("num_beams", 1), do_sample=False, max_new_tokens=cap,
                                     **extra)
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(idx) // 2
            batch(pieces, idx[:half], texts, cut)
            batch(pieces, idx[half:], texts, cut)
            return
        if causal:
            gen = gen[:, enc["input_ids"].shape[1]:]
        for i, g in zip(idx, gen):
            if causal:  # the continuation ends at its first eos (pad is eos too); none means it hit the cap
                stop = (g == tok.eos_token_id).nonzero()
                g, cut[i] = (g[:stop[0, 0]], False) if stop.numel() else (g, True)
            else:
                cut[i] = int((g != tok.pad_token_id).sum()) >= cap
            texts[i] = clean_mt(tok.decode(g, skip_special_tokens=True))

    def translate(pieces):
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        texts, cut, size = [""] * len(pieces), [False] * len(pieces), h.get("batch_size", 16)
        for b in range(0, len(order), size):
            batch(pieces, order[b:b + size], texts, cut)
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


def check_path(path, role):
    """Where `role`'s checks of a translation file go: <stem>.check.jsonl for the judge, <stem>.check_<role>.jsonl for
    another checker, so several checkers can score the same translations."""
    return f"{stem(path)}.check.jsonl" if role == "judge" else f"{stem(path)}.check_{role}.jsonl"


def quality(chk):
    """A check's score, higher is better: a metric checker's `quality`, or a model checker's expected meaning."""
    return None if not chk else chk.get("quality", chk.get("meaning_expected"))


def check(cfg, path, role=None, concurrency=None, backend=None, client=None):
    """A checker model (`role`, default translate.checker) reads the English and Thai prompt: is the task bound to
    English, and how fluent and faithful (1-5) is the translation? `meaning` comes last in the JSON, so its token is
    the last digit and its logprobs give `meaning_expected`, a continuous score for the threshold. A role with a
    `metric:` block (MetricX, CometKiwi) runs in-process instead (`metric_check`)."""
    tc = cfg["translate"]
    role = role or tc.get("checker") or "judge"
    if "metric" in cfg[role]:
        return metric_check(cfg, path, role)
    jc = {**judge.judge_config(cfg), "max_tokens": cfg["translate"]["check_max_tokens"]}
    ep = server.endpoint(cfg, role, backend)
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
                           role=role, backend=ep["backend"], model=ep["model"])


# ---------------------------------------------------------------- metric checkers (in-process)

SPLITS = ((re.compile(r"\n\s*\n"), "\n\n"), (re.compile(r"\n"), "\n"))  # paragraphs, then lines
METRICX_SCORE_ID = 250089  # <extra_id_10>: MetricX-24 reads its score from this logit


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


# ---------------------------------------------------------------- translator comparison

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


def compare_pick(cfg, per_set, out_dir, exclude_run="data/full_th", seed=0):
    """The pool of a later translator comparison: for each Thai set, a seeded stratified sample of `per_set` of the
    translator's records (data/translate/<set>.jsonl, failed translations included) whose questions the run has not
    distilled yet (no samples in `exclude_run`). They and their checks are copied to <out_dir>/<set>.judge.jsonl and
    <set>.judge.check.jsonl, so the existing translations need no GPU."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in (n for n in cfg["datasets"] if data.translate_source(cfg, n)):
        path = TRANSLATE_DIR / f"{ds}.jsonl"
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
        run(cfg, ds, role, ids=[r["id"] for r in report.iter_jsonl(pick)], reference=False,
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
    key_path = f"{str(out).removesuffix('.xlsx')}_key.json"
    Path(key_path).write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"[compare] {len(key)} translations of {len(keys)} prompts -> {out} (key: {key_path})")
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
    key = {int(k): v for k, v in json.loads(Path(f"{str(xlsx).removesuffix('.xlsx')}_key.json")
                                          .read_text(encoding="utf-8")).items()}
    grades, bad = read_grades(xlsx)
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
    mean = lambda xs: sum(xs) / len(xs) if xs else None
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
        path = TRANSLATE_DIR / f"{ds}.jsonl"
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
    tc, out_dir, src_dir = cfg["translate"], Path(out_dir), Path(src_dir or TRANSLATE_DIR)
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
    key = {int(k): v for k, v in json.loads(Path(f"{str(xlsx).removesuffix('.xlsx')}_key.json")
                                          .read_text(encoding="utf-8")).items()}
    grades, unreadable = read_grades(xlsx)
    picks = json.loads(Path(picks or Path(next(iter(key.values()))["path"]).parent / "picks.json")
                       .read_text(encoding="utf-8"))
    graded = {f"{k['dataset']}/{k['id']}": grades[n] for n, k in key.items() if grades.get(n, (None,))[0] is not None}
    mean = lambda xs: sum(xs) / len(xs) if xs else None
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


