"""Offline tests for the Thai sets: translating new questions, checking and building them (gates included), and the
Thai-aware parts of the later stages. The comparisons' tests are in test_compare.py."""
import copy
import json
import math
import sys
import types

import pytest
import yaml

from distill import compress, data, generate, hub, rate, report, selection, translate, verify
from helpers import CFG, SOURCE, THAI, completion, gen_row, mock_client


def test_source_rows_skip_used_ids_and_nest(source):
    rows, pool, used = translate.source_rows(CFG, "no_robots_th", limit=6)
    assert (pool, used) == (8, 4) and not {r["id"] for r in rows} & {"p0", "p1", "p2", "p3"}
    first, _, _ = translate.source_rows(CFG, "no_robots_th", limit=3)
    assert first == rows[:3]  # the comparison's prompts are the first of the full run's


def test_source_rows_need_the_english_ratings_and_a_large_enough_pool(source, monkeypatch):
    with pytest.raises(ValueError, match="only 8 unused"):
        translate.source_rows(CFG, "no_robots_th", limit=9)
    monkeypatch.setattr(data, "load_complexity", lambda ds: {})
    with pytest.raises(FileNotFoundError, match="cannot be excluded"):
        translate.source_rows(CFG, "no_robots_th", limit=3)


def test_translate_requests_records_and_resume(source, tmp_path):
    out = tmp_path / "t.jsonl"
    bodies = []
    translate.run(CFG, "no_robots_th", "teacher", limit=4, out=out, client=mock_client(lambda p, b: completion(THAI),
                                                                                     bodies))
    sampling = CFG["translate"]["sampling"]
    assert {k: bodies[0][k] for k in sampling} == sampling and bodies[0]["model"] == CFG["teacher"]["vllm"]["model"]
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "<text>\nName 3" in bodies[0]["messages"][1]["content"] and "Thai" in bodies[0]["messages"][0]["content"]
    recs = report.read_jsonl(out)
    assert len(recs) == 4 and len(bodies) == 4 + sum(r["hint_ok"] for r in recs)  # references only when hint_ok
    r = recs[0]
    assert r["input"] == THAI and r["input_en"].startswith("Name 3") and r["dataset"] == "no_robots_th"
    assert r["translator"]["role"] == "teacher" and r["rule_flags"] == []
    assert all((x["reference"] == THAI) == x["hint_ok"] for x in recs)
    bodies.clear()
    translate.run(CFG, "no_robots_th", "teacher", limit=6, out=out, reference=False,
                  client=mock_client(lambda p, b: completion(THAI), bodies))
    assert len(bodies) == 2  # resumed: only the two new prompts, and no references


def test_gemma_thought_block_is_not_part_of_the_translation(source, tmp_path):
    reply = lambda p, b: completion("<|channel>thought\n<channel|>" + THAI)
    translate.run(CFG, "no_robots_th", "judge", limit=1, out=tmp_path / "t.jsonl", reference=False,
                  client=mock_client(reply))
    assert report.read_jsonl(tmp_path / "t.jsonl")[0]["input"] == THAI


def test_typhoon_gets_its_card_prompt_and_dry_run_calls_nothing(source, tmp_path, capsys):
    bodies = []
    translate.run(CFG, "no_robots_th", "typhoon", limit=1, out=tmp_path / "t.jsonl", reference=False,
                  client=mock_client(lambda p, b: completion(THAI), bodies))
    (msg,) = bodies[0]["messages"]
    assert msg["role"] == "user" and msg["content"].startswith("Translate the following English text into Thai")
    translate.run(CFG, "no_robots_th", "teacher", limit=2, dry_run=True, client=mock_client(lambda p, b: 1 / 0))
    assert "4 prompts used in English excluded, 8 left, 2 sampled" in capsys.readouterr().out


def rec(en="Name 3 fruits high in vitamin C, from https://a.com.", th=THAI + " จาก https://a.com", **kw):
    return {"input_en": en, "input": th, "finish_reason": "stop", **kw}


@pytest.mark.parametrize("overrides, flag", [
    ({"finish_reason": "length"}, "not_stop"),
    ({"th": "Name 3 fruits high in vitamin C, from https://a.com."}, "not_thai"),
    ({"th": "ขออภัย ฉันไม่สามารถแปลข้อความนี้ได้ https://a.com 3"}, "refusal"),
    ({"th": "ผล 3 https://a.com", "en": "Name 3 fruits high in vitamin C. " * 10 + "https://a.com"}, "length_ratio"),
    ({"th": THAI.replace("3", "4") + " จาก https://a.com"}, "numbers_changed"),
    ({"en": "Run it:\n```print(1)```", "th": "รันโค้ดนี้:\n```พิมพ์(1)```"}, "code_changed"),
    ({"th": THAI + " จากเว็บไซต์"}, "urls_changed"),
    ({"en": "Which is true?\nA. One\nB. Two", "th": "ข้อใดเป็นจริง\nก. หนึ่ง\nข. สอง"}, "options_missing"),
    ({"en": "Translate into French: good morning", "th": "แปลเป็นภาษาฝรั่งเศส: good morning"}, "english_task_pattern"),
])
def test_rule_flags(overrides, flag):
    assert translate.rule_flags(rec(), CFG) == []
    kw = {k: v for k, v in overrides.items() if k not in ("en", "th")}
    r = rec(**{k: overrides[k] for k in ("en", "th") if k in overrides}, **kw)
    assert flag in translate.rule_flags(r, CFG)


def test_numbers_read_thai_digits_and_thousands_separators():
    assert not translate.numbers("Pay 1,000 baht in 2024") - translate.numbers("จ่าย ๑๐๐๐ บาท ในปี 2024")


def digit(token, probs):
    return {"token": token, "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]}


def test_check_parses_meaning_last_and_resumes(tmp_path):
    src = tmp_path / "t.jsonl"
    report.write_jsonl(src, [{"id": "a", "input_en": "Hi?", "input": "สวัสดี?"},
                             {"id": "b", "input_en": "Bye?", "input": ""}])  # an empty translation is not checked
    content = '{"english_only": false, "critique": "Fine.", "fluency": 4, "meaning": 4}'
    logprobs = {"content": [digit("4", {"4": 0.9, "9": 0.1}), digit("4", {"4": 0.6, "5": 0.4})]}
    bodies = []
    translate.check(CFG, src, client=mock_client(lambda p, b: completion(content, logprobs=logprobs), bodies))
    (out,) = report.read_jsonl(tmp_path / "t.check.jsonl")
    assert out["meaning"] == 4 and out["meaning_expected"] == pytest.approx(4.4) and out["english_only"] is False
    fmt = bodies[0]["response_format"]["json_schema"]
    assert fmt["name"] == "translation_check" and list(fmt["schema"]["properties"])[-1] == "meaning"
    assert "[English]\nHi?" in bodies[0]["messages"][1]["content"] and bodies[0]["temperature"] == 0
    bodies.clear()
    translate.check(CFG, src, client=mock_client(lambda p, b: completion(content), bodies))
    assert bodies == []


def test_a_second_checker_writes_its_own_file_with_its_own_model(tmp_path):
    src = tmp_path / "t.jsonl"
    report.write_jsonl(src, [{"id": "a", "input_en": "Hi?", "input": "สวัสดี?"}])
    content = '{"english_only": false, "critique": "Fine.", "fluency": 5, "meaning": 5}'
    bodies = []
    translate.check(CFG, src, role="checker", client=mock_client(lambda p, b: completion(content), bodies))
    assert bodies[0]["model"] == CFG["checker"]["vllm"]["model"] and bodies[0]["logprobs"] is True
    (out,) = report.read_jsonl(tmp_path / "t.check_checker.jsonl")
    assert out["judge_model"] == CFG["checker"]["vllm"]["model"] and not (tmp_path / "t.check.jsonl").exists()


def test_build_reads_the_configured_checker_and_sorts_by_id(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PREPARED_DIR", tmp_path / "prepared")
    cfg = copy.deepcopy(CFG)
    cfg["translate"]["checker"] = "checker"
    path = tmp_path / "no_robots_th.jsonl"
    base = {"source_dataset": "no_robots", "input_en": "Hi?", "input": "สวัสดี?", "rule_flags": [], "gold": [],
            "translator": {"model": "gemma"}}
    report.write_jsonl(path, [{**base, "id": i} for i in ("c", "a", "b")])  # in the order translations finished
    ok = {"english_only": False, "fluency": 5, "meaning": 5, "meaning_expected": 4.9, "judge_model": "qwen3"}
    report.write_jsonl(translate.check_path(path, "checker"), [{**ok, "id": i} for i in ("a", "b", "c")])
    report.write_jsonl(translate.check_path(path, "judge"), [{**ok, "id": "a", "meaning_expected": 1.0}])
    assert translate.build(cfg, "no_robots_th", path)["kept"] == 3  # the judge's low score is not read
    rows = report.read_jsonl(tmp_path / "prepared" / "no_robots_th.jsonl")
    assert [r["id"] for r in rows] == ["a", "b", "c"] and rows[0]["translation"]["checker"] == "qwen3"


def test_build_drops_failed_translations_and_rebuilds_option_golds(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PREPARED_DIR", tmp_path / "prepared")
    cfg = copy.deepcopy(CFG)
    cfg["translate"].update(min_meaning=4.0, min_similarity=0.7)
    th = "ข้อใดถูก\n\nA. หนึ่ง\nB. สอง\n\nตอบด้วยตัวอักษร"
    base = {"source_dataset": "logiqa", "input_en": "Which?\n\nA. One\nB. Two", "input": th, "strata": "x",
            "closed": True, "gold": ["B", "B. Two", "Two"], "hint_ok": True, "reference": None, "rule_flags": [],
            "translator": {"model": "qwen"}}
    path = tmp_path / "logiqa_th.jsonl"
    report.write_jsonl(path, [{**base, "id": i} for i in ("ok", "flagged", "english", "weak", "far", "unchecked")])
    path_rows = report.read_jsonl(path)
    path_rows[1]["rule_flags"] = ["not_thai"]
    report.write_jsonl(path, path_rows)
    ok = {"english_only": False, "fluency": 5, "meaning": 5, "meaning_expected": 4.9}
    report.write_jsonl(tmp_path / "logiqa_th.check.jsonl", [
        {"id": "ok", **ok}, {"id": "flagged", **ok}, {"id": "english", **ok, "english_only": True},
        {"id": "weak", **ok, "meaning_expected": 3.2}, {"id": "far", **ok}])
    report.write_jsonl(tmp_path / "logiqa_th.similarity.jsonl",
                       [{"id": i, "similarity": 0.5 if i == "far" else 0.9} for i in ("ok", "english", "weak", "far")])
    stats = translate.build(cfg, "logiqa_th", path)
    assert stats == {"translated": 6, "dropped: rule: not_thai": 1, "dropped: english_only": 1, "dropped: meaning": 1,
                     "dropped: similarity": 1, "dropped: unchecked": 1, "kept": 1}
    (row,) = report.read_jsonl(tmp_path / "prepared" / "logiqa_th.jsonl")
    assert row["gold"] == ["B", "B. สอง", "สอง"] and row["source_dataset"] == "logiqa_th" and row["lang"] == "th"
    assert row["closed"] and row["input_en"].startswith("Which?") and not row["hint_ok"]
    assert row["translation"] == {"model": "qwen", "checker": None, "meaning": 4.9, "fluency": 5, "similarity": 0.9}
    assert json.loads((tmp_path / "prepared" / "stats.json").read_text(encoding="utf-8"))["logiqa_th"]["kept"] == 1


def test_translated_sets_are_skipped_by_prepare_and_keep_exact_golds(tmp_path, monkeypatch):
    assert data.exact_gold(CFG) == {"logiqa", "logiqa_th"}
    monkeypatch.chdir(tmp_path)
    cfg = {"datasets": {k: CFG["datasets"][k] for k in ("logiqa", "logiqa_th")}, "verify": CFG["verify"]}
    row = {"text": json.dumps({"text": "P.", "question": "Q?", "options": ["a", "b"], "answer": 0,
                               "type": {"Categorical Reasoning": True}})}
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda hf_id, split: [row]))
    data.prepare(cfg)
    assert sorted(p.name for p in data.PREPARED_DIR.iterdir()) == ["logiqa.jsonl", "stats.json"]
    with pytest.raises(ValueError, match="build-translated"):
        data.prepare(cfg, only="logiqa_th")


def test_a_gate_flags_rows_instead_of_dropping_them(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PREPARED_DIR", tmp_path / "prepared")
    cfg = copy.deepcopy(CFG)
    cfg["translate"]["gates"] = [{"role": "metricx", "min_quality": -3.0}]
    path = tmp_path / "no_robots_th.jsonl"
    report.write_jsonl(path, [{"id": i, "source_dataset": "no_robots", "input_en": "Hi?", "input": "สวัสดี?",
                               "rule_flags": [], "gold": [], "translator": {"model": "gemma"}} for i in "abc"])
    ok = {"english_only": False, "fluency": 5, "meaning": 5, "meaning_expected": 4.9}
    report.write_jsonl(translate.check_path(path, "judge"), [{**ok, "id": i} for i in "abc"])
    report.write_jsonl(translate.check_path(path, "metricx"), [{"id": "a", "quality": -1.0}, {"id": "b", "quality": -5.0}])
    stats = translate.build(cfg, "no_robots_th", path)
    assert stats["kept"] == 3 and stats["gated: metricx"] == 2  # b scores too low; c has no score
    rows = {r["id"]: r for r in report.read_jsonl(tmp_path / "prepared" / "no_robots_th.jsonl")}
    assert "gate_failed" not in rows["a"]["translation"] and rows["b"]["translation"]["gate_failed"] == ["metricx"]
    assert data.gated_ids(cfg, "no_robots_th") == {"b", "c"} and data.gated_ids(cfg, "no_robots") == set()


def test_gated_rows_leave_after_the_sample_and_in_select(tmp_path, monkeypatch):
    rows = [{"id": f"p{i}", "strata": f"s{i % 3}", "hint_ok": False, "closed": False,
             "translation": {"gate_failed": ["metricx"]} if i % 4 == 0 else {}} for i in range(24)]
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: rows)
    expected = [r for r in data.sample(rows, 10, 0) if not r["translation"]]
    assert data.select_rows(CFG, "no_robots_th", limit=10) == expected  # the same draw, less the gated rows
    monkeypatch.setattr(data, "PREPARED_DIR", tmp_path)
    report.write_jsonl(tmp_path / "no_robots_th.jsonl", rows)
    monkeypatch.setattr(selection, "load_complexity", lambda ds: {})
    samples = [gen_row(i, source_dataset="no_robots_th", rationale="1. A step.", answer="Ok.") for i in range(2)]
    src = tmp_path / "s.kept.jsonl"
    report.write_jsonl(src, samples)
    report.write_jsonl(tmp_path / "s.kept.judge.jsonl", [{"id": f"p{i}", "sample_idx": 0, "score": 8} for i in range(2)])
    stats = selection.run(CFG, src, tmp_path / "f.jsonl", count=len)
    assert [r["id"] for r in report.read_jsonl(tmp_path / "f.jsonl")] == ["p1"]  # p0 failed the gate
    assert stats["dropped: translation gate"] == 1


def test_rate_uses_the_english_original(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: [
        {"id": "a", "input": THAI, "input_en": "Name 3 fruits.", "strata": None, "closed": False}])
    bodies = []
    rate.run(CFG, "no_robots_th", out=tmp_path / "r.jsonl",
             client=mock_client(lambda p, b: completion("ReasoningRequired: 1"), bodies))
    assert "Name 3 fruits." in bodies[0]["messages"][-1]["content"] and THAI not in bodies[0]["messages"][-1]["content"]


def test_generate_keeps_the_english_prompt_and_carries_the_translation(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: [
        {**SOURCE[0], "input": THAI, "input_en": SOURCE[0]["input"], "lang": "th", "translation": {"model": "q"}}])
    bodies = []
    client = mock_client(lambda p, b: completion("คำตอบ", reasoning="Because.", reasoning_tokens=5), bodies)
    generate.run(CFG, "no_robots_th", n=1, out=tmp_path / "g.jsonl", client=client)
    system, user = (m["content"] for m in bodies[0]["messages"])
    assert "same language as the user's prompt" in system and user == THAI
    sample = report.read_jsonl(tmp_path / "g.jsonl")[0]
    assert sample["lang"] == "th" and sample["input_en"].startswith("Name 3") and sample["translation"] == {"model": "q"}


def test_thai_final_answers_and_words():
    assert verify.is_correct("เหตุผล...\nคำตอบสุดท้าย: B", ["B"])
    assert verify.is_correct("Final answer: ไม่ใช่", ["ไม่ใช่"]) and not verify.is_correct("Final answer: ใช่", ["ไม่ใช่"])
    assert verify.words("ฉันชอบกินข้าวผัด มากที่สุด") == ["ฉัน", "ชอบ", "กิน", "ข้าวผัด", "มาก", "ที่สุด"]
    assert verify.words("two words") == ["two", "words"] and verify.words(None) == []


def test_select_tells_thai_rationales_apart():
    a = selection.trigrams("1. แมวทุกตัวเป็นสัตว์เลี้ยงลูกด้วยนม\n\n2. ดังนั้นคำตอบคือ B")
    b = selection.trigrams("1. นกบางชนิดบินไม่ได้ เช่น นกเพนกวิน\n\n2. ดังนั้นคำตอบคือ C")
    assert len(a) > 5 and selection.jaccard(a, b) < 0.7


def test_filter_flags_an_english_rationale_for_a_thai_question():
    fcfg = CFG["filter"]
    row = gen_row(0, input="ช่วยอธิบายว่าทำไมท้องฟ้าถึงมีสีฟ้าในตอนกลางวัน และเป็นสีส้มในตอนเย็น",
                  answer="ท้องฟ้ามีสีฟ้าเพราะแสงสีน้ำเงินกระเจิงในบรรยากาศได้มากกว่าแสงสีอื่น ส่วนตอนเย็นแสงเดินทางไกลขึ้น",
                  rationale="1. Sunlight scatters off air molecules, and blue light scatters the most.\n\n"
                            "2. At sunset light travels through more air, so red and orange remain.",
                  cot_method="compressed")
    assert "language_mismatch" in report.rejection_reasons(row, fcfg)
    thai_steps = "1. แสงอาทิตย์กระเจิงเมื่อกระทบโมเลกุลอากาศ โดยแสงสีน้ำเงินกระเจิงมากที่สุด\n\n2. ตอนเย็นแสงผ่านอากาศมากขึ้น"
    assert report.rejection_reasons({**row, "rationale": thai_steps}, fcfg) == []


THAI_ANSWER = "ฟังก์ชันนี้รับรายการสินค้าแล้วรวมราคาทั้งหมด โดยวนลูปผ่านแต่ละรายการและบวกราคาเข้าด้วยกันก่อนคืนค่าผลรวม"


THAI_STEPS = "1. ฟังก์ชันต้องรับรายการสินค้าและคืนผลรวมของราคา\n\n2. วนลูปผ่านแต่ละรายการแล้วบวกราคาเข้าไปในตัวแปรผลรวม"


CODE_PROMPT = ("เขียน Python function calculate_total_price(items: list[dict]) -> float that returns the sum of "
               "item['price'] for every item in the shopping cart list")


def test_filter_expects_thai_on_a_thai_row_whose_prompt_reads_as_english():
    fcfg = CFG["filter"]
    row = gen_row(0, input=CODE_PROMPT, answer=THAI_ANSWER, rationale=THAI_STEPS, cot_method="compressed")
    assert report.detect_language(CODE_PROMPT, fcfg["language"]["min_chars"]).name == "ENGLISH"
    assert "language_mismatch" in report.rejection_reasons(row, fcfg)  # without `lang`: the old, wrong rejection
    assert report.rejection_reasons({**row, "lang": "th"}, fcfg) == []
    english = "1. The function sums the price of every item in the list.\n\n2. It loops over the items and adds each price."
    assert "language_mismatch" in report.rejection_reasons({**row, "lang": "th", "rationale": english}, fcfg)


def compress_row(i, rationale=None, **kw):
    return gen_row(i, input=CODE_PROMPT, answer=THAI_ANSWER, reasoning="Sum the prices.", **kw,
                   **({"rationale": rationale, "cot_method": "compressed"} if rationale is not None else {}))


def test_compress_names_the_target_language_on_translated_rows(tmp_path):
    src = tmp_path / "raw.jsonl"
    report.write_jsonl(src, [compress_row(0, lang="th", source_dataset="no_robots_th"), compress_row(1)])
    bodies = []
    compress.run(CFG, src, tmp_path / "b.jsonl", client=mock_client(lambda p, b: completion(THAI_STEPS), bodies))
    users = {b["messages"][1]["content"].count("Write every step in Thai, the language of the question"): b
             for b in bodies}
    assert set(users) == {0, 1} and "Write in the same language as the question" in users[0]["messages"][1]["content"]
    hashes = {r["id"]: r["compress_prompt_hash"] for r in report.read_jsonl(tmp_path / "b.jsonl")}
    assert hashes["p0"] != hashes["p1"]


def test_redo_recompresses_only_thai_rows_with_english_steps(tmp_path):
    english = "1. The function sums the price of every item in the list.\n\n2. It loops over the items and adds each price."
    raw = [compress_row(0, lang="th"), compress_row(1, lang="th"), compress_row(2)]
    src, out = tmp_path / "raw.jsonl", tmp_path / "b.jsonl"
    report.write_jsonl(src, raw)
    report.write_jsonl(out, [{**raw[0], "rationale": english, "cot_method": "compressed"},
                             {**raw[1], "rationale": THAI_STEPS, "cot_method": "compressed"},
                             {**raw[2], "rationale": english, "cot_method": "compressed"}])  # English row: not redone
    bodies = []
    compress.run(CFG, src, out, redo_wrong_language=True,
                 client=mock_client(lambda p, b: completion(THAI_STEPS), bodies))
    assert len(bodies) == 1 and CODE_PROMPT in bodies[0]["messages"][1]["content"]
    recs = report.read_jsonl(out)
    assert sorted(r["id"] for r in recs) == ["p0", "p1", "p2"]  # no key twice
    assert {r["id"]: r["rationale"] for r in recs} == {"p0": THAI_STEPS, "p1": THAI_STEPS, "p2": english}


def test_rate_asks_again_for_a_rating_without_a_level(tmp_path, monkeypatch):
    rows = [{"id": f"q{i}", "input": f"Question {i}?", "strata": None, "closed": False} for i in range(2)]
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: rows)
    out = tmp_path / "ratings.jsonl"
    report.write_jsonl(out, [{"id": "q0", "complexity": None}, {"id": "q1", "complexity": 2}])
    bodies = []
    rate.run(CFG, "x", out=out, client=mock_client(lambda p, b: completion("ReasoningRequired: 4"), bodies))
    assert len(bodies) == 1 and "Question 0?" in bodies[0]["messages"][-1]["content"]
    latest = {r["id"]: r["complexity"] for r in report.read_jsonl(out)}  # the last line wins, as in load_complexity
    assert latest == {"q0": 4, "q1": 2} and CFG["rate"]["max_tokens"] == 2048


def test_final_rows_carry_the_translation():
    r = gen_row(0, source_dataset="logiqa_th", lang="th", input_en="Which?", translation={"model": "qwen"})
    final = selection.final_row(r, 2, {"method": "rule+judge"}, len)
    assert final["lang"] == "th" and final["input_en"] == "Which?" and final["translation"] == {"model": "qwen"}
    assert "lang" not in selection.final_row(gen_row(1), 2, {}, len)


def test_card_keeps_english_configs_and_names_thai_subsets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = [{"id": "a", "complexity": 1, "input": "q", "rationale": "1. r", "answer": "a"}]
    for f in ("data/full/no_robots_open_B.final.jsonl", "data/full_th/no_robots_th_open_B.final.jsonl",
              "data/full_th/logiqa_th_closed_B.final.jsonl", "data/full_th/logiqa_th_closed_B_raw.jsonl"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        report.write_jsonl(tmp_path / f, row)
    (tmp_path / "data/prepared").mkdir()
    text = hub.card("data/full_th", "org/data", runs=["data/full", "data/full_th"])
    meta = yaml.safe_load(text.split("---")[1])
    subsets = {c["config_name"]: [(f["split"], f["path"]) for f in c["data_files"]] for c in meta["configs"]}
    assert list(subsets) == ["final", "final_th", "final_th_closed"]  # raw files get no subset
    assert subsets["final"] == [("no_robots", "data/full/no_robots_open_B.final.jsonl")]
    assert subsets["final_th"] == [("no_robots", "data/full_th/no_robots_th_open_B.final.jsonl")]
    assert subsets["final_th_closed"] == [("logiqa", "data/full_th/logiqa_th_closed_B.final.jsonl")]
    assert meta["language"] == ["en", "th"] and "| logiqa (Thai, closed) | 1 | — | 1 |" in text
    assert "### no_robots (Thai): complexity 1" in text



def test_parse_check_needs_both_scores_and_reads_reasoning_content():
    with pytest.raises(ValueError, match="no valid scores"):
        translate.parse_check(completion('{"english_only": true, "critique": "x", "meaning": 7}'))
    choice = {"message": {"content": "", "reasoning_content": '{"english_only": false, "critique": "ok", '
                                                            '"fluency": 5, "meaning": 4}'}}
    assert translate.parse_check({"choices": [choice]})[2] == 4
