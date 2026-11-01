"""Offline tests for the Thai sets: translating new questions, checking and building them, the translator comparison,
and the Thai-aware parts of the later stages."""
import copy
import json
import math
import sys
import types

import pytest
import yaml

from distill import compress, data, generate, hub, rate, report, selection, translate, verify
from helpers import CFG, completion, gen_row, mock_client

SOURCE = [{"id": f"p{i}", "input": f"Name 3 fruits high in vitamin C, take {'abcdefghijkl'[i]}.", "strata": f"s{i % 2}", "closed": False,
           "reference": f"Reference {i}.", "hint_ok": i % 2 == 0, "source_dataset": "no_robots", "gold": []}
          for i in range(12)]
THAI = "บอกชื่อผลไม้ 3 ชนิดที่มีวิตามินซีสูง"


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: SOURCE)
    monkeypatch.setattr(data, "load_complexity", lambda ds: {f"p{i}": 2 for i in range(4)})  # used in English


# ---------------------------------------------------------------- choosing and translating the questions

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


LONG = "Long sentence number one is here. " * 30


def test_split_and_join_keep_the_structure_and_translate_only_prose():
    text = ("Which is true?\n\nA. Cats are mammals. They purr.\nB. Fish fly.\n\n```\nprint('hi')\n```\n"
            "- keep the list marker\n" + LONG)
    fake = lambda pieces: ([f"<{p}>" for p in pieces], [False] * len(pieces))
    ((out, cut),) = translate.translate_texts([text], fake, 120)
    lines = out.split("\n")
    assert lines[:4] == ["<Which is true?>", "", "A. <Cats are mammals. They purr.>", "B. <Fish fly.>"]
    assert "```\nprint('hi')\n```" in out and "- <keep the list marker>" in out and not cut
    assert all(len(p) <= 120 for p in translate.chunks(LONG.strip(), 120)) and lines[-1].count("<") > 1
    assert all(len(p) <= 50 for p in translate.chunks("word " * 100 + "x" * 80, 50))  # cut at spaces, then hard


@pytest.fixture
def fake_engine(monkeypatch):
    """A stand-in for loading a transformers model: THAI for every piece, cut when it mentions 'take f'."""
    loads = []

    def load(h):
        loads.append(h["model"])
        return lambda pieces: ([THAI for _ in pieces], ["take f" in p for p in pieces])
    translate.hf_engine.cache_clear()
    monkeypatch.setattr(translate, "load_engine", load)
    yield loads
    translate.hf_engine.cache_clear()


def test_an_hf_role_runs_in_process_on_the_given_ids_only(source, fake_engine, tmp_path):
    out = tmp_path / "t.jsonl"
    translate.run(CFG, "no_robots_th", "nllb", limit=8, ids=["p4", "p5"], reference=False, out=out)
    recs = {r["id"]: r for r in report.read_jsonl(out)}
    assert sorted(recs) == ["p4", "p5"] and recs["p4"]["input"] == THAI and recs["p4"]["rule_flags"] == []
    assert recs["p4"]["translator"] == {"model": "facebook/nllb-200-3.3B", "role": "nllb", "engine": "transformers",
                                        "num_beams": 4}
    assert recs["p5"]["finish_reason"] == "length" and recs["p4"]["finish_reason"] == "stop"  # p5 is "take f"
    translate.run(CFG, "no_robots_th", "nllb", limit=8, ids=["p6"], reference=False, out=tmp_path / "u.jsonl")
    assert fake_engine == ["facebook/nllb-200-3.3B"]  # loaded once per process
    with pytest.raises(ValueError, match="not in the set's sample"):
        translate.run(CFG, "no_robots_th", "nllb", limit=8, ids=["p0"], out=tmp_path / "v.jsonl")  # used in English


def test_seed_x_prompt_and_its_language_tag():
    assert CFG["seedx"]["hf"]["prompt"].format(text="Hi") == "Translate the following English sentence into Thai:\nHi <th>"
    assert translate.clean_mt("สวัสดี <th>") == "สวัสดี" and translate.clean_mt(" สวัสดี ") == "สวัสดี"


def test_compare_pick_takes_undistilled_translations_with_their_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(translate, "TRANSLATE_DIR", tmp_path / "translate")
    (tmp_path / "translate").mkdir()
    (tmp_path / "run").mkdir()
    recs = [{"id": f"q{i}", "strata": f"s{i % 3}", "input": THAI} for i in range(12)]
    report.write_jsonl(tmp_path / "translate" / "logiqa_th.jsonl", recs)
    report.write_jsonl(tmp_path / "translate" / "logiqa_th.check.jsonl", [{"id": r["id"], "meaning": 5} for r in recs])
    report.write_jsonl(tmp_path / "run" / "logiqa_th_closed_B_raw.jsonl", [{"id": f"q{i}"} for i in range(6)])
    translate.compare_pick(CFG, 4, tmp_path / "pool", exclude_run=tmp_path / "run")
    picked = report.read_jsonl(tmp_path / "pool" / "logiqa_th.judge.jsonl")
    checks = report.read_jsonl(tmp_path / "pool" / "logiqa_th.judge.check.jsonl")
    assert len(picked) == 4 and not {r["id"] for r in picked} & {f"q{i}" for i in range(6)}
    assert [c["id"] for c in checks] == [r["id"] for r in picked]
    assert not (tmp_path / "pool" / "no_robots_th.judge.jsonl").exists()  # no translations for that set: skipped


def test_compare_translate_covers_every_set_with_one_model_load(source, fake_engine, tmp_path):
    cfg = copy.deepcopy(CFG)
    for ds in ("no_robots_th", "helpsteer2_th"):
        cfg["datasets"][ds]["translate"]["limit"] = 6
        report.write_jsonl(tmp_path / f"{ds}.judge.jsonl", [{"id": "p4"}, {"id": "p7"}])
    translate.compare_translate(cfg, "madlad", tmp_path)
    for ds in ("no_robots_th", "helpsteer2_th"):
        assert sorted(r["id"] for r in report.read_jsonl(tmp_path / f"{ds}.madlad.jsonl")) == ["p4", "p7"]
    assert fake_engine == ["google/madlad400-10b-mt"]


def test_numbers_read_thai_digits_and_thousands_separators():
    assert not translate.numbers("Pay 1,000 baht in 2024") - translate.numbers("จ่าย ๑๐๐๐ บาท ในปี 2024")


# ---------------------------------------------------------------- the judge's check and embedding similarity

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


def test_a_larger_limit_continues_a_smaller_one():
    rows = [{"id": f"p{i}", "strata": f"s{i % 7}"} for i in range(1000)]
    assert data.sample(rows, 800, 0)[:400] == data.sample(rows, 400, 0)


def test_check_rejects_a_reply_without_scores():
    with pytest.raises(ValueError, match="no valid scores"):
        translate.parse_check(completion('{"english_only": true, "critique": "x", "meaning": 7}'))


def test_similarity_with_a_fake_encoder(tmp_path):
    src = tmp_path / "t.jsonl"
    report.write_jsonl(src, [{"id": "a", "input_en": "same", "input": "same"},
                             {"id": "b", "input_en": "same", "input": "other"}])
    vec = {"same": [1.0, 0.0], "other": [0.0, 2.0]}
    translate.similarity(CFG, src, encode=lambda texts: [vec[t] for t in texts])
    assert {r["id"]: r["similarity"] for r in report.read_jsonl(tmp_path / "t.similarity.jsonl")} == {"a": 1.0, "b": 0.0}


# ---------------------------------------------------------------- build

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


# ---------------------------------------------------------------- translator comparison

def comparison_files(tmp_path):
    """Two prompts, each translated by three translators, with checks and similarities."""
    paths = []
    for role, quality in (("teacher", 5), ("judge", 4), ("typhoon", 2)):
        path = tmp_path / f"no_robots_th.{role}.jsonl"
        report.write_jsonl(path, [{"id": f"q{i}", "dataset": "no_robots_th", "source_dataset": "no_robots",
                                   "input_en": f"English prompt {i}", "input": f"คำแปลที่ {i} แบบ {'กขค'[len(paths)]}",
                                   "translator": {"role": role, "model": f"model-{role}"},
                                   "rule_flags": ["not_thai"] if role == "typhoon" else []} for i in range(2)])
        report.write_jsonl(f"{str(path)[:-6]}.check.jsonl", [
            {"id": f"q{i}", "english_only": i == 1, "fluency": quality, "meaning": quality,
             "meaning_expected": quality - 0.1} for i in range(2)])
        report.write_jsonl(f"{str(path)[:-6]}.similarity.jsonl",
                           [{"id": f"q{i}", "similarity": 0.5 + quality / 10} for i in range(2)])
        paths.append(path)
    return paths


def test_compare_sheet_groups_prompts_and_hides_the_translators(tmp_path):
    from openpyxl import load_workbook
    out = tmp_path / "compare.xlsx"
    key = translate.compare_sheet(comparison_files(tmp_path), out)
    rows = list(load_workbook(out)["grades"].iter_rows(values_only=True))
    assert list(rows[0]) == translate.HEADER and len(rows) == 7
    english = [r[2] for r in rows[1:]]
    assert english[0] == english[1] == english[2] and english[3] == english[4] == english[5] != english[0]
    assert not any("model-" in str(v) or "teacher" in str(v) for r in rows for v in r)  # blind
    assert set(json.loads((tmp_path / "compare_key.json").read_text(encoding="utf-8"))) == {str(n) for n in key}
    assert {k["translator"] for k in key.values()} == {"teacher", "judge", "typhoon"}


def test_compare_report_scores_translators_and_signals(tmp_path):
    from openpyxl import load_workbook
    out = tmp_path / "compare.xlsx"
    key = translate.compare_sheet(comparison_files(tmp_path), out)
    wb = load_workbook(out)
    ws = wb["grades"]
    grade = {"teacher": 5, "judge": 4, "typhoon": 2}
    for row in ws.iter_rows(min_row=2):
        g = grade[key[row[0].value]["translator"]]
        row[4].value, row[5].value = g, g
    ws.cell(row=2, column=7, value="odd | phrase")
    wb.save(out)
    translate.compare_report(out, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Graded by hand: 6 of 6 translations for meaning, 6 for fluency" in text
    assert "compared on the 2 prompts" in text and "Unreadable" not in text
    assert "| teacher | model-teacher | 2 | 5.00 | 5.00 | 0% | 0% |" in text
    assert "| typhoon | model-typhoon | 2 | 2.00 | 2.00 | 100% | 100% |" in text
    assert "| teacher | -0.10 | 50% |" in text  # the judge's lean on each translator, and its English-only share
    assert "| judge meaning (expected) | meaning | 6 | 1.00 | 1.00 |" in text
    assert "| embedding similarity | 0.8 | 100% | 100% |" in text  # the bad ones (0.7) out, the good (0.9, 1.0) in
    assert "| judge meaning (expected) | 4.0 | 100% | 50% |" in text  # the judge's own 3.9 falls below
    assert "0 grades changed" in text
    assert "| not_thai | 2 | 2.00 |" in text and "odd / phrase" in text and "English prompt 1" in text


def test_compare_report_uses_only_graded_cells_and_complete_prompts(tmp_path):
    from openpyxl import load_workbook
    out = tmp_path / "compare.xlsx"
    key = translate.compare_sheet(comparison_files(tmp_path), out)
    wb = load_workbook(out)
    ws = wb["grades"]
    grade = {"teacher": 5, "judge": 4, "typhoon": 2}
    for row in list(ws.iter_rows(min_row=2))[:3]:  # the first prompt: all three translations
        row[4].value = row[5].value = grade[key[row[0].value]["translator"]]
    ws.cell(row=5, column=5, value=5)       # the second prompt: one meaning grade, no fluency
    ws.cell(row=6, column=5, value="4.5")   # a typo
    ws.cell(row=6, column=6, value="3")     # fluency typed as text
    wb.save(out)
    grades, bad = translate.read_grades(out)
    assert sorted(grades) == [1, 2, 3, 4, 5] and grades[4][1] is None and grades[5][:2] == (None, 3)
    assert bad == [(5, "meaning", "4.5")]
    translate.compare_report(out, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Graded by hand: 4 of 6 translations for meaning, 4 for fluency" in text
    assert "compared on the 1 prompts" in text and "Only 1 complete prompts" in text
    assert "Unreadable grades, left out: row 5 meaning '4.5'." in text
    assert "| teacher | model-teacher | 1 | 5.00 | 5.00 | 0% |" in text  # the half-graded prompt is not compared
    assert "| judge fluency | fluency | 4 |" in text and "| judge meaning (expected) | meaning | 4 |" in text


def test_compare_report_puts_a_second_checker_beside_the_judge(tmp_path):
    from openpyxl import load_workbook
    paths = comparison_files(tmp_path)
    for path in paths:  # the second checker: lenient on the teacher (its own family), strict on the rest
        role = path.name.split(".")[1]
        report.write_jsonl(translate.check_path(path, "checker"), [
            {"id": f"q{i}", "english_only": False, "fluency": 5, "meaning": 5 if role == "teacher" else 2,
             "meaning_expected": 5.0 if role == "teacher" else 2.0, "judge_model": "model-checker"} for i in range(2)])
    typhoon = report.read_jsonl(paths[2])
    typhoon[0]["input"] = typhoon[0]["input_en"]  # an untranslated copy
    report.write_jsonl(paths[2], typhoon)
    out = tmp_path / "compare.xlsx"
    key = translate.compare_sheet(paths, out)
    wb = load_workbook(out)
    for row in wb["grades"].iter_rows(min_row=2):
        row[4].value = row[5].value = 5 if key[row[0].value]["translator"] != "typhoon" else 4  # copy graded 4
    wb.save(out)
    translate.compare_report(out, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "`judge` (judge), `checker` (model-checker)" in text and "1 grades changed" in text
    assert "| typhoon | model-typhoon | 2 | 2.50 |" in text  # the copy counts as 1: (1 + 4) / 2
    assert "| judge: meaning − user | judge: English-only | checker: meaning − user | checker: English-only |" in text
    assert "| teacher | -0.10 | 50% | +0.00 | 0% |" in text
    assert "| checker meaning (expected) | meaning | 6 |" in text and "| checker meaning (expected) | 4.0 |" in text


def test_reply_json_falls_back_to_reasoning_content():
    choice = {"message": {"content": "", "reasoning_content": '{"english_only": false, "critique": "ok", '
                                                            '"fluency": 5, "meaning": 4}'}}
    assert translate.parse_check({"choices": [choice]})[2] == 4


def test_grade_reads_excel_cells():
    assert [translate.grade(v) for v in (None, "", " ", 4, 4.0, "5")] == [None, None, None, 4, 4, 5]
    for v in (0, 6, 4.5, "x"):
        with pytest.raises(ValueError):
            translate.grade(v)


def test_spearman_handles_ties():
    assert translate.ranks([3, 1, 3]) == [2.5, 1.0, 2.5]
    assert translate.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert translate.spearman([1, 1, 1], [1, 2, 3]) is None


# ---------------------------------------------------------------- Thai in the later stages

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
