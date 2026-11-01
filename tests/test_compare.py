"""Offline tests for the experiments kept for the project report: the in-process translators and metric checkers
(inproc), and the translator and checker comparisons (compare)."""
import copy
import json
import types

import pytest

from distill import compare, inproc, report, translate
from helpers import CFG, THAI


LONG = "Long sentence number one is here. " * 30



def test_split_and_join_keep_the_structure_and_translate_only_prose():
    text = ("Which is true?\n\nA. Cats are mammals. They purr.\nB. Fish fly.\n\n```\nprint('hi')\n```\n"
            "- keep the list marker\n" + LONG)
    fake = lambda pieces: ([f"<{p}>" for p in pieces], [False] * len(pieces))
    ((out, cut),) = inproc.translate_texts([text], fake, 120)
    lines = out.split("\n")
    assert lines[:4] == ["<Which is true?>", "", "A. <Cats are mammals. They purr.>", "B. <Fish fly.>"]
    assert "```\nprint('hi')\n```" in out and "- <keep the list marker>" in out and not cut
    assert all(len(p) <= 120 for p in inproc.chunks(LONG.strip(), 120)) and lines[-1].count("<") > 1
    assert all(len(p) <= 50 for p in inproc.chunks("word " * 100 + "x" * 80, 50))  # cut at spaces, then hard



@pytest.fixture
def fake_engine(monkeypatch):
    """A stand-in for loading a transformers model: THAI for every piece, cut when it mentions 'take f'."""
    loads = []

    def load(h):
        loads.append(h["model"])
        return lambda pieces: ([THAI for _ in pieces], ["take f" in p for p in pieces])
    inproc.hf_engine.cache_clear()
    monkeypatch.setattr(inproc, "load_engine", load)
    yield loads
    inproc.hf_engine.cache_clear()



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



class BareTokenizer:
    """Seed-X's tokenizer under transformers 5: special tokens unset, no bos added to a prompt."""
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    bos_token = eos_token = pad_token = None
    padding_side = "right"
    bos_token_id = property(lambda self: self.vocab.get(self.bos_token))

    def convert_ids_to_tokens(self, i):
        return {v: k for k, v in self.vocab.items()}[i]

    def __call__(self, text):
        return {"input_ids": [5]}



def test_a_bare_causal_tokenizer_gets_its_special_tokens_from_the_model_config():
    tok = BareTokenizer()
    assert inproc.fix_special_tokens(tok, types.SimpleNamespace(bos_token_id=1, eos_token_id=2)) is True
    assert (tok.bos_token, tok.eos_token, tok.pad_token, tok.padding_side) == ("<s>", "</s>", "</s>", "left")



def test_seed_x_prompt_and_its_language_tag():
    assert CFG["seedx"]["hf"]["prompt"].format(text="Hi") == "Translate the following English sentence into Thai:\nHi <th>"
    assert inproc.clean_mt("สวัสดี <th>") == "สวัสดี" and inproc.clean_mt(" สวัสดี ") == "สวัสดี"



def test_compare_pick_takes_undistilled_translations_with_their_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(translate, "TRANSLATE_DIR", tmp_path / "translate")
    (tmp_path / "translate").mkdir()
    (tmp_path / "run").mkdir()
    recs = [{"id": f"q{i}", "strata": f"s{i % 3}", "input": THAI} for i in range(12)]
    report.write_jsonl(tmp_path / "translate" / "logiqa_th.jsonl", recs)
    report.write_jsonl(tmp_path / "translate" / "logiqa_th.check.jsonl", [{"id": r["id"], "meaning": 5} for r in recs])
    report.write_jsonl(tmp_path / "run" / "logiqa_th_closed_B_raw.jsonl", [{"id": f"q{i}"} for i in range(6)])
    compare.compare_pick(CFG, 4, tmp_path / "pool", exclude_run=tmp_path / "run")
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
    compare.compare_translate(cfg, "madlad", tmp_path)
    for ds in ("no_robots_th", "helpsteer2_th"):
        assert sorted(r["id"] for r in report.read_jsonl(tmp_path / f"{ds}.madlad.jsonl")) == ["p4", "p7"]
    assert fake_engine == ["google/madlad400-10b-mt"]



def test_similarity_with_a_fake_encoder(tmp_path):
    src = tmp_path / "t.jsonl"
    report.write_jsonl(src, [{"id": "a", "input_en": "same", "input": "same"},
                             {"id": "b", "input_en": "same", "input": "other"}])
    vec = {"same": [1.0, 0.0], "other": [0.0, 2.0]}
    compare.similarity(CFG, src, encode=lambda texts: [vec[t] for t in texts])
    assert {r["id"]: r["similarity"] for r in report.read_jsonl(tmp_path / "t.similarity.jsonl")} == {"a": 1.0, "b": 0.0}



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
    key = compare.compare_sheet(comparison_files(tmp_path), out)
    rows = list(load_workbook(out)["grades"].iter_rows(values_only=True))
    assert list(rows[0]) == compare.HEADER and len(rows) == 7
    english = [r[2] for r in rows[1:]]
    assert english[0] == english[1] == english[2] and english[3] == english[4] == english[5] != english[0]
    assert not any("model-" in str(v) or "teacher" in str(v) for r in rows for v in r)  # blind
    assert set(json.loads((tmp_path / "compare_key.json").read_text(encoding="utf-8"))) == {str(n) for n in key}
    assert {k["translator"] for k in key.values()} == {"teacher", "judge", "typhoon"}



def test_compare_report_scores_translators_and_signals(tmp_path):
    from openpyxl import load_workbook
    out = tmp_path / "compare.xlsx"
    key = compare.compare_sheet(comparison_files(tmp_path), out)
    wb = load_workbook(out)
    ws = wb["grades"]
    grade = {"teacher": 5, "judge": 4, "typhoon": 2}
    for row in ws.iter_rows(min_row=2):
        g = grade[key[row[0].value]["translator"]]
        row[4].value, row[5].value = g, g
    ws.cell(row=2, column=7, value="odd | phrase")
    wb.save(out)
    compare.compare_report(out, tmp_path / "report.md")
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
    key = compare.compare_sheet(comparison_files(tmp_path), out)
    wb = load_workbook(out)
    ws = wb["grades"]
    grade = {"teacher": 5, "judge": 4, "typhoon": 2}
    for row in list(ws.iter_rows(min_row=2))[:3]:  # the first prompt: all three translations
        row[4].value = row[5].value = grade[key[row[0].value]["translator"]]
    ws.cell(row=5, column=5, value=5)       # the second prompt: one meaning grade, no fluency
    ws.cell(row=6, column=5, value="4.5")   # a typo
    ws.cell(row=6, column=6, value="3")     # fluency typed as text
    wb.save(out)
    grades, bad = compare.read_grades(out)
    assert sorted(grades) == [1, 2, 3, 4, 5] and grades[4][1] is None and grades[5][:2] == (None, 3)
    assert bad == [(5, "meaning", "4.5")]
    compare.compare_report(out, tmp_path / "report.md")
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
    key = compare.compare_sheet(paths, out)
    wb = load_workbook(out)
    for row in wb["grades"].iter_rows(min_row=2):
        row[4].value = row[5].value = 5 if key[row[0].value]["translator"] != "typhoon" else 4  # copy graded 4
    wb.save(out)
    compare.compare_report(out, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "`judge` (judge), `checker` (model-checker)" in text and "1 grades changed" in text
    assert "| typhoon | model-typhoon | 2 | 2.50 |" in text  # the copy counts as 1: (1 + 4) / 2
    assert "| judge: meaning − user | judge: English-only | checker: meaning − user | checker: English-only |" in text
    assert "| teacher | -0.10 | 50% | +0.00 | 0% |" in text
    assert "| checker meaning (expected) | meaning | 6 |" in text and "| checker meaning (expected) | 4.0 |" in text



def test_compare_report_reads_a_metric_checker_on_its_own_scale(tmp_path):
    from openpyxl import load_workbook
    paths = comparison_files(tmp_path)
    for path in paths:
        q = {"teacher": -0.5, "judge": -1.0, "typhoon": -9.0}[path.name.split(".")[1]]
        report.write_jsonl(translate.check_path(path, "metricx"), [
            {"id": f"q{i}", "quality": q, "raw": -q, "pieces": 1, "truncated": False, "judge_model": "model-metricx"}
            for i in range(2)])
    out = tmp_path / "compare.xlsx"
    key = compare.compare_sheet(paths, out)
    wb = load_workbook(out)
    for row in wb["grades"].iter_rows(min_row=2):
        row[4].value = row[5].value = {"teacher": 5, "judge": 4, "typhoon": 2}[key[row[0].value]["translator"]]
    wb.save(out)
    compare.compare_report(out, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "`metricx` (model-metricx)" in text and "| metricx quality | meaning | 6 | 1.00 | 1.00 |" in text
    assert "metricx fluency" not in text and "metricx: meaning" not in text  # no 1-5 scale: no fluency, no lean
    assert "| metricx quality | -1.000 | 100% | 100% |" in text  # the score that keeps the good translations



def test_aligned_pieces_split_long_pairs_by_paragraph_then_line():
    fits = lambda en, th: len(en) + len(th) <= 40
    assert inproc.aligned_pieces("short", "สั้น", fits) == ([("short", "สั้น")], False)
    en = "aaaaaaaa\n\nbbbbbbbb\n\ncccccccccc\ndddddddddd"
    th = "กกกกกกกก\n\nขขขขขขขข\n\nคคคคคคคคคค\nงงงงงงงงงง"
    assert inproc.aligned_pieces(en, th, fits) == ([("aaaaaaaa\n\nbbbbbbbb", "กกกกกกกก\n\nขขขขขขขข"),
                                                       ("cccccccccc", "คคคคคคคคคค"), ("dddddddddd", "งงงงงงงงงง")], False)
    en, th = "x" * 30 + "\n\n" + "y" * 30, "ก" * 50  # the translation merged the paragraphs: no alignment
    assert inproc.aligned_pieces(en, th, fits) == ([(en, th)], True)
    assert inproc.metricx_text("Hi", "สวัสดี") == "source: Hi candidate: สวัสดี"



@pytest.fixture
def fake_metric(monkeypatch):
    """A stand-in for loading a metric model: the raw score is the Thai piece's length; English up to 20 chars fits."""
    loads = []

    def load(m):
        loads.append(m["model"])
        return (lambda pairs: [float(len(th)) for _, th in pairs]), (lambda en, th: len(en) <= 20)
    inproc.metric_engine.cache_clear()
    monkeypatch.setattr(inproc, "load_metric", load)
    yield loads
    inproc.metric_engine.cache_clear()



def test_metric_check_scores_the_worst_piece_and_resumes(fake_metric, tmp_path):
    src = tmp_path / "t.jsonl"
    report.write_jsonl(src, [{"id": "a", "input_en": "Hi?", "input": "สวัสดี?"},
                             {"id": "b", "input_en": "First paragraph.\n\nSecond one here.",
                              "input": "ย่อหน้าแรก\n\nย่อหน้าที่สองยาวกว่า"},
                             {"id": "c", "input_en": "Bye?", "input": ""}])  # an empty translation is not scored
    translate.check(CFG, src, role="metricx")
    out = {r["id"]: r for r in report.read_jsonl(tmp_path / "t.check_metricx.jsonl")}
    model = CFG["metricx"]["metric"]["model"]
    assert out["a"] == {"id": "a", "quality": -7.0, "raw": 7.0, "pieces": 1, "truncated": False, "judge_model": model}
    assert sorted(out) == ["a", "b"] and out["b"]["pieces"] == 2
    assert out["b"]["raw"] == len("ย่อหน้าที่สองยาวกว่า")  # MetricX: the larger error of the two pieces decides
    translate.check(CFG, src, role="metricx")
    assert len(report.read_jsonl(tmp_path / "t.check_metricx.jsonl")) == 2 and fake_metric == [model]
    translate.check(CFG, src, role="cometkiwi")
    kiwi = {r["id"]: r for r in report.read_jsonl(tmp_path / "t.check_cometkiwi.jsonl")}
    assert kiwi["b"]["quality"] == kiwi["b"]["raw"] == len("ย่อหน้าแรก")  # CometKiwi: higher is better, the lower decides



def checker_universe(tmp_path, monkeypatch):
    """60 translations of no_robots_th. Left out of the universe: p0 (rule flag), p1 (English-only), p2 (no MetricX
    score). The judge rejects p3-p10; MetricX's lowest are p3-p5 and p20-p24; Typhoon's are p3, p20 and p30-p35."""
    monkeypatch.setattr(translate, "TRANSLATE_DIR", tmp_path / "translate")
    (tmp_path / "translate").mkdir()
    path = tmp_path / "translate" / "no_robots_th.jsonl"
    ids = [f"p{i}" for i in range(60)]
    report.write_jsonl(path, [{"id": i, "dataset": "no_robots_th", "source_dataset": "no_robots", "strata": None,
                               "input_en": f"Prompt {i}", "input": f"คำแปล {i}", "translator": {"role": "judge",
                               "model": "gemma"}, "rule_flags": ["not_thai"] if i == "p0" else []} for i in ids])
    num = lambda i: int(i[1:])
    report.write_jsonl(translate.check_path(path, "judge"), [
        {"id": i, "english_only": i == "p1", "fluency": 5, "meaning": 5,
         "meaning_expected": 3.0 if 3 <= num(i) <= 10 else 4.8} for i in ids])
    low = {3: -9, 4: -8, 5: -7, 20: -6, 21: -6, 22: -5, 23: -5, 24: -4}
    report.write_jsonl(translate.check_path(path, "metricx"), [
        {"id": i, "quality": low.get(num(i), -1.0)} for i in ids if i != "p2"])
    weak = {3: 1.0, 20: 1.5, **{n: 2.0 for n in range(30, 36)}}
    report.write_jsonl(translate.check_path(path, "typhoon25"), [
        {"id": i, "english_only": i == "p40", "fluency": 5, "meaning": 5, "meaning_expected": weak.get(num(i), 4.9)}
        for i in ids])



def test_checker_pick_draws_each_disagreement_group(tmp_path, monkeypatch):
    checker_universe(tmp_path, monkeypatch)
    picks = compare.checker_pick(CFG, tmp_path / "pool", ["metricx", "typhoon25", "cometkiwi"], per_group=5,
                                   random_n=6)
    assert (picks["universe"], picks["judge_rejects"], picks["missing"]) == (57, 8, ["cometkiwi"])
    assert picks["sizes"] == {"only metricx rejects": 5, "only typhoon25 rejects": 7, "only judge rejects": 5,
                              "all pass": 38}
    assert picks["thresholds"] == {"metricx": -1.0, "typhoon25": 4.9} and picks["overlap"] == {"metricx": 3,
                                                                                               "typhoon25": 1}
    assert picks["english_only"] == {"typhoon25": 1} and picks["rejects"] == {"metricx": 8, "typhoon25": 8}
    rows = picks["rows"]
    assert len(rows) == 20  # 5 + 4 more for Typhoon (p20 is in both) + 5 + 6
    assert rows["no_robots_th/p20"]["groups"] == ["only metricx rejects", "only typhoon25 rejects"]
    assert sum("only typhoon25 rejects" in r["groups"] for r in rows.values()) == 5
    written = report.read_jsonl(tmp_path / "pool" / "no_robots_th.jsonl")
    assert sorted(f"no_robots_th/{r['id']}" for r in written) == sorted(rows)
    again = compare.checker_pick(CFG, tmp_path / "again", ["metricx", "typhoon25"], per_group=5, random_n=6)
    assert again["rows"] == rows  # seeded



def test_checker_sample_takes_gate_candidates_with_the_judges_checks(tmp_path, monkeypatch):
    checker_universe(tmp_path, monkeypatch)
    compare.checker_sample(CFG, 10, tmp_path / "sample")
    picked = report.read_jsonl(tmp_path / "sample" / "no_robots_th.jsonl")
    ids = [r["id"] for r in picked]
    assert len(ids) == 10 and not {"p0", "p1"} & set(ids)  # no rule flag, not English-only
    assert [c["id"] for c in report.read_jsonl(tmp_path / "sample" / "no_robots_th.check.jsonl")] == ids
    compare.checker_sample(CFG, 10, tmp_path / "again")
    assert [r["id"] for r in report.read_jsonl(tmp_path / "again" / "no_robots_th.jsonl")] == ids  # seeded
    report.write_jsonl(tmp_path / "sample" / "no_robots_th.check_metricx.jsonl", [{"id": i, "quality": -1.0} for i in ids])
    picks = compare.checker_pick(CFG, tmp_path / "pool", ["metricx"], src_dir=tmp_path / "sample")
    assert picks["universe"] == 10  # the sample, not data/translate



def test_checker_report_applies_the_decision_rule(tmp_path, monkeypatch):
    from openpyxl import load_workbook
    checker_universe(tmp_path, monkeypatch)
    picks = compare.checker_pick(CFG, tmp_path / "pool", ["metricx", "typhoon25"], per_group=5, random_n=6)
    out = tmp_path / "checkers.xlsx"
    key = compare.compare_sheet([tmp_path / "pool" / "no_robots_th.jsonl"], out)
    wb = load_workbook(out)
    for row in wb["grades"].iter_rows(min_row=2):
        groups = picks["rows"][f"no_robots_th/{key[row[0].value]['id']}"]["groups"]
        row[4].value = row[5].value = 2 if "only metricx rejects" in groups else 5  # MetricX alone is right
    wb["grades"].cell(row=2, column=7, value="a note")
    wb.save(out)
    stats = compare.checker_report(out, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert stats["only metricx rejects"][:2] == (5, 5) and stats["only typhoon25 rejects"][:2] == (1, 5)
    assert "| metricx | -1.000 | 8 | 3 | — |" in text and "| typhoon25 | 4.900 | 8 | 1 | 1 |" in text
    assert "| only metricx rejects | 5 | 5 | 5 | 100% (57%–100%) | 2.00 | 2.00 | 0.002 |" in text
    assert "Add `metricx` as a second gate: `translate.gates: [{role: metricx, min_quality: -1.0}]`" in text
    assert "Most translations only the judge rejects were graded good" in text and "a note" in text



def test_stats_helpers():
    assert [compare.grade(v) for v in (None, "", " ", 4, 4.0, "5")] == [None, None, None, 4, 4, 5]
    for v in (0, 6, 4.5, "x"):
        with pytest.raises(ValueError):
            compare.grade(v)
    assert compare.ranks([3, 1, 3]) == [2.5, 1.0, 2.5]
    assert compare.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0) and compare.spearman([1, 1, 1], [1, 2, 3]) is None
    assert compare.fisher_greater(5, 0, 0, 6) == pytest.approx(1 / 462) and compare.fisher_greater(0, 5, 0, 6) == 1.0
    lo, hi = compare.wilson(5, 5)
    assert lo == pytest.approx(0.5655, abs=1e-3) and hi == 1.0 and compare.wilson(0, 0) == (None, None)
