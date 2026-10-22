"""Offline tests for the stages after generation: rating, compression, rule checks, judging, selection and the report."""
import json
import math

import pytest

from distill import compress, data, generate, judge, rate, report, selection, server, verify
from helpers import CFG, completion, gen_row, mock_client


# ---------------------------------------------------------------- rate

@pytest.mark.parametrize("text, level", [
    ("Analysis: two steps.\nReasoningRequired: 2", 2),
    ("Analysis: many steps.\n**ReasoningRequired:** [4]", 4),
    ("I cannot rate this.", None),
])
def test_parse_rating(text, level):
    assert rate.parse_rating(text)[0] == level


def test_rate_writes_ratings_and_resumes(tmp_path, monkeypatch):
    rows = [{"id": f"q{i}", "input": f"Question {i}?", "strata": None, "closed": False} for i in range(3)]
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: rows)
    out = tmp_path / "ratings.jsonl"
    out.write_text(json.dumps({"id": "q0", "complexity": 1}) + "\n", encoding="utf-8")
    bodies = []
    reply = lambda path, body: completion("Analysis: a few steps.\nReasoningRequired: 3")
    rate.run(CFG, "x", out=out, client=mock_client(reply, bodies))
    assert len(bodies) == 2 and bodies[0]["temperature"] == 0
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "<question> Question" in bodies[0]["messages"][-1]["content"]
    assert {r["id"]: r["complexity"] for r in report.read_jsonl(out)} == {"q0": 1, "q1": 3, "q2": 3}


# ---------------------------------------------------------------- compress

def test_compress_step_limit_follows_complexity(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_complexity", lambda ds: {"p0": 4})
    src = tmp_path / "raw.jsonl"
    report.write_jsonl(src, [gen_row(0), gen_row(1)])  # p1 is unrated
    bodies = []
    compress.run(CFG, src, tmp_path / "b.jsonl", client=mock_client(lambda p, b: completion("1. Step."), bodies))
    prompts = {b["messages"][1]["content"].split("\n")[1]: b["messages"][1]["content"] for b in bodies}
    assert "at most 8 short numbered steps (under 320 words" in prompts["Question 0?"]
    assert "at most 6 short numbered steps (under 240 words" in prompts["Question 1?"]
    assert {r["id"]: r["max_steps"] for r in report.read_jsonl(tmp_path / "b.jsonl")} == {"p0": 8, "p1": 6}


def test_compress_rewrites_finished_samples_only(tmp_path):
    src = tmp_path / "raw.jsonl"
    report.write_jsonl(src, [gen_row(0), gen_row(1, finish_reason="length", answer="")])
    bodies = []
    compress.run(CFG, src, tmp_path / "b.jsonl", client=mock_client(lambda p, b: completion("1. Short.\n2. Steps."), bodies))
    assert len(bodies) == 1 and "Long trace." in bodies[0]["messages"][1]["content"]
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    out = {r["id"]: r for r in report.read_jsonl(tmp_path / "b.jsonl")}
    assert out["p0"]["rationale"] == "1. Short.\n\n2. Steps." and out["p0"]["cot_method"] == "compressed"
    assert out["p1"]["rationale"] == "" and out["p1"]["reasoning"] == "Long trace."
    bodies.clear()
    compress.run(CFG, src, tmp_path / "b.jsonl", client=mock_client(lambda p, b: completion("x"), bodies))
    assert bodies == []


# ---------------------------------------------------------------- verify

@pytest.mark.parametrize("answer, gold, ok", [
    ("Some reasoning.\nFinal answer: Paris", ["Paris"], True),
    ("Final answer: **Sad**.", ["Sad\n****\n"], True),
    ("The value is \\boxed{39.0}", ["39"], True),
    ("Final answer: c = -18", ["-18"], True),
    ("Final answer: 17", ["18"], False),
    ("Final answer: requires everyone's hard work.", ["requires everyone's hard work"], True),
    ("Final answer: A", ["A"], True),
    ("Final answer: B", ["A"], False),
    ("Final answer: the cat", ["cat", "a dog"], True),
    ("No marker at all: Paris", ["Paris"], False),
])
def test_is_correct(answer, gold, ok):
    assert verify.is_correct(answer, gold) is ok


def test_extract_boxed_takes_the_last_with_nested_braces():
    assert verify.extract_boxed("\\boxed{a} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert verify.extract_boxed("no box") is None


def test_count_steps():
    assert verify.count_steps("1. Read.\n2) Think.\nStep 3: Answer.") == 3
    assert verify.count_steps("First this.\nThen that.") == 2
    assert verify.count_steps("") == 0


@pytest.mark.parametrize("raw, formatted", [
    ("1. Read the task.\n2. Check the rule,\nwhich wraps here.\n3. Answer.",
     "1. Read the task.\n\n2. Check the rule, which wraps here.\n\n3. Answer."),
    ("1. A.\n\n\n2. B.", "1. A.\n\n2. B."),
    ("First this.\nThen that.", "First this.\n\nThen that."),
    ("One paragraph only.", "One paragraph only."),
    ("", ""),
])
def test_format_steps_separates_steps_with_a_blank_line(raw, formatted):
    assert verify.format_steps(raw) == formatted
    assert verify.count_steps(formatted) == verify.count_steps(raw)


# ---------------------------------------------------------------- judge

def test_judge_messages_put_the_response_last():
    system, user, _ = generate.load_template("judge", CFG["judge"]["prompt_version"])
    msgs = judge.judge_messages(system, user, "Q?", "Ref.", "1. Step.", "Ans.")
    text = msgs[1]["content"]
    assert msgs[0]["content"] == system and '{"critique"' in system
    assert text.index("Q?") < text.index("Ref.") < text.index("1. Step.") < text.index("Ans.")
    assert "[Reasoning]" not in judge.judge_messages(system, user, "Q?", None, "", "Ans.")[1]["content"]


def test_parse_grade_with_logprobs():
    top = [{"token": "7", "logprob": math.log(0.6)}, {"token": " 8", "logprob": math.log(0.3)},
           {"token": "x", "logprob": math.log(0.1)}]
    lp = {"content": [{"token": "{", "logprob": 0.0, "top_logprobs": []},
                      {"token": "7", "logprob": math.log(0.6), "top_logprobs": top}]}
    score, expected, critique = judge.parse_grade(completion('{"critique": "Good.", "score": 7}', logprobs=lp))
    assert score == 7 and critique == "Good." and expected == pytest.approx((7 * 0.6 + 8 * 0.3) / 0.9)


def test_parse_grade_strips_thought_block_and_falls_back_without_logprobs():
    score, expected, _ = judge.parse_grade(completion('<|channel>thought\n<channel|>{"critique": "Ok.", "score": 5}'))
    assert score == 5 and expected == 5.0


def test_parse_grade_rejects_a_reply_without_score():
    with pytest.raises(ValueError):
        judge.parse_grade(completion("I think it is fine."))


def test_judge_candidates():
    rows = [gen_row(0), gen_row(0, sample_idx=1),                                                # open: all
            gen_row(1, closed=True, gold=["Paris"], answer="Final answer: Paris"),                # passes
            gen_row(1, sample_idx=1, closed=True, gold=["Paris"], answer="Final answer: Rome"),   # fails, sibling passes
            gen_row(2, closed=True, gold=["X"], answer="Final answer: Y")]                        # none pass: all
    assert {(r["id"], r["sample_idx"]) for r in judge.candidates(rows)} == {("p0", 0), ("p0", 1), ("p1", 0), ("p2", 0)}


def test_judge_candidates_all_includes_rule_failing_samples():
    rows = [gen_row(1, closed=True, gold=["Paris"], answer="Final answer: Paris"),
            gen_row(1, sample_idx=1, closed=True, gold=["Paris"], answer="Final answer: Rome")]
    assert len(judge.candidates(rows)) == 1 and len(judge.candidates(rows, all_samples=True)) == 2


def test_judge_config_thinking_override():
    assert judge.judge_config(CFG)["max_tokens"] == CFG["judge"]["max_tokens"]
    assert judge.judge_config(CFG, prompt_version="v1")["prompt_version"] == "v1"
    assert generate.load_template("judge", "v1")[2] != generate.load_template("judge", CFG["judge"]["prompt_version"])[2]
    on = judge.judge_config(CFG, thinking=True)
    assert on["thinking"] and on["max_tokens"] == CFG["judge"]["max_tokens_thinking"]
    body = judge.judge_body(server.endpoint(CFG, "judge", "vllm"), on, [{"role": "user", "content": "x"}])
    assert body["chat_template_kwargs"] == {"enable_thinking": True} and body["max_tokens"] == 4096


def test_validate_records_agreement_thinking_and_cost(tmp_path, monkeypatch):
    rows = [{"prompt": "Q1?", "response": "good", "helpfulness": 4}, {"prompt": "Q1?", "response": "bad", "helpfulness": 1},
            {"prompt": "Q2?", "response": "fine", "helpfulness": 2}, {"prompt": "Q2?", "response": "great", "helpfulness": 3},
            {"prompt": "Q3?", "response": "same", "helpfulness": 2}, {"prompt": "Q3?", "response": "same2", "helpfulness": 2}]
    monkeypatch.setattr("datasets.load_dataset", lambda name, split: rows)
    scores = {"good": 8, "bad": 3, "fine": 7, "great": 5}  # agrees on Q1, disagrees on Q2; Q3 has equal ratings

    def reply(path, body):
        answer = body["messages"][1]["content"].rsplit("[Answer]\n", 1)[1]
        return completion(json.dumps({"critique": "x", "score": scores[answer]}), completion_tokens=40)
    out = tmp_path / "validation.md"
    agreement = judge.validate(CFG, 10, out=out, backend="vllm", client=mock_client(reply), thinking=True)
    assert agreement == 0.5
    row = out.read_text(encoding="utf-8").splitlines()[-1]
    assert "| on |" in row and "| 2 | 50.0% |" in row and "| 40 |" in row


def test_judge_run_writes_scores_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: [{"id": "p0", "gold": [], "reference": "Ref answer.",
                                                                   "hint_ok": True}])
    src = tmp_path / "x.kept.jsonl"
    report.write_jsonl(src, [gen_row(0, rationale="1. Think.")])
    bodies = []
    judge.run(CFG, [src], backend="vllm", client=mock_client(lambda p, b: completion('{"critique": "Fine.", "score": 8}'),
                                                             bodies))
    body = bodies[0]
    assert body["temperature"] == 0 and body["logprobs"] and body["response_format"]["type"] == "json_schema"
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and "Ref answer." in body["messages"][1]["content"]
    rec = report.read_jsonl(tmp_path / "x.kept.judge.jsonl")[0]
    assert rec["score"] == 8 and rec["expected_score"] == 8.0 and rec["reference_used"]
    bodies.clear()
    judge.run(CFG, [src], backend="vllm", client=mock_client(lambda p, b: completion("{}"), bodies))
    assert bodies == []


def test_judge_openrouter_body_has_no_logprobs():
    body = judge.judge_body(server.endpoint(CFG, "judge", "openrouter"), CFG["judge"], [{"role": "user", "content": "x"}])
    assert "logprobs" not in body and body["reasoning"] == {"enabled": False}


def test_agreement_counts_ties_half():
    assert judge.agreement([(8, 5, 4, 2), (3, 6, 4, 1), (5, 5, 3, 1)]) == pytest.approx(1.5 / 3)


# ---------------------------------------------------------------- report: judge vs rules, best-of-k

def test_auc_counts_ties_half():
    assert report.auc([9, 8], [3, 8]) == pytest.approx((2 + 1.5) / 4)


def test_judge_vs_rules():
    def judged(i, answer, score):
        return {**gen_row(1, sample_idx=i, closed=True, gold=["Paris"], answer=f"Final answer: {answer}"),
                "_judge": {"score": score, "expected_score": float(score)}}
    rows = [judged(0, "Paris", 8), judged(1, "Paris", 6), judged(2, "Rome", 7), judged(3, "Lyon", 2)]
    stats = report.judge_vs_rules(rows, 7)
    assert stats["judge vs rules: judged correct / wrong samples"] == "2 / 2"
    assert stats["judge vs rules: AUC (correct scored above wrong)"] == "0.75"
    assert stats["judge vs rules: wrong scored ≥ 7 (false accept)"] == "50%"
    assert report.judge_vs_rules(rows[:2], 7) == {}  # no wrong samples judged


def test_best_of_k_uses_the_first_k_samples():
    def s(i, k, score):
        return {**gen_row(i, sample_idx=k), "_judge": {"score": score, "expected_score": float(score)}}
    rows = [s(0, 0, 5), s(0, 1, 8), s(0, 2, 6), s(0, 3, 9), s(1, 0, 7), s(1, 1, 4), s(1, 2, 4), s(1, 3, 4)]
    table = report.best_of_k_lines("v", rows, 7)
    assert "| 1 | 6.00 | 50% |" in table and "| 2 | 7.50 | 100% |" in table and "| 4 | 8.00 | 100% |" in table


# ---------------------------------------------------------------- select

def write_judged(tmp_path, rows, scores):
    src = tmp_path / "s.kept.jsonl"
    report.write_jsonl(src, rows)
    report.write_jsonl(tmp_path / "s.kept.judge.jsonl", [{"id": i, "sample_idx": k, "score": s, "expected_score": e}
                                                         for (i, k), (s, e) in scores.items()])
    return src


def test_select_keeps_best_open_and_distinct_correct_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "load_complexity", lambda ds: {"p0": 3, "p1": 1})
    open_samples = [("1. Short.", "Ok."), ("1. A longer but equally good rationale here.", "Ok then."), ("1. Bad.", "No.")]
    closed_samples = [("1. France's capital is Paris, by history.", "Final answer: Paris"),
                      ("1. France's capital is Paris, by history.", "Final answer: Paris"),  # same rationale
                      ("2. Paris hosts the government of France, so it is the capital.", "Final answer: Paris"),
                      ("1. Rome.", "Final answer: Rome")]                                   # wrong answer
    rows = [gen_row(0, sample_idx=k, rationale=r, answer=a) for k, (r, a) in enumerate(open_samples)]
    rows += [gen_row(1, sample_idx=k, closed=True, gold=["Paris"], rationale=r, answer=a)
             for k, (r, a) in enumerate(closed_samples)]
    scores = {("p0", 0): (8, 8.0), ("p0", 1): (8, 8.0), ("p0", 2): (3, 3.0),
              ("p1", 0): (9, 8.9), ("p1", 1): (9, 8.8), ("p1", 2): (7, 7.1), ("p1", 3): (9, 9.0)}
    words = lambda text: len(text.split())
    stats = selection.run(CFG, write_judged(tmp_path, rows, scores), tmp_path / "final.jsonl", count=words)
    out = report.read_jsonl(tmp_path / "final.jsonl")
    keys = {(r["id"], r["sample_idx"]) for r in out}
    assert keys == {("p0", 0), ("p1", 0), ("p1", 2)}       # tie -> shorter; duplicate and wrong answers skipped
    assert [r["id"] for r in out] == ["p1", "p1", "p0"]    # curriculum: complexity 1 before 3
    assert out[0]["selection"]["method"] == "rule+judge"
    assert out[-1]["complexity"] == 3 and out[-1]["n_steps"] == 1
    assert stats["kept: judge"] == 1 and stats["kept: rule+judge"] == 1


def test_final_rows_have_student_tokens_and_a_teacher_group(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "load_complexity", lambda ds: {"p0": 2})
    row = gen_row(0, rationale="1. One.\n\n2. Two.", answer="The answer.", reasoning="Long raw thinking.",
                  reasoning_tokens=610, answer_tokens=12, prompt_tokens=60, max_tokens=32768, temperature=1.0,
                  cot_method="compressed", max_steps=3, compress_prompt_hash="abc")
    src = write_judged(tmp_path, [row], {("p0", 0): (8, 8.2)})
    selection.run(CFG, src, tmp_path / "f.jsonl", count=lambda text: len(text.split()))
    final = report.read_jsonl(tmp_path / "f.jsonl")[0]
    assert final["student_tokens"] == {"input": 2, "rationale": 4, "answer": 2, "total": 8 + report.STUDENT_TEMPLATE_TOKENS}
    assert final["teacher"]["thinking_tokens"] == 610 and final["teacher"]["max_tokens"] == 32768
    assert final["teacher"]["sampling"] == {"temperature": 1.0} and final["teacher"]["compress_prompt_hash"] == "abc"
    assert final["complexity"] == 2 and final["max_steps"] == 3 and final["n_steps"] == 2
    for gone in ("reasoning", "reasoning_tokens", "answer_tokens", "prompt_tokens", "finish_reason", "cost", "provider"):
        assert gone not in final


def test_select_drops_low_scores_and_unjudged_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "load_complexity", lambda ds: {})
    src = write_judged(tmp_path, [gen_row(0, rationale="1. x."), gen_row(1, rationale="1. y.")], {("p0", 0): (5, 5.2)})
    stats = selection.run(CFG, src, tmp_path / "f.jsonl", count=len)
    assert stats["rows_written"] == 0 and stats["dropped: no judged sample"] == 1
    assert stats[f"dropped: best score < {CFG['select']['min_score']}"] == 1


# ---------------------------------------------------------------- analyze

def test_analyze_compares_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "load_complexity", lambda ds: {"p0": 2})
    a = [gen_row(0, sample_idx=k, mode="cot", thinking=False, cot_method="prompted", rationale="1. Go.") for k in range(2)]
    b = [gen_row(0, mode="open", thinking=True, cot_method="compressed", rationale="1. Go.\n2. Done."),
         gen_row(1, mode="open", thinking=True, cot_method="compressed", closed=True, gold=["A"], answer="Final answer: A")]
    report.write_jsonl(tmp_path / "a.kept.jsonl", a)
    report.write_jsonl(tmp_path / "a.kept.judge.jsonl", [{"id": "p0", "sample_idx": 0, "score": 8, "expected_score": 7.6,
                                                         "critique": "Good."}])
    b.append(gen_row(1, sample_idx=1, mode="open", thinking=True, cot_method="compressed", closed=True, gold=["A"],
                     answer="Final answer: B"))
    report.write_jsonl(tmp_path / "b.kept.judge.jsonl", [
        {"id": "p1", "sample_idx": 0, "score": 8, "expected_score": 8.0, "critique": "Right."},
        {"id": "p1", "sample_idx": 1, "score": 3, "expected_score": 3.0, "critique": "Wrong."}])
    report.write_jsonl(tmp_path / "b.kept.jsonl", b)
    out = tmp_path / "r.md"
    report.analyze([tmp_path / "a.kept.jsonl", tmp_path / "b.kept.jsonl"], out)
    text = out.read_text(encoding="utf-8")
    assert "cot/off/prompted" in text and "open/on/compressed" in text and "Matched prompts" in text
    assert "## no_robots / closed" in text and "closed pass@1 / pass@N | 50% / 100%" in text
    assert "mean score / mean best-of-N" in text and "By complexity level" in text
    assert "judge scores with logprob detail" in text and "judge vs rules: AUC (correct scored above wrong) | 1.00" in text
    assert "Best-of-k (open/on/compressed; 1 prompts)" in text and "pass@k" in text
