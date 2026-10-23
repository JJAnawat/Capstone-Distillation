"""Offline tests: response parsing, request bodies, filters, resume, dedup and sampling. No network."""
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest

from distill import cli, data, generate, report, server
from helpers import CFG, completion, mock_client

FCFG = CFG["filter"]


def row(**kw):
    base = {"id": "x", "input": "What is the capital of France? Please explain briefly why.",
            "answer": "The capital of France is Paris, which has been its political centre for centuries.",
            "reasoning": "The user asks about France.", "finish_reason": "stop", "mode": "open",
            "reasoning_tokens": 10, "think_unclosed": False}
    return {**base, **kw}


# ---------------------------------------------------------------- parsing

def test_parse_reasoning_field():
    p = generate.parse_response(completion("\n\n391", reasoning="17*23=391", reasoning_tokens=40))
    assert p["reasoning"] == "17*23=391" and p["answer"] == "391"
    assert p["reasoning_tokens"] == 40 and p["answer_tokens"] == 10 and p["cost"] == 0.001


def test_parse_reasoning_content_field():  # vLLM's name for the reasoning field
    j = completion("391")
    j["choices"][0]["message"] = {"content": "391", "reasoning_content": "17*23=391"}
    p = generate.parse_response(j)
    assert p["reasoning"] == "17*23=391" and p["answer"] == "391"


def test_parse_thinking_off_reply_filed_as_reasoning():
    p = generate.parse_response(completion("", reasoning="Hello there."), thinking=False)
    assert p["answer"] == "Hello there." and p["reasoning"] == ""


def test_parse_inline_think():
    p = generate.parse_response(completion("<think>step one</think>\n\nFinal: 4"))
    assert p["reasoning"] == "step one" and p["answer"] == "Final: 4" and not p["think_unclosed"]


def test_parse_closing_tag_only():
    p = generate.parse_response(completion("step one\n</think>\nFinal: 4"))
    assert p["reasoning"] == "step one" and p["answer"] == "Final: 4"


def test_parse_unclosed_think():
    p = generate.parse_response(completion("<think>still thinking...", finish="length"))
    assert p["think_unclosed"] and p["answer"] == "" and p["reasoning"] == "still thinking..."


def test_parse_missing_usage():
    j = completion("hi")
    del j["usage"]
    p = generate.parse_response(j)
    assert p["prompt_tokens"] == 0 and p["cost"] is None


def test_parse_cot_rationale():
    p = generate.parse_response(completion("<reasoning>1. A.\n2. B.</reasoning>\n\nThe answer is B."),
                                thinking=False, cot=True)
    assert p["rationale"] == "1. A.\n\n2. B." and p["answer"] == "The answer is B."  # steps separated by a blank line


def test_parse_cot_without_tags_has_empty_rationale():
    p = generate.parse_response(completion("Just an answer."), thinking=False, cot=True)
    assert p["rationale"] == "" and p["answer"] == "Just an answer."


# ---------------------------------------------------------------- request bodies and server plumbing

def test_build_body_backends():
    msgs, sampling = [{"role": "user", "content": "hi"}], CFG["sampling"]["thinking_off"]
    v = server.build_body(server.endpoint(CFG, "teacher", "vllm"), msgs, sampling, False, 7, 100)
    assert v["chat_template_kwargs"] == {"enable_thinking": False} and "provider" not in v and "reasoning" not in v
    assert {k: v[k] for k in sampling} == sampling and v["seed"] == 7 and v["max_tokens"] == 100
    o = server.build_body(server.endpoint(CFG, "teacher", "openrouter"), msgs, sampling, True, 7, 100)
    assert o["reasoning"] == {"enabled": True} and o["provider"] and "chat_template_kwargs" not in o
    assert "min_p" not in o and "repetition_penalty" not in o and o["top_k"] == 20  # neutral values left out
    assert server.endpoint(CFG, "judge", "openrouter")["api_base"] == CFG["openrouter_api_base"]


def test_sampling_profiles_send_every_card_field():
    for profile in CFG["sampling"].values():
        assert set(profile) == {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}


def test_wait_ready_polls_until_the_model_is_served():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200 if calls.count("/health") >= 3 else 503)
        return httpx.Response(200, json={"data": [{"id": CFG["teacher"]["vllm"]["model"]}]})
    server.wait_ready(CFG, "teacher", client=httpx.Client(transport=httpx.MockTransport(handler)), poll_s=0)
    assert calls.count("/health") == 3


def test_serve_command_appends_extra_vllm_args(monkeypatch):
    monkeypatch.setenv("DISTILL_SERVE_ARGS", """--async-scheduling --speculative-config '{"method": "x", "n": 2}'""")
    cmd = server.serve_command(CFG, "teacher")
    assert cmd[:3] == ["vllm", "serve", CFG["teacher"]["vllm"]["model"]] and "--reasoning-parser=qwen3" in cmd
    assert cmd[-3:] == ["--async-scheduling", "--speculative-config", '{"method": "x", "n": 2}']
    # FlashInfer's sampler needs a CUDA compiler the VM lacks; both servers use vLLM's PyTorch sampler.
    assert server.serve_env(CFG, "teacher") == server.serve_env(CFG, "judge") == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def memory_readings(*free_gib, total_gib=40):
    readings = iter([(f * 1024, total_gib * 1024) for f in free_gib])
    return lambda: next(readings)


def test_fit_gpu_keeps_the_config_on_a_free_gpu_and_shrinks_beside_others():
    cmd = server.serve_command(CFG, "teacher")
    assert server.fit_gpu(CFG, "teacher", cmd, memory_readings(39.9)) == cmd  # 0.92 fits
    assert server.fit_gpu(CFG, "teacher", cmd, lambda: None) == cmd           # no nvidia-smi: the laptop
    shrunk = server.fit_gpu(CFG, "teacher", cmd, memory_readings(30))         # another user holds 10 GiB
    assert shrunk[-1] == "--gpu-memory-utilization=0.7" and server.requested_utilization(shrunk) == 0.7
    assert sum(a.startswith("--gpu-memory-utilization") for a in shrunk) == 1
    assert server.with_utilization(["vllm", "--gpu-memory-utilization", "0.9", "--port=1"], 0.5) == [
        "vllm", "--port=1", "--gpu-memory-utilization=0.5"]


def test_fit_gpu_waits_for_memory_then_gives_up():
    sleeps = []
    cmd = server.serve_command(CFG, "judge")
    # 20 GiB free is short of the judge's minimum; the second reading (36 GiB) fits.
    assert server.fit_gpu(CFG, "judge", cmd, memory_readings(20, 36), sleeps.append)[-1].endswith("=0.85")
    assert len(sleeps) == 1
    cfg = {**CFG, "gpu_share": {**CFG["gpu_share"], "wait_s": 0}}
    with pytest.raises(RuntimeError, match="judge needs"):
        server.fit_gpu(cfg, "judge", cmd, lambda: (20 * 1024, 40 * 1024), lambda s: None)


def test_wait_ready_fails_fast_when_the_server_dies(monkeypatch):
    monkeypatch.setattr(server, "alive", lambda pid: False)
    handler = lambda request: httpx.Response(503)
    with pytest.raises(RuntimeError, match="exited during startup"):
        server.wait_ready(CFG, "teacher", client=httpx.Client(transport=httpx.MockTransport(handler)), poll_s=0,
                          pid=12345)


def test_wait_ready_rejects_a_server_with_another_model():
    handler = lambda request: httpx.Response(200, json={"data": [{"id": "other/model"}]})
    with pytest.raises(RuntimeError):
        server.wait_ready(CFG, "judge", client=httpx.Client(transport=httpx.MockTransport(handler)), poll_s=0)


def test_ctrl_c_drops_queued_requests_and_returns_at_once(tmp_path, monkeypatch):
    # On the VM, Ctrl-C also stops vLLM, so requests in flight never get a reply: run_pool must not wait for them.
    release, started = threading.Event(), []

    def work(i):
        started.append(i)
        if i:
            release.wait(10)
        return {"id": i}, 1

    def interrupt(self, tokens=0, failed=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(server.Progress, "update", interrupt)
    out, t0 = tmp_path / "out.jsonl", time.time()
    with pytest.raises(KeyboardInterrupt):
        server.run_pool("t", work, list(range(50)), out, 2, lambda i: {"id": i})
    release.set()
    assert time.time() - t0 < 5 and len(started) <= 3
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == [0]  # finished record kept


def test_cli_exits_on_ctrl_c_without_joining_threads(monkeypatch):
    codes = []
    monkeypatch.setattr(cli, "run", lambda argv: (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr(cli.os, "_exit", codes.append)
    cli.main(["prepare"])
    assert codes == [130]


# ---------------------------------------------------------------- filters

def test_filter_accepts_clean_row():
    assert report.rejection_reasons(row(), FCFG) == []


@pytest.mark.parametrize("overrides, reason", [
    ({"answer": "  "}, "missing_answer"),
    ({"answer": "Paris </think> is the capital of France, as everyone knows."}, "malformed_think_tag"),
    ({"think_unclosed": True}, "malformed_think_tag"),
    ({"finish_reason": "length"}, "finish_not_stop"),
    ({"answer": "La capitale de la France est Paris, qui est son centre politique depuis des siècles."},
     "language_mismatch"),
    ({"reasoning": "wait let me check the answer again. " * 40}, "repetition"),
    ({"mode": "hint", "reasoning": "The provided answer says Paris."}, "hint_leakage"),
    ({"reasoning_tokens": 5000}, "reasoning_too_long"),
])
def test_filter_rejects(overrides, reason):
    assert reason in report.rejection_reasons(row(**overrides), FCFG, max_reasoning_tokens=4000)


def test_language_check_skipped_for_language_identification():
    r = row(input="Верхнерейнская низменность Which language is this?", answer="This text is written in Russian.")
    assert "language_mismatch" not in report.rejection_reasons(r, FCFG)


def test_language_check_skipped_for_translation_prompts():
    r = row(input="Translate this sentence into French: the capital of France is Paris.",
            answer="La capitale de la France est Paris, qui est son centre politique depuis des siècles.")
    assert "language_mismatch" not in report.rejection_reasons(r, FCFG)


def test_filter_rationale_and_student_budget(monkeypatch):
    monkeypatch.setattr(report, "student_tokens", lambda r, tok: 3000 if "huge" in r["answer"] else 100)
    base = row(rationale="1. France's capital is Paris.", cot_method="prompted", mode="cot")
    limits = {"max_rationale_words": 150, "student": ("tok", 2048)}
    assert report.rejection_reasons(base, FCFG, **limits) == []
    assert "missing_rationale" in report.rejection_reasons({**base, "rationale": " "}, FCFG, **limits)
    assert "rationale_too_long" in report.rejection_reasons({**base, "rationale": "word " * 200}, FCFG, **limits)
    assert "over_student_budget" in report.rejection_reasons({**base, "answer": base["answer"] + " huge"}, FCFG,
                                                             **limits)
    # A looping raw trace does not matter once a short rationale replaces it.
    looping = {**base, "cot_method": "compressed", "reasoning": "wait let me check the answer again. " * 40}
    assert report.rejection_reasons(looping, FCFG, **limits) == []


def test_filter_file_writes_outputs(tmp_path):
    src = tmp_path / "gen.jsonl"
    report.write_jsonl(src, [row(id="a"), row(id="b", finish_reason="length")])
    s = report.filter_file(src, tmp_path / "kept.jsonl", FCFG)
    assert s["kept"] == 1 and s["by_reason"] == {"finish_not_stop": 1}
    assert report.read_jsonl(tmp_path / "kept.rejected.jsonl")[0]["id"] == "b"


# ---------------------------------------------------------------- data

def test_normalized_ids_dedupe_whitespace_and_case():
    assert data.make_id("Hello   World\n") == data.make_id("hello world")
    assert data.make_id("hello world") != data.make_id("hello worlds")


def test_sample_is_deterministic_and_stratified():
    rows = [{"id": f"{s}{i}", "strata": s} for s in "abcde" for i in range(20)]
    first, again = data.sample(rows, 10, seed=1), data.sample(rows, 10, seed=1)
    assert [r["id"] for r in first] == [r["id"] for r in again]
    assert {r["strata"] for r in first} == set("abcde")
    assert [r["id"] for r in data.sample(rows, 10, seed=2)] != [r["id"] for r in first]


def test_ultrafeedback_gold_answers_and_closed_flag():
    comp = [{"response": "4", "fine-grained_score": 5, "overall_score": 9}]
    ds = [{"instruction": "Solve 2 + 2.", "source": "flan_v2_flan2021", "correct_answers": ["4"], "completions": comp},
          {"instruction": "Write a poem.", "source": "evol_instruct", "correct_answers": ["None"], "completions": comp}]
    solved, poem = data._ultrafeedback(ds, CFG["datasets"]["ultrafeedback"])
    assert solved["gold"] == ["4"] and poem["gold"] == []
    v = CFG["verify"]
    assert data.is_closed(solved, v) and not data.is_closed(poem, v)
    assert not data.is_closed({**solved, "strata": "truthful_qa"}, v)
    assert not data.is_closed({**solved, "gold": ["a much longer free text answer"]}, v)


def test_select_rows_only_closed_and_same_subset(monkeypatch):
    rows = [{**p, "closed": i % 2 == 0} for i, p in enumerate(PROMPTS)]
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: rows)
    assert [r["id"] for r in data.select_rows(CFG, "x", only="closed")] == ["p0", "p2"]
    assert [r["id"] for r in data.select_rows(CFG, "x", only="open")] == ["p1", "p3"]
    assert data.select_rows(CFG, "x", limit=2, seed=3) == data.select_rows(CFG, "x", limit=2, seed=3)
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: [r for r in rows if r["closed"]])
    assert data.subsets(CFG, "x") == ["closed"]  # pipeline.sh then makes no empty open set
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: rows)
    assert data.subsets(CFG, "x") == ["open", "closed"]


def logiqa_row(premise="All cats purr. Tom is a cat.", answer=1):
    return {"text": json.dumps({"id": 1, "answer": answer, "text": premise, "question": "What follows?",
                                "options": ["Tom barks.", " Tom purrs. ", "Tom flies.", "Nothing."],
                                "type": {"Conjunctive Reasoning": False, "Categorical Reasoning": True}})}


def test_logiqa_keeps_english_multiple_choice_rows_with_letter_golds():
    zh = {"text": json.dumps({"example_id": 2, "answer": 0, "text": "所有猫", "question": "?", "options": ["A.x"]})}
    nli = {"text": json.dumps({"id": 3, "premise": "P", "hypothesis": "H", "label": "entailment", "type": {}})}
    two = {"text": nli["text"] + logiqa_row()["text"]}  # two mirror rows hold two objects back to back
    (row, again) = data._logiqa([zh, logiqa_row(), nli, two], CFG["datasets"]["logiqa"])
    assert again == row
    assert row["input"].endswith("A. Tom barks.\nB. Tom purrs.\nC. Tom flies.\nD. Nothing.\n\n"
                                 "Answer with the letter of the correct option.")
    assert row["gold"] == ["B", "B. Tom purrs.", "Tom purrs."] and row["strata"] == "Categorical Reasoning"


def test_missing_prepared_file_is_an_error_not_a_full_prepare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "prepare", lambda cfg, only=None: pytest.fail("must not rebuild every dataset"))
    with pytest.raises(FileNotFoundError, match="prepare --dataset logiqa"):
        data.load_prompts(CFG, "logiqa")


def test_cli_loads_vm_env_for_commands_typed_without_sourcing_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vm_env").write_text("export HF_HOME=/dev/shm/someone/hf\n")
    monkeypatch.setenv("HF_HOME", "x")
    monkeypatch.delenv("HF_HOME")  # restored to its original state afterwards
    monkeypatch.setattr(cli.data, "subsets", lambda cfg, name: ["closed"])
    cli.main(["--config", str(Path(__file__).parent.parent / "config.yaml"), "subsets", "logiqa"])
    assert os.environ["HF_HOME"] == "/dev/shm/someone/hf"


def test_prepare_one_dataset_leaves_earlier_ones_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"datasets": {k: CFG["datasets"][k] for k in ("no_robots", "logiqa")}, "verify": CFG["verify"]}
    dup = logiqa_row(premise="Seen before.")
    (dup_prompt,) = data._logiqa([dup], cfg["datasets"]["logiqa"])
    data.PREPARED_DIR.mkdir(parents=True)
    earlier = json.dumps({"id": data.make_id(dup_prompt["input"]), "input": dup_prompt["input"]}) + "\n"
    (data.PREPARED_DIR / "no_robots.jsonl").write_text(earlier, encoding="utf-8")
    (data.PREPARED_DIR / "stats.json").write_text(json.dumps({"no_robots": {"kept": 1}}), encoding="utf-8")
    fake = types.SimpleNamespace(load_dataset=lambda hf_id, split: [dup, logiqa_row()])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    data.prepare(cfg, only="logiqa")
    assert (data.PREPARED_DIR / "no_robots.jsonl").read_text(encoding="utf-8") == earlier  # untouched
    (row,) = [json.loads(line) for line in open(data.PREPARED_DIR / "logiqa.jsonl", encoding="utf-8")]
    assert row["closed"] and row["gold"][0] == "B" and row["source_dataset"] == "logiqa"  # the duplicate is gone
    stats = json.loads((data.PREPARED_DIR / "stats.json").read_text(encoding="utf-8"))
    assert stats["no_robots"] == {"kept": 1} and stats["logiqa"]["duplicate"] == 1


# ---------------------------------------------------------------- generate: samples, resume, templates, hint isolation

PROMPTS = [{"id": f"p{i}", "input": f"Question {i}?", "strata": None, "reference": f"SECRET-REF-{i}",
            "hint_ok": True, "source_dataset": "no_robots"} for i in range(4)]


def fake_client(seen_bodies):
    return mock_client(lambda path, body: completion("Answer.", reasoning="Because.", reasoning_tokens=5), seen_bodies)


def test_resume_skips_done_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: PROMPTS)
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"id": "p0"}) + "\n" + json.dumps({"id": "p2"}) + "\n", encoding="utf-8")
    bodies = []
    generate.run(CFG, "no_robots", n=1, out=out, client=fake_client(bodies), concurrency=2)
    asked = sorted(b["messages"][1]["content"] for b in bodies)
    assert asked == ["Question 1?", "Question 3?"]
    assert sorted(r["id"] for r in report.read_jsonl(out)) == ["p0", "p1", "p2", "p3"]
    # Second run has nothing left to do.
    bodies.clear()
    generate.run(CFG, "no_robots", n=1, out=out, client=fake_client(bodies))
    assert bodies == []


def test_n_samples_resume_with_distinct_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: PROMPTS[:2])
    out = tmp_path / "n.jsonl"
    out.write_text(json.dumps({"id": "p0", "sample_idx": 1}) + "\n", encoding="utf-8")
    bodies = []
    generate.run(CFG, "no_robots", n=3, out=out, client=fake_client(bodies))
    assert len(bodies) == 5 and len({b["seed"] for b in bodies}) == 5
    assert sorted((r["id"], r["sample_idx"]) for r in report.read_jsonl(out)) == [
        (f"p{i}", k) for i in range(2) for k in range(3)]
    assert report.read_jsonl(f"{out}.runs.jsonl")[-1]["done"] == 5


def test_vllm_body_and_record_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: PROMPTS[:1])
    bodies = []
    generate.run(CFG, "no_robots", n=1, out=tmp_path / "v.jsonl", client=fake_client(bodies))
    sampling = CFG["sampling"]["thinking_on"]
    assert {k: bodies[0][k] for k in sampling} == sampling
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": True} and "provider" not in bodies[0]
    rec = report.read_jsonl(tmp_path / "v.jsonl")[0]
    assert rec["provider"] == "local-vllm" and rec["min_p"] == 0.0 and rec["sample_idx"] == 0
    assert rec["seed"] == generate.sample_seed(CFG["generate"]["seed"], "p0", 0)


def test_closed_prompts_get_final_answer_template(tmp_path, monkeypatch):
    rows = [{**PROMPTS[0], "closed": True, "gold": ["Paris"]}, PROMPTS[1]]
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: rows)
    bodies = []
    generate.run(CFG, "ultrafeedback", mode="cot", thinking=False, n=1, out=tmp_path / "c.jsonl",
                 client=fake_client(bodies))
    systems = {b["messages"][1]["content"]: b["messages"][0]["content"] for b in bodies}
    assert "Final answer" in systems["Question 0?"] and "Final answer" not in systems["Question 1?"]
    assert all("<reasoning>" in s for s in systems.values())
    assert all(b["max_tokens"] == CFG["cot"]["max_tokens"] for b in bodies)
    recs = {r["id"]: r for r in report.read_jsonl(tmp_path / "c.jsonl")}
    assert recs["p0"]["gold"] == ["Paris"] and "gold" not in recs["p1"] and recs["p0"]["cot_method"] == "prompted"


def test_samples_and_step_limit_follow_complexity(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: PROMPTS[:3])
    monkeypatch.setattr(data, "load_complexity", lambda ds: {"p0": 1, "p1": 5})  # p2 is unrated
    out = tmp_path / "lv.jsonl"
    out.write_text(json.dumps({"id": "p1", "sample_idx": 0}) + "\n", encoding="utf-8")
    bodies = []
    generate.run(CFG, "no_robots", mode="cot", thinking=False, out=out, client=fake_client(bodies))
    per_prompt = {}
    for b in bodies:
        per_prompt.setdefault(b["messages"][1]["content"], []).append(b["messages"][0]["content"])
    assert {q: len(s) for q, s in per_prompt.items()} == {"Question 0?": 2, "Question 1?": 3, "Question 2?": 4}
    assert "at most 4, under 160 words" in per_prompt["Question 0?"][0]
    assert "at most 10, under 400 words" in per_prompt["Question 1?"][0]
    assert "at most 6, under 240 words" in per_prompt["Question 2?"][0]
    recs = report.read_jsonl(out)[1:]
    assert {(r["id"], r["max_steps"]) for r in recs} == {("p0", 4), ("p1", 10), ("p2", 6)}


def test_cot_mode_requires_thinking_off():
    with pytest.raises(ValueError):
        generate.run(CFG, "no_robots", mode="cot", thinking=True, out="unused.jsonl")


def test_vllm_reasoning_tokens_counted_with_tokenize(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: PROMPTS[:1])

    def handler(path, body):
        if path == "/tokenize":
            return {"count": 30}
        return completion("Answer.", reasoning="Long thoughts.", reasoning_tokens=0, completion_tokens=50)
    generate.run(CFG, "no_robots", n=1, out=tmp_path / "t.jsonl", client=mock_client(handler))
    rec = report.read_jsonl(tmp_path / "t.jsonl")[0]
    assert rec["reasoning_tokens"] == 30 and rec["answer_tokens"] == 20


def test_hint_reaches_teacher_but_not_stored_input(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: PROMPTS[:1])
    out = tmp_path / "hint.jsonl"
    bodies = []
    generate.run(CFG, "no_robots", mode="hint", n=1, out=out, client=fake_client(bodies))
    assert "SECRET-REF-0" in json.dumps(bodies[0])
    rec = report.read_jsonl(out)[0]
    assert rec["input"] == "Question 0?" and "SECRET-REF" not in json.dumps(rec)
    assert rec["mode"] == "hint" and len(rec["prompt_hash"]) == 16


def test_budget_stops_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: PROMPTS)
    bodies = []
    state = generate.run(CFG, "no_robots", n=1, out=tmp_path / "b.jsonl", client=fake_client(bodies),
                         concurrency=1, max_cost=0.0015, backend="openrouter")
    assert state["ok"] == 2  # each call costs 0.001; stops after crossing 0.0015
