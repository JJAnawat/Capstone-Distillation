# CLAUDE.md

Context for future sessions. Commands, schemas and stage details live in README.md; this file records the project,
the decisions and the reasons behind them.

## Project
Capstone: turn a tiny Thai model into a small **reasoning model** by distilling short chain-of-thought (CoT) from a teacher.
- **Student:** `wongwian-org/wongwian-micro-instruct` (MIT). 272M Llama, trained from scratch on Thai-dominant data.
  **2,048-token context** (prompt + response). Its card recommends 150–300 new tokens for chat (512 long-form),
  temperature 0.6–0.8, repetition_penalty 1.2–1.4 and no_repeat_ngram_size 4, so it loops easily. Plain-text chat
  template `system:` / `ผู้ใช้:` / `ผู้ช่วย:`, no thinking tags. Tokenizer: SentencePiece Unigram, 32K vocab.
- **Goal:** better non-math logical reasoning, instruction following and safety (refuse harmful requests without
  over-refusing). Don't change the architecture, and don't hurt Thai fluency or inference efficiency. Math isn't a
  target skill, but the math items in the existing datasets are still distilled (user's call).
- **This repo is stage 1:** teacher data generation. Training and translation into Thai are separate stages.
  Translation is undecided, so generation is in English for now.

## Pipeline (details in README.md)
prepare → rate → generate → compress (variant B only) → filter → judge → select → analyze
- **Teacher:** Qwen3.5-9B (bf16) behind `vllm serve`. **Judge:** Gemma 4 31B QAT W4A16 behind `vllm serve`.
  They can't share the A100 40GB, so teacher stages and judge stages run as separate GPU phases.
- **OpenRouter** is a second backend (`--backend openrouter`), used for laptop smoke tests.

## Where things run
- **Laptop (Windows 11):** all coding and the offline tests. There is no GPU. Shells: PowerShell and Git Bash.
- **GitHub:** the user commits and pushes. Don't commit or push unless asked.
- **GPU VM:** an A100 40GB inside a company machine, reached over SSH (Linux). The user pulls there and runs
  `scripts/run_pilot.sh` or `scripts/run_full.sh` in tmux. Claude can't reach the VM unless the user sets up an ssh alias.

## Key decisions and why
- **Every training target is short CoT + answer, never Qwen's raw thinking.**
  - Raw traces don't fit a 2,048 context: in the OpenRouter pilot the median was 1,739 tokens and p90 was 5,045.
  - Models of 3B or smaller learn poorly from long CoT (Li et al. 2025, "Small Models Struggle to Learn from Strong
    Reasoners"). Short rationales with many samples per question do work at 125M–1.3B (SCoTD, Li et al. 2023).
  - Raw traces stay in the generation files for analysis only; `select` drops them.
- **Two ways to make the CoT short, compared in the pilot.** Pick one after the pilot.
  - A "prompted": thinking off, at most 5 steps inside `<reasoning>` tags.
  - B "compressed": thinking on, then `compress` rewrites the trace into at most 5 steps.
- **N = 8 samples per prompt.** The user wanted 6–10.
- **Closed-ended means a known short answer.**
  - Source: UltraFeedback `correct_answers` of 4 words or fewer, from flan_v2_niv2, flan_v2_p3 and flan_v2_flan2021.
    For other sources the field is the literal string "None" or long free text.
  - Rules decide correctness, and the judge still ranks the reasoning of the correct samples, because a yes/no or
    multiple-choice answer can be right with wrong reasoning.
  - truthful_qa isn't rule-checked: its short golds are answers like "I have no comment".
- **Selection:** open prompts keep their best sample (judge score ≥ 7 of 9). Closed prompts keep up to 3 distinct
  correct samples, since several rationales per question help small students (SCoTD).
- **Sampling:** the Qwen3.5 model card's "general" profiles (thinking on 1.0/0.95/20, off 0.7/0.8/20, presence 1.5),
  with all six fields sent explicitly. Thinking-off output is capped at 4,096 tokens. The judge and the rater
  deliberately use temperature 0.
- **Curriculum:**
  - Every question is rated 1–5 with the DeReason prompt (Hu et al. 2026, arXiv:2603.11193, App. A.1).
  - `select` writes the data from easy to hard.
  - The paper's split (SFT on levels ≤3, RL on ≥4) is left to the training stage.
- **Judge:** a pointwise rubric over reasoning + answer on a 1–9 scale. A single digit is a single token, so logprobs
  give an expected score that breaks ties. It is checked against HelpSteer2 human preferences (`judge --validate`).

## Commands
- Laptop: `uv venv --python 3.12 && uv pip install -e ".[dev]"`, then `uv run pytest`.
- Laptop smoke test (Git Bash; needs `OPENROUTER_API_KEY` in `.env`; costs cents): `bash scripts/smoke_openrouter.sh`.
- VM: the README's "VM sessions" section. Session 1: `vm_check.sh`, install, download, a `LIMIT=5` smoke run.
  Session 2: `run_pilot.sh`. Session 3: `bench.sh` and judge validation with thinking on. Each session ends with
  `scripts/collect.sh <name> <out_dir>`, then a commit and push of `reports/`.
- **How results reach Claude:** the user runs everything on the VM, then pushes `reports/<name>/` and the Markdown
  reports. After a pull on the laptop, read `reports/<name>/` and the report. Larger data comes as
  `data/<name>_data.tar.gz` over scp.

## Conventions
- `config.yaml` is the single source of truth; CLI flags override it. `DISTILL_CONFIG=path` swaps in another config
  file, for example a VM-local copy with a different port.
- `notebooks/colab_gpu_test.ipynb` is an earlier Colab throughput test, replaced by the VM pipeline.
- Prompts live in `prompts/{name}_{version}.txt` with `### system` / `### user` sections. Closed-ended prompts use
  `{mode}_closed_{version}` when that file exists.
  - Templates go through `str.format`, so literal braces must be doubled.
  - The judge's system prompt is the exception: it is used verbatim.
- Every stage appends to its output and resumes by key: `(id, sample_idx)`, or `id` for ratings. Failures go to
  `<out>.errors.jsonl` and are retried on the next run. Each run appends a summary to `<out>.runs.jsonl`.
- Tests are offline (httpx.MockTransport) and never import vLLM, which only installs on Linux.
- Match the existing style: compact functions, module docstrings, few comments.

## Gotchas
- A local vLLM server fills unsent sampling fields from the model's generation_config.json, so always send all six.
- httpx defaults to 100 connections. `server.make_client` raises the limit to the configured concurrency.
- vLLM returns reasoning in `reasoning_content`. With thinking off, its parser can put the whole reply there;
  `parse_response` handles both.
- vLLM's usage has no reasoning-token count, so `generate` asks `/tokenize`.
- With thinking off, Gemma 4 still emits an empty thought block (`<|channel>thought…<channel|>`); `judge.parse_grade`
  strips it.
- The Windows console is cp1252. Set `PYTHONIOENCODING=utf-8` for ad-hoc scripts that print Thai.
- `.sh` files must keep LF line endings (`.gitattributes`), or bash on the VM fails with `$'\r'`.

## Licensing
No Robots is CC BY-NC 4.0: everything derived from it is non-commercial, so keep `source_dataset` on every row.
Confirm that running it on company hardware is acceptable. See docs/LICENSES.md.

## Status (2026-10-21)
- Done: the OpenRouter pilot (30 prompts, thinking on/off; reports/pilot.md).
- Built: the local-vLLM pipeline with N samples, rating, compress, filter budgets, judge, rule checks and curriculum
  selection. Also the VM tooling: `vm_check.sh`, `collect.sh`, `bench.sh`, clean Ctrl-C stop and resume,
  `judge --all` / `--thinking`, and report sections for best-of-k, judge vs rules and logprob detail.
  - Tested offline (73 tests), plus dry runs of `pipeline.sh`, `collect.sh` and `bench.sh` against a fake
    OpenAI-compatible server.
  - It has not yet run against real models.
- The user got VM access on 2026-10-21: exclusive, a few hours per day. The OpenRouter smoke test is skipped.
- Next:
  1. The user pushes to a private GitHub repo.
  2. VM session 1: setup and smoke.
  3. Session 2: pilot.
  4. Session 3: bench and judge thinking on vs off.
  5. Decide A or B, N, the judge setup and the server flags; size the full run in daily sessions.
- What decides each question:
  - **A vs B:** matched-prompt scores, pass@8, yield and student-token fit, plus a hand review. B costs ~6× the teacher
    tokens, so it must win clearly.
  - **N:** the best-of-k curve. If k=4 comes within ~0.1 score and ~2 points of yield of k=8, use N=4.
  - **Judge:** HelpSteer2 agreement of about 70% or more; judge-vs-rules AUC and false-accept rate.
