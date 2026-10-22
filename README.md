# Teacher-output generation for distillation

This pipeline is stage 1 of the distillation project. The student is `wongwian-org/wongwian-micro-instruct`: a 272M
Thai model with a **2,048-token context**. The pipeline asks a teacher (Qwen3.5-9B) for **short chain-of-thought +
answer** pairs, samples **N per prompt**, grades them with a judge (Gemma 4 31B) or with rules, keeps the best, and
writes them **easy → hard** for a curriculum. Raw long thinking is never a training target: it doesn't fit the
student, and small models learn poorly from it. Training and translation into Thai are out of scope.
Project context and design decisions are in [CLAUDE.md](CLAUDE.md).

```
distill/data.py       load → dedupe → drop multi-turn → gold answers / closed flag → seeded stratified sample
distill/server.py     endpoints (vllm | openrouter), request bodies, retries, thread-pool runner, vllm serve / wait
distill/rate.py       question complexity 1-5 (DeReason prompt)
distill/generate.py   N samples per prompt, response parsing, resume, budget
distill/compress.py   variant B: thinking trace → short rationale
distill/verify.py     rule checks for closed-ended prompts
distill/judge.py      rubric scores 1-9 for rationale + answer; validation against human preferences
distill/selection.py  best-of-N / distinct correct samples, curriculum order
distill/report.py     filter (heuristic rejections) and analyze (Markdown report)
distill/cli.py        command line
config.yaml           every setting: models, servers, sampling, thresholds, datasets
prompts/              versioned templates: {name}_{version}.txt
scripts/              pipeline.sh (+ lib.sh), run_pilot.sh / run_full.sh / bench.sh / vm_check.sh / collect.sh (GPU VM),
                      smoke_openrouter.sh (laptop)
```

## Pipeline
```
prepare  (CPU)  load, dedupe, gold answers
── teacher phase: Qwen3.5-9B ────────────────────────────────────────────────────────────
rate      complexity 1-5 per question
generate  N=8 samples per prompt   A: --mode cot --thinking off   |   B: --mode open --thinking on
compress  B only: trace + answer → ≤5-step rationale
── CPU ───────────────────────────────────────────────────────────────────────────────────
filter    heuristic checks, incl. rationale length and the student's context budget
── judge phase: Gemma 4 31B ──────────────────────────────────────────────────────────────
judge     rationale + answer, rubric score 1-9
── CPU ───────────────────────────────────────────────────────────────────────────────────
select    best samples, curriculum order, raw traces dropped
analyze   A vs B, by complexity level: scores, pass rates, lengths, throughput, examples
```
The teacher (≈18 GB) and the judge (≈20 GB) can't share an A100 40GB, so each phase runs its own `vllm serve`.
Every GPU stage takes `--backend vllm|openrouter`. vllm is a local server on the GPU VM; openrouter is for laptop
smoke tests.

## Setup
```bash
# Laptop (Windows): code and offline tests. No GPU needed.
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest                                   # offline tests, no network, no vLLM
echo OPENROUTER_API_KEY=sk-or-... > .env        # only for the OpenRouter smoke test; never commit

# GPU VM (Linux, over SSH): after git clone / pull
uv venv --python 3.12 && uv pip install -e ".[dev,vm]"
hf download Qwen/Qwen3.5-9B && hf download google/gemma-4-31B-it-qat-w4a16-ct   # ≈38 GB; set HF_HOME to a big disk
uv run distill prepare
```
On a company network, check first that the VM can reach huggingface.co (set a proxy or `HF_ENDPOINT` if needed).

## VM sessions
The GPU VM is used in sessions of a few hours. Everything runs inside tmux. **Ctrl-C** stops cleanly: the server is
shut down and its GPU memory released. Re-running the same command resumes. After each session,
`scripts/collect.sh` copies the small results (reports, run summaries, rejection and selection counts, error samples,
server-log highlights, complexity counts) into `reports/<name>/` for one commit and push.

```bash
# Session 1: setup and smoke test (~1-1.5 h, mostly downloads)
git clone <repo> && cd Distillation
bash scripts/vm_check.sh | tee reports/vm_check.txt          # GPU, disk, network, tools
uv venv --python 3.12 && uv pip install -e ".[dev,vm]"
bash scripts/vm_check.sh | tee reports/vm_check.txt          # again: vLLM version and the flags we use
uv run pytest
hf download Qwen/Qwen3.5-9B && hf download google/gemma-4-31B-it-qat-w4a16-ct   # ≈38 GB
uv run distill prepare
tmux new -s smoke 'LIMIT=5 VALIDATE=20 OUT=data/smoke_vm REPORT=reports/smoke_vm.md bash scripts/run_pilot.sh 2>&1 | tee -ai logs/smoke.log'
bash scripts/collect.sh smoke_vm data/smoke_vm && git add reports && git commit -m "Results: smoke_vm" && git push

# Session 2: pilot, both variants, 100 prompts per set, 8 samples (~2-2.5 h; can span sessions)
tmux new -s pilot 'bash scripts/run_pilot.sh 2>&1 | tee -ai logs/pilot.log'
bash scripts/collect.sh pilot2 data/pilot2 --data           # --data also packs kept + judge files for scp

# Session 3: experiments (~1.5 h)
tmux new -s bench 'bash scripts/bench.sh 2>&1 | tee -ai logs/bench.log'   # throughput settings → reports/bench.md
tmux new -s judge 'bash -c "source scripts/lib.sh && start_server judge && distill judge --validate 300 --thinking on"'
bash scripts/collect.sh bench data/bench

# Full run with the pilot's winning variant
tmux new -s full 'VARIANTS=A bash scripts/run_full.sh 2>&1 | tee -ai logs/full.log'
```
Monitor with `tail -f logs/pilot.log logs/vllm_teacher.log` (tmux: `Ctrl-b d` detaches, `tmux attach -t pilot`
returns).

**Machine settings** go in `.vm_env` in the repo root. It is untracked, and the scripts source it, because
`~/.bashrc` is often skipped by tmux and non-interactive shells. Source it yourself for commands typed by hand. On the
current VM the root disk is nearly full, so the model, dataset and vLLM caches live on the shared filesystem:
```bash
export HF_HOME=/lambda/nfs/Fusion-SFT/aicookie/hf
export VLLM_CACHE_ROOT=/lambda/nfs/Fusion-SFT/aicookie/vllm-cache
```

`scripts/pipeline.sh` does the work:
1. Starts the teacher server; rates, generates and compresses; stops it.
2. Filters.
3. Starts the judge server; validates the judge (once per output directory) and judges; stops it.
4. Selects and reports.

Settings come from environment variables: `LIMIT`, `N`, `VARIANTS`, `DATASETS`, `VALIDATE`, `JUDGE_ALL_CLOSED`, `OUT`,
`REPORT`, `BACKEND`.
- `BY_SET=1` runs each set through teacher, judge and select before starting the next. That costs a few minutes of
  model swaps per set, but a time-boxed run that gets cut off still leaves finished training data for the sets it
  completed.
- `run_pilot.sh` sets `JUDGE_ALL_CLOSED=1`: rule-failing closed samples are judged too, so the report can show how well
  the judge separates right from wrong answers.
- `scripts/bench.sh` reruns the pilot's no_robots generation under other server settings: concurrency 128,
  `--async-scheduling`, and MTP speculative decoding (passed through `DISTILL_SERVE_ARGS`). It tabulates tok/s against
  the pilot.
- A laptop smoke test through OpenRouter is also available: `bash scripts/smoke_openrouter.sh` (Git Bash; costs cents).

Single stages, for example by hand on the VM:
```bash
uv run distill serve teacher                     # vllm serve with the flags in config.yaml (foreground)
uv run distill wait teacher                      # blocks until the server is up and serving the teacher
uv run distill rate --dataset no_robots --limit 100
uv run distill generate --dataset no_robots --limit 100 --n 8 --mode cot --thinking off --out data/x/no_robots_A.jsonl
uv run distill generate --dataset no_robots --limit 100 --n 8 --mode open --thinking on --out data/x/no_robots_B_raw.jsonl
uv run distill compress data/x/no_robots_B_raw.jsonl --out data/x/no_robots_B.jsonl
uv run distill filter data/x/no_robots_A.jsonl --out data/x/no_robots_A.kept.jsonl
uv run distill serve judge                       # after stopping the teacher server
uv run distill judge --validate 300 data/x/no_robots_A.kept.jsonl   # --all: every sample; --thinking on|off
uv run distill select data/x/no_robots_A.kept.jsonl --out data/x/no_robots_A.final.jsonl
uv run distill analyze data/x/*.kept.jsonl --out reports/x.md
#   other generate flags: --only closed|open  --model  --concurrency  --seed  --max-tokens  --prompt-version
#                         --max-cost (OpenRouter)  --include-reference  --backend openrouter
```
Datasets: `no_robots`, `helpsteer2`, `ultrafeedback`. To add one, add an entry in `config.yaml` and a loader in
`data.LOADERS`.

**Resume:** every stage appends to its output and skips keys already there: `(id, sample_idx)`, or `id` for ratings.
Requests that fail after all retries go to `<out>.errors.jsonl` and are retried on the next run.
**Monitoring:** each GPU stage prints a progress line every minute (done/total, tok/s, ETA) and appends a summary to
`<out>.runs.jsonl`. `analyze` turns those summaries into a throughput table.
**Budget:** `--max-cost` (OpenRouter) stops new requests once the spent USD reaches the limit.

## Data preparation
- **Prompt as the only input.** The input is `prompt` (No Robots, HelpSteer2) or `instruction` (UltraFeedback).
- **Dedup.** The ID is `sha256(normalize(prompt))[:16]`, where normalize means NFKC, collapsed whitespace and casefold.
  Duplicates are removed within each dataset and across datasets, in config order.
- **Multi-turn rows are excluded:** No Robots "Chat" rows (their `prompt` is a system persona) and HelpSteer2 prompts
  with `<extra_id_1>` turn markers. Counts are in `data/prepared/stats.json`.
- **Gold answers.** UltraFeedback's `correct_answers`, without the literal "None" that other sources carry.
- **Closed-ended prompts** have short gold answers (≤4 words) from flan_v2_niv2, flan_v2_p3 or flan_v2_flan2021
  (10,603 prompts). They are rule-checked. truthful_qa golds such as "I have no comment" aren't exact-matchable, so
  those prompts go to the judge with the gold as its reference.
- **Reference** (used by the judge, in hint mode, or with `--include-reference`): No Robots uses the human answer,
  HelpSteer2 its best-rated response, UltraFeedback the completion with the highest `fine-grained_score`.
- **Hint eligibility** is set by `hint.min_ratings`: all No Robots rows; HelpSteer2 helpfulness, correctness and
  coherence all ≥ 4; UltraFeedback fine-grained score 5.0 and overall score ≥ 8. The judge only uses a dataset
  response as its reference when it is hint-eligible.

## Generation
- **Sampling** follows the Qwen3.5-9B model card's general-task settings, and every field is sent, because a local
  vLLM server fills unsent fields from the model's generation_config.json.
  - Thinking on: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0.
  - Thinking off: 0.7, 0.8, 20, 0, 1.5, 1.0.
- **Seeds:** each sample's seed is derived from `(seed, id, sample_idx)`. Samples are reproducible per request, but
  not bit-exact under different batching.
- **Modes:**
  - `cot` (variant A): thinking off. The model reasons in at most 5 numbered steps inside `<reasoning>` tags, then
    answers. The steps are stored as `rationale`. Output is capped at `cot.max_tokens` (4,096).
  - `open`: the plain assistant prompt. With thinking on, this is variant B, whose trace goes to `compress`.
  - `hint` (experimental): the reference answer is a private note in the system prompt and is never stored in
    `input`. The filter's `hint_leakage` check catches phrases like "provided answer".
- **Closed-ended prompts** use `{mode}_closed_{version}` templates, which ask for a final line `Final answer: <answer>`.
- **Complexity sets the effort.** The rating made before generation picks the samples per prompt
  (`generate.n_by_complexity`: 2 for level 1, otherwise `generate.n` = 4; `--n` fixes it for all). It also picks
  the rationale's step limit (`cot.max_steps_by_complexity`: 4, 5, 6, 8, 10 steps for levels 1–5, about 40 words a
  step), filled into `{max_steps}` / `{max_words}` in the cot and compress prompts. Unrated prompts get 6 steps.
  The filter only rejects runaway rationales (over 600 words); the student's 2,048-token budget is the real limit.
- **Thinking:** vLLM gets `chat_template_kwargs.enable_thinking`; OpenRouter gets `reasoning.enabled`.
  - vLLM returns reasoning in `reasoning_content`, and its token count comes from the server's `/tokenize` endpoint.
  - With thinking off, a reply filed under reasoning is moved back to the answer.
  - Inline `<think>…</think>` is split out.
- **compress** (variant B) rewrites each finished trace into at most 5 numbered steps that reach the same final
  answer, with no new facts and no harmful details. Unfinished samples are passed through with an empty rationale, so
  the filter records why they were dropped.
- **rate** scores each question 1–5 with the "Reasoning Complexity Rating Prompt" of DeReason (Hu et al. 2026,
  arXiv:2603.11193, Appendix A.1), at temperature 0. Results go to `data/prepared/<dataset>.complexity.jsonl`.

## Filter reasons
| reason | rule |
|---|---|
| `missing_answer` | the answer is empty |
| `malformed_think_tag` | an unclosed `<think>`, or a stray `<think>`/`</think>` in the answer |
| `finish_not_stop` | `finish_reason != "stop"` |
| `language_mismatch` | lingua detects different languages for the prompt and the answer. Code blocks are stripped first. The check is skipped for short texts and for prompts that match `language_skip_patterns` (e.g. translation requests) |
| `repetition` | `1 − unique/total` 8-gram ratio > 0.3 in the answer or rationale, or in the raw reasoning of rows without a rationale (character n-grams for scripts without spaces). A compressed sample's raw trace is not trained on, so it is not checked |
| `hint_leakage` | hint mode only: the reasoning, rationale or answer contains a phrase from `leakage_phrases` |
| `reasoning_too_long` | `reasoning_tokens > --max-reasoning-tokens` |
| `missing_rationale` | a cot or compressed sample has no rationale |
| `rationale_too_long` | the rationale has more than `cot.max_words` (150) words |
| `over_student_budget` | input + rationale + answer, in the student's tokenizer plus 32 template tokens, exceed `student.max_total_tokens` (2,048; `--max-student-tokens 0` disables). English is a proxy: re-check after translation |

The filter writes three files: `<out>` (kept rows), `<out stem>.rejected.jsonl` (each rejected row with all its
reasons) and `<out stem>.rejections.json` (counts).

## Judging, rules and selection
- **Judge:** Gemma 4 31B (`google/gemma-4-31B-it-qat-w4a16-ct` on vLLM; `google/gemma-4-31b-it` on OpenRouter),
  temperature 0, thinking off, JSON output `{critique, score}`.
  - The rubric (`prompts/judge_v1.txt`) grades both parts. Reasoning: valid, supports the answer, concise, no harmful
    details. Answer: every instruction met, correct, concise, the prompt's language; declines only clearly harmful
    requests.
  - The gold or a hint-eligible dataset answer is shown as a possibly imperfect reference.
  - The rubric, question and reference come first and the graded response last, so the server's prefix cache computes
    the shared part once per prompt.
  - Scores run 1–9, so a score is one token. On vLLM, its top logprobs give an `expected_score` that breaks ties.
- **What gets judged:** every sample of an open prompt. For a closed prompt, only the samples whose final answer
  passes the rule (`verify.is_correct`: normalized match, or numeric match for number golds). If none pass, all of
  them go to the judge with the gold as reference, because some golds are right but not exactly matchable.
  `judge --all` judges every sample, including rule-failing ones.
- **Validation:** `judge --validate N` judges both responses of HelpSteer2 *validation* prompts whose two responses
  got different human helpfulness ratings. It appends a row to `reports/judge_validation.md`: agreement, ties, thinking
  on/off, completion tokens per call, minutes. `--thinking on|off` overrides `judge.thinking` (thinking uses
  `judge.max_tokens_thinking`). `--prompt-version` picks the rubric. v2, the default, is stricter: v1 gave almost
  every Qwen sample a 9 in the smoke run. The pilot validates both (`VALIDATE_PROMPTS="v1 v2"`).
- **Report sections that guide the full run:**
  - **Best-of-k** per variant: what the first k samples of each prompt would give (mean best score, yield, pass@k), to
    choose N.
  - **Judge vs rules** on closed prompts (needs `--all`): mean score of correct vs wrong samples, AUC, and the
    false-accept rate (wrong samples scoring ≥ min_score).
  - **Logprob detail**: the share of judge scores whose expected score differs from the integer, which shows that
    logprobs arrive.
- **select:**
  - Open prompts keep their best sample if it scores ≥ `select.min_score` (7).
  - Closed prompts keep up to `k_closed` (3) distinct rule-passing samples (rationale word-trigram Jaccard ≤ 0.7).
  - Ties go to the shorter sample.
  - Output is written in curriculum order: complexity 1 → 5, unrated last, shuffled within each level; or
    `--order random`.
  - The raw `reasoning` field is dropped. `<out stem>.select_stats.json` counts kept and dropped prompts and rows per
    complexity level.

## Output schemas (one JSON object per line)
**Generated sample** (`generate`, then `compress` adds `rationale`, `cot_method`, `compress_model`, `compress_prompt_hash`):

| field | description |
|---|---|
| `id`, `sample_idx` | hash of the normalized prompt (stable across runs); sample number 0..N-1 |
| `source_dataset` | `no_robots` / `helpsteer2` / `ultrafeedback` |
| `input` | the user prompt sent to the teacher (never contains the hint) |
| `closed`, `gold` | closed-ended flag; gold answers (closed prompts only) |
| `reasoning` | raw teacher thinking (empty when thinking is off); analysis only |
| `rationale` | short numbered reasoning, one step per paragraph with steps separated by a blank line (`\n\n`; `verify.format_steps`), the training target together with `answer` |
| `answer` | teacher final answer, with think and reasoning tags removed |
| `finish_reason` | `stop`, `length`, … |
| `prompt_tokens`, `reasoning_tokens`, `answer_tokens` | `answer_tokens = completion_tokens − reasoning_tokens` |
| `think_unclosed` | true when the answer started a `<think>` that never closed |
| `provider`, `cost` | `local-vllm` and 0 locally; the upstream provider and USD cost on OpenRouter |
| `cot_method` | `prompted` (A), `compressed` (B) or null |
| `model`, `mode`, `thinking` | teacher ID, `open`/`cot`/`hint`, bool |
| `prompt_version`, `prompt_hash` | template version and first 16 hex characters of its sha256 |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, `seed`, `max_tokens` | sampling settings actually sent |
| `timestamp` | UTC ISO-8601 |
| `reference_response` | only with `--include-reference`; for spot-checking, never for training |

**Judge score** (`<kept stem>.judge.jsonl`): `id`, `sample_idx`, `score` (1–9), `expected_score`, `critique`,
`reference_used`, `judge_model`, `judge_prompt_hash`.
**Complexity rating** (`data/prepared/<dataset>.complexity.jsonl`): `id`, `complexity` (1–5 or null), `analysis`,
`model`, `prompt_hash`.
**Final row** (`select`, `*.final.jsonl`): the training data.

| field | meaning |
|---|---|
| `id`, `sample_idx`, `source_dataset` | which prompt and which of its samples |
| `input`, `rationale`, `answer` | **the training sample**: prompt, then reasoning steps separated by a blank line, then the answer |
| `closed`, `gold` | whether the prompt has a known short answer, and that answer (closed only) |
| `complexity`, `max_steps`, `n_steps` | DeReason rating 1–5 (curriculum order), the step limit it gave, and the steps actually written |
| `student_tokens` | `input`, `rationale`, `answer` and `total` in the **student's** tokenizer; total adds 32 chat-template tokens and must stay ≤ 2,048 |
| `selection` | `method` (rule+judge or judge), the judge's `score` / `expected_score`, `n_candidates`, `n_passing` |
| `teacher` | how Qwen made it: `model`, `mode`, `thinking`, `cot_method`, prompt versions/hashes, `sampling`, `seed`, `timestamp`. `max_tokens` is the **teacher's** generation cap, not the student's. `thinking_tokens` is the raw thinking (variant B), which is not in this file. `reply_tokens` is the visible reply in Qwen's tokens (for A it includes the rationale) |

Raw thinking and per-call bookkeeping (`finish_reason`, `cost`, …) stay in the generation and kept files.

## Licensing
See [docs/LICENSES.md](docs/LICENSES.md). **Everything derived from No Robots is non-commercial (CC BY-NC 4.0).**
`samples/` holds a small pilot sample, and generated data is otherwise not committed.
