# Teacher-output generation for distillation

This pipeline is stage 1 of the distillation project. The student is `wongwian-org/wongwian-micro-instruct`: a 272M
Thai model with a **2,048-token context**. The pipeline asks a teacher (Qwen3.5-9B) for **short chain-of-thought +
answer** pairs, samples **N per prompt**, grades them with a judge (Gemma 4 31B) or with rules, keeps the best, and
writes them **easy → hard** for a curriculum. Raw long thinking is never a training target: it doesn't fit the
student, and small models learn poorly from it. A **Thai set** is added next to the English one: new questions are
translated first, checked, and then distilled the same way (see [Thai sets](#thai-sets)). Training is out of scope.
Project context is in [CLAUDE.md](CLAUDE.md); every experiment and its results are in
[docs/experiments.md](docs/experiments.md).

```
distill/data.py       load → dedupe → drop multi-turn → gold answers / closed flag → seeded stratified sample
distill/server.py     vLLM endpoints, request bodies, retries, thread-pool runner, vllm serve / wait
distill/rate.py       question complexity 1-5 (DeReason prompt)
distill/generate.py   N samples per prompt, response parsing, resume
distill/compress.py   variant B: thinking trace → short rationale
distill/verify.py     rule checks for closed-ended prompts
distill/judge.py      rubric scores 1-9 for rationale + answer; validation against human preferences
distill/selection.py  best-of-N / distinct correct samples, curriculum order
distill/report.py     filter (heuristic rejections) and analyze (Markdown report)
distill/translate.py  Thai sets: translate new questions, check them (rules and a checker model), build the
                      prepared file
distill/inproc.py     experiments: sentence-level translators and metric checkers run in-process (transformers)
distill/compare.py    experiments: the translator and checker comparisons (grading sheets, reports, pools)
distill/cli.py        command line
config.yaml           every setting: models, servers, sampling, thresholds, datasets
prompts/              versioned templates: {name}_{version}.txt
scripts/              pipeline.sh (+ lib.sh), run_thai.sh, vm_check.sh; the experiments' translate_compare.sh,
                      translate_compare2.sh, second_checker.sh, checker_compare.sh
reports/              results: pilots, full runs, judge validation, the comparisons
```

## Pipeline
```
prepare  (CPU)  load, dedupe, gold answers
── teacher phase: Qwen3.5-9B ────────────────────────────────────────────────────────────
rate      complexity 1-5 per question
generate  N samples per prompt (2 at level 1, else 4)   B: --mode open --thinking on   |   A: --mode cot --thinking off
compress  B only: trace + answer → short numbered steps (4-10 by complexity)
── CPU ───────────────────────────────────────────────────────────────────────────────────
filter    heuristic checks, incl. rationale length and the student's context budget
── judge phase: Gemma 4 31B ──────────────────────────────────────────────────────────────
judge     rationale + answer, rubric score 1-9
── CPU ───────────────────────────────────────────────────────────────────────────────────
select    best samples, curriculum order, raw traces dropped
analyze   by set and complexity level: scores, pass rates, lengths, throughput, examples
```
The teacher (≈18 GB) and the judge (≈20 GB) can't share an A100 40GB, so each phase runs its own local `vllm serve`.
Variant B (thinking on, then `compress`) won the pilot and makes all the data; variant A (thinking off, steps inside
`<reasoning>` tags) stays for comparison (`VARIANTS="A B"`).

## Setup
```bash
# Laptop (Windows): code and offline tests. No GPU needed.
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest                                   # offline tests, no network, no vLLM

# GPU VM (Linux, over SSH): after git clone / pull
uv venv --python 3.12 && uv pip install -e ".[dev,vm]"     # extras: qe (similarity), mt (in-process translators)
bash scripts/vm_check.sh                                   # GPU, disk, network, vLLM version and flags
hf download Qwen/Qwen3.5-9B && hf download google/gemma-4-31B-it-qat-w4a16-ct   # ≈38 GB into $HF_HOME
uv run distill prepare
```
**Machine settings** go in `.vm_env` in the repo root (untracked). `distill` and the scripts load it, because
`~/.bashrc` is often skipped by tmux and non-interactive shells. On the current VM the root disk is nearly full, so the
caches live in the home on a shared filesystem (not `/dev/shm`, which is wiped when the user is fully logged out):
```bash
export HF_HOME=$HOME/hf
export VLLM_CACHE_ROOT=$HOME/vllm-cache
export HF_TOKEN=hf_...   # only for Hub backups: a fine-grained token with write access to the backup repo alone
```

## Running on the VM
Everything runs inside tmux. **Ctrl-C** stops cleanly: the server is shut down and its GPU memory released, and
re-running the same command resumes. Launch with `tee -ai` (a plain `tee` dies on Ctrl-C).
```bash
# English full run (done 2026-10-27: 12,432 final rows). The same command resumes; a larger LIMIT adds prompts.
tmux new -s full 'BY_SET=1 LIMIT=2500 LIMIT_CLOSED=500 bash scripts/pipeline.sh 2>&1 | tee -ai logs/full.log'
# LogiQA (closed only) runs on its own, so LIMIT_CLOSED=500 stays UltraFeedback's. Prepare it once first.
distill prepare --dataset logiqa
tmux new -s logiqa 'DATASETS=logiqa BY_SET=1 LIMIT_CLOSED=2500 bash scripts/pipeline.sh 2>&1 | tee -ai logs/full.log'

# A time-boxed GPU window: Ctrl-C 10 min before the end (the run stops its server and resumes next time).
# S = seconds until then; -X cancel first, because C-c only leaves copy mode when someone is scrolling the pane.
tmux new -d -s stopper "sleep $S; tmux send-keys -t full -X cancel 2>/dev/null; tmux send-keys -t full C-c"
```
Monitor with `tail -f logs/full.log logs/vllm_teacher.log` (tmux: `Ctrl-b d` detaches, `tmux attach -t full`
returns). Every GPU stage prints a progress line each minute (done/total, tok/s, ETA) and appends a summary to
`<out>.runs.jsonl`; `analyze` turns those into a throughput table.

`scripts/pipeline.sh` does the work:
1. Starts the teacher server; rates, generates and compresses; stops it.
2. Filters.
3. Starts the judge server; validates the judge (if `VALIDATE` > 0, once per output directory) and judges; stops it.
4. Selects, backs up to the Hub and reports. The report covers every set in `OUT`, also those from earlier sessions.

Settings come from environment variables (defaults in brackets): `LIMIT` [0 = all], `LIMIT_CLOSED` [LIMIT], `N`,
`VARIANTS` [B], `DATASETS`, `VALIDATE` [0], `JUDGE_ALL_CLOSED`, `BY_SET`, `REDO_LANGUAGE`, `OUT` [data/full],
`REPORT` [reports/full.md]. `BY_SET=1` runs each set through teacher, judge and select before the next: a few minutes
of model swaps per set, but a run cut short still leaves finished data for the sets it completed.

- **A dead server restarts itself.** If 16 requests in a row fail at the connection level (after their retries), the
  stage stops with exit code 75 instead of failing every queued request. `with_server` (`scripts/lib.sh`) then restarts
  the server and resumes the stage, at most twice.
- **The GPU can be shared.** `start_server` refuses only if our own processes hold the GPU; other users' processes are
  listed and never stopped. `distill serve` lowers `--gpu-memory-utilization` to the free memory minus
  `gpu_share.headroom_gib`; only the KV cache shrinks. If not even the role's `min_gpu_gib` fits, it waits up to
  `gpu_share.wait_s` and then fails.
- **Hub backup.** Generated data lives only on the VM, so with `hub.repo` set the pipeline runs `distill push $OUT`
  after every finished set. It uploads the run directory and the complexity ratings to that dataset repo, which must
  be private (it is created private, and a public one is refused). A failed upload only prints a warning.
  `hf download <repo> --repo-type dataset --local-dir .` restores the files at their paths, and every stage resumes
  from them. Each push also rewrites the dataset card (from `docs/dataset_card.md`) for every run in `hub.runs`: one
  subset per file stage (`final`, `judge`, `samples`, `rejected`, `ratings`) with one split per set, and separate
  subsets for closed and Thai sets (`final_closed`, `final_th`, `final_th_closed`, …), whose extra columns can't share
  a schema. The Thai prepared files are backed up too, since rebuilding them takes a GPU.

**Using the data.** The training rows are the `final` subsets. Each row's `input` is the prompt, and its target is
`rationale` (numbered steps separated by blank lines) followed by `answer`.
```python
from datasets import load_dataset   # after `hf auth login` with an account in the org
repo = "capstone-thai-slm/capstone-distill-data"
chat = load_dataset(repo, "final", split="no_robots")           # also helpsteer2, ultrafeedback
logiqa = load_dataset(repo, "final_closed", split="logiqa")     # also ultrafeedback (its closed set)
thai = load_dataset(repo, "final_th", split="no_robots")        # also helpsteer2, ultrafeedback
thai_logiqa = load_dataset(repo, "final_th_closed", split="logiqa")
```
The models are not in the repo: they are unchanged Hugging Face downloads in `$HF_HOME` on the VM. `distill serve
teacher` (or `judge`) runs one by hand behind an OpenAI-style API at `localhost:8000/v1`, during a GPU window.

Single stages, by hand on the VM:
```bash
uv run distill serve teacher                     # vllm serve with the flags in config.yaml (foreground)
uv run distill wait teacher                      # blocks until the server is up and serving the teacher
uv run distill rate --dataset no_robots --limit 100
uv run distill generate --dataset no_robots --limit 100 --mode open --thinking on --out data/x/no_robots_B_raw.jsonl
uv run distill compress data/x/no_robots_B_raw.jsonl --out data/x/no_robots_B.jsonl
uv run distill filter data/x/no_robots_B.jsonl --out data/x/no_robots_B.kept.jsonl
uv run distill serve judge                       # after stopping the teacher server
uv run distill judge data/x/no_robots_B.kept.jsonl          # --validate N, --all, --thinking on|off
uv run distill select data/x/no_robots_B.kept.jsonl --out data/x/no_robots_B.final.jsonl
uv run distill analyze data/x/*.kept.jsonl --out reports/x.md
#   other generate flags: --only closed|open  --n  --model  --concurrency  --seed  --max-tokens  --prompt-version
#                         --include-reference
```
Datasets: `no_robots`, `helpsteer2`, `ultrafeedback`, `logiqa`, and the Thai sets `<name>_th`. To add one, add an
entry in `config.yaml` and a loader in `data.LOADERS`. Every stage appends to its output and skips keys already there
(`(id, sample_idx)`, or `id` for ratings); requests that fail after all retries go to `<out>.errors.jsonl` and are
retried on the next run.

## Data preparation
- **Prompt as the only input.** The input is `prompt` (No Robots, HelpSteer2) or `instruction` (UltraFeedback).
  LogiQA 2.0 (`datatune/LogiQA2.0`) stores each row as a JSON string in `text`. Only its 12,567 English
  multiple-choice rows are used (the mirror also holds Chinese and NLI rows). The prompt is the premise, the question,
  options `A.`–`D.`, then "Answer with the letter of the correct option."
- **Adding a dataset later:** put it last in `config.yaml` and run `distill prepare --dataset NAME`. That builds only
  this dataset, deduped against the earlier ones, and leaves their files and samples untouched. `distill subsets NAME`
  prints which sets it has (`open`, `closed`), and `pipeline.sh` makes one of each.
- **Dedup.** The ID is `sha256(normalize(prompt))[:16]`, where normalize means NFKC, collapsed whitespace and casefold.
  Duplicates are removed within each dataset and across datasets, in config order.
- **Multi-turn rows are excluded:** No Robots "Chat" rows (their `prompt` is a system persona) and HelpSteer2 prompts
  with `<extra_id_1>` turn markers. Counts are in `data/prepared/stats.json`.
- **Closed-ended prompts** have short gold answers (≤4 words) from UltraFeedback's `correct_answers` in flan_v2_niv2,
  flan_v2_p3 or flan_v2_flan2021 (10,603 prompts). They are rule-checked. truthful_qa golds such as "I have no
  comment" aren't exact-matchable, so those prompts go to the judge with the gold as its reference.
  - A dataset with `closed: true` (LogiQA) is closed wherever it has golds: the letter, the letter with its option,
    and the option text.
  - If no sample of a closed prompt passes the rule check, its samples go to the judge, since a noisy gold may be
    right without matching exactly. With `exact_gold: true` (LogiQA) such a prompt is dropped instead.
- **Reference** (shown to the judge, used in hint mode or with `--include-reference`): No Robots uses the human answer,
  HelpSteer2 its best-rated response, UltraFeedback the completion with the highest `fine-grained_score`. It is used
  only when hint-eligible (`hint.min_ratings`).

## Generation
- **Sampling** follows the Qwen3.5-9B card's general-task settings, and every field is sent, because a local vLLM
  server fills unsent fields from the model's generation_config.json. Thinking on: temperature 1.0, top_p 0.95,
  top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0. Thinking off: 0.7, 0.8, 20, 0, 1.5, 1.0.
- **Seeds:** each sample's seed is derived from `(seed, id, sample_idx)`: reproducible per request, not bit-exact
  under different batching.
- **Modes:**
  - `open`: the plain assistant prompt. With thinking on, this is variant B, whose trace goes to `compress`.
  - `cot` (variant A): thinking off; numbered steps inside `<reasoning>` tags, then the answer. Capped at
    `cot.max_tokens` (4,096).
  - `hint` (experimental): the reference answer is a private note in the system prompt, never stored in `input`.
- **Closed-ended prompts** use `{mode}_closed_{version}` templates, which ask for a final line `Final answer: <answer>`.
- **Complexity sets the effort.** The rating picks the samples per prompt (`generate.n_by_complexity`: 2 for level 1,
  otherwise 4) and the step limit (`cot.max_steps_by_complexity`: 4, 5, 6, 8, 10 for levels 1–5, 6 when unrated,
  about 40 words a step). The filter only rejects runaway rationales (over 600 words); the student's budget is the real
  limit.
- **Thinking:** vLLM gets `chat_template_kwargs.enable_thinking` and returns the thinking in `reasoning` (before vLLM
  0.32: `reasoning_content`; `server.reasoning_text` reads both); its
  token count comes from `/tokenize`. With thinking off, a reply filed under reasoning is moved back to the answer, and
  inline `<think>…</think>` is split out.
- **compress** rewrites each finished trace into numbered steps that reach the same answer, with no new facts and no
  harmful details. Unfinished samples pass through with an empty rationale, so the filter records why they drop.
- **rate** scores each question 1–5 with the "Reasoning Complexity Rating Prompt" of DeReason (Hu et al. 2026,
  arXiv:2603.11193, Appendix A.1), at temperature 0, into `data/prepared/<dataset>.complexity.jsonl`. A rating without
  a level is asked again on the next run.

## Filter reasons
| reason | rule |
|---|---|
| `missing_answer` | the answer is empty |
| `malformed_think_tag` | an unclosed `<think>`, or a stray `<think>`/`</think>` in the answer |
| `finish_not_stop` | `finish_reason != "stop"` |
| `language_mismatch` | the answer or rationale is in another language than expected: Thai on Thai rows (`lang: th`), else lingua's reading of the prompt. Code blocks are stripped first; skipped for short texts and for prompts matching `language_skip_patterns` (e.g. translation requests) |
| `repetition` | `1 − unique/total` 8-gram ratio > 0.3 in the answer or the rationale (character n-grams for scripts without spaces). The raw thinking is never trained on, so it is not checked |
| `hint_leakage` | hint mode only: the reasoning, rationale or answer contains a phrase from `leakage_phrases` |
| `reasoning_too_long` | `reasoning_tokens > --max-reasoning-tokens` |
| `missing_rationale` | a cot or compressed sample has no rationale |
| `rationale_too_long` | the rationale has more than `cot.max_words` (600) words; Thai is split into words by PyThaiNLP |
| `over_student_budget` | input + rationale + answer in the student's tokenizer, plus 32 template tokens, exceed `student.max_total_tokens` (2,048; `--max-student-tokens 0` disables) |

The filter writes `<out>` (kept rows), `<out stem>.rejected.jsonl` (each rejected row with its reasons) and
`<out stem>.rejections.json` (counts).

## Judging, rules and selection
- **Judge:** Gemma 4 31B (`google/gemma-4-31B-it-qat-w4a16-ct`), temperature 0, thinking off, JSON `{critique, score}`.
  - The rubric (`prompts/judge_v2.txt`) grades both parts. Reasoning: valid, supports the answer, concise, no harmful
    details. Answer: every instruction met, correct, concise, the prompt's language; declines only clearly harmful
    requests. The gold or a hint-eligible dataset answer is shown as a possibly imperfect reference.
  - The rubric, question and reference come first and the graded response last, so the prefix cache computes the
    shared part once per prompt.
  - Scores run 1–9, so a score is one token; its top logprobs give an `expected_score` that breaks ties.
- **What gets judged:** every sample of an open prompt. For a closed prompt, only the samples whose final answer passes
  the rule (`verify.is_correct`: normalized match, or numeric match for number golds); if none pass, all of them (see
  Data preparation). `judge --all` judges every sample.
- **Validation:** `judge --validate N` judges both responses of HelpSteer2 *validation* prompts whose two responses got
  different human helpfulness ratings, and appends agreement, ties, thinking on/off, tokens per call and minutes to
  `reports/judge_validation.md`. `--thinking on|off` and `--prompt-version` override the config.
- **The report** adds best-of-k per variant (to choose N), judge vs rules on closed prompts (needs `--all`), and the
  share of scores whose logprobs moved the expected score.
- **select:** open prompts keep their best sample if it scores ≥ `select.min_score` (7); closed prompts keep up to
  `k_closed` (3) distinct rule-passing samples (rationale trigram Jaccard ≤ 0.7); ties go to the shorter sample. Output
  is in curriculum order (complexity 1 → 5, unrated last, shuffled within each level) or `--order random`. The raw
  `reasoning` is dropped; `<out stem>.select_stats.json` counts kept and dropped prompts and rows per level.

## Thai sets
About 6k Thai rows are added next to the English ones (about 1:2). The English rows are not translated. Instead,
**new questions are translated first and then distilled**, so Qwen writes the Thai rationale and answer itself.
- **Sets:** a Thai set is a dataset named `<source>_th` with a `translate:` block in `config.yaml` (`from`, `only`,
  `limit`): `no_robots_th`, `helpsteer2_th`, `ultrafeedback_th` (open) and `logiqa_th` (closed). UltraFeedback's closed
  set is left out (noisy golds, English-bound tasks); its share goes to LogiQA. `distill build-translated` writes the
  prepared file in the usual schema, so every later stage reads it like any other dataset.
- **New questions only:** the English run rated every question first, so the source's ratings file lists every id it
  used. `translate` leaves those out, then takes a seeded stratified sample. A missing ratings file is an error.
- **What is translated:** the prompt, and the reference answer when the judge would use it (`hint_ok`); the reference
  is never trained on. LogiQA keeps its option letters, and its gold option text is read back from the Thai options.
- **Checks, before any generation:**
  - `rule_flags`: `not_stop`, `not_thai` (Thai share below `translate.rules.min_thai_share`, 0.2), `refusal`,
    `length_ratio`, `numbers_changed`, `code_changed`, `urls_changed`, `options_missing`, `english_task_pattern`
    (`filter.language_skip_patterns`), `repetition`.
  - `check-translation`: the checker (`translate.checker`, Gemma) answers `english_only` (the task depends on English
    itself: grammar, rhyme, wordplay, letter counts, translating into another language), `fluency` and `meaning`
    (1–5), at temperature 0 with a JSON schema. `meaning` comes last, so its logprobs give `meaning_expected`.
  - `build-translated` drops a translation with any rule flag, `english_only`, no check, or `meaning_expected` below
    `translate.min_meaning` (4.0), and writes the prompts sorted by id. A configured gate (`translate.gates`) flags a
    row instead (`translation.gate_failed`), so the seeded sample stays the same; none is in use.
- **Later stages:** `rate` rates the English original; the filter expects Thai on Thai rows; `verify` accepts
  `คำตอบสุดท้าย:` as well as `Final answer:`. The teacher keeps its English prompts, which ask for the question's
  language; `compress_v4` (on Thai rows) says "Write every step in Thai", because Qwen thinks in English and v3 let
  ~590 of night 1's rationales come out in English. `compress --redo-wrong-language` (`REDO_LANGUAGE=1`) redoes them.

```bash
distill translate --dataset logiqa_th --dry-run            # pool left after excluding the English ids; no server
tmux new -s thai 'LIMIT=500 bash scripts/run_thai.sh 2>&1 | tee -ai logs/thai.log'    # night 1: prep + 500 per set
# night 2: every remaining open prompt, LogiQA up to 1,200, and night 1's English rationales redone
tmux new -s thai 'REDO_LANGUAGE=1 LIMIT=0 LIMIT_CLOSED=1200 bash scripts/run_thai.sh 2>&1 | tee -ai logs/thai.log'
```
- **Prep** runs once, for each set without a prepared file: translate, check, build, and a backup of `data/translate`.
  The prompt pool then stays fixed.
- **Distillation** runs `pipeline.sh` on `LIMIT` prompts per Thai set (variant B, `OUT=data/full_th`). The sampler is
  nested, so a larger `LIMIT` only adds prompts and `LIMIT=0` takes the rest. With a limit each model loads once per
  run (`BY_SET=0`); an unlimited run goes set by set.
- **Timing (night 1):** 4.2–4.6 s per prompt in an open set and 7.2 s in LogiQA (generate + compress + judge), plus
  ~10 min of model loads per run; prep took 86 min. LogiQA gives about 2.3 rows per prompt (up to 3 correct samples).

Single steps:
```bash
distill translate --dataset no_robots_th [--translator teacher --limit 13 --no-reference --out FILE]
distill check-translation data/translate/no_robots_th.jsonl      # the judge's server; --role for another checker
distill build-translated --dataset no_robots_th                  # -> data/prepared/no_robots_th.jsonl
```

**Translation record** (`data/translate/<set>.jsonl`): `id` (the English prompt's id), `dataset`, `source_dataset`,
`input_en`, `input` (Thai), `finish_reason`, `reference_en`, `reference` (Thai, or null), `strata`, `closed`, `gold`
(English), `hint_ok`, `translator` (`model`, `role`, `prompt_version`, `prompt_hash`, `sampling`), `tokens`,
`rule_flags`. **Check** (`<stem>.check.jsonl`): `id`, `english_only`, `fluency`, `meaning`, `meaning_expected`,
`critique`, `judge_model`, `check_prompt_hash`; another checker writes `<stem>.check_<role>.jsonl`. The Thai prepared
row adds `lang`, `input_en` and `translation` to the usual fields, and final rows carry them on.

## Experiments
The comparisons behind the Thai decisions stay runnable, for the project report. Results and decisions:
[docs/experiments.md](docs/experiments.md).
- **Translator comparisons:** `scripts/translate_compare.sh` (Qwen, Gemma and Typhoon Translate; G1) and
  `scripts/translate_compare2.sh` (Seed-X, MADLAD and NLLB as `hf:` roles in `inproc.py`, which split prompts into
  sentence-sized pieces and keep code blocks, option labels and list markers). `distill compare-sheet` writes a blind
  grading sheet with its key; `distill compare-report` reads the graded sheet (partly graded is fine; untranslated
  copies count as meaning 1) and reports each translator and how well each automatic signal agrees with the grades.
- **Checker comparisons:** `scripts/second_checker.sh` (Qwen3-32B) and `scripts/checker_compare.sh` (MetricX-24 and
  CometKiwi as in-process `metric:` roles, Typhoon 2.5 behind vLLM). A metric check stores `quality` (higher is
  better), `raw`, `pieces` and `truncated`; a pair longer than the model's input is read in aligned paragraphs or lines,
  and the worst piece decides. `distill checker-sample`, `checker-pick` and `checker-report` build the disagreement pool
  and apply the decision rule. CometKiwi is gated and needs its own venv:
  ```bash
  uv venv .venv-comet --python 3.12 && uv pip install --python .venv-comet -e . "unbabel-comet>=2.2" "setuptools<81"
  uv pip install --python .venv-comet "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
  ```
- `distill similarity` (BGE-M3 cosine, extra `qe`) was ruled out by G1 and is manual only.

## Output schemas (one JSON object per line)
**Generated sample** (`generate`, then `compress` adds `rationale`, `cot_method`, `compress_model`, `compress_prompt_hash`):

| field | description |
|---|---|
| `id`, `sample_idx` | hash of the normalized prompt (stable across runs); sample number 0..N-1 |
| `source_dataset` | `no_robots` / `helpsteer2` / `ultrafeedback` / `logiqa`, or a Thai set |
| `input` | the user prompt sent to the teacher (never contains the hint) |
| `closed`, `gold` | closed-ended flag; gold answers (closed prompts only) |
| `reasoning` | raw teacher thinking (empty when thinking is off); analysis only |
| `rationale` | short numbered reasoning, one step per paragraph separated by a blank line (`verify.format_steps`); the training target together with `answer` |
| `answer` | teacher final answer, with think and reasoning tags removed |
| `finish_reason` | `stop`, `length`, … |
| `prompt_tokens`, `reasoning_tokens`, `answer_tokens` | `answer_tokens = completion_tokens − reasoning_tokens` |
| `think_unclosed` | true when the answer started a `<think>` that never closed |
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
| `teacher` | how Qwen made it: `model`, `mode`, `thinking`, `cot_method`, prompt versions/hashes, `sampling`, `seed`, `timestamp`. `max_tokens` is the **teacher's** generation cap. `thinking_tokens` is the raw thinking (variant B), not in this file. `reply_tokens` is the visible reply in Qwen's tokens |
| `lang`, `input_en`, `translation` | Thai rows only: `th`, the English original of `input`, and the translation's `model`, `checker`, `meaning` (expected, 1–5), `fluency` and `similarity` (null: not used) |

Raw thinking and per-call bookkeeping stay in the generation and kept files.

## Licensing
See [docs/LICENSES.md](docs/LICENSES.md). **Everything derived from No Robots is non-commercial (CC BY-NC 4.0)**, and
LogiQA 2.0 is CC BY-NC-SA 4.0. Generated data is not committed; it lives on the VM and in the private Hub repo.
