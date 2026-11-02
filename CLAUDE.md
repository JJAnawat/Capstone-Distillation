# CLAUDE.md

Context for future sessions: the project, how the user works, the decisions and the traps. Commands, schemas and stage
details are in README.md; every experiment with its numbers is in `docs/experiments.md`.

## Context rule
**Don't read anything under `data/` (laptop or VM) into context**, `data/vm/` included. Those files are large JSONL
and .xlsx and would flood the session. When a number is needed, run a short script that prints counts or a few fields,
and only when the task needs it. The same goes for `logs/` on the VM: grep or tail, never cat.

## Project
Capstone: turn a tiny Thai model into a small **reasoning model** by distilling short chain-of-thought from a teacher.
- **Student:** `wongwian-org/wongwian-micro-instruct` (MIT), 272M Llama trained from scratch on Thai-dominant data.
  **2,048-token context** (prompt + response); it loops easily (card: 150–300 new tokens, temperature 0.6–0.8,
  repetition_penalty 1.2–1.4). Plain-text chat template `system:` / `ผู้ใช้:` / `ผู้ช่วย:`, no thinking tags.
- **Goal:** better non-math logical reasoning, instruction following and safety (refuse harmful requests without
  over-refusing), without changing the architecture or hurting Thai fluency. Math items are still distilled (user's
  call).
- **This repo is stage 1, teacher data generation.** Its output is data, not a model; training is a separate stage.
  The English set is done (12,432 rows); a Thai set (~6k rows) is half done.

## Pipeline
prepare → rate → generate → compress (variant B) → filter → judge → select → analyze. Thai sets first run translate →
check-translation → build-translated, then the same stages.
- **Teacher:** Qwen3.5-9B (bf16). **Judge, translator and translation checker:** Gemma 4 31B QAT W4A16. Both run
  behind `vllm serve` and can't share the A100 40GB, so teacher and judge stages are separate GPU phases.
- `distill/translate.py` is the Thai pipeline path; `compare.py` and `inproc.py` hold the finished experiments, kept
  for the project report.

## How the user works
- **CPU phase, then GPU phase:** finish and report the CPU work (code, offline tests, VM setup, dry checks) before any
  GPU step. GPU time comes in windows the user announces.
- **No smoke tests:** no trial runs; offline tests and dry checks cover the CPU phase.
- **The user approves every command**, commits and SSH included. Claude commits and pushes.
- **Small pilots, graded by hand:** the user grades blind .xlsx sheets (meaning and fluency 1–5), often partly.
- **Delete unused models** from `~/hf` once a decision is made; keep the code and results the final report needs.
- When asking the user to choose, show a concrete example per option.

## Where things run
- **Laptop (Windows 11):** coding and offline tests (`uv run pytest`); no GPU. Shells: PowerShell and Git Bash.
- **GPU VM (A100 40GB, Linux):** Claude drives it with `ssh -o BatchMode=yes aicookie@193.122.152.236 '…'`.
  - **The repo is `/lambda/nfs/Fusion-SFT/aicookie/Capstone-Distillation`** (`~/Capstone-Distillation`). Run `pwd`
    first. `/home/aicookie/Capstone-Distillation` is an old clone the user keeps; never work there.
  - Code gets there by commit, push, then `git pull` on the VM. Over SSH use `.venv/bin/...`.
  - Results come back by SSH (grep/tail) or scp into `data/vm/<name>/` on the laptop (gitignored).
  - `.vm_env` (untracked) sets `HF_HOME=$HOME/hf` and `VLLM_CACHE_ROOT`; `distill` loads it by itself.

## Key decisions (details and numbers in docs/experiments.md)
- **Targets are short CoT + answer, never Qwen's raw thinking** (raw traces don't fit 2,048 tokens).
- **Variant B** (thinking on, then `compress`) won the pilot; variant A stays in the code for comparison.
- **Rationale format (user's requirement):** numbered steps, one per paragraph, separated by a blank line;
  `verify.format_steps` enforces it.
- **Samples and steps follow complexity** (DeReason rating 1–5): N=2 at level 1, else 4; step limit 4/5/6/8/10,
  about 40 words a step. `select` writes the data from easy to hard.
- **Selection:** open prompts keep their best sample (judge ≥ 7 of 9); closed prompts keep up to 3 distinct correct
  samples (SCoTD). LogiQA is `exact_gold`: no correct letter means dropped, not judged.
- **Sampling:** the Qwen3.5 card's "general" profiles with all six fields sent; judge and rater at temperature 0.
- **Thai set:** new questions only (ids in the English ratings files are excluded), about 1:2 Thai:English, half of
  each English set; UltraFeedback closed skipped, its share to LogiQA (capped at 1,200 prompts).
  - Translate the question, then distill: Qwen writes the Thai rationale itself. Teacher prompts stay English;
    `compress_v4` asks for Thai steps on Thai rows.
  - Gemma translates and checks (meaning ≥ 4.0, English-only flag) plus rule flags. Two translator comparisons and
    two checker comparisons found nothing better and no self-preference.
  - The reference answer is translated only for the judge and the golds.
- **Hub backup:** `distill push` copies runs and ratings to the private dataset repo
  `capstone-thai-slm/capstone-distill-data`. It stays private until a deliberate release. The token lives only in
  `.vm_env`; the user pastes tokens into the chat, so suggest putting future ones on the VM directly.

## Conventions
- `config.yaml` is the single source of truth; CLI flags override it; `DISTILL_CONFIG=path` swaps the file.
- Prompts are `prompts/{name}_{version}.txt` with `### system` / `### user`; closed prompts use
  `{mode}_closed_{version}`. Templates go through `str.format` (double the braces), except the judge's system prompt.
- Every stage appends and resumes by key (`(id, sample_idx)`, or `id`); failures go to `<out>.errors.jsonl` and are
  retried next run; each run appends a summary to `<out>.runs.jsonl`.
- Tests are offline (httpx.MockTransport) and never import vLLM. Shared fixtures in `tests/conftest.py`.
- Match the existing style: compact functions, module docstrings, few comments.

## Gotchas
- **vLLM:** a local server fills unsent sampling fields from generation_config.json, so send all six. vLLM 0.32 sends
  the thinking as `reasoning` (`reasoning_content` is the deprecated old name; `server.reasoning_text` reads both).
  Dropping `reasoning` in the cleanup cost night 2 its first 2,174 samples. With thinking off the parser can file the
  whole reply there (`parse_response` handles it). Usage has no reasoning-token count, so `generate` asks `/tokenize`. Gemma 4 emits an empty thought block even
  with thinking off (`judge.strip_thought`).
- **httpx** allows 100 connections by default; `server.make_client` raises it to the concurrency.
- **Ctrl-C in a tmux pane** hits vLLM too and can deadlock on open connections: `run_pool` drops queued work and
  `cli.main` exits with `os._exit(130)`. Launch with `tee -ai` (a plain `tee` dies). In copy mode, `tmux send-keys C-c`
  only leaves copy mode, so stoppers send `-X cancel` first. Timed stoppers live untracked in `logs/stopper*.sh`.
- **A dead server restarts itself:** after 16 connection failures in a row `distill` exits 75 and `with_server`
  restarts the server, twice at most.
- **The GPU is shared** despite "exclusive" windows (another user's ~2 GB app). `distill serve` shrinks the KV cache to
  fit (`gpu_share`, `min_gpu_gib`) and waits up to 15 min.
- **No MTP speculative decoding:** it ran out of memory at full load on night 2.
- **Don't `git pull` on the VM while a script runs:** bash reads scripts as it executes them.
- **VM disk:** the root disk belongs to others and is nearly full; models, datasets and caches live in the home on
  the shared mount (admin-approved). The HF cache keeps files in a shared `~/hf/hub/blobs/<prefix>/<sha256>`, so
  deleting a model's folder frees nothing: delete only the blobs no kept model links to.
- **/dev/shm is wiped** whenever the user is fully logged out; keep nothing there.
- **Windows:** the console is cp1252 (`PYTHONIOENCODING=utf-8` for Thai output); `.sh` files must keep LF endings; Git
  Bash heredocs that rewrite files mangle escape sequences, so use the Edit tool for those.
- `.venv-comet` (CometKiwi) needs its own numpy 1 / transformers 4, `setuptools<81` and torch pinned to the cu130 build
  (driver 580 supports CUDA 13.0). It was deleted after the checker comparison; README has the recipe.

## Licensing
No Robots is CC BY-NC 4.0 and LogiQA 2.0 CC BY-NC-SA 4.0: derived data is non-commercial, so keep `source_dataset` on
every row. Open TODOs (company hardware, Qwen terms) are in docs/LICENSES.md.

## Status (2026-11-02)
- **Done:** the English full run (12,432 final rows), Thai night 1 (2,336 rows), both translator comparisons and both
  checker comparisons (Gemma stays translator and checker, no second gate).
- **Thai night 2 (2026-11-01) got through everything but the judging** (details in docs/experiments.md §7).
  - The teacher phase is complete for all four Thai sets. Filter is done. No Robots is fully judged.
  - The stopper cut the judge phase at 17:50 UTC. Left to judge: ~2,400 HelpSteer2, ~2,000 UltraFeedback and ~2,500
    LogiQA samples. No final rows from night 2 yet; `data/full_th` was pushed to the Hub on 2026-11-02.
- **Next, in the user's next GPU window (~45 min): resume with the same command.** Rate, generate and compress find
  nothing left; the teacher server still loads for ~6 min. Judging, select, push and the report follow.
  `tmux new -s thai 'REDO_LANGUAGE=1 LIMIT=0 LIMIT_CLOSED=1200 bash scripts/run_thai.sh 2>&1 | tee -ai logs/thai.log'`
  - Arm a timed stopper if the window has an end time: `logs/stopper_thai.sh` (untracked); set its STOP and KILL
    times first.
  - Expected: about 5.9–6.1k Thai rows in all. After that, the Thai set is done.
  - The `*.bak-night2` files in `data/full_th` (No Robots before the repair) can be deleted once the final rows look
    right.
- **Open with the user (no GPU):** LogiQA answers often repeat the rationale; how a row maps onto the student's chat
  template (training stage); judge validation with thinking on; the closed-gold audit; the licensing TODOs; narrowing
  the VM token to the dataset repo and deleting the old `JuyJuy` token.
