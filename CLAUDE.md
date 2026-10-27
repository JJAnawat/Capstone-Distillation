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
- **GitHub:** Claude commits and pushes, and the user approves each command.
- **GPU VM:** an A100 40GB inside a company machine (Linux).
  - **Claude drives it over SSH** from the laptop (`ssh -o BatchMode=yes aicookie@193.122.152.236 '…'`, key login),
    and the user approves each command.
  - Code gets there by commit, push, then `git pull` on the VM.
  - GPU time comes in windows the user announces (night 1 ended 10:00 Thailand time, 03:00 UTC). Light CPU work on
    the VM is fine outside them.

## Key decisions and why
- **Every training target is short CoT + answer, never Qwen's raw thinking.**
  - Raw traces don't fit a 2,048 context: in the OpenRouter pilot the median was 1,739 tokens and p90 was 5,045.
  - Models of 3B or smaller learn poorly from long CoT (Li et al. 2025, "Small Models Struggle to Learn from Strong
    Reasoners"). Short rationales with many samples per question do work at 125M–1.3B (SCoTD, Li et al. 2023).
  - Raw traces stay in the generation files for analysis only; `select` drops them.
- **Variant B won the pilot and is used for the full run:** thinking on, then `compress` rewrites the trace into
  short steps. Variant A (thinking off, steps inside `<reasoning>` tags) stays in the code for comparison.
- **Rationale format (user's requirement):** numbered steps, one per paragraph, separated by a blank line (`\n\n`).
  The prompts ask for it (`cot_v3`, `compress_v3`), and `verify.format_steps` enforces it on every stored rationale.
  `count_steps` uses the same split.
- **Samples and step limit follow complexity** (decided after the pilot, 2026-10-23).
  - N=2 for level-1 prompts and N=4 otherwise: B's best-of-2 already reached 95% usable at level 1.
  - Step limit 4/5/6/8/10 for levels 1–5, 6 when unrated, about 40 words per step. The user found 2/3/5/8/10 at 25
    words too harsh. The filter's `max_words` is 600, only against runaway rationales. The v3 prompts ask for facts
    and inferences, not plans.
  - The fixed 5-step cap had made rationales pad to 5 steps even on easy prompts.
  - The pilot itself used N=8 and a fixed 5 steps.
- **First full run:** variant B, `LIMIT=2500` per set, except the UltraFeedback closed set, which gets
  `LIMIT_CLOSED=500` because of its noisy known answers. About 8k prompts, current sampling.
  - It runs with `BY_SET=1` in tmux. A `stopper` tmux session sends Ctrl-C 10 min before the GPU window ends.
    Night 3's version (`logs/stopper.sh` on the VM, untracked) waits for clock times, not a sleep count. At 3 min
    before the end it also kills any of our processes still on the GPU.
  - The bench ran on 2026-10-23 (MTP only): 1.12x the pilot. MTP is off again because it crashed night 2 (Gotchas).
  - The sampler is nested, so raising `LIMIT` later only adds prompts.
  - **Night 1 (2026-10-22 17:52 to 23:34 UTC):** No Robots (2,296 final rows) and HelpSteer2 (2,322) finished.
    - Each set took about 2h45m: about 2h15m in the teacher phase and 30–40 min judging.
    - The run then stopped. Another user's app was on the GPU (see Gotchas), and the old start check refused to start
      a server, so the GPU sat unused until 02:50.
  - **Night 2 (2026-10-23 16:21 to about 22:00 UTC, with MTP on):** UltraFeedback open (2,304 final rows) and closed
    (846 rows from 495 prompts: 320 rule+judge, 138 judge fallback) finished, about 4h20m for both.
    - LogiQA's teacher crashed at 21:17, 35 min in (MTP, see Gotchas), with 1,126 of its 9,910 samples.
    - Until the auto-restart existed, the pipeline then ran on against the dead server: about 4 GPU hours lost.
  - **Night 3 (2026-10-27 17:08 to 22:19 UTC, MTP off):** LogiQA resumed and finished in 5h11m, with no failures and
    no restarts. That is 4,664 final rows from 1,946 of its 2,500 prompts.
    - Times: server start 6 min each, generate (8,784 samples) 3h21m, compress (9,910) 57 min, judge (8,306) 37 min.
    - Yield: 90% of prompts had a correct letter (pass@4; pass@1 84%). Select dropped 250 prompts with no correct
      sample and 304 whose best score was below 7.
    - The judge often marks down answers that repeat the rationale instead of giving just the letter. Not yet
      reviewed with the user.
- **Hub backup:** generated data lives only on the VM. `distill push` copies a run and the complexity ratings to a
  private Hugging Face dataset repo (`hub.repo`), and the pipeline runs it after every finished set. The token goes
  in `.vm_env`, never in the config.
  - Repo: `capstone-thai-slm/capstone-distill-data`, private on purpose. Night 1 went up on 2026-10-23 (29 files,
    469 MiB) to `JuyJuy/...`, then moved into the team org. Teammates join the org with the read role and use their
    own tokens.
  - The VM's token is fine-grained. The user pastes tokens into the chat, so suggest putting future ones on the VM
    directly.
  - `push` also writes the dataset card from `docs/dataset_card.md`. It has a subset per stage and a split per set,
    and closed sets get their own subsets (`gold` broke a shared schema: `CastError`). All 9 subsets were
    load-tested on 2026-10-23.
  - The Hub's table viewer doesn't work here: it needs PRO or an Enterprise org for private datasets. Readers use
    the card and `load_dataset(repo, "final", split="no_robots")`.
  - It stays private until a deliberate release: the final rows with a dataset card, once the company-hardware and
    Qwen terms questions in docs/LICENSES.md are settled.
- **Final rows** carry `student_tokens`, measured in the student tokenizer, and a `teacher` group. `reasoning_tokens`
  meant raw thinking, not the rationale, and confused the user, so it is now `teacher.thinking_tokens`.
- **Closed-ended means a known short answer.**
  - Source: UltraFeedback `correct_answers` of 4 words or fewer, from flan_v2_niv2, flan_v2_p3 and flan_v2_flan2021.
    For other sources the field is the literal string "None" or long free text.
  - Rules decide correctness, and the judge still ranks the reasoning of the correct samples, because a yes/no or
    multiple-choice answer can be right with wrong reasoning.
  - truthful_qa isn't rule-checked: its short golds are answers like "I have no comment".
- **LogiQA 2.0 was added for non-math logical reasoning** (user's choice, 2026-10-23).
  - Source: `datatune/LogiQA2.0`. After dedupe it has 11,678 English multiple-choice prompts, all closed; the run
    takes 2,500.
  - **License:** CC BY-NC-SA 4.0 upstream, so non-commercial and share-alike. The mirror's MIT tag is wrong.
  - **`exact_gold: true`:** a prompt with no correct letter is dropped, not judged. The noisy-gold judge fallback
    would admit wrong answers, and the pilot's judge accepted 63–74% of wrong closed answers.
  - It runs as its own invocation (`DATASETS=logiqa LIMIT_CLOSED=2500`), so `LIMIT_CLOSED=500` stays UltraFeedback's.
  - Its reasoning types are uneven: Categorical 7,291, Sufficient Conditional 4,176, the others under 100 each. The
    equal-allocation sampler takes every prompt of the rare types.
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
- **How results reach Claude:** read them over SSH (`logs/full.log`, `data/full/`), and scp data to
  `data/vm/<name>/` on the laptop (gitignored). `collect.sh` and `reports/` are for the user's own copies.

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
- Don't `git pull` on the VM while a pipeline script runs: bash reads scripts as it executes them.
- In Git Bash, Python heredocs that rewrite files turned escape sequences into real newlines. Use the Edit tool for
  text that contains them.
- **Ctrl-C in a tmux pane** sends SIGINT to every process in it, vLLM included.
  - On 2026-10-23 vLLM then waited for its open connections while `generate` waited, on the same connections, for
    replies that never came. That deadlock lasts until the 2 h request timeout.
  - Fix: `run_pool` now drops queued work, and `cli.main` exits with `os._exit(130)` without joining the threads.
  - Launch with `tee -ai`; a plain `tee` dies too and the rest of the log is lost.
  - In copy mode (an attached user scrolling), `tmux send-keys C-c` only leaves copy mode, so timed stoppers send
    `tmux send-keys -X cancel` first.
- **The GPU is shared** in spite of the "exclusive" window. On night 1 another user (`ubuntu`) ran a web app
  (uvicorn on port 8010, about 1.7 GB) on it.
  - `start_server` now refuses only when our own processes hold the GPU, and `stop_server` waits only for ours.
  - `distill serve` lowers `--gpu-memory-utilization` to fit beside other processes (`gpu_share`, `min_gpu_gib`).
    Weights are fixed (teacher 16.7 GiB, judge 18.7 GiB), so only the KV cache shrinks (full: 18.5 and 16.4 GiB).
    Below the minimum it waits up to 15 min, then fails.
  - If someone else's process grows during a run, either side can run out of memory.
- **No MTP speculative decoding on this VM.**
  - The bench (800 samples) measured 1.12x. On night 2 its rejection sampler then ran out of GPU memory
    (`rejection_sampler._verify_in_chunks`) and killed the teacher.
  - That memory falls outside vLLM's startup profile. The PyTorch sampler, Qwen's 248k vocab, the other user's 2 GiB
    and a KV cache 99.8% full on LogiQA's long reasoning all add to it. UltraFeedback had logged the same OOM twice
    and survived.
  - Short benches don't show it. Re-enable it only after a long full-load test with a memory margin.
  - It may not even help at full load. On night 3, LogiQA without MTP ran at 2,100–2,325 tok/s, the same speed it
    had with MTP on night 2.
- **A dead server restarts itself** (added after night 2).
  - `run_pool` raises `ServerDown` after 16 connection-level failures in a row, and `distill` exits with 75.
  - `pipeline.sh`'s `with_server` then restarts the server and resumes the command, twice at most.
  - Before this, every queued request retried for about 38 min against the dead server, and then the empty stages
    ran on.
- **VM disk:** the root disk `/` (495 GB) belongs to others and is nearly full (1.0 GB free on 2026-10-27, falling
  about 0.6 GB a day). Nothing of ours lives there.
  - `aicookie`'s home is on the shared virtiofs mount (`/lambda/nfs/Fusion-SFT/aicookie`, owned by us and
    writable), so the repo, venv and pipeline outputs are already there. Earlier notes said the share was off-limits;
    that was wrong. Only the share's top level needs the `fusion` group, which we don't have and don't need.
  - The share reports a placeholder size (8.0 EiB), so its real limit can't be seen from the VM. The admin approved
    keeping the models there (2026-10-27).
  - So models, datasets and vLLM caches go in the home: `HF_HOME=$HOME/hf` and `VLLM_CACHE_ROOT=$HOME/vllm-cache`,
    set in `.vm_env` (untracked, sourced by `scripts/lib.sh` and `vm_check.sh`). Unlike `/dev/shm`, they survive
    logouts.
  - Over SSH, use `.venv/bin/...`: non-interactive shells don't activate the venv.
- **/dev/shm is wiped whenever the user is fully logged out** (systemd `RemoveIPC` at its default, yes), even without
  a reboot. The models lived there until 2026-10-27 and had to be downloaded again each session (about 36 s for
  40 GB). Don't put anything there that must survive.
  - On 2026-10-23 a failed write there cut short a command chain, `source .vm_env` included. A missing prepared file
    then made `distill` silently run a full `prepare` onto the root disk: 1.9 GB, deleted afterwards, with no data
    changed.
  - Now `distill` loads `.vm_env` by itself, and a missing prepared file raises an error.

## Licensing
No Robots is CC BY-NC 4.0: everything derived from it is non-commercial, so keep `source_dataset` on every row.
Confirm that running it on company hardware is acceptable. See docs/LICENSES.md.

## Status (2026-10-28)
- **Built and in use on the VM:** the full pipeline, with resume, clean Ctrl-C, per-set mode, the shared-GPU fit,
  and the Hub backup with its dataset card. 89 offline tests.
- **Pilot (2026-10-22: 100 prompts per set, N=8; reports/pilot2.md, reports/judge_validation.md):**
  - **Judge:** HelpSteer2 agreement was 71.0% for v1 and 70.2% for v2. v2 stays for its wider score spread.
  - **Judge vs rules on closed questions:** AUC about 0.5, mostly from noisy FLAN golds. A small human audit is
    still open.
  - **B vs A:** B scored higher per sample (mean +0.3–0.45; closed pass@1 62% vs 53%), and its rationales state
    facts, not plans. B gains almost nothing after best-of-4.
  - **Cost:** B at N=4 takes about 5.7 s per prompt, judging included.
- **The full run is done** (details under Key decisions). All five sets have finished: 12,432 final rows, 4,664 of
  them from LogiQA.
  - Their files are in `data/full` on the VM and on the Hub. The LogiQA files were checked against the Hub byte for
    byte on 2026-10-28.
  - The laptop has every set's final rows, select stats and `reports/full.md` in `data/vm/full_night3/`, and night
    1's full files in `data/vm/full_night1/`. `hf download` gets the rest.
- **Next, with the user:** review `reports/full.md` and some samples, LogiQA's verbose answers first (Night 3 above).
- **Models moved off /dev/shm (2026-10-27):** `.vm_env` now sets `HF_HOME=$HOME/hf` and
  `VLLM_CACHE_ROOT=$HOME/vllm-cache` (the old file is `.vm_env.bak-shm`), and both models are in `~/hf` (40 GB;
  the files sit in `hub/blobs`, so per-model `du` looks tiny). The home now holds 53 GB. Loading from the share
  is fine: each server starts in about 6 min. Weights take 107 s for the teacher and 139 s for the judge.
- **Open, later:**
  - judge validation with thinking on;
  - the closed-gold audit;
  - Thai translation (undecided);
  - the licensing TODOs in docs/LICENSES.md.
- **Left to the user:** narrow the VM token to the dataset repo only, and delete the old `JuyJuy` token.
