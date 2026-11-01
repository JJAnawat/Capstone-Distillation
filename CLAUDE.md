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
- **This repo is stage 1:** teacher data generation. Training is a separate stage. The English set is done; a **Thai
  set** is being added by translating new questions first and then distilling them (decided 2026-10-31, see Key
  decisions).

## Pipeline (details in README.md)
prepare → rate → generate → compress (variant B only) → filter → judge → select → analyze
Thai sets first run translate → check-translation → similarity → build-translated, then the same stages.
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
- **Work is split into a CPU phase and a GPU phase** (user's rule, 2026-10-31). Claude finishes the CPU phase (code,
  offline tests, VM setup and dry checks) and reports. Then the GPU steps run in the windows.
- **No smoke tests** (the user, 2026-10-31): neither OpenRouter runs nor `LIMIT=5` trial runs. Offline tests and the
  VM's dry checks cover the CPU phase.

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
    the card and `load_dataset(repo, "final", split="no_robots")`. Closed sets load from `final_closed`
    (`split="logiqa"` or `"ultrafeedback"`), which was tested on 2026-10-28. README's "Using the data" has both.
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
- **Thai sets** (discussed with the user, 2026-10-31; README "Thai sets" has the mechanics).
  - **Added next to the English rows, about 1:2** (about 6.2k Thai rows), not a re-translation of them. The ratio
    used in training is the training stage's choice.
  - **Translate the question first, then distill** (the user's research): Qwen writes the Thai rationale and answer
    itself, so no target is translated text. Variant B fits this: Qwen can think in English, and compress writes
    the Thai steps.
  - **Questions never used in English** (the user's choice over pairing). The ids in the source's ratings file are
    excluded, because every English question was rated first. Offsets were rejected: ids are explicit.
  - **Each Thai set gets half its English set's rows.** UltraFeedback closed is skipped (noisy golds, English-bound
    FLAN tasks, short golds translate badly), and its share goes to LogiQA.
  - **The reference answer is translated only for the judge and the golds.**
  - **Prompts bound to English are flagged and dropped:** the judge's `english_only`, plus the
    `language_skip_patterns` rule.
  - **Translator comparison (G1, 2026-10-31):** 52 prompts (13 per set) by Qwen, Gemma and Typhoon Translate 1.5,
    which makes 156 translations. The user graded 117 by hand, blind (meaning and fluency, 1–5), in an .xlsx. Results
    are in `reports/translate_compare.md`.
    - **Gemma won:** meaning 4.77 and fluency 4.69, against Qwen 4.36 / 4.15 and Typhoon 3.51 / 3.54. Typhoon left
      13 of its 52 translations as untranslated English copies. So `translate.translator: judge`.
    - **Checks:**
      - The judge's meaning score works: AUC 0.79 and Spearman 0.53 against the grades.
      - Its fluency score agrees too (AUC 0.82).
      - BGE-M3 similarity doesn't (AUC 0.36): untranslated copies score 1.0, and flipped meanings look normal. It
        was dropped from the run.
      - So `min_meaning: 4.0` and `rules.min_thai_share: 0.2`. At 0.2 the copies (0.0) are still caught, and prompts
        heavy with code or names (0.35–0.48, graded 5) are kept. Gemma's translations then lose about 6%.
    - **Grading caveats:**
      - 7 untranslated copies were graded meaning 5 with fluency 1, so the report counts copies as 1, as the guide's
        scale says.
      - Gemma caught two LogiQA option flips (rows 120 and 134) that were graded 5. Claude checked both, and Gemma
        was right.
    - **Same family (the user's concern):** Gemma translating and checking. Its lean is +0.23 on its own
      translations and +0.19 on Qwen's, so there's little self-preference. But it flagged none of its own as
      English-only. The user chose to test a checker from another family: Qwen3-32B-AWQ, the `checker` role, run by
      `scripts/second_checker.sh`.
    - **The second checker lost** (2026-10-31, 18:49–18:56 UTC, 7 min). Its meaning score reaches AUC 0.69 and
      Spearman 0.35, against Gemma's 0.79 / 0.53. At 4.0 it rejects 12% of bad translations, against Gemma's 56%.
      - It scored 11 of the 13 untranslated English copies 4–5 ("accurate"). It gave both LogiQA option flips 5,
        where Gemma gave 2–3.
      - It leans +0.54 on Qwen's translations and +1.15 on Typhoon's.
      - **On Gemma's translations it leans +0.20, where Gemma leans +0.23:** an independent checker rates them as
        Gemma does, so there's no sign that Gemma favours its own work.
      - So `translate.checker` stays `judge`, with `min_meaning: 4.0`.
    - **A second translator comparison** (the user's call, 2026-11-01). The user felt G1's pool was too small to
      crown Gemma.
      - On G1's 39 complete prompts Gemma's lead is significant: against Qwen, meaning +0.41 (95% CI +0.15 to +0.69,
        sign test p = 0.03) and fluency +0.54 (p = 0.0004); against Typhoon, +1.26 and +1.15. But no dedicated
        translation model had been tried.
      - **Candidates:** Seed-X-PPO-7B (OpenMDW), MADLAD-400 10B (Apache-2.0) and NLLB-200 3.3B (CC BY-NC). They are
        the `hf:` roles `seedx`, `madlad` and `nllb`, run in-process with transformers and split into sentences.
      - **Ruled out:**
        - Hunyuan-MT-7B, the WMT25 winner: its license forbids using outputs to improve another model.
        - TranslateGemma: gated, no Thai on its card, and Gemma's family.
        - Tower-Plus: no Thai.
        - SeamlessM4T v2: close to NLLB.
      - **Pool:** 40 undistilled questions (10 per set; `compare-pick`), with Gemma's existing translations, so 160
        rows graded blind. It detects a meaning gap of about 0.4.
      - **Decision rule:** switch only if a rival beats Gemma clearly on meaning (sign test p < 0.05) and is no worse
        on fluency. Otherwise Gemma stays: night 1 used it, and switching means mixed translators.
      - **Night 2 waits for this decision.**
      - **CPU phase done (2026-11-01):**
        - The VM's `~/hf` holds NLLB (16.4 GiB), MADLAD (39.9 GiB; its two unused GGUF files were deleted) and Seed-X
          (14 GiB).
        - The pool is built: `data/translate_compare2` on the VM, 10 per set, drawn from 900 undistilled questions
          per open set and 1,150 for LogiQA.
        - Seed-X ships a bare `tokenizer.json`, so under transformers 5 it has no bos, eos or pad.
          `translate.fix_special_tokens` takes them from the model config (`<s>` = 1, `</s>` = 2, pad = eos), and
          outputs are cut at the first `</s>`. A CPU check on the VM confirmed it.
      - **The GPU run (2026-11-01, 06:18–06:50 UTC, with two fixes along the way):**
        - MADLAD ran out of memory on UltraFeedback's long passages at batch 16 × 4 beams. Now a batch that runs out
          is retried as two halves, MADLAD's batch is 8, and the translators run with
          `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
        - Seed-X had 17 of 40 translations cut off: Thai takes it 3.7× the tokens of English (max 5.4×), against at
          most 2× for NLLB and MADLAD. Now the output limit is set per model (`max_new_ratio`, `max_new_tokens`):
          Seed-X 6× and 1,024, the others 2.5× and 512. The cut files are kept in `seedx_cut_v1/`.
        - After the rerun, 1 Seed-X translation is still cut: a prompt with unfenced JavaScript, which the
          English-only rule drops anyway.
        - **Waiting for the user's grades:** the laptop has the sheet in `data/vm/translate_compare2/` (160 rows, 40
          per translator, blind). Then `distill compare-report data/vm/translate_compare2/translate_compare2.xlsx
          --out reports/translate_compare2.md`.
  - **The teacher keeps its English prompts.** Claude drafted Thai versions with a prompt test. The user judged
    the drafts poor and dropped both (2026-10-31). The English prompts already ask for the prompt's language, and
    the filter rejects English rationales for Thai questions.
  - The judge rubric stays English. Thai rows keep `input_en` and the English id, and have their own Hub subsets
    (`final_th`, `final_th_closed`).
- **Judge:** a pointwise rubric over reasoning + answer on a 1–9 scale. A single digit is a single token, so logprobs
  give an expected score that breaks ties. It is checked against HelpSteer2 human preferences (`judge --validate`).

## Commands
- Laptop: `uv venv --python 3.12 && uv pip install -e ".[dev]"`, then `uv run pytest`.
- Laptop smoke test (Git Bash; needs `OPENROUTER_API_KEY` in `.env`; costs cents): `bash scripts/smoke_openrouter.sh`.
- Thai sets on the VM (README "Thai sets"):
  - `scripts/translate_compare.sh` (the translator comparison) and `scripts/second_checker.sh` (the other-family
    checker);
  - then `LIMIT=N bash scripts/run_thai.sh`: prep once, then distill N prompts per set. A larger N on the next
    night continues, and N=0 takes the rest.
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
  text that contains them. `sed`'s `c\` command drops backslashes too.
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
- **VM disk:** the root disk `/` (495 GB) belongs to others. It had 1.0 GB free on 2026-10-27 and 52 GB on
  2026-10-31.
  - **The repo is `/lambda/nfs/Fusion-SFT/aicookie/Capstone-Distillation`** (`$HOME/Capstone-Distillation`). Run
    `pwd` first over SSH and work only there. `/home/aicookie/Capstone-Distillation` is an old, deprecated clone
    (9.7 GB on the root disk, at a9cff7c, with its own venv). `/home/aicookie` is a real directory, not `$HOME`.
  - The repo's venv had been copied from that clone. Until 2026-10-31, 60 files in `.venv/bin` still pointed at
    the old path, so every script ran the old venv's `distill` and `vllm`: the `activate` script's `VIRTUAL_ENV`
    and the shebangs of `vllm`, `hf`, `pytest` and others. They were rewritten that day, and nothing points there
    now.
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

## Status (2026-10-31)
- **Built and in use on the VM:** the full pipeline, with resume, clean Ctrl-C, per-set mode, the shared-GPU fit,
  and the Hub backup with its dataset card.
- **Thai sets: the CPU phase is done** (2026-10-31): `distill/translate.py`, the scripts and the Thai-aware filter,
  select, verify and hub. The teacher keeps its English prompts.
  - **The VM is ready:**
    - it has installed the `qe` extra (sentence-transformers 6.2.0; vLLM, torch and transformers unchanged);
    - the venv no longer points at the old clone (Gotchas), and the tests pass there;
    - `~/hf` holds only Qwen3.5-9B, Gemma 4 31B and the student tokenizer.
      - Typhoon Translate and BGE-M3 were deleted after the comparison (11.8 GiB), and Qwen3-32B-AWQ after it lost
        the checker test (18 GiB).
      - The user wants unused models deleted. The old clone on the root disk stays: the user's choice.
  - **Dry pool check** (the same on both machines), unused prompts left vs the limit: No Robots 6,205 / 1,400,
    HelpSteer2 4,692 / 1,400, UltraFeedback open 48,572 / 1,400, LogiQA 9,178 / 1,650.
  - The laptop has the VM's ratings files in `data/prepared/`. Its OpenRouter key has expired, and no smoke tests
    are wanted anyway.
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
- **Nothing runs on the VM now:** no tmux sessions, and none of our processes on the GPU. Stage 1's output is data,
  not a model. The teacher, judge and student are unchanged downloads in `~/hf` on the VM. The user asked where
  "the model" was, so say this plainly.
- **Next: the Thai GPU phase**, in the user's windows:
  1. **G1, the translator comparison, is done:** it ran on 2026-10-31 from 16:50 to 17:07 UTC (17 min), and the
     user graded it. Results are under Key decisions.
     - The graded sheet is on the laptop in `data/vm/translate_compare/` (gitignored) and on the VM at
       `reports/translate_compare.xlsx`.
  2. **G2, night 1 (about 6 h):**
     - `bash scripts/second_checker.sh` is done: Gemma stays the checker (Key decisions).
     - **Night 1, `LIMIT=500`: done, 2026-10-31 19:04 to 23:31 UTC (4h27m), with no failures.** The stopper
       (`logs/stopper_thai.sh`, untracked) found nothing left to stop. Data is in `data/full_th` and on the Hub.
       - **Prep took 86 min:** translation 65 min (561–695 tok/s on Gemma), checks 17 min. Prepared prompts:
         No Robots 1,275, HelpSteer2 1,292, UltraFeedback 1,152 and LogiQA 1,523, from 1,400 / 1,400 / 1,400 /
         1,650. UltraFeedback lost 18%, 96 of them to `english_task_pattern` (translation tasks).
       - **Distillation took 3h01m:** 20.4 s per prompt per set, so Thai costs no more than English, plus about
         10 min a run for model loads.
       - **2,336 final rows:** No Robots 393, HelpSteer2 400, UltraFeedback 410, LogiQA 1,133 (2.27 rows per prompt;
         72 prompts had no correct letter). Student tokens: median 381, p90 623, max 1,953.
       - **The language filter rejected 16–24% of open-set samples.**
         - About 590 because `compress` wrote English steps for a Thai question, from Qwen's English thinking.
         - About 110 because lingua read a Thai prompt heavy with English names or code as English, so a Thai
           answer looked mismatched.
         - 28 / 54 / 49 open prompts lost every sample to this.
       - LogiQA: 34 of 500 got no complexity rating, because the rater's analysis ran past `rate.max_tokens` 1,024
         (English: 158 of 2,500, the same cause).
       - **Why LogiQA gives about 2 rows per prompt:**
         - By design, a closed prompt keeps up to 3 distinct correct samples (`k_closed: 3`, SCoTD), and most had
           all 4 correct.
         - Thai gives 2.27 rows per prompt against English 1.87. Only 16% of prompts were dropped (English 22%):
           1% for a best score below 7, against English's 12%, where the judge marked down verbose answers. And 77%
           of kept prompts kept all 3 samples, against 60%.
     - **The fixes for night 2 (the user's decisions, 2026-11-01):**
       - The filter expects Thai on `lang: th` rows (`report.expected_language`).
       - Compress uses `compress_v4` on translated rows. It is v3 with "Write every step in {language}" instead of
         "the same language as the question". The user approved changing that one sentence, after rejecting the
         Thai prompts.
       - `REDO_LANGUAGE=1` redoes night 1's English-step samples (`compress --redo-wrong-language`).
       - `rate.max_tokens` is 2048, and ratings without a level are asked again.
       - Thai LogiQA is capped at 1,200 prompts, about 2,700 rows (`LIMIT_CLOSED=1200`).
       - **Dry check on night 1's data** (VM CPU, scratch output):
         - The fixed filter keeps 93 more samples (No Robots +23, HelpSteer2 +33, UltraFeedback +35).
         - The redo will compress 715 English-step records again (211 / 311 / 166 / 27), which takes a few minutes.
       - The copied `Final answer:` line stays in the rationales. It's in 91% of Thai and 74% of English LogiQA
         rationales; the user left it for the training stage.
  3. **First the second translator comparison** (Key decisions):
     - `bash scripts/translate_compare2.sh` (~20 min of GPU);
     - the user grades `reports/translate_compare2.xlsx` (160 rows);
     - `compare-report … --out reports/translate_compare2.md`;
     - decide, then delete the losing models from `~/hf`.
     If a rival wins, a follow-up plan re-translates the undistilled questions before night 2.
  4. **G2, night 2 (about 4.4 h):** `REDO_LANGUAGE=1 LIMIT=0 LIMIT_CLOSED=1200 bash scripts/run_thai.sh`.
     - It takes every remaining open prompt (No Robots 775, HelpSteer2 792, UltraFeedback 652) and 700 more LogiQA,
       plus the redo.
     - The estimate uses night 1's rates (open 4.2–4.6 s per prompt, LogiQA 7.2 s) plus about 20 min for model
       loads, the redo and the re-rating.
     - **Expected: about 5.9–6.1k Thai rows, and the Thai set is done in two nights.** Night 3 isn't needed.
- **Also open with the user** (no GPU): LogiQA's verbose answers (Night 3 above): `answer` often repeats the
  rationale. And how a row maps onto the student's chat template, which belongs to the training stage.
- **Models moved off /dev/shm (2026-10-27):** `.vm_env` now sets `HF_HOME=$HOME/hf` and
  `VLLM_CACHE_ROOT=$HOME/vllm-cache` (the old file is `.vm_env.bak-shm`), and both models are in `~/hf` (40 GB;
  the files sit in `hub/blobs`, so per-model `du` looks tiny). The home now holds 53 GB. Loading from the share
  is fine: each server starts in about 6 min. Weights take 107 s for the teacher and 139 s for the judge.
- **Open, later:**
  - judge validation with thinking on;
  - the closed-gold audit;
  - the licensing TODOs in docs/LICENSES.md.
- **Left to the user:** narrow the VM token to the dataset repo only, and delete the old `JuyJuy` token.
