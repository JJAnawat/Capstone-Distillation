# Experiments

The record of every experiment in stage 1 (teacher data generation), for the project report: setup, results and the
decision each one led to. The generated reports are in `reports/`; the code of the finished experiments stays in
`distill/compare.py`, `distill/inproc.py` and the `scripts/*compare*.sh` and `second_checker.sh` scripts.

## Decisions at a glance

| decision | evidence | section |
|---|---|---|
| Train on short CoT + answer, never Qwen's raw thinking | raw traces: median 1,739 tokens, p90 5,045, against a 2,048 context | 1 |
| Variant B (thinking on, then `compress`) | higher scores per sample, rationales state facts | 2 |
| Samples and step limit follow complexity | B's best-of-2 reached 95% usable at level 1 | 2 |
| Judge rubric v2 | same agreement as v1, wider score spread | 2 |
| No MTP speculative decoding | 1.12x in a short bench, then an out-of-memory crash at full load | 3 |
| Gemma 4 31B translates the Thai questions | best on meaning and fluency in G1, and safe with unmarked code in G2 | 5, 8 |
| Gemma checks the translations (meaning ≥ 4.0, English-only flag) | no other checker more accurate; no self-preference found | 5, 6, 9 |
| No embedding similarity check | AUC 0.36: untranslated copies score 1.0 | 5 |

## 1. OpenRouter pilot (2026-10-13; `reports/pilot.md`)
- Qwen3.5-9B through OpenRouter, thinking on, on the three English open sets.
- Its raw thinking traces had a median of 1,739 tokens and a p90 of 5,045. The student has a 2,048-token context for
  prompt and response.
- Models of 3B or smaller learn poorly from long CoT (Li et al. 2025). Short rationales with many samples per question
  do work at 125M–1.3B (SCoTD, Li et al. 2023).
- **Decision:** every target is a short rationale plus the answer. Raw traces are kept for analysis only.

## 2. Pilot 2: variant A vs B, the judge (2026-10-22; `reports/pilot2.md`, `pilot2_A_vs_B.md`, `judge_validation.md`)
- **Setup:** 100 prompts per set, N=8 samples, a fixed limit of 5 steps.
  - Variant A: thinking off, the rationale inside `<reasoning>` tags.
  - Variant B: thinking on, then `compress` rewrites the trace into short numbered steps.
- **A vs B:** B scored higher per sample (mean +0.3–0.45; closed pass@1 62% against 53%), and its rationales state
  facts rather than plans. B gains almost nothing after best-of-4. B at N=4 costs about 5.7 s per prompt, judging
  included.
- **Judge:** agreement with HelpSteer2 human preferences was 71.0% for rubric v1 and 70.2% for v2. v1 gave almost
  every Qwen sample a 9, so v2 stays for its wider spread.
- **Judge vs rules on closed questions:** AUC about 0.5, mostly from noisy FLAN golds.
- **Decisions:**
  - Variant B for the full run. A stays in the code (`VARIANTS=A`) for comparison.
  - N=2 for level-1 prompts and N=4 otherwise: B's best-of-2 already reached 95% usable at level 1.
  - Step limit 4/5/6/8/10 for complexity levels 1–5 (6 unrated), about 40 words a step. The fixed 5-step cap had made
    easy rationales pad to 5 steps; the user found 2/3/5/8/10 at 25 words too harsh.

## 3. MTP speculative decoding (2026-10-23; `reports/bench.md`)
- An 800-sample bench measured 1.12x the pilot's throughput with MTP.
- On night 2 of the full run its rejection sampler ran out of GPU memory (`rejection_sampler._verify_in_chunks`) and
  killed the teacher 35 min into LogiQA. That memory falls outside vLLM's startup profile, and short benches don't
  show it.
- On night 3, LogiQA without MTP ran at 2,100–2,325 tok/s, the same speed as with MTP on night 2.
- **Decision:** MTP stays off.

## 4. English full run (2026-10-22 to 10-27; `reports/full.md`)
Variant B, `LIMIT=2500` per set, `LIMIT_CLOSED=500` for UltraFeedback's noisy closed set. **12,432 final rows.**

| night | sets | rows | time | notes |
|---|---|---:|---|---|
| 1 (10-22) | No Robots, HelpSteer2 | 2,296 + 2,322 | ~2h45m a set (2h15m teacher, 30–40 min judge) | then the GPU sat idle: another user's app blocked the old start check |
| 2 (10-23) | UltraFeedback open, closed | 2,304 + 846 | 4h20m for both | closed: 495 prompts, 320 rule+judge, 138 judge fallback; LogiQA's teacher then crashed (MTP), ~4 GPU hours lost before the auto-restart existed |
| 3 (10-27) | LogiQA | 4,664 | 5h11m, no failures | 1,946 of 2,500 prompts kept |

- **Night 3 timing:** server start 6 min each; generate (8,784 samples) 3h21m; compress (9,910) 57 min; judge (8,306)
  37 min.
- **LogiQA yield:** 90% of prompts had a correct letter (pass@4; pass@1 84%). Select dropped 250 prompts with no
  correct sample and 304 whose best score was below 7. The judge often marks down answers that repeat the rationale
  instead of giving the letter (open with the user).

## 5. Thai translator comparison, G1 (2026-10-31; `reports/translate_compare.md`)
- **Setup:** 52 new questions (13 per Thai set) translated by Qwen3.5-9B, Gemma 4 31B and Typhoon Translate 1.5 4B:
  156 translations. The user graded 117 by hand, blind (meaning and fluency, 1–5). 17 min of GPU.
- **Translators:**

| translator | meaning | fluency |
|---|---:|---:|
| Gemma 4 31B | 4.77 | 4.69 |
| Qwen3.5-9B | 4.36 | 4.15 |
| Typhoon Translate 1.5 | 3.51 | 3.54 (13 of 52 left as untranslated English) |

  Against Qwen on the 39 fully graded prompts, Gemma leads on meaning by +0.41 (95% CI +0.15 to +0.69, sign test
  p = 0.03) and on fluency by +0.54 (p = 0.0004); against Typhoon by +1.26 and +1.15.
- **Automatic checks against the grades:** Gemma's meaning score AUC 0.79 (Spearman 0.53), its fluency score AUC 0.82;
  BGE-M3 embedding similarity AUC 0.36 (untranslated copies score 1.0, flipped meanings look normal).
- **Grading notes:** 7 untranslated copies graded meaning 5 with fluency 1 count as meaning 1, as the grading scale
  says. Gemma caught two LogiQA answer-option flips that had been graded 5 (rows 120 and 134); both checked and real.
- **Decisions:** Gemma translates; Gemma's check with `min_meaning: 4.0`; `min_thai_share: 0.2` (catches copies at
  0.0, keeps code-heavy prompts at 0.35–0.48); similarity dropped. Gemma's translations lose about 6% to the checks.

## 6. A checker from another family (2026-10-31; `reports/translate_compare.md`)
- **Why:** Gemma both translates and checks; the user worried about self-preference. Qwen3-32B-AWQ scored the same 156
  translations (7 min of GPU).
- **Result:** AUC 0.69 and Spearman 0.35, against Gemma's 0.79 / 0.53. At 4.0 it rejects 12% of the bad translations,
  against Gemma's 56%. It scored 11 of 13 untranslated copies 4–5 and both LogiQA option flips 5.
- **Self-preference test:** Qwen3-32B leans +0.20 above the user on Gemma's translations, where Gemma leans +0.23.
  Gemma leans +0.19 on Qwen's and +0.27 on Typhoon's. No sign that Gemma favours its own work.
- **Decision:** Gemma stays the checker.

## 7. Thai night 1 (2026-10-31; `reports/full_th.md`)
- `LIMIT=500` per set, 19:04–23:31 UTC (4h27m), no failures.
- **Prep (86 min):** translation 65 min (561–695 tok/s on Gemma), checks 17 min. Prepared prompts: No Robots 1,275,
  HelpSteer2 1,292, UltraFeedback 1,152 and LogiQA 1,523, from 1,400 / 1,400 / 1,400 / 1,650. UltraFeedback lost 18%,
  96 of them to `english_task_pattern` (translation tasks).
- **Distillation (3h01m):** 20.4 s per prompt per set, so Thai costs no more than English, plus ~10 min of model loads.
- **2,336 final rows:** No Robots 393, HelpSteer2 400, UltraFeedback 410, LogiQA 1,133 (2.27 rows per prompt; 72
  prompts had no correct letter). Student tokens: median 381, p90 623, max 1,953.
- **Language filter:** it rejected 16–24% of open-set samples, ~590 because `compress` wrote English steps from Qwen's
  English thinking and ~110 because lingua read a Thai prompt full of names or code as English.
- **Why LogiQA gives ~2 rows per prompt:** a closed prompt keeps up to 3 distinct correct samples (`k_closed: 3`). Thai
  gives 2.27 against English 1.87: only 16% of prompts were dropped (English 22%), 1% for a best score below 7 against
  12%, and 77% of kept prompts kept all 3 samples against 60%.
- **Fixes for night 2 (the user's decisions):** the filter expects Thai on Thai rows; `compress_v4` says "Write every
  step in Thai"; `REDO_LANGUAGE=1` redoes the English-step samples; `rate.max_tokens` 2048 (34 of 500 LogiQA prompts
  had no rating); Thai LogiQA capped at 1,200 prompts. A dry check on night 1's data: the fixed filter keeps 93 more
  samples, and the redo recompresses 715 records.

### Thai night 2 (2026-11-01, 13:02–17:50 UTC; window until 18:00)
- **Command:** `REDO_LANGUAGE=1 LIMIT=0 LIMIT_CLOSED=1200 bash scripts/run_thai.sh` (`BY_SET=0`).
- **A false start (13:02–13:48):**
  - The code cleanup had dropped the read of vLLM 0.32's `reasoning` field (it now keeps only `reasoning_content`,
    the deprecated name). So the first 2,174 No Robots samples were stored without their thinking, and compress
    couldn't use them.
  - The run was stopped, the field fixed (`server.reasoning_text` reads both names), and those samples removed and
    regenerated. That cost ~45 min of GPU.
- **Teacher phase (13:51–17:25):** 0 failures, all at 2,500–2,900 tok/s.
  - Generated: No Robots 2,174 samples, HelpSteer2 2,454, UltraFeedback 1,920, LogiQA 2,782 (1,200 prompts).
  - Compress redid 715 English-step rationales (211 / 311 / 166 / 27), as the dry check predicted.
  - Re-rating: 34 LogiQA prompts left unrated on night 1 were rated again.
- **Filter:**

| set | kept | rejected | language mismatch | night 1's language rejections |
|---|---:|---:|---:|---:|
| No Robots | 3,579 | 91 | 57 (1.6%) | 16–24% across the open sets |
| HelpSteer2 | 3,867 | 159 | 99 (2.5%) | |
| UltraFeedback | 3,218 | 176 | 113 (3.3%) | |
| LogiQA | 4,748 | 24 | 23 | |

  `compress_v4` and the Thai-aware filter cut the language rejections about tenfold.
- **Judge phase (from ~17:32):** No Robots fully judged (2,341 new samples). HelpSteer2 got 202 of 2,681 before the
  stopper's Ctrl-C at 17:50.
- **Left for the next window:** judging the rest of HelpSteer2, UltraFeedback and LogiQA, then select, push and the
  report.

### Thai night 2, resume (2026-11-08, 16:00–16:55 UTC; window until 17:00)
- **Same command,** launched at 16:00 by a timed starter in tmux; the stopper's Ctrl-C came at 16:55.
- **Teacher phase (16:00–16:21):** a few retries only (21 ratings, 11 compressions). Another user's jobs started
  after vLLM had sized its KV cache (11.4 GiB in use at launch, then a new 2 GB process): the teacher ran out of
  memory at 16:06, and `with_server` restarted it at 16:11 (1 of 2).
- **Judge phase (16:21–16:55), 140–190 tok/s:** HelpSteer2 (2,433 samples) and UltraFeedback (2,000, 1 failure)
  fully judged; LogiQA 515 of 2,204.
- **Select (on CPU after the stop)** for the three open sets:

| set | prompts | rows | dropped: best score < 7 |
|---|---:|---:|---:|
| No Robots | 1,262 | 1,052 | 210 (17%; English 6%) |
| HelpSteer2 | 1,271 | 1,161 | 110 |
| UltraFeedback | 1,118 | 1,005 | 113 |

- **Left:** ~1,700 LogiQA samples to judge (~10 min), then select, push and the report: ~25 min of GPU with the
  teacher's load. Interim totals in `reports/dataset_status.md`.

## 8. Second translator comparison, G2 (2026-11-01)
- **Why:** the user found G1's pool too small to crown Gemma, and no dedicated translation model had been tried.
- **Candidates:** Seed-X-PPO-7B, MADLAD-400 10B and NLLB-200 3.3B, run in-process with transformers on sentence-sized
  pieces (`inproc.py`). Ruled out: Hunyuan-MT-7B (its license forbids using outputs to improve another model),
  TranslateGemma (gated, no Thai, Gemma's family), Tower-Plus (no Thai), SeamlessM4T v2 (close to NLLB).
- **Pool:** 40 undistilled questions (10 per set) with Gemma's existing translations: 160 rows. GPU run 06:18–06:50
  UTC with two fixes: MADLAD ran out of memory at batch 16 × 4 beams (now halved batches), and Seed-X cut 17 of 40
  translations because Thai costs it 3.7× the tokens of English (now a per-model output limit).
- **Outcome:** the user chose Gemma without grading the sheet. On the pool's one prompt with unmarked code, Gemma kept
  all 25 code lines and the others 8–9: the sentence-level models see one sentence at a time, and only fenced code is
  protected. Across 5,850 night-1 translations Gemma split no Thai word with a space; MADLAD did in 2 of 40.

## 9. Checker comparison (2026-11-01; `reports/checker_compare.md`, `reports/translate_compare.md`)
- **Candidates:** MetricX-24 XXL (a reference-free translation-quality metric on mT5), CometKiwi XL (dropped: its
  gated repo stayed closed) and Typhoon 2.5 30B-A3B (a Thai-tuned Qwen3 MoE, quantized to FP8 by vLLM while loading).
- **Design:** G1 has only 2 bad Gemma translations, too few to test a checker on Gemma's real errors. So the
  candidates scored a seeded sample of 300 translations per Thai set; each rejected as many as Gemma (2.6%); and the
  user graded a blind pool of where they disagree: 38 of 64 rows. GPU run 10:32–10:47 UTC (15 min).
- **G1's grades (AUC):**

| checker | all 117 graded | rows a gate acts on (85 good, 11 bad) |
|---|---:|---:|
| Gemma 4 31B | 0.79 | 0.53 |
| MetricX-24 XXL | 0.84 | 0.65 |
| Typhoon 2.5 | 0.60 | 0.50 |
| Qwen3-32B | 0.69 | 0.55 |

  Gemma's 0.79 came largely from untranslated copies, which the rules already drop.
- **The disagreement pool:** of the rows only MetricX rejects, 2 of 8 graded are bad; only Typhoon 2 of 10; only Gemma
  2 of 10; all pass 0 of 11. G1 agrees: at its threshold MetricX rejects none of G1's rule-passing rows, and looser
  thresholds cost as many good translations as they catch bad ones.
- **Each checker's character:**

| checker | catches | fails on | damage as the checker |
|---|---|---|---|
| Gemma | untranslated copies; flipped meanings, even one word in a long passage (6 of the 8 good-graded rows it rejected hold a real error, e.g. "aren't afraid" → "don't dare") | its own partial translations (an embedded sentence, options or examples left in English, critique "None"); an occasional misread | some mixed-language prompts pass; a few good long passages are lost |
| MetricX | untranslated text | treats kept examples, data and code as errors | task-template prompts are 4% of the sample but 45% of its rejections (UltraFeedback 25% vs 58%) |
| Typhoon 2.5 | little | grades the request's content or premise ("illegal, cannot translate"; "the sun rises in the east") | deletes harmful and false-premise prompts the safety and reasoning goals need |
| Qwen3-32B | little | passes copies (11 of 13 scored 4–5) and both option flips | wrong-answer LogiQA rows reach training |

- **Swap or not:** no candidate is more accurate than Gemma, and a swap would also cost ~1 h of GPU (adding Gemma's
  145 rejected questions back reshuffles LogiQA's seeded sample). The user's blind grades: 48 of 50 Gemma translations
  that pass the checks are good (96%).
- **Timing per checker on 5,850 translations:** Gemma 17 min (already loaded for translation); MetricX ~6 min plus a
  1 min load; Typhoon ~6 min plus a 10.5 min load.
- **Decision:** no second gate; Gemma stays the only checker. Its weakness is inconsistent "keep in English" choices,
  which a translation-prompt rule would target better than another checker.
