# Translator comparison

Graded by hand: 117 of 156 translations for meaning, 117 for fluency (1-5). Translators are compared on the 39 prompts whose translations are all graded for meaning; rule flags cover all their translations. The signals use every graded translation. Bad = meaning ≤ 3.

Untranslated copies (no Thai letters) count as meaning 1, as the grading guide's scale says: 7 grades changed.

Checkers: `judge` (google/gemma-4-31B-it-qat-w4a16-ct).

## Translators

| translator | model | prompts | meaning | fluency | meaning ≤ 2 | rule-flagged |
|---|---|---:|---:|---:|---:|---:|
| judge | google/gemma-4-31B-it-qat-w4a16-ct | 39 | 4.77 | 4.69 | 0% | 12% |
| teacher | Qwen/Qwen3.5-9B | 39 | 4.36 | 4.15 | 3% | 17% |
| typhoon | typhoon-ai/typhoon-translate1.5-4b | 39 | 3.51 | 3.54 | 31% | 31% |

## Each checker's lean, by translator

Mean of the checker's meaning score minus the user's grade (+ = more lenient than the user), and the share of translations it calls bound to English. A checker that favours its own family leans further up on that family's translations.

| translator | judge: meaning − user | judge: English-only |
|---|---:|---:|
| judge | +0.23 | 0% |
| teacher | +0.19 | 4% |
| typhoon | +0.27 | 8% |

## Automatic signals vs the grades

Spearman: rank correlation with the user's grade. AUC: how often a good translation (grade ≥ 4) scores above a bad one (≤ 3).

| signal | vs | rows | Spearman | AUC good > bad |
|---|---|---:|---:|---:|
| judge meaning (expected) | meaning | 117 | 0.53 | 0.79 |
| judge fluency | fluency | 117 | 0.62 | 0.82 |
| embedding similarity | meaning | 117 | -0.18 | 0.36 |

## Thresholds

Share of bad translations rejected and of good ones kept at each threshold.

| signal | threshold | bad rejected | good kept |
|---|---:|---:|---:|
| judge meaning (expected) | 3.0 | 52% | 96% |
| judge meaning (expected) | 3.5 | 52% | 95% |
| judge meaning (expected) | 4.0 | 56% | 93% |
| judge meaning (expected) | 4.5 | 60% | 92% |
| judge meaning (expected) | 4.8 | 60% | 92% |
| embedding similarity | 0.6 | 0% | 100% |
| embedding similarity | 0.7 | 16% | 96% |
| embedding similarity | 0.8 | 24% | 76% |
| embedding similarity | 0.9 | 48% | 28% |

## Rule flags

| flag | translations | mean user meaning of the flagged |
|---|---:|---:|
| not_thai | 19 | 1.86 |
| numbers_changed | 6 | 4.50 |
| length_ratio | 3 | 2.00 |
| options_missing | 3 | 5.00 |
| english_task_pattern | 3 | 3.00 |

## Called bound to English by a checker

| set | id | by | prompt |
|---|---|---|---|
| helpsteer2_th | e6e38b74fc477030 | judge | From the provided list of "to-get supplies", sort out the supplies that start with the letter "A" and "B". Output should be not more than 30 words.  R |
| no_robots_th | 03d14063670eabd2 | judge | Hi I'm working on a Python programming prompt and I am running into a bug. Here is the code prompt I'm trying to solve:  You are given a string of wor |
| ultrafeedback_th | 0d1630c75c12272a | judge | Detailed Instructions: Given a sentence in Japanese, provide an equivalent paraphrased translation in Korean that retains the same meaning both throug |
| ultrafeedback_th | c158c1330be40491 | judge | Given the question: What is the final step of the following process: -  An egg is laid by a mother bird - A baby bird forms within the egg if it is fe |

## Grader notes

| row | translator | meaning | note |
|---:|---|---:|---|
| 2 | typhoon | 1 | No translation |
| 4 | typhoon | 1 | No translation |
| 22 | judge | 5 | Some parts are clearly translated, but some also seem natural especially the conjuctions. |
| 23 | typhoon | 1 | No translation |
| 24 | teacher | 3 | First parts (instruction and The Grand Palace details) are natural and precise, but the later parts (Floating Markets and Wat Phra That) are badly translated. |
