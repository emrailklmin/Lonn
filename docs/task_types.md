# Task types

> The benchmark and training notes below describe Prototype 3.0. Prototype 4.0
> assigns supplied or derived scores to the seven unmeasured types; see
> [Prototype 4.0](prototype4.md).

Issue #19, branch `prototype3.0`. The new router replaces the four broad topics with about a dozen **task types**. The classifier picks a task type (plus complexity) for each prompt. For each measured task type, the answer bank gives every pool model a measured accuracy and mean cost, and the router trades these off. The mapping lives in `config/task_types.yaml`; `tests/test_task_types.py` checks it against `data/benchmark/manifest.json`.

## Rule for splitting and merging

**Two groups are separate task types only if the models perform differently on them.** Otherwise a split only adds classifier errors and gives the router nothing new. Differences in difficulty *within* one kind of task are left to the complexity signal, not turned into extra types.

## Evidence (answer bank, 690 questions, 2026-10-22)

Accuracy / mean billed cost per question in thousandths of a USD (so `0.36` = 0.00036 USD).

| Group | n | gpt-oss-20b | gpt-oss-120b | glm-5.3-flash | v4.1-flash | v4-pro | gemini-3.8-flash | muse-spark | sonnet-5.5 |
|---|---:|---|---|---|---|---|---|---|---|
| ARC-Challenge | 30 | 97 / 0.02 | 97 / 0.04 | 100 / 0.03 | 97 / 0.41 | 100 / 0.45 | 100 / 0.71 | 100 / 3.36 | 100 / 1.50 |
| HumanEval+ and MBPP+ | 80 | 91 / 0.07 | 92 / 0.13 | 88 / 0.20 | 95 / 0.36 | 90 / 2.05 | 91 / 2.93 | 94 / 5.20 | 90 / 2.04 |
| LiveCodeBench easy | 30 | 97 / 0.08 | 97 / 0.13 | 100 / 0.26 | 100 / 0.35 | 97 / 2.86 | 97 / 2.31 | 100 / 4.11 | 100 / 2.37 |
| LiveCodeBench medium | 40 | 68 / 0.51 | 68 / 0.69 | 45 / 0.86 | 75 / 3.00 | 48 / 21.12 | 80 / 10.73 | 78 / 18.72 | 92 / 9.82 |
| LiveCodeBench hard | 50 | 38 / 0.86 | 56 / 1.21 | 4 / 1.73 | 26 / 5.09 | 16 / 25.63 | 54 / 20.18 | 28 / 31.57 | 80 / 27.02 |
| MATH-500 easy | 40 | 100 / 0.08 | 100 / 0.16 | 98 / 0.04 | 100 / 0.25 | 100 / 0.95 | 100 / 2.11 | 100 / 3.61 | 100 / 2.02 |
| MATH-500 medium | 30 | 97 / 0.07 | 90 / 0.24 | 100 / 0.07 | 100 / 0.34 | 100 / 1.95 | 100 / 2.75 | 100 / 4.99 | 100 / 2.32 |
| MATH-500 hard | 80 | 85 / 0.22 | 90 / 0.44 | 89 / 0.15 | 96 / 0.95 | 95 / 3.10 | 99 / 5.36 | 98 / 6.78 | 100 / 4.36 |
| MMLU-Pro, calculation subjects | 59 | 80 / 0.14 | 90 / 0.13 | 86 / 0.27 | 93 / 0.62 | 90 / 2.10 | 95 / 2.13 | 95 / 6.12 | 95 / 3.05 |
| MMLU-Pro, recall subjects | 61 | 64 / 0.07 | 61 / 0.07 | 69 / 0.43 | 75 / 1.00 | 74 / 2.30 | 80 / 2.26 | 82 / 6.76 | 75 / 3.37 |
| GPQA Diamond | 50 | 58 / 0.34 | 68 / 0.33 | 42 / 0.90 | 66 / 3.14 | 62 / 6.10 | 88 / 5.15 | 86 / 15.67 | 84 / 8.13 |
| IFEval | 80 | 66 / 0.42 | 86 / 0.36 | 84 / 0.36 | 89 / 1.93 | 94 / 2.15 | 95 / 6.92 | 92 / 9.61 | 95 / 12.85 |
| WritingBench business | 60 | 2 / 0.43 | 7 / 1.29 | 38 / 1.55 | 20 / 3.82 | 22 / 8.49 | 40 / 18.06 | 53 / 20.67 | 48 / 53.99 |

What it shows:
- **Difficulty matters within a type:** on LiveCodeBench, easy problems are solved by almost every model, while hard ones range from 4% to 80%. MATH-500 shows the same pattern, less strongly. The complexity signal should move the choice within a type.
- **MMLU-Pro splits in two:** on calculation subjects, gpt-oss-120b passes 90% for almost nothing; on recall subjects, the cheap models pass only about 62%.
- **HumanEval+ and MBPP+ behave alike**, so they share one type.
- **Writing is where cheap models fail hardest:** 2–7% for gpt-oss.

The MMLU-Pro split is a judgment call made before looking at per-subject results: math, physics, chemistry, engineering, business, economics and computer science count as calculation-heavy; biology, health, history, law, philosophy, psychology and "other" count as recall. With only 8–9 sampled questions per subject, finer splits are not supported by the data.

## Task types

| Task type | Measured | Anchors (routing table) | Training rows (excluding the benchmark sample) | Difficulty label |
|---|---|---|---|---|
| `math_problem` | yes | MATH-500 | 350 from MATH-500; full MATH (~12,500) to download | MATH level 1–2 easy, 3 medium, 4–5 hard |
| `code_function` | yes | HumanEval+, MBPP+ | 124 + 338; full MBPP (~974) to download | – |
| `code_algorithmic` | yes | LiveCodeBench | **only 55** from v6; v1–v5 (~880) to download | LiveCodeBench easy / medium / hard |
| `science_expert` | yes | GPQA Diamond | 148; GPQA extended minus diamond (~348) to download | – |
| `knowledge_quantitative` | yes | MMLU-Pro calculation subjects | MMLU-Pro rows of those subjects (thousands) | – |
| `knowledge_recall` | yes | MMLU-Pro recall subjects, ARC-Challenge | MMLU-Pro rows, 1,142 ARC test; ARC train + validation (~1,418) to download | ARC easy, MMLU-Pro medium |
| `instruction_following` | yes | IFEval | 461 | – |
| `business_writing` | yes | WritingBench business | 129 | – |
| `summarization` | no | – | to choose | – |
| `translation` | no | – | to choose | – |
| `information_extraction` | no | – | to choose | – |
| `creative_writing` | no | – | to choose | – |
| `conversation` | no | – | to choose | – |
| `tool_calling` | no | – | BFCL v3, minus the 160 sampled | – |
| `other` | no | – | – | – |

The `description` of each type in the YAML doubles as the criterion text for Laya's Choice question, so it is written for real user prompts, not for benchmark formats.

## Decisions (2026-10-22)

- **Q19, unmeasured types:** summarization, translation, information extraction, creative writing, conversation and tool calling are separate types now, without a routing row. The router uses its fallback default for them until they are measured, but the classifier already learns them, so production traffic can be counted per type. Tool calling has BFCL data but no answers yet (deferred on 2026-10-12); a request that carries tools is also a direct signal.
- **Q20, ARC:** part of `knowledge_recall` as its easy end, not a separate "simple factual" type. Easy versus hard is the complexity signal's job.
- **Q21, the format trap:** benchmark items carry formats users rarely send, such as options A–J. Training uses both the benchmark rendering and a plain variant without options, answer-format lines or test-harness boilerplate, so the classifier learns the task and not the format.
- **Q22, location:** `config/task_types.yaml` with this page. `TopicCategory` in `src/lonn/schemas.py` stays as it is until the router is rebuilt.

## Open items

- **Training sources for unmeasured types** are still to be chosen, with licences checked. A classifier cannot learn a type it has never seen.
- **Sources marked `to_download`** need a pinned revision and a licence check, as in `docs/benchmark_sources.md`. LiveCodeBench v1–v5 matter most, because `code_algorithmic` has only 55 unused rows otherwise.
- **Class balance:** MMLU-Pro and MATH offer thousands of rows and IFEval and WritingBench a few hundred, so training needs per-type caps or weights.
- **Licences:** GPQA must never be committed or published in plain text, and LiveCodeBench is internal-only. The terms question in `docs/benchmark_sources.md` (training on model outputs) applies to the routing table too.
