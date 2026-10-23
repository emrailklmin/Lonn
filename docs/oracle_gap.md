# Closing the gap to the oracle

Issue #19, 2026-10-23. The task router passes 84.1% of the 690 benchmark questions at 0.93 USD. The oracle, which picks the cheapest model that answers each question correctly, passes 94.1% at 1.08 USD. This page records which strategies were tried against that gap, all replayed on the answer bank: no paid calls, task router cross-fitted (`docs/task_router.md`), true task types as the signal.

## Where the gap is

| Dataset | Router | Oracle | Questions lost | Share of gap | Router spend | Oracle spend |
|---|---:|---:|---:|---:|---:|---:|
| LiveCodeBench | 70.0% | 91.7% | 26 | 37% | 0.09 USD | 0.42 USD |
| WritingBench | 38.3% | 75.0% | 22 | 31% | 0.09 USD | 0.48 USD |
| MMLU-Pro | 84.2% | 93.3% | 11 | 16% | 0.16 USD | 0.04 USD |
| GPQA, IFEval, MATH-500 | | | 3 each | 4% each | | |
| ARC, MBPP+, HumanEval+ | | | 0–1 | ≤ 1% | | |

**Two task types hold 68% of the gap.** The router spends little on them, while the oracle spends about five times as much.

## Strategies tried

### 1. Complexity (difficulty within a task type)

The table gets an easy, medium and hard row per task type, and the router uses the prompt's difficulty. The table below shows the best pass rate at equal cost, **with difficulty known perfectly**, which is an upper bound:

| Difficulty signal | 0.6 USD | 0.9 USD | 1.2 USD | 2.0 USD |
|---|---:|---:|---:|---:|
| none | 80.5% | 83.4% | 84.0% | 85.5% |
| how many of the 8 models solve it | 81.9% | 83.2% | 84.4% | 85.9% |
| how many of the 3 cheapest models solve it | 82.8% | 84.4% | 85.0% | 86.7% |

- **At most +1–2 points, or about 25% lower cost at equal quality.**
- **Prompt length predicts difficulty only for algorithmic code.** Short LiveCodeBench problems are solved by 6.9 of 8 models, long ones by 3.0. For the other types it says almost nothing, and as a routing signal it does not help.
- **The datasets' own levels (MATH, LiveCodeBench) do not help either.**

### 2. Agreement cascade (multiple choice and maths)

Two cheap models answer. If their final answers agree (maths compared with math-verify), that answer is returned; otherwise a strong model answers.

- The best pair (gpt-oss-120b + glm-5.3-flash, escalating to Gemini 3.8 Flash) passes **84.9% at 1.07 USD**, against 83.9% at 0.93 USD.
- That is about +1 point. The router already passes 91.7% of these questions.

### 3. Self-check cascade (all types)

The router's choice answers. If the reply is cut off at the token limit, has no final answer, or (for LiveCodeBench) **fails the example tests shown in the prompt**, the router escalates once to a stronger model, chosen at half the router's λ.

How reliable the example tests are as a check:
- **LiveCodeBench:** code that passes the public tests is correct 95.0% of the time, and the tests catch 90.9% of wrong answers.
- **MBPP+:** the single example assert catches almost nothing.

Pareto frontier, 5 folds × 5 repeats:

| Budget for 690 questions | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 | 2.5 USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Router | 79.9% | 81.6% | 83.8% | 84.1% | 84.4% | 85.3% | 86.1% |
| Self-check + code tests | **81.0%** | **82.3%** | 82.9% | 83.6% | 84.3% | **86.1%** | **88.7%** |

- **It helps at low and high budgets, by +1 to +2.6 points.** For example, 90% costs 3.75 USD instead of more than 5 USD.
- **In the middle it adds nothing yet.** The cascade's escalation steps are coarse there; a finer escalation policy may close that.
- **Most of the gain is on writing**, where truncated answers are the main failure, **and on algorithmic code.**

## Conclusions

1. **Most of the oracle's lead cannot be predicted from the prompt.** The oracle knows which specific model happened to be right. Task type and difficulty together explain only a few points of it.
2. **What does help is checking the answer before returning it:** truncation, a missing final answer, example tests for code, and agreement between models. Each adds +1–3 points at equal cost, but adds latency (a second call when the check fails) and, for code, needs a sandbox in production.
3. **Writing is the largest single gap.** Even the best model passes only 53% of WritingBench, and many failures are answers cut off at 8,192 tokens. A larger output budget for writing, or a cheap judge before returning, could matter more there than routing; both need paid calls to test.
4. **A better task classifier does not help on the benchmark,** where it is already at the upper bound. It does help on everyday prompts (`docs/classifier_bakeoff.md`).

## Built: check and escalate

The self-check cascade is now part of the router and the API (`docs/task_router.md`). It uses only checks that production can run on any prompt: cut off, empty, and code that fails the prompt's own worked examples. It dropped the benchmark-only "no `Answer:` line" check. Escalation follows a value rule: a second call is made only when it is worth its expected cost at the request's λ. The cascade is equal to or better than the router at every budget, by up to +1.9 points, and escalates 6–8% of requests.
