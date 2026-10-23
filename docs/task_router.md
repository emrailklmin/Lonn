# Task-type router

Issue #19, branch `prototype3.0`. A router that decides from **task type**, **complexity** and **measured cost** instead of four topics, static aptitudes and list prices. The app and the API use it, with the task classifier from `docs/classifier_bakeoff.md`. The old router (`src/lonn/router.py`) remains in the code but is no longer used.

## How it decides

1. **Routing table** (`src/lonn/routing_table.py`, `config/routing_table.json`). For each task type (`config/task_types.yaml`, `docs/task_types.md`), each difficulty level and each pool model, it holds the accuracy and the mean billed cost per request from the answer bank. That cost includes reasoning tokens. A difficulty cell is shrunk towards its task type's average with 5 pseudo-answers, so a small cell cannot swing the choice on its own.
2. **Expected outcome per model** (`src/lonn/task_router.py`). The classifier's output is a `TaskSignal`: a probability per task type, plus a complexity in [0, 1] or `None`.
   - **Mix:** expected accuracy and cost are a probability-weighted mix of the task types' rows, so an uncertain classification blends smoothly.
   - **Unmeasured types:** probability on summarization, other and similar types uses the `global` row, the mean over all measured types.
   - **Complexity** (`use_complexity=True`, **off by default**, see the results): moves the row from the `easy` cell (0) through `medium` (0.5) to `hard` (1), linearly. `None`, or a task type without difficulty levels, uses the task type's average.
3. **Choice:** the model with the highest `accuracy − λ · cost_usd`. λ (default 20) is how many accuracy points one USD per request is worth; higher means cheaper. `mode='threshold'` instead picks the cheapest model with expected accuracy ≥ τ.

**Defaults (decision 2026-10-22): λ 20, complexity off.** That point beats the current router on both pass rate and cost. Complexity stays available and is switched on once a complexity signal improves the cross-fitted curve.

The router never calls an API. Models missing from the table are not ranked.

```bash
python3 scripts/benchmark/build_routing_table.py      # rebuild config/routing_table.json from the bank
python3 scripts/benchmark/eval_task_router.py --name task-true-utility       # defaults; sweep covers both complexity settings
python3 scripts/benchmark/eval_task_router.py --name task-true-complexity --use-complexity
python3 scripts/benchmark/eval_task_router.py --name task-true-threshold --mode threshold
python3 scripts/benchmark/answer_checks.py                      # once, then:
python3 scripts/benchmark/eval_task_router.py --name task-true-cascade --cascade
```

## Checking answers and escalating (cascade)

`/generate` checks each answer before returning it (`src/lonn/answer_check.py`):

| Check | Fails when | Correct when it fails (benchmark) |
|---|---|---:|
| `truncated` | the reply stopped at the token limit | 0.9% of 546 |
| `empty` | no final text | – |
| `failed_examples` | the prompt has worked examples of a programming task (AtCoder-style "Sample Input/Output", or LeetCode-style "Example: Input … Output …" with a `class Solution` starter), and the reply's last Python block does not reproduce them in the sandbox | 0.0% of 27 |

Answers that pass are correct 87.7% of the time. The examples read from the prompt match LiveCodeBench's public tests exactly for all 120 questions. The code check needs the macOS sandbox (`src/lonn/code_sandbox.py`, shared with the benchmark graders); elsewhere it is skipped.

**Escalation rule** (`escalation_target`): a failed answer is almost never right, so a second call is worth making when the best other model's own score, `accuracy − λ·cost`, is positive. At a high λ it often is not, and the first answer is returned as it is. There is at most one escalation. The response lists every attempt, its check result and its cost; `verify: false` turns the check off.

Replayed on the answer bank (`scripts/benchmark/answer_checks.py` stores every saved answer's check result in `results/benchmark_checks/answer_checks.jsonl`; `eval_task_router.py --cascade`, cross-fitted 5×5, Pareto frontier):

| Budget for 690 questions | 0.4 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Router | 79.1% | 79.9% | 81.6% | 83.9% | 84.6% | 85.3% | 86.3% | 87.3% |
| **Router + check and escalate** | **80.7%** | **81.4%** | **82.3%** | 83.8% | **85.7%** | **86.9%** | **88.2%** | **89.0%** |

- **Equal or better at every budget,** by up to +1.9 points.
- **6–8% of requests are escalated,** and only those take longer (two calls).
- At λ 20 the cascade passes 85.9% at 1.62 USD, so to keep today's cost, raise λ.

## Evaluation: cross-fitted, with the truth as classifier

The table comes from the 690 benchmark answers, so scoring on the same questions would let the router see the answers. `eval_task_router.py` therefore **cross-fits**:
1. Split the questions into 5 folds, stratified per dataset.
2. Build the table from 4 folds and score the held-out fold.
3. Repeat with 10 random splits and average.

`tests/test_eval_task_router.py` checks that a question is never scored by a table that contains it.

The signal is the truth, so these numbers are an upper bound for a classifier: the task type comes from the dataset, and complexity from the dataset's own difficulty label (MATH levels, LiveCodeBench easy/medium/hard, ARC easy and MMLU-Pro medium). Everything else follows production: the saved answer of the chosen model counts, and the next candidate is used only if the saved call failed.

## Results (2026-10-22, 690 questions, 8 models, output budget 8192)

| Policy | Pass | Cost | Most-chosen model |
|---|---:|---:|---|
| Current router (stock Laya, alpha 0.5) | 78.1% | 1.10 USD | flash, 94% |
| **Task router, λ 20, without complexity (default)** | **84.1%** | **0.93 USD** | 42% |
| Task router, λ 20, with complexity | 84.5% | 1.19 USD | flash, 47% |
| Task router, λ 30, with complexity | 82.9% | 0.90 USD | 42% |
| Task router, λ 10, without complexity | 87.3% | 2.88 USD | 35% |
| Task router, λ 80, with complexity | 78.9% | 0.40 USD | 41% |
| Best fixed model (Claude Sonnet 5.5) | 88.1% | 7.58 USD | 100% |
| Oracle (cheapest correct per question) | 94.1% | 1.08 USD | |

The full sweep (λ 0–160, with and without complexity) is in `results/router_eval/task-true-utility/summary.json`; the plots are in `reports/router_eval/task-true-utility/` (not committed).

What it shows:
- **The routing logic was the main bottleneck.** With the same eight models and the same answers, the task router beats the current router at lower cost: 84.1% at 0.93 USD against 78.1% at 1.10 USD, which closes about 37% of the gap to the oracle. It reaches 78.9%, the current router's level, for 0.40 USD, about a third of the cost.
- **No collapse:** no model gets more than about 50% of the choices at any λ, against 94% today.
- **Difficulty labels do not help yet.** With and without complexity, the curves overlap: at about 0.9 USD, 82.9% with and 83.4–84.1% without. Shrinkage does not change this (0, 5 and 20 pseudo-answers give the same picture). The datasets' coarse labels may simply not separate the questions the cheap models fail on; more questions per cell may also be needed. So complexity stays in the design (decision Q16) but is off by default until it earns its place. The per-prompt "would a cheap model fail?" signal may do better than these labels.
- **What remains** (about 84% against the oracle's 94%) is variation between questions inside a task type. Only per-prompt information can capture it.

## Known limits

- These are upper bounds: a trained classifier will make mistakes on task type and complexity.
- The same differences from production as in `docs/router_evaluation.md` apply: the format system prompt, the output budget (8192 here), temperature and seed. Costs are billed costs at an 8192-token output budget.
- The table has 30–150 questions per task type, so accuracy differences of a few points between models are noise.
