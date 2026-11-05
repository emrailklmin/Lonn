# Prototype 4.0: task aptitudes and benchmark replay

Prototype 4.0 keeps Prototype 3.0's classifier, measured routing table, model
pool, lambda, answer checks, and evaluation settings. It adds user-supplied
aptitude estimates for six task types. These numbers are **not measured pass
rates** and their calibration and original source are not established here.

## Routing data

- The eight measured task types and their benchmark accuracy and cost values
  remain unchanged in `config/routing_table.json`.
- The six supplied categories are defined once in `config/task_aptitudes.yaml`:
  summarization, translation, information extraction, creative writing,
  conversation, and tool calling. They use the model's existing global benchmark
  mean cost, since task-specific costs were not supplied.
- `other` is calculated for each model as the arithmetic mean of the eight
  measured task scores and the six supplied scores. Its cost also uses the global
  benchmark mean. The global row itself continues to average only measured tasks.
- `GET /models` exposes these 15 effective scores under `task_aptitudes`, with
  `source` (`measured`, `supplied`, or `derived`), sample size where measured,
  and cost provenance. The Streamlit model view reads from the same table. The
  older `topic_aptitudes` field remains for the legacy router and clients; it is
  not used by the current task router.

The router still weights task scores and costs by classifier probabilities and
maximizes `estimated_quality - lam * expected_cost_usd`. The API's historical
`expected_accuracy` field may now mix benchmark pass rates and supplied estimates;
it must not be interpreted as a calibrated probability of success. Tool-calling
scores do not add tool support to the text-only executor.

## Evaluation method

The original 690-question answer bank, classifier signals, answer checks,
graders, 8,192-token answer budget, and model pool are reused without new API
calls or model training. Each run uses the existing five folds, 10 repeats,
smoothing 5, lambda 20 headline, and the original lambda/complexity sweep.
No setting was chosen using the Prototype 4.0 results.

For each fold, the eight measured rows are built only from the training folds.
The six supplied values are then overlaid, and `other` is calculated using that
fold's eight measured scores and the six fixed estimates. The router scores the
held-out fold. The original global benchmark cost is computed from the training
fold and reused for supplied and derived categories. This avoids putting a held-
out answer into its own measured aptitude or derived `other` value.

The raw benchmark prompt files are not committed in this checkout. The replay
therefore uses question IDs and task/difficulty labels from the committed
Prototype 3.0 `task-true-utility/decisions.jsonl`, with dataset/category metadata
and grades from the committed answer bank. Saved classifier signals and answer
checks already contain the information required for replay; no prompt text is
regenerated. The recovered 690 IDs match the benchmark manifest and reproduce
the original first-fold assignments. The Prototype 3.0 shipped-classifier,
escalation and predicted-complexity runs were independently reproduced with
identical pass rates, costs and model-selection distributions.

## Results

All costs below are USD for the same 690 questions, averaged across repeats.
Savings are compared with always selecting Claude Sonnet 5.5, whose saved calls
cost $7.581904 in this benchmark. Cascade costs include the second model call.

| Configuration, lambda 20 | 3.0 pass | 4.0 pass | 3.0 cost | 4.0 cost | 3.0 savings | 4.0 savings |
|---|---:|---:|---:|---:|---:|---:|
| True task labels | 84.09% | 84.09% | $0.928335 | $0.928335 | 87.76% | 87.76% |
| True task labels + checks/escalation | 86.30% | 86.30% | $1.666475 | $1.666475 | 78.02% | 78.02% |
| **Shipped classifier** | **83.70%** | **83.35%** | **$0.907041** | **$0.867253** | **88.04%** | **88.56%** |
| **Shipped classifier + checks/escalation** | **85.75%** | **85.55%** | **$1.581869** | **$1.544106** | **79.14%** | **79.63%** |
| Predicted complexity | 83.06% | 82.96% | $1.077770 | $1.039976 | 85.78% | 86.28% |
| Predicted complexity + checks/escalation | 84.91% | 84.94% | $1.622259 | $1.580534 | 78.60% | 79.15% |
| Earlier BGE classifier | 83.99% | 83.55% | $0.917010 | $0.894343 | 87.91% | 88.20% |
| GTE linear classifier | 83.61% | 83.45% | $0.892324 | $0.863864 | 88.23% | 88.61% |
| GTE kNN classifier | 83.46% | 83.01% | $0.938601 | $0.846806 | 87.62% | 88.83% |
| Laya zero-shot classifier | 80.78% | 80.36% | $0.787035 | $0.708352 | 89.62% | 90.66% |

The API defaults to answer checks; the Streamlit routing lab does not perform
the API's check-and-escalate step. The predicted-complexity runs are retained
for comparison and are not the default. The true-label runs use the correct
dataset task type instead of the shipped classifier and are not live results.
All full run artifacts are under `results/router_eval/p4-*`; the original
Prototype 3.0 artifacts remain alongside them.

### Model selection distribution

Percentage of final selected models, averaged over cross-fitting repeats. In
the escalation run, the second model is counted when escalation succeeds.

| Model | 3.0 classifier | 4.0 classifier | 3.0 + escalation | 4.0 + escalation |
|---|---:|---:|---:|---:|
| Claude Sonnet 5.5 | 0.00% | 0.00% | 0.67% | 0.67% |
| Muse Spark 1.3 | 0.00% | 0.00% | 1.96% | 1.97% |
| Gemini 3.8 Flash | 20.58% | 20.33% | 20.83% | 20.59% |
| DeepSeek V4 Pro | 7.20% | 7.28% | 7.26% | 7.42% |
| DeepSeek V4.1 Flash | 42.97% | 39.41% | 44.81% | 41.32% |
| GLM 5.3 Flash | 9.17% | 14.43% | 4.99% | 9.86% |
| GPT-OSS 120B | 20.07% | 18.55% | 17.16% | 15.84% |
| GPT-OSS 20B | 0.00% | 0.00% | 2.33% | 2.33% |

Rounding can make a column differ slightly from 100%.

### Why the scores changed

The 690 questions have ground-truth labels only in the eight measured types.
When the evaluation supplies those exact labels, Prototype 4.0 is identical to
3.0, including with escalation. The shipped classifier, however, outputs a
distribution across all 15 types. Its saved signals place some weight on the
seven formerly unmeasured types for 560 of the 690 questions, averaging about
5.18% probability mass across the set. The new values therefore change some
model rankings even for questions whose ground-truth type was already measured.
In the first cross-fitting repeat, the shipped-classifier run selects a different
first model for 41 of 690 questions.

This replay does **not** measure summarization, translation, extraction,
creative-writing, conversation or tool-calling quality. The small differences
in overall pass rate and cost cannot establish that the supplied estimates
improve those tasks. The answer bank and these settings were also used during
development. A separate held-out evaluation representative of real traffic,
with graded answers for the newly covered categories and a fixed routing policy,
is needed to assess performance on genuinely unseen data.

## Reproducing the runs

From the repository root, with the existing requirements installed:

```bash
run_eval() {
  local name="$1"
  shift
  python3 scripts/benchmark/eval_task_router.py --name "replay-$name" \
    --saved-questions results/router_eval/task-true-utility/decisions.jsonl "$@"
}

run_eval task-true-utility
run_eval task-true-cascade --cascade
run_eval bge-wildchat --signals results/classifier_bakeoff/bge-small-linear-wildchat/benchmark_signals.jsonl
run_eval bge-wildchat-cascade --cascade --signals results/classifier_bakeoff/bge-small-linear-wildchat/benchmark_signals.jsonl
run_eval bge --signals results/classifier_bakeoff/bge-small-linear/benchmark_signals.jsonl
run_eval gte-linear --signals results/classifier_bakeoff/gte-base-linear/benchmark_signals.jsonl
run_eval gte-knn --signals results/classifier_bakeoff/gte-base-knn/benchmark_signals.jsonl
run_eval laya --signals results/classifier_bakeoff/laya-zeroshot/benchmark_signals.jsonl
run_eval complexity --use-complexity --levels-from-signals --signals results/classifier_bakeoff/bge-small-linear-wildchat-complexity/benchmark_signals.jsonl
run_eval complexity-cascade --cascade --use-complexity --levels-from-signals --signals results/classifier_bakeoff/bge-small-linear-wildchat-complexity/benchmark_signals.jsonl
```

Add `--legacy-global-fallback` to reproduce the corresponding 3.0 headline.
Each new run stores its configuration, supplied-aptitude file hash, decisions,
summary and model distribution; it also appends a row to `results/router_eval/runs.csv`.
The `git_dirty` flag in the saved runs records that the Prototype 4.0 changes
were uncommitted when the replay was run.

## Local verification note

The existing macOS code-check sandbox permits the virtualenv but denies the rest
of the home directory. In this checkout, the virtualenv's base Python standard
library is installed under the home directory, so live code-example checks can
fail to start Python. The replay is unaffected because it uses the previously
saved answer checks. The offline suite passed with a temporary copy of the same
interpreter outside the home directory; three live Laya tests were excluded to
avoid model downloads. The sandbox implementation was not changed in this branch.
