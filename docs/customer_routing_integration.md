# Customer portal routing integration investigation

The production routing formula remains `quality = Σ p(task) × aptitude(model, task)`, `expected_cost = Σ p(task) × benchmark_cost(model, task)`, and `utility = quality − λ × expected_cost`. The classifier's complete distribution is preserved, without a top-category or keyword override.

## Findings recorded before implementation

The committed customer-portal snapshot loaded `RoutingTable.load()`. The current working source already used `load_runtime_table()` and the backend on port 8765 confirmed 15 dimensions per model. The development proxy configuration still pointed to port 8000, where an older process returned errors. Vite now targets the updated backend on 8765 by default; `LONN_DEMO_API_URL` allows an explicit alternative.

Runtime loading preserves the original eight measured rows, overlays the six supplied scores from `config/task_aptitudes.yaml`, and calculates `other` from each model's other 14 scores. The six supplied rows and `other` reuse the existing model-specific global benchmark cost. All 15 probabilities affect routing. The API previously evaluated all eight candidates but only exposed five; it now exposes the complete ranking while preserving the five-candidate fallback limit and one optional escalation.

The portal used literal λ values 0–1 with default 0.2; the original router's default is 20. There was no hidden conversion. This made cost penalties 100 times smaller at the demo default than at the benchmark default. Backend request and policy validation also rejected λ 20, and the startup migration replaced any stored organization λ above 1 with 0.2. Both are corrected: a benchmark-scale value such as 20 now reaches the router unchanged and persists across startup.

## Exact prompt: `What is 5+5?`

The real pinned, locally cached BGE classifier was run without network access or a complexity head. It returned `complexity=None` and these 15 probabilities (displayed as percentages):

| Task | Probability (%) |
|---|---:|
| math_problem | 19.793184159 |
| code_function | 0.111948836 |
| code_algorithmic | 0.000048354 |
| science_expert | 0.120583016 |
| knowledge_quantitative | 0.662576461 |
| knowledge_recall | 3.082808463 |
| instruction_following | 0.766265080 |
| business_writing | 0.024379024 |
| summarization | 0.028862541 |
| translation | 0.143020591 |
| information_extraction | 0.014165498 |
| creative_writing | 3.032944301 |
| conversation | 71.094567088 |
| tool_calling | 0.000123817 |
| other | 1.124522772 |

## Complete eight-model ranking

All eight models are eligible under the inspected Engineering application policy. Expected costs are probability-weighted **benchmark mean costs per request**, not a token forecast for this short prompt. Supplied task scores are estimates rather than measured correctness probabilities for this question.

| Rank at λ 20 | Model | Weighted aptitude | Expected cost (USD) | Utility at λ 20 | Rank at λ 0.2 | Utility at λ 0.2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | glm-5.3-flash | 0.874333719 | 0.000484480 | 0.864644122 | 5 | 0.874236823 |
| 2 | gpt-oss-120b | 0.835924387 | 0.000390839 | 0.828107611 | 7 | 0.835846219 |
| 3 | deepseek-v4.1-flash | 0.851933300 | 0.001547126 | 0.820990780 | 6 | 0.851623875 |
| 4 | deepseek-v4-pro | 0.887868857 | 0.004631992 | 0.795229020 | 4 | 0.886942459 |
| 5 | gemini-3.8-flash | 0.908594513 | 0.005963109 | 0.789332337 | 2 | 0.907401891 |
| 6 | gpt-oss-20b | 0.785981098 | 0.000240445 | 0.781172204 | 8 | 0.785933009 |
| 7 | claude-sonnet-5.5 | 0.935179047 | 0.010436859 | 0.726441876 | 1 | 0.933091675 |
| 8 | muse-spark-1.3 | 0.901831509 | 0.009812258 | 0.705586352 | 3 | 0.899869058 |

Claude also wins at the old UI maximum, λ 1. Its utility is 0.924742188, compared with GLM's 0.873849239. At literal λ 20, GLM wins with utility 0.864644122. This change comes entirely from the existing cost term.

`/route` performs no generation, fallback or escalation. If the first answer fails its existing checks, the prospective escalation target is Gemini at λ 0.2 and GPT-OSS 120B at λ 20. Provider failures trigger the existing fallback through at most the first five ranked candidates. A normal non-empty arithmetic response does not trigger an arithmetic correctness checker; the existing checker checks empty/truncated output and supported programming examples.

## Complexity

Production constructs `TaskClassifier()` without a complexity head and leaves `DEFAULT_USE_COMPLEXITY=False`. Merely enabling the routing flag while complexity remains `None` produces exactly the same ranking. As an explicitly counterfactual experiment, forcing the signal to easy (0) and enabling complexity still selects Claude at λ 0.2 and GLM at λ 20. No such forced score or heuristic was shipped.

The router therefore cannot distinguish trivial and difficult prompts with similar task probabilities and uses benchmark-average costs. The conversation-heavy classification also affects the result. These are limitations, but neither requires inventing a new routing equation. Existing predicted-complexity development results at λ 20 are worse than the no-complexity results: 83.06% at $1.077770 versus 83.70% at $0.907041 (no cascade); 84.91% at $1.622259 versus 85.75% at $1.581869 (with cascade). These are the older measured-only comparisons in `results/router_eval/complexity-predicted*/summary.json`, not a new estimate of unseen-data performance.

## Fixed-point benchmark replay

Reused Prototype 4.0's existing evaluator, original 690 questions reconstructed from saved decisions and answer-bank metadata, saved shipped-classifier benchmark signals, saved model answers/grades/checks, five training/test folds and ten repeated splits. Measured aptitudes were built from training folds only, fixed supplied aptitudes were overlaid within each fold, and `other` was calculated within each fold. Complexity remained off. Compared only the predeclared λ values 0.2, 1 and 20, with and without the existing check/escalation flow. No parameters were optimized and no provider calls were made during replay. Costs below are saved benchmark provider costs, without customer Lönn fees.

| Configuration | Literal λ | Correct (%) | Provider cost for 690 (USD) | Escalated (%) |
|---|---:|---:|---:|---:|
| utility | 0.2 | 89.04 | 5.653143 | 0.00 |
| utility | 1 | 89.20 | 5.335242 | 0.00 |
| utility | 20 | 83.35 | 0.867253 | 0.00 |
| cascade | 0.2 | 89.22 | 6.381693 | 2.43 |
| cascade | 1 | 89.30 | 5.982812 | 2.33 |
| cascade | 20 | 85.55 | 1.544106 | 8.58 |

This supports exposing the original λ units and retaining the original default 20 as an established quality/cost operating point. It does not establish an optimal λ for every customer. These are development benchmark results; cross-fitting prevents question-level table leakage, but the benchmark has already informed router development and is not a genuinely unseen production test.

## Implemented policy and request display

All demo λ controls and validation now accept literal 0–160, matching the existing sweep range, and defaults inherit the router's `DEFAULT_LAMBDA=20`. There is no nonlinear or multiplied conversion of user inputs. Existing explicitly stored values, including 0.2 and zero-valued locks, remain literal and are preserved. The inspected demo admin has a saved personal preference of 0.2 and Content Studio has a team lock of 0; those will still apply until explicitly changed. Settings includes a `20 · Benchmark default` preset.

Each generation response returns the effective `lam`, `lambda_source` and `lambda_locked` from the request's original policy snapshot. The customer app displays these beside each response and updates its controls from that server response. A stale connection menu or policy change during inference cannot relabel a completed request with a different λ. SDK results expose the same optional metadata. Security eligibility continues to precede routing, direct selection, fallback and escalation.

## Live verification

After restarting the updated backend and Vite proxy, the actual `/route` call through port 5173 returned eight ranked models and all 15 classifier probabilities. Omitting λ used the existing saved personal 0.2 and selected Claude. Explicit λ 20 selected GLM, with no input conversion. All seven pre-existing live records, account settings, personal preferences and team policies were checked byte-for-byte unchanged before the diagnostic generation.

One live `What is 5+5?` request was then submitted through the customer app's λ 20 preset, without changing any saved preference. GLM 5.3 Flash returned `5 + 5 = **10**`, with `finish_reason=stop`, answer check `ok`, one attempted model, zero fallbacks and no escalation. The receipt displayed `Effective λ 20 · request`. Usage was 19 input tokens and 12 output tokens; reported provider cost was $0.000003426 and total customer cost including the existing Lönn fee was $0.000016775294. The request retained all eight model estimates in telemetry. Request ID: `77e325c0-6704-49da-b7ea-29ba526b5534`.

Regression tests cover the full 15-category probability mixture for all eight candidates, supplied/derived scores, the unchanged five-candidate fallback cap, literal λ values up to 160, scope enforcement, request-time policy snapshots, and preservation of saved literal values. TypeScript checks and the production build pass.

The local diagnostic JSON files are in `reports/routing-integration/what-is-5-plus-5-before.json` and `reports/routing-integration/calibration.json`. The former includes full-precision model estimates and classifier/table/config hashes. The latter records the replay evaluator source hash and per-configuration selection distributions. Evaluation source: `prototype4.0:scripts/benchmark/eval_task_router.py`; signals: `results/classifier_bakeoff/bge-small-linear-wildchat/benchmark_signals.jsonl`; questions: `results/router_eval/task-true-cascade/decisions.jsonl` plus `results/benchmark/` metadata. These reuse the existing evaluation code rather than introducing another evaluator.
