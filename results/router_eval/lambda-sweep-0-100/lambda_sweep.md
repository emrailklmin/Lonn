# Lambda sweep 0–100 (#19)

Task-type router with the shipped classifier (bge-small + linear, WildChat), complexity off, no cascade.
Cross-fitted 5 folds x 10 repeats over 690 benchmark questions. No API calls.
Cost is the total billed cost in USD for all 690 questions. Utility = accuracy − λ × cost_usd per question.

Reproduce: `python3 scripts/benchmark/eval_task_router.py --name lambda-sweep-0-100 --signals results/classifier_bakeoff/bge-small-linear-wildchat/benchmark_signals.jsonl --lambdas 0 10 20 30 40 50 60 70 80 90 100`

| λ | Accuracy | Cost (USD) | Cost per question (USD) | Oracle gap closed | Top model share |
|---:|---:|---:|---:|---:|---:|
| 0 | 88.6% | 6.115 | 0.00886 | 66% | 48% |
| 10 | 87.0% | 3.119 | 0.00452 | 56% | 33% |
| 20 | 83.7% | 0.907 | 0.00131 | 35% | 43% |
| 30 | 82.7% | 0.812 | 0.00118 | 29% | 45% |
| 40 | 81.0% | 0.643 | 0.00093 | 18% | 45% |
| 50 | 80.1% | 0.484 | 0.00070 | 13% | 43% |
| 60 | 79.8% | 0.435 | 0.00063 | 10% | 42% |
| 70 | 79.4% | 0.409 | 0.00059 | 8% | 45% |
| 80 | 78.9% | 0.383 | 0.00056 | 5% | 48% |
| 90 | 78.5% | 0.363 | 0.00053 | 3% | 50% |
| 100 | 78.2% | 0.334 | 0.00048 | 1% | 51% |

Reference points:

- Oracle: 94.1% at 1.077 USD
- Best fixed model (anthropic/claude-sonnet-5.5): 88.1% at 7.582 USD
- Old router (laya-stock-a05): 78.1% at 1.099 USD
