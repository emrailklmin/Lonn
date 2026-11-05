# Lambda sweep 0–100 with answer check (#19)

Task-type router with the shipped classifier (bge-small + linear, WildChat), complexity off, with answer checks and one value-based escalation (cascade).
Cross-fitted 5 folds x 10 repeats over 690 benchmark questions. No API calls. Cost is the total for all 690 questions.

Reproduce: `python3 scripts/benchmark/eval_task_router.py --name lambda-sweep-0-100-cascade --signals results/classifier_bakeoff/bge-small-linear-wildchat/benchmark_signals.jsonl --cascade --lambdas 0 10 20 30 40 50 60 70 80 90 100`

| λ | Correct | Cost (USD) | Compared with always using Sonnet | Escalated |
|---:|---:|---:|---:|---:|
| 0 | 89.0% | 6.90 | +0.9 points, −9% cost | 3% |
| 10 | 89.6% | 3.91 | +1.5 points, −48% cost | 6% |
| 20 | 85.8% | 1.58 | −2.4 points, −79% cost | 8% |
| 30 | 84.0% | 1.00 | −4.2 points, −87% cost | 9% |
| 40 | 82.3% | 0.82 | −5.8 points, −89% cost | 9% |
| 50 | 81.4% | 0.63 | −6.7 points, −92% cost | 9% |
| 60 | 81.2% | 0.53 | −7.0 points, −93% cost | 8% |
| 70 | 81.0% | 0.48 | −7.1 points, −94% cost | 7% |
| 80 | 80.7% | 0.44 | −7.4 points, −94% cost | 7% |
| 90 | 80.6% | 0.42 | −7.5 points, −94% cost | 7% |
| 100 | 80.3% | 0.39 | −7.9 points, −95% cost | 7% |

Always anthropic/claude-sonnet-5.5: 88.1% at 7.58 USD. Oracle: 94.1% at 1.08 USD.
Without answer check: see ../lambda-sweep-0-100/lambda_sweep.md.
