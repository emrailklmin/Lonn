# Lambda sweep 0–100 with answer check (#19)

Task-type router with the shipped classifier (bge-small + linear, WildChat), complexity off, with answer checks and one value-based escalation (cascade).
Cross-fitted 5 folds x 10 repeats over 690 benchmark questions. No API calls. Cost is the total for all 690 questions.

Reproduce:

```bash
python3 scripts/benchmark/eval_task_router.py --name lambda-sweep-0-100-cascade --signals results/classifier_bakeoff/bge-small-linear-wildchat/benchmark_signals.jsonl --cascade --lambdas 0 5 10 15 20 30 40 50 60 70 80 90 100
python3 scripts/benchmark/eval_fixed_cascade.py --lambdas 0 20
```

| Configuration | Correct | Cost (USD) | Compared with always using Sonnet | Escalated |
|---|---:|---:|---:|---:|
| Lönn Router, λ = 0 | 89.0% | 6.90 | +0.9 points, −9% cost | 3% |
| Lönn Router, λ = 5 | 89.2% | 4.73 | +1.1 points, −38% cost | 3% |
| Lönn Router, λ = 10 | 89.6% | 3.91 | +1.5 points, −48% cost | 6% |
| Lönn Router, λ = 15 | 88.1% | 2.71 | +0.0 points, −64% cost | 8% |
| Lönn Router, λ = 20 | 85.8% | 1.58 | −2.4 points, −79% cost | 8% |
| Lönn Router, λ = 30 | 84.0% | 1.00 | −4.2 points, −87% cost | 9% |
| Lönn Router, λ = 40 | 82.3% | 0.82 | −5.8 points, −89% cost | 9% |
| Lönn Router, λ = 50 | 81.4% | 0.63 | −6.7 points, −92% cost | 9% |
| Lönn Router, λ = 60 | 81.2% | 0.53 | −7.0 points, −93% cost | 8% |
| Lönn Router, λ = 70 | 81.0% | 0.48 | −7.1 points, −94% cost | 7% |
| Lönn Router, λ = 80 | 80.7% | 0.44 | −7.4 points, −94% cost | 7% |
| Lönn Router, λ = 90 | 80.6% | 0.42 | −7.5 points, −94% cost | 7% |
| Lönn Router, λ = 100 | 80.3% | 0.39 | −7.9 points, −95% cost | 7% |
| Always Claude Sonnet 5.5, no answer check | 88.1% | 7.58 | – | – |
| Always Claude Sonnet 5.5 + answer check, λ = 20 | 88.3% | 7.66 | +0.2 points, +1% cost | 4.5% |
| Always Claude Sonnet 5.5 + answer check, λ = 0 | 89.4% | 8.47 | +1.3 points, +12% cost | 4.5% |
| Oracle | 94.1% | 1.08 | – | – |

With always-Sonnet, a failed check escalates to the best *other* model at accuracy − λ × cost (Sonnet itself is excluded).
At λ = 20 that is mostly glm-5.3-flash (66%) and gpt-oss-120b (29%); at λ = 0 it is muse-spark-1.3 (68%) and gemini-3.8-flash (32%).
Without answer check: see ../lambda-sweep-0-100/lambda_sweep.md.
