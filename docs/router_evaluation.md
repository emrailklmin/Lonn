# Router evaluation against the answer bank

Issue #18. `scripts/benchmark/eval_router.py` measures how good the router's choices are by **replaying** the 690 benchmark questions against the answer bank from #17, without calling any model.

## How it follows the architecture

The evaluation uses the existing code unchanged (`docs/Prototype_architecture.md`, sections 5 and 7). It only adds new files.

| Step | Production | Evaluation |
|---|---|---|
| Input | the client's `prompt` | the question's user message (`templates.user_message`), without the benchmark system prompt |
| Analysis | `ComplexityScorer.score_prompt(prompt)` | the same call, cached per question and scorer in `reports/router_eval_cache/` |
| Routing | `rank_models(signal, pool, alpha, limit=5)` | the same call; the pool is `load_model_pool()` restricted to models with saved answers |
| Execution | `execute(prompt, model)` | `ResultBank.lookup(prompt_id, model_id)`: the saved, graded answer |
| Fallback | next candidate **only if the provider call fails** | next candidate only if the saved pair is an error row; a wrong answer is final |
| Cost | estimated from list prices | the billed cost from the answer bank (the router still decides with list prices) |

## Running

```bash
python3 scripts/benchmark/eval_router.py --name stock-a05 --description "Stock Laya, alpha 0.5"
python3 scripts/benchmark/eval_router.py --name finetuned-a07 --scorer LonnAI/laya-lonn --alpha 0.7
python3 scripts/benchmark/eval_router.py --name topics --scorer true-topic --alpha-sweep
python3 scripts/benchmark/eval_router.py --name cheap-pool --models gpt-oss deepseek
python3 scripts/benchmark/eval_router.py --name stock-prod --max-tokens-equivalent 4096
```

- **`--name`:** required. A rerun with the same name replaces that run; a new name adds one.
- **`--description`:** free text describing the router.
- **`--scorer`:** which scorer the router uses.
  - `stock` is Laya's default checkpoint.
  - `LonnAI/laya-lonn`, or a local directory, is a fine-tuned checkpoint. It is set through the existing `LONN_LAYA_MODEL` variable.
  - `true-topic` uses the dataset's topic label as the topic, and its difficulty as complexity (easy 0.25, medium 0.5, hard 0.75). It is an ablation that needs no Laya.
- **`--models`:** ID substrings of the models the router may choose from. The default is every active pool model with saved answers.
- **`--alpha`:** the routing preference, default 0.5. `--alpha-sweep` adds alpha 0.0 to 1.0 and an SVG plot (`reports/router_eval/<name>/alpha_sweep.svg`).
- **`--max-tokens-equivalent`:** 8192 (as generated, the default) or 4096 (the production executor's default). With 4096, answers that needed more output tokens count as cut off (failed), and their cost is scaled to 4096 output tokens.

Laya (about 850 MB) is downloaded on first use unless `--scorer true-topic`. No API calls are made.

## Results (committed)

```
results/router_eval/
├── runs.csv                 one row per run: the leaderboard
└── <name>/
    ├── config.json          name, description, date, git commit, scorer, alpha, models, prices
    ├── decisions.jsonl      per question: signal, ranking, chosen model, fallbacks, correct, cost, latency
    └── summary.json         every policy overall and per category / dataset / difficulty, plus the alpha sweep
```

**Policies compared on the same questions:**
- **router:** the evaluated router.
- **oracle:** the cheapest correct model per question, i.e. the best achievable. If no model is right, the cheapest answer counts as a failure.
- **best fixed:** the single model with the most passes (ties go to the cheaper one).
- **cheapest:** the lowest list price.
- **random:** the expected value over the pool.
- **every single model.**
- **`quality_fallback_hypothetical`:** retry down the ranking until an answer is right, paying for every attempt. This is **not** what production does.

**Pass rules:** the same as in the answer bank. WritingBench passes at a judge mean of at least 7.

## Known differences from production

1. **System prompt:** the saved answers were produced with the benchmark's format system prompt (for example `Answer: X`); production sends only the user prompt.
2. **Output budget:** the answers were generated with up to 8192 output tokens, while the executor defaults to 4096. Use `--max-tokens-equivalent 4096` to compare.
3. **Temperature and seed:** the bank used temperature 0 and a fixed seed; the executor uses provider defaults.
4. **Tool calling (BFCL):** not in the bank, and the executor has no tool support.
5. **One error row:** Muse Spark × `mmlu_pro-066` is handled by the fallback rule.
