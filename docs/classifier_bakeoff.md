# Classifier bake-off

Issue #19, decision Q14: the task-type classifier is chosen by comparison, not up front, and Laya stays only if it wins. This page reports stage A: candidates that need no GPU.

```bash
python3 scripts/classifier_bakeoff.py        # ~45 min on an M3; models and embeddings go to the system temp folder
python3 scripts/benchmark/eval_task_router.py --name bakeoff-bge-small-linear \
    --signals results/classifier_bakeoff/bge-small-linear/benchmark_signals.jsonl
```

## Candidates

| Candidate | Model | Training |
|---|---|---|
| `laya-zeroshot` | stock Laya (ModernBERT-large, 421M) | none: one Choice question over the 15 task-type descriptions in `config/task_types.yaml` |
| `gte-base-linear` | `Alibaba-NLP/gte-modernbert-base` (149M, Apache-2.0), frozen | a linear layer on the embeddings, with class weights |
| `gte-base-knn` | same embeddings | the 15 nearest training prompts vote |
| `bge-small-linear` | `BAAI/bge-small-en-v1.5` (33M, MIT), frozen | a linear layer, with class weights |

All candidates are trained on the train split of `LonnAI/lonn-task-data` plus local GPQA (31,886 + 888 rows) and scored on its validation split (3,520 + 104 rows). Embedders read 512 tokens and keep the start and end of longer prompts. Laya has room for about 312 prompt tokens next to a 15-option question.

## Results (2026-10-22)

**Routing:** each candidate's predicted task-type distribution for the 690 benchmark questions drives the task router (λ 20, complexity off, cross-fitted as in `docs/task_router.md`).

| Candidate | Task-type accuracy | Macro-F1 | Plain variant | Prompts > 370 tokens | **Router pass** | **Router cost** | Latency (CPU, 1 prompt) |
|---|---:|---:|---:|---:|---:|---:|---|
| truth (upper bound) | 100% | 1.00 | | | **84.1%** | **0.93 USD** | |
| `bge-small-linear` | 80.0% | 0.79 | 78.3% | 85.9% | **84.0%** | 0.92 USD | 5 ms short, 34 ms at 512 tokens |
| `gte-base-linear` | 84.7% | 0.83 | 83.0% | 90.2% | 83.6% | 0.89 USD | 16 ms short, 1.4 s at 512 tokens* |
| `gte-base-knn` | 81.4% | 0.79 | 86.5% | 84.0% | 83.5% | 0.94 USD | as gte-base |
| `laya-zeroshot` | 31.1% | 0.25 | 32.7% | 44.4% | 80.8% | 0.79 USD | 260 ms (MPS) |
| current router (stock Laya, 4 topics) | | | | | 78.1% | 1.10 USD | |

\* ModernBERT hits a slow attention path on CPU for long inputs. It is probably fixable (eager attention or ONNX) but is not fixed here.

Per-type scores and the main confusions are in `results/classifier_bakeoff/<candidate>/metrics.json`, and routing runs are in `results/router_eval/bakeoff-*`.

## What it shows

- **Any trained classifier gets the router to the truth's level.** With a 33M-parameter model and a linear layer, the router passes 84.0% at 0.92 USD, against 84.1% at 0.93 USD with perfect task types. A better classifier has at most about 0.1 points left to gain on this benchmark. **The classifier is no longer the bottleneck.** What remains (84% against the oracle's 94%) is differences between questions within a task type.
- **Stock Laya does not know the task types:** 31% accuracy. It mixes up knowledge questions with expert science and conversation with code. Even so, the new router logic lifts it from 78.1% to 80.8% at lower cost. A fine-tuned Laya would likely land with the others, but there is little left for it to win on routing.
- **Long prompts are not a problem** for the 512-token embedders (86–90% on prompts over 370 tokens, as good as or better than short ones). Laya's 312-token room is a weakness here (44%).
- **Weak spots are the fuzzy Dolly categories:** summarization vs. information extraction, knowledge questions vs. conversation, and creative writing. These are mostly unmeasured types that route the same way today, so they cost little now. They matter once those types get measurements.

## Spot check on everyday prompts

The bake-off scores benchmark-style prompts, like the training data. Nine hand-written everyday prompts sent through the running API with the shipped classifier (2026-10-22) show where it breaks:

| Prompt | Classified as | Should be |
|---|---|---|
| "What is 17 * 23?" | instruction_following (0.90) | math_problem |
| "Given an array of n integers (n up to 2·10^5), count pairs … Read from stdin." | code_function (0.93) | code_algorithmic |
| "Write a business proposal for a coffee subscription startup …" | creative_writing (0.55) | business_writing |
| "Summarize this paragraph in two sentences: …" | instruction_following (0.68) | summarization |

Five of the nine were right (a quartic equation, a short Python function, a translation, a history question as conversation, a haiku). Short and casual prompts look little like the benchmark items the classifier was trained on, so **the 84% router result is optimistic for real traffic.** The next data step is realistic prompts, such as WildChat (ODC-BY) or hand-labelled examples, plus a held-out set of everyday prompts to measure this.

## Retrained with everyday prompts (2026-10-23)

The classifier was retrained with 2,971 labelled WildChat prompts (`docs/task_data.md`). It was scored on 300 held-out everyday prompts with gold labels, and **routing regret** compares the model the router picks against its pick with the gold task type:

| Classifier | Task type right | Same model as with gold | Expected accuracy lost | Extra cost per request |
|---|---:|---:|---:|---:|
| bge-small + linear, benchmark data only | 31.0% | 54.7% | 2.8 pts | +0.40 mUSD |
| **bge-small + linear, + WildChat** (shipped) | **51.3%** | **69.3%** | **1.7 pts** | −0.12 mUSD |

On the benchmark the router is unchanged: 83.7% at 0.91 USD, against 84.0% at 0.92 USD (`results/classifier_bakeoff/bge-small-linear-wildchat`). Validation accuracy fell from 80.0% to 77.4%, because the validation split now also holds the harder WildChat rows.

Up-weighting the WildChat rows (×3 to ×30; measured on the first 265 gold prompts) raised everyday task-type accuracy from 53.2% to at most 58.5%, but did not change model choice (69–70%) and lowered validation accuracy, so it is not used.

The main remaining confusions are among the open-ended types: knowledge questions, conversation, creative writing and business writing. Even the two gold labellers disagreed on 12% of prompts. These types partly route alike, which is why model choice agrees more often than the task type does.

## Recommendation

**Use `bge-small-linear` as the classifier for now.**
- It reaches the router's upper bound.
- It is the fastest (34 ms on CPU at full length) and smallest (33M parameters, about 130 MB).
- It has an MIT licence and needs no GPU.

`gte-base-linear` is the backup if task-type accuracy itself starts to matter, once its CPU latency on long prompts is fixed.

**Stage B (fine-tuning on a rented GPU) is not worth it yet for task types.** It becomes relevant for per-prompt difficulty ("would a cheap model fail?"), which is where the remaining gap is. Laya's yes/no question format is a natural fit there, so the case for Laya can be re-made then, with data.

## Disk

Embedding models and embeddings live in the system temp folder (`$TMPDIR/lonn-bakeoff`, 590 MB), not the Hugging Face cache. macOS clears it over time. To remove it now: `rm -rf "$TMPDIR/lonn-bakeoff"`.

`laya-zeroshot` downloaded stock Laya's multilingual checkpoint (644 MB) into the Hugging Face cache. Laya routes prompts that look non-English (some translation prompts) to it. The app needs it too, for non-English prompts, so it was left in place.
