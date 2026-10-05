# Fine-tuned Laya classifier

The `laya-finetune` branch adds a fine-tuned version of the Laya prompt classifier
and changes how complexity is scored. This page covers what changed, how to use
the model, and how to retrain it.

## Using the fine-tuned model

The model is in a private Hugging Face repo:
[LonnAI/laya-lonn](https://huggingface.co/LonnAI/laya-lonn).

1. Get access: ask an admin of the [LonnAI](https://huggingface.co/LonnAI)
   organization to add you as a member (read access is enough).
2. Check out this branch and install dependencies (see the README's Setup).
3. Log in to Hugging Face once. The browser option is easiest:

   ```bash
   .venv/bin/hf auth login
   ```

4. Add this line to your `.env`:

   ```text
   LONN_LAYA_MODEL=LonnAI/laya-lonn
   ```

5. Check it works. The first run downloads the model (about 1.6 GB):

   ```bash
   python3 examples/scorer_demo.py
   ```

Without `LONN_LAYA_MODEL`, everything still works with the stock Laya model.
Only English prompts use the fine-tuned model; Laya sends other languages to
its unchanged multilingual model.

The model only works as intended with this branch's question wording in
`src/lonn/complexity.py`. Do not use it with the older `prototype` code.

## What changed from the prototype branch

### Behaviour of `ComplexityScorer`

| | Before | Now |
|---|---|---|
| Topic | Stock Laya | Fine-tuned Laya when `LONN_LAYA_MODEL` is set |
| Complexity question | "How difficult is this task?" with five levels (trivial to very hard) | Yes/no: "Is this task hard for an AI assistant to do well?" |
| Complexity value | Average of the five levels, scaled to 0-1 | Laya's confidence that the answer is yes, 0-1 |
| Long prompts | Only the start was read (Laya stops at about 370 prompt tokens) | Start and end are kept, middle dropped |

`score_prompt` still returns the same `ComplexityResult`: topic probabilities
that sum to 1, and complexity between 0 and 1. Calling code does not change.

### Reading the complexity value

- Higher means harder. On the pilot prompts, most scores fall between about
  0.15 and 0.65. For example, "What is the capital of France?" scores about
  0.3 and a proof question about 0.45.
- Treat it as a ranking, not as a probability that a model fails. A value of
  0.8 is unusually hard, but it does not mean a cheap model fails 80% of the time.
- The router (`src/lonn/router.py`) does not use complexity yet. A cut-off such
  as "above X, use a stronger model" still needs to be added, and its value
  should be chosen from labelled data or evaluation results.

Why the change: the five-level question put almost every prompt at level 1 or
2, so it barely separated easy from hard prompts. The yes/no question separates
the prompts labelled 3-5 from easier ones much better (AUC about 0.9 versus
0.72, where 0.5 is chance). Wordings phrased the other way round ("can a small
model do it?") gave inverted answers, so keep it positive.

### Results on held-out prompts

45 prompts the model never trained on:

| | Stock Laya | Fine-tuned |
|---|---:|---:|
| Topic accuracy | 67% | 76% |
| Complexity, separating hard prompts (AUC) | 0.92 | 0.89 |

The topic gain comes mostly from general-knowledge prompts (summaries, factual
questions), which stock Laya usually labelled creative writing. The complexity
difference is within noise; the test set has only about 10 hard prompts.

### New and moved files

| File | Purpose |
|---|---|
| `scripts/finetune_laya.py` | Trains the model on labelled prompts and reports before/after results |
| `scripts/push_laya.py` | Uploads a trained model to a private Hugging Face repo |
| `data/labels/pilot.jsonl` | 228 labelled training prompts |
| `data/labels/pilot_spotcheck.md` | 30 of those prompts for a human to check the labels |
| `docs/labeling_guide.md` | Rules for labelling topic and complexity |
| `examples/scorer_demo.py` | Try the scorer interactively (was `classify_prompt.py`) |
| `examples/scorer_benchmark.py` | Measure the scorer's speed on your machine |

`tests/test_complexity.py` covers the new complexity question, long-prompt
shortening and `LONN_LAYA_MODEL`. It also fixes 12 tests that failed on the
prototype branch.

## Retraining

```bash
python3 scripts/finetune_laya.py
python3 scripts/push_laya.py --repo LonnAI/laya-lonn
```

- Training takes about 15-20 minutes on an Apple-silicon Mac and needs several
  GB of free memory, so close other heavy apps first.
- Only the small decision part of Laya is trained; its 400M-parameter text
  encoder stays frozen. 20% of prompts are held out and never trained on.
- The script prints stock versus fine-tuned results and writes them, with a
  model card, into `models/laya-lonn/` (not committed to git).
- Retrain after changing the question wording in `src/lonn/complexity.py`.

## Limitations

- The training set is small: 228 prompts, mostly real chat prompts from
  WildChat plus public datasets (each row records its source).
- The labels were written by Claude following `docs/labeling_guide.md`. The
  human spot check in `data/labels/pilot_spotcheck.md` is not finished yet.
- Most prompts are labelled complexity 2 and only one is labelled 5. Training
  complexity on these labels does not improve it; more prompts labelled 3-5,
  or measured "did the cheap model fail?" results, should.
