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
   python3 legacy/examples/scorer_demo.py
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

45 prompts the model never trained on, scored with each version's own code:

| | Prototype (stock Laya, 1-5 question) | Stock Laya, yes/no question | Fine-tuned |
|---|---:|---:|---:|
| Topic accuracy | 67% | 67% | 76% |
| General knowledge correct | 1 of 15 | 1 of 15 | 9 of 15 |
| Creative writing correct | 17 of 17 | 17 of 17 | 13 of 17 |
| Complexity, separating hard prompts (AUC) | 0.77 | 0.92 | 0.89 |
| Complexity, rank correlation with labels | 0.18 | 0.32 | 0.31 |
| Speed per prompt (Apple-silicon Mac) | 134 ms | 133 ms | 143 ms |

The topic gain comes from general-knowledge prompts (summaries, factual
questions), which stock Laya almost always labelled creative writing. The
fine-tuned model gives up some creative-writing prompts in exchange.

Only 4 of the 45 prompts are labelled hard (3-5), so the AUC figures are rough.
Across all 228 pilot prompts, which is fair for the two untrained versions only,
the yes/no question also beats the 1-5 question: AUC 0.81 versus 0.72, rank
correlation 0.37 versus 0.27.

### New and moved files

| File | Purpose |
|---|---|
| `legacy/scripts/finetune_laya.py` | Trains the model on labelled prompts and reports before/after results |
| `legacy/scripts/push_laya.py` | Uploads a trained model to a private Hugging Face repo |
| `legacy/data/labels/pilot.jsonl` | 228 labelled training prompts |
| `legacy/data/labels/pilot_spotcheck.md` | 30 of those prompts for a human to check the labels |
| `legacy/docs/labeling_guide.md` | Rules for labelling topic and complexity |
| `legacy/examples/scorer_demo.py` | Try the scorer interactively (was `classify_prompt.py`) |
| `legacy/examples/scorer_benchmark.py` | Measure the scorer's speed on your machine |

`tests/test_complexity.py` covers the new complexity question, long-prompt
shortening and `LONN_LAYA_MODEL`. It also fixes 12 tests that failed on the
prototype branch.

## Retraining

```bash
python3 legacy/scripts/finetune_laya.py
python3 legacy/scripts/push_laya.py --repo LonnAI/laya-lonn
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
- The labels were written by Claude following `legacy/docs/labeling_guide.md`. The
  human spot check in `legacy/data/labels/pilot_spotcheck.md` is not finished yet.
- Most prompts are labelled complexity 2 and only one is labelled 5. Training
  complexity on these labels does not improve it; more prompts labelled 3-5,
  or measured "did the cheap model fail?" results, should.
