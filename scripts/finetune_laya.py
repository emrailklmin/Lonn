"""Fine-tune Laya's decision head on Lonn's labelled prompts.

Usage:
    python3 scripts/finetune_laya.py
    python3 scripts/finetune_laya.py --data data/labels/pilot.jsonl --out models/laya-lonn

Only the English checkpoint is trained: Laya's router sends non-English prompts to
its multilingual checkpoint, so those are skipped. The ModernBERT encoder is frozen
and only the decision head (head layers, type embedding, option scorer) is updated;
with a few hundred labels, training the 400M-parameter encoder would overfit.

A stratified share of prompts (by main topic, fixed by --seed) is held out and never
trained on. Base and fine-tuned models are both scored on it through
`ComplexityScorer`, so prompts are shortened and answers parsed exactly as in the app,
and the results are written to <out>/metrics.json and README.md.

Complexity is a yes/no "is this task hard?" question, trained on the labelled
level mapped to a probability, (level - 1) / 4. On the pilot labels this kept
complexity close to base Laya (AUC 0.89 vs 0.92) while topic accuracy rose from
67% to 76%. `--complexity-target base` instead trains complexity towards base
Laya's own answers; that held complexity but topic fell to 64%.
The output directory is a complete Laya checkpoint: `laya.load(<out>)` and
`Router(models={"english": <out or Hub repo>})` both load it.

Relies on Laya internals (`Agent._encode_state`, `DecisionModel` layout), so the
laya version is pinned in requirements.txt.
"""

import argparse
from collections import Counter, defaultdict
import gc
from datetime import datetime, timezone
import hashlib
import json
from math import fsum
import os
from pathlib import Path
import random
import shutil
import sys
import warnings

os.environ.setdefault("USE_TF", "0")  # Must precede the transformers import inside laya.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

import laya
from laya.agent import Agent
from laya.common import QTYPES, temp_bucket
from laya.revisions import snapshot_revision
from laya.router import BUNDLE_REPO, Router

from lonn.complexity import QUESTIONS, ComplexityScorer

QUESTION_IDS = list(QUESTIONS)
INTERNAL = {qid: Agent._to_internal(q) for qid, q in QUESTIONS.items()}
CATEGORY_KEYS = list(QUESTIONS["category"]["criteria"])
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
MAX_LEVEL = 5  # docs/labeling_guide.md scores complexity 1-5.
HARD_LEVEL = 3  # Labels at or above this count as hard when computing AUC.


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def main_topic(example):
    return max(example["topics"], key=example["topics"].get)


def load_examples(path):
    """Read labelled prompts, keeping those Laya routes to the English checkpoint."""
    router = Router(preload=False)  # Routing only; no checkpoint is loaded.
    examples, skipped = [], 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            unknown = set(row["topics"]) - set(CATEGORY_KEYS)
            if unknown:
                raise ValueError(f"Unknown topics {sorted(unknown)} in {path}")
            if not 1 <= row["complexity"] <= MAX_LEVEL:
                raise ValueError(f"Complexity {row['complexity']!r} outside 1-{MAX_LEVEL}")
            if router.route(row["prompt"], QUESTIONS).model != "english":
                skipped += 1
                continue
            examples.append(row)
    return examples, skipped


def stratified_split(examples, fraction, rng):
    """Hold out `fraction` of each main topic, so small topics are represented."""
    by_topic = defaultdict(list)
    for example in examples:
        by_topic[main_topic(example)].append(example)
    kept, held_out = [], []
    for topic in sorted(by_topic):
        group = by_topic[topic][:]
        rng.shuffle(group)
        n = round(len(group) * fraction)
        held_out += group[:n]
        kept += group[n:]
    return kept, held_out


def targets(example):
    """Soft targets in option order: topic distribution, and [not hard, hard]."""
    total = fsum(example["topics"].values())
    category = [example["topics"].get(key, 0.0) / total for key in CATEGORY_KEYS]
    hard = (example["complexity"] - 1) / (MAX_LEVEL - 1)
    return {"category": category, "complexity": [1.0 - hard, hard]}


def fingerprint(examples):
    digest = hashlib.sha1()
    for prompt in sorted(e["prompt"] for e in examples):
        digest.update(prompt.encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode(scorer, agent, examples):
    """Run the frozen encoder once per (prompt, question) row and keep its output.

    Prompts are shortened as `ComplexityScorer` shortens them at runtime. Each row
    stores what the head needs: encoder states, marker positions, question type, the
    runtime temperature for its option count, and the soft target.
    """
    rows = []
    for example in examples:
        prompt = scorer._fit(example["prompt"], "english", QUESTIONS)
        items = agent._encode_state(prompt, QUESTION_IDS, INTERNAL)
        goal = targets(example)
        for qid, item in zip(QUESTION_IDS, items):
            ids = torch.tensor([item["ids"]], device=agent.device)
            hidden = agent.model.encoder(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
            k = len(item["markers"])
            qtype = item["qtype"]
            rows.append({
                "hidden": hidden[0].to("cpu", torch.float16),
                "markers": item["markers"],
                "qtype": qtype,
                "temperature": agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype]),
                "target": goal[qid],
            })
    return rows


def collate(rows, device):
    n, length = len(rows), max(len(r["hidden"]) for r in rows)
    k = max(len(r["markers"]) for r in rows)
    hidden = torch.zeros(n, length, rows[0]["hidden"].shape[-1])
    mask = torch.zeros(n, length, dtype=torch.long)
    marker_pos = torch.zeros(n, k, dtype=torch.long)
    marker_mask = torch.zeros(n, k, dtype=torch.bool)
    target = torch.zeros(n, k)
    for i, row in enumerate(rows):
        hidden[i, : len(row["hidden"])] = row["hidden"].float()
        mask[i, : len(row["hidden"])] = 1
        marker_pos[i, : len(row["markers"])] = torch.tensor(row["markers"])
        marker_mask[i, : len(row["markers"])] = True
        target[i, : len(row["target"])] = torch.tensor(row["target"])
    batch = {
        "hidden": hidden, "attention_mask": mask, "marker_pos": marker_pos, "marker_mask": marker_mask,
        "qtype": torch.tensor([r["qtype"] for r in rows]),
        "temperature": torch.tensor([r["temperature"] for r in rows]),
        "target": target,
    }
    return {name: value.to(device) for name, value in batch.items()}


def head_logits(model, batch):
    """`DecisionModel.forward` from after the encoder, without the unused act head."""
    h = batch["hidden"] + model.type_emb(batch["qtype"])[:, None, :]
    pad = ~batch["attention_mask"].bool()
    for layer in model.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    idx = batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    logits = model.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
    return logits.masked_fill(~batch["marker_mask"], -1e4)


def batch_loss(model, batch):
    """Soft-target cross-entropy on temperature-scaled logits, as Laya scores at runtime."""
    logits = head_logits(model, batch) / batch["temperature"][:, None]
    return -(batch["target"] * F.log_softmax(logits, -1)).sum(-1).mean()


@torch.no_grad()
def anchor_complexity(model, rows, device):
    """Replace complexity targets with the untrained model's own answers.

    Topic and complexity share one head, so training topic alone would let the
    complexity answer drift. Training complexity towards the base model's
    probabilities keeps it where it was while the head learns topic.
    """
    model.eval()
    for row in rows:
        if row["qtype"] != QTYPES["noul"]:
            continue
        batch = collate([row], device)
        logits = head_logits(model, batch) / batch["temperature"][:, None]
        row["target"] = F.softmax(logits[0, : len(row["markers"])], -1).tolist()


def trainable_parameters(model):
    modules = (model.head, model.type_emb, model.scorer)
    for parameter in model.parameters():
        parameter.requires_grad = False
    params = [p for module in modules for p in module.parameters()]
    for parameter in params:
        parameter.requires_grad = True
    return modules, params


@torch.no_grad()
def mean_loss(model, rows, device, batch_size):
    losses = [batch_loss(model, collate(rows[i:i + batch_size], device)).item() * len(rows[i:i + batch_size])
              for i in range(0, len(rows), batch_size)]
    return sum(losses) / len(rows)


def head_state(model):
    """CPU copy of everything but the frozen encoder, which is most of the memory."""
    return {name: tensor.detach().to("cpu", copy=True)
            for name, tensor in model.state_dict().items() if not name.startswith("encoder.")}


def train(agent, train_rows, val_rows, args, rng):
    """Train the head, keeping the weights with the lowest validation loss."""
    model, device = agent.model, agent.device
    modules, params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    model.eval()
    best_loss = mean_loss(model, val_rows, device, args.batch_size)
    best_state = head_state(model)
    history = [{"epoch": 0, "val_loss": round(best_loss, 4)}]
    print(f"epoch  0  val loss {best_loss:.4f}")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        for module in modules:
            module.train()  # Dropout on in the head only; the encoder is not run here.
        order = train_rows[:]
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            loss = batch_loss(model, collate(order[i:i + args.batch_size], device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += loss.item() * len(order[i:i + args.batch_size])
        model.eval()
        val = mean_loss(model, val_rows, device, args.batch_size)
        history.append({"epoch": epoch, "train_loss": round(total / len(order), 4), "val_loss": round(val, 4)})
        print(f"epoch {epoch:2d}  train loss {total / len(order):.4f}  val loss {val:.4f}")
        if val < best_loss - 1e-4:
            best_loss, best_state, stale = val, head_state(model), 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"stopping: no validation improvement for {args.patience} epochs")
                break
    model.load_state_dict(best_state, strict=False)
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    return history, best_epoch


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def ranks(values):
    """Ranks with ties averaged, for rank correlation and AUC."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values))
    result[order] = np.arange(len(values))
    for value in np.unique(values):
        tied = values == value
        result[tied] = result[tied].mean()
    return result


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def auc(scores, positive):
    """Chance that a random positive scores above a random negative."""
    r, positive = ranks(scores), np.asarray(positive, dtype=bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if not n_pos or not n_neg:
        return float("nan")
    return float((r[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def evaluate(scorer, examples):
    """Score held-out prompts with `ComplexityScorer`, exactly as the app does.

    Complexity is judged by ranking, since the router compares it against a
    threshold: Spearman correlation with the labelled level, and AUC for telling
    labels HARD_LEVEL-5 apart from easier ones.
    """
    topic_hits, scores, levels = 0, [], []
    confusion = Counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for example in examples:
            result = scorer.score_prompt(example["prompt"])
            predicted_topic = max(result.topic_scores, key=result.topic_scores.get).value
            topic_hits += predicted_topic == main_topic(example)
            confusion[f"{main_topic(example)} -> {predicted_topic}"] += 1
            scores.append(result.complexity)
            levels.append(example["complexity"])
    levels = np.array(levels)
    return {
        "prompts": len(examples),
        "topic_accuracy": round(topic_hits / len(examples), 4),
        "complexity_spearman": round(spearman(scores, levels), 4),
        "complexity_auc_hard": round(auc(scores, levels >= HARD_LEVEL), 4),
        "complexity_p10_p90": [round(float(v), 4) for v in np.percentile(scores, [10, 90])],
        "topic_confusion": dict(sorted(confusion.items())),
    }


def length_baseline(tok, examples):
    """Rank quality of prompt length alone, the bar complexity should clear."""
    lengths = [len(tok(e["prompt"], add_special_tokens=False)["input_ids"]) for e in examples]
    levels = np.array([e["complexity"] for e in examples])
    return {"complexity_spearman": round(spearman(lengths, levels), 4),
            "complexity_auc_hard": round(auc(lengths, levels >= HARD_LEVEL), 4)}


def scorer_for(checkpoint, device):
    """A `ComplexityScorer` whose English checkpoint is `checkpoint`."""
    return ComplexityScorer(Router(models={"english": str(checkpoint)}, device=device, preload=False))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(agent, base_dir, out_dir, meta):
    """Write a directory `laya.load` accepts: config, weights, tokenizer, encoder config."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("tokenizer", "encoder"):
        shutil.copytree(base_dir / sub, out_dir / sub, dirs_exist_ok=True)
    cfg = dict(agent.cfg, lonn_finetune=meta)
    (out_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    state = {name: tensor.detach().to("cpu").contiguous() for name, tensor in agent.model.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"))


def write_model_card(out_dir, meta, metrics):
    before, after = metrics["base"], metrics["fine_tuned"]
    rows = [
        ("Topic accuracy", "topic_accuracy", "{:.0%}"),
        ("Complexity rank correlation (Spearman)", "complexity_spearman", "{:.2f}"),
        (f"Complexity AUC, labels {HARD_LEVEL}-{MAX_LEVEL} vs easier", "complexity_auc_hard", "{:.2f}"),
    ]
    table = "\n".join(f"| {label} | {fmt.format(before[key])} | {fmt.format(after[key])} |"
                      for label, key, fmt in rows)
    data = meta["data"]
    card = f"""---
license: apache-2.0
base_model: {BUNDLE_REPO}
tags:
  - laya
  - prompt-classification
  - routing
---

# Laya fine-tuned for Lonn prompt routing

The English [Laya]({f"https://huggingface.co/{BUNDLE_REPO}"}) checkpoint with its decision head
fine-tuned to answer Lonn's two routing questions: the prompt's topic (coding, math_reasoning,
creative_writing, general_knowledge_extraction) and whether the task is hard (a yes/no
probability used as the complexity score). The ModernBERT encoder is unchanged.

## Usage

```python
from laya import Router

router = Router(models={{"english": "<this repo id>"}})
```

In the Lonn repository, set `LONN_LAYA_MODEL=<this repo id>` in `.env`. The repository is
private, so run `hf auth login` once with an account that has access.

The model is trained for the exact question wording in `src/lonn/complexity.py`
(`QUESTIONS`). Changing that wording without retraining moves it away from what was learned.

## Training

- Data: `{data['path']}`, {data['english_prompts']} English prompts
  ({data['skipped_non_english']} non-English skipped), labels per `docs/labeling_guide.md`.
- Split, stratified by main topic (seed {meta['seed']}): {data['train']} train,
  {data['validation']} validation (early stopping), {data['test']} held-out test.
- Trained: head layers, type embedding and option scorer. Frozen: encoder, act head.
- Topic targets: the labels. Complexity targets: {"base Laya's own answers, so complexity stays as it was" if meta['complexity_target'] == "base" else "the labelled levels"}.
- Best epoch {meta['best_epoch']} of {meta['epochs_run']}, learning rate {meta['lr']},
  base revision `{meta['base_revision']}`.

## Held-out results ({before['prompts']} prompts)

| Metric | Base Laya | Fine-tuned |
|---|---:|---:|
{table}

For comparison, prompt length alone gives a rank correlation of
{metrics['length_baseline']['complexity_spearman']:.2f} and an AUC of
{metrics['length_baseline']['complexity_auc_hard']:.2f}.

## Limitations

- Small training set; labels were produced by a model following the labeling guide and only
  partly checked by people. Treat the held-out numbers as indicative.
- Non-English prompts still go to Laya's unchanged multilingual checkpoint.
"""
    (out_dir / "README.md").write_text(card)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/labels/pilot.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "models/laya-lonn")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.15, help="share of the training part")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--complexity-target", choices=("labels", "base"), default="labels",
                        help="labels: train complexity on the labelled levels (default); "
                             "base: train it towards base Laya's own answers")
    parser.add_argument("--device", help="cpu, mps or cuda; default picks the fastest available")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    examples, skipped = load_examples(args.data)
    rest, test = stratified_split(examples, args.test_fraction, rng)
    train_set, val_set = stratified_split(rest, args.val_fraction, rng)
    print(f"{len(examples)} English prompts ({skipped} non-English skipped): "
          f"{len(train_set)} train, {len(val_set)} validation, {len(test)} test")

    from huggingface_hub import snapshot_download

    base_dir = Path(snapshot_download(BUNDLE_REPO, allow_patterns=list(CHECKPOINT_FILES)))
    print(f"Loading base checkpoint {BUNDLE_REPO} ...")
    scorer = scorer_for(base_dir, args.device)
    agent = scorer.laya.load("english")

    print("Scoring base model on the held-out set ...")
    base_metrics = evaluate(scorer, test)

    print("Encoding prompts with the frozen encoder ...")
    train_rows, val_rows = encode(scorer, agent, train_set), encode(scorer, agent, val_set)
    if args.complexity_target == "base":
        anchor_complexity(agent.model, train_rows + val_rows, agent.device)
    history, best_epoch = train(agent, train_rows, val_rows, args, rng)

    meta = {
        "base_model": BUNDLE_REPO,
        "base_revision": snapshot_revision(str(base_dir)),
        "laya_version": laya.__version__ if hasattr(laya, "__version__") else None,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_modules": ["head", "type_emb", "scorer"],
        "questions": QUESTIONS,
        "seed": args.seed,
        "lr": args.lr,
        "complexity_target": args.complexity_target,
        "epochs_run": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "data": {
            "path": str(args.data.resolve().relative_to(ROOT)) if args.data.resolve().is_relative_to(ROOT)
            else str(args.data),
            "english_prompts": len(examples),
            "skipped_non_english": skipped,
            "train": len(train_set),
            "validation": len(val_set),
            "test": len(test),
            "test_fingerprint": fingerprint(test),
        },
    }
    save_checkpoint(agent, base_dir, args.out, meta)
    baseline = length_baseline(agent.tok, test)

    # Release the base model and cached encoder states before loading a second
    # 1.6 GB checkpoint; holding both pushed a 16 GB machine deep into swap.
    del scorer, agent, train_rows, val_rows
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("Scoring the saved checkpoint on the held-out set ...")
    tuned_metrics = evaluate(scorer_for(args.out, args.device), test)

    metrics = {"base": base_metrics, "fine_tuned": tuned_metrics,
               "length_baseline": baseline, "history": history, **meta}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    write_model_card(args.out, meta, metrics)

    print(f"\nHeld-out results ({len(test)} prompts)      base   fine-tuned   length only")
    for key in ("topic_accuracy", "complexity_spearman", "complexity_auc_hard"):
        length = metrics["length_baseline"].get(key)
        print(f"  {key:<28} {base_metrics[key]:>8.3f} {tuned_metrics[key]:>10.3f}"
              + (f" {length:>12.3f}" if length is not None else ""))
    print(f"\nSaved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
