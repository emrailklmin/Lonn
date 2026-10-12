"""Train a MIMO topic classifier on frozen BERT sentence embeddings (experiment).

Usage:
    python3 scripts/train_topic_classifier_mimo.py
    python3 scripts/train_topic_classifier_mimo.py --heads 6 --hidden 1024 --layers 2 --epochs 100

MIMO (multi-input multi-output, Havasi et al. 2021) trains an ensemble of M
members inside one network. Embeddings come from the same frozen encoder as
scripts/train_topic_classifier.py; the network is

    M x 384 inputs (concatenated) -> [Linear -> ReLU -> Dropout] x layers -> M heads x 4 topics

The M members share the hidden neurons, so the hidden layers must be wide
enough to hold M sub-networks; the defaults are much wider than the single
network in scripts/train_topic_classifier.py.

- Training: each of the M input slots gets its own independent shuffle of the
  training prompts, and head m is trained on the label of the prompt in slot m.
  Each head must learn to ignore the other slots, so the heads act as separate
  ensemble members that share one hidden layer.
- Diversity: members only become a real ensemble if they learn from the data
  differently. Training uses shuffled mini-batches, and with --bootstrap each
  head draws its slot inputs from its own resample (with replacement) of the
  training prompts, as in bagging.
- Testing: the same prompt fills all M slots; the prediction is the mean of
  the M heads' probabilities.

Uncertainty, per prompt, in nats (maximum ln 4 = 1.386 for four topics):
- total     = entropy of the mean prediction
- aleatoric = mean entropy of each head's prediction: every member is unsure,
              so the prompt itself is ambiguous or its label noisy
- epistemic = total - aleatoric (mutual information): the members disagree,
              so the model lacks knowledge about prompts like this one

Same 70/30 stratified split and labels as the first version; no validation.
The trained network is written to models/topic_classifier_mimo.pt (gitignored).
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_topic_classifier import ROOT, TOPICS, embed, load_rows, main_topic, report, split, targets


class MimoTopicNet(nn.Module):
    def __init__(self, dim_in, heads, hidden, n_out, dropout, layers=1):
        super().__init__()
        self.heads, self.n_out = heads, n_out
        blocks, width = [], heads * dim_in
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden, heads * n_out)  # All heads in one layer, split below.

    def forward(self, x):
        """x: (batch, heads, dim) -> logits (batch, heads, n_out)."""
        hidden = self.body(x.flatten(1))
        return self.out(hidden).view(-1, self.heads, self.n_out)


def epoch_batches(pools, batch_size, generator):
    """Yield (batch, heads) index tensors: each head walks its own shuffled pool."""
    orders = [pool[torch.randperm(len(pool), generator=generator)] for pool in pools]
    n = min(len(order) for order in orders)
    for start in range(0, n, batch_size):
        yield torch.stack([order[start:start + batch_size] for order in orders], 1)


def entropy(p, dim=-1):
    return -(p * p.clamp_min(1e-12).log()).sum(dim)


@torch.no_grad()
def predict(model, x):
    """Mean prediction, per-head predictions, and the uncertainty split."""
    model.eval()
    member_probs = F.softmax(model(x.unsqueeze(1).expand(-1, model.heads, -1)), -1)  # (N, M, K)
    mean = member_probs.mean(1)
    total = entropy(mean)
    aleatoric = entropy(member_probs).mean(1)
    return mean, member_probs, {"total": total, "aleatoric": aleatoric, "epistemic": total - aleatoric}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/labels/pilot.jsonl")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--holdout", type=Path, help="extra test-only labelled prompts, reported "
                        "separately and combined with the pilot test split")
    parser.add_argument("--extra", type=Path, help="extra labelled prompts added to training only "
                        "(e.g. data/external/topics.jsonl from scripts/build_external_topics.py)")
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 penalty on weights, added to the loss")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True,
                        help="give each head its own resample of the training prompts (default on)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "models/topic_classifier_mimo.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    train_rows, test_rows = split(load_rows(args.data), args.test_fraction, random.Random(args.seed))
    if args.extra:
        extra = load_rows(args.extra)
        train_rows = train_rows + extra
        print(f"Added {len(extra)} extra training prompts from {args.extra}")
    print(f"{len(train_rows)} train / {len(test_rows)} test prompts")
    print(f"Embedding on {device} ...")
    x_train = embed([r["prompt"] for r in train_rows], device)
    x_test = embed([r["prompt"] for r in test_rows], device)
    y_train = targets(train_rows)

    model = MimoTopicNet(x_train.shape[1], args.heads, args.hidden, len(TOPICS), args.dropout, args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
    print(f"Network: {args.heads} x {x_train.shape[1]} -> {args.layers} x {args.hidden} -> {args.heads} heads x {len(TOPICS)} "
          f"({sum(p.numel() for p in model.parameters()):,} parameters), dropout {args.dropout}, "
          f"L2 {args.l2}, batch {args.batch_size}, bootstrap {args.bootstrap}, {args.epochs} epochs")

    n = len(x_train)
    pools = [torch.randint(n, (n,), generator=generator) if args.bootstrap else torch.arange(n)
             for _ in range(args.heads)]
    if args.bootstrap:
        seen = sum(len(set(p.tolist())) for p in pools) / args.heads / n
        print(f"  bootstrap: each head sees {seen:.0%} of the distinct training prompts on average")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx in epoch_batches(pools, args.batch_size, generator):  # (batch, heads)
            xb, yb = x_train[idx], y_train[idx]  # (batch, heads, dim) and (batch, heads, K)
            data_loss = -(yb * F.log_softmax(model(xb), -1)).sum(-1).mean()  # Mean over prompts and heads.
            l2_penalty = args.l2 * sum(w.pow(2).sum() for w in weights)
            loss = data_loss + l2_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in (1, args.epochs) or epoch % 100 == 0:
            print(f"  epoch {epoch:4d}  cross-entropy {data_loss.item():.4f}  L2 penalty {l2_penalty.item():.4f}")

    train_mean, _, _ = predict(model, x_train)
    test_mean, test_members, unc = predict(model, x_test)
    report("Train (ensemble mean)", train_mean, train_rows)
    test_acc = report("Test (ensemble mean)", test_mean, test_rows)

    truth = torch.tensor([TOPICS.index(main_topic(r)) for r in test_rows])
    member_acc = (test_members.argmax(-1) == truth.unsqueeze(1)).float().mean(0)
    print("  individual heads on test: " + "  ".join(f"{a:.0%}" for a in member_acc.tolist()))
    agree = (test_members.argmax(-1) == test_mean.argmax(-1, keepdim=True)).float().mean().item()
    print(f"  heads agreeing with the ensemble's choice: {agree:.0%}")

    if args.holdout:
        holdout_rows = load_rows(args.holdout)
        x_holdout = embed([r["prompt"] for r in holdout_rows], device)
        holdout_mean, _, _ = predict(model, x_holdout)
        report(f"Holdout ({args.holdout.name})", holdout_mean, holdout_rows)
        test_rows = test_rows + holdout_rows
        test_mean, test_members, unc = predict(model, torch.cat([x_test, x_holdout]))
        report("Pilot test + holdout (ensemble mean)", test_mean, test_rows)
        truth = torch.tensor([TOPICS.index(main_topic(r)) for r in test_rows])

    correct = test_mean.argmax(-1) == truth
    print(f"\nUncertainty on test prompts (nats; max {math.log(len(TOPICS)):.3f})")
    print(f"  {'':<18}{'total':>8}{'aleatoric':>11}{'epistemic':>11}")
    for name, mask in [("correct", correct), ("wrong", ~correct), ("all", torch.ones_like(correct))]:
        if mask.any():
            print(f"  {name:<10} n={int(mask.sum()):<4}" + "".join(
                f"{unc[k][mask].mean().item():>{w}.3f}" for k, w in (("total", 8), ("aleatoric", 11), ("epistemic", 11))))

    print("\nMost uncertain test prompts (label -> predicted):")
    for i in unc["total"].argsort(descending=True)[:6].tolist():
        kind = "model lacks knowledge" if unc["epistemic"][i] > 0.5 * unc["total"][i] else "ambiguous prompt/label"
        print(f"  total {unc['total'][i]:.2f}  aleatoric {unc['aleatoric'][i]:.2f}  epistemic {unc['epistemic'][i]:.2f}"
              f"  [{kind}]  {main_topic(test_rows[i])} -> {TOPICS[test_mean[i].argmax().item()]}"
              f"  | {' '.join(test_rows[i]['prompt'].split())[:60]}")

    probes = ["asdkjh qwe zxcmn poiu lkjhg", "Hur lagar man en god kanelbulle?",
              "Code me a nail saloon app that is like Airbnb but for free lance nail artists",
              "Prove that sqrt(3) + sqrt(5) is irrational."]
    _, _, probe_unc = predict(model, embed(probes, device))
    print("\nProbe prompts (unlike the training data):")
    for i, text in enumerate(probes):
        print(f"  total {probe_unc['total'][i]:.2f}  aleatoric {probe_unc['aleatoric'][i]:.2f}"
              f"  epistemic {probe_unc['epistemic'][i]:.3f}  | {text}")

    majority = Counter(main_topic(r) for r in train_rows).most_common(1)[0][0]
    baseline = sum(main_topic(r) == majority for r in test_rows) / len(test_rows)
    print(f"\nBaseline (always '{majority}'): {baseline:.0%} on test")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"topics": TOPICS, "heads": args.heads, "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                "l2": args.l2, "state_dict": model.state_dict(), "test_accuracy": test_acc}, args.out)
    print(f"Saved network to {args.out}")


if __name__ == "__main__":
    main()
