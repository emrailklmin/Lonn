"""Train a small topic classifier on frozen BERT sentence embeddings (experiment).

Usage:
    python3 scripts/train_topic_classifier.py
    python3 scripts/train_topic_classifier.py --hidden 128 --epochs 300 --dropout 0.1 --l2 1e-3

Each prompt is embedded once with a frozen BERT sentence encoder
(sentence-transformers/all-MiniLM-L6-v2: 6 layers, 384 dimensions, mean pooling).
A feed-forward network then maps the embedding to the four topics:

    384 -> Linear -> ReLU -> Dropout -> 4

Training minimises soft-label cross-entropy plus an L2 penalty, l2 * sum of
squared weights (biases excluded), with Adam. Labels are the soft topic distributions in data/labels/pilot.jsonl, split 70/30
into train and test, stratified by main topic. There is no validation set: the
network trains for a fixed number of epochs and is scored once on the test set.
The trained network is written to models/topic_classifier.pt (gitignored).
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random

os.environ.setdefault("USE_TF", "0")  # Avoids a known TensorFlow import hang.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
TOPICS = ["coding", "math_reasoning", "creative_writing", "general_knowledge_extraction"]
MAX_TOKENS = 256  # The encoder was trained on inputs up to this length.


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main_topic(row):
    return max(row["topics"], key=row["topics"].get)


def split(rows, test_fraction, rng):
    """Stratified train/test split by main topic."""
    by_topic = defaultdict(list)
    for row in rows:
        by_topic[main_topic(row)].append(row)
    train, test = [], []
    for topic in sorted(by_topic):
        group = by_topic[topic][:]
        rng.shuffle(group)
        n_test = round(len(group) * test_fraction)
        test += group[:n_test]
        train += group[n_test:]
    return train, test


@torch.no_grad()
def embed(texts, device, batch_size=32):
    """Mean-pooled, L2-normalised sentence embeddings from the frozen encoder."""
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to(device).eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=MAX_TOKENS, return_tensors="pt").to(device)
        hidden = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        chunks.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(chunks)


def targets(rows):
    """Soft topic distributions in TOPICS order."""
    out = torch.zeros(len(rows), len(TOPICS))
    for i, row in enumerate(rows):
        total = sum(row["topics"].values())
        for topic, weight in row["topics"].items():
            out[i, TOPICS.index(topic)] = weight / total
    return out


class TopicNet(nn.Module):
    def __init__(self, dim_in, hidden, n_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def report(name, logits, rows):
    predicted = [TOPICS[i] for i in logits.argmax(-1).tolist()]
    truth = [main_topic(r) for r in rows]
    hits = [p == t for p, t in zip(predicted, truth)]
    print(f"\n{name}: accuracy {sum(hits) / len(hits):.0%} ({sum(hits)}/{len(hits)})")
    for topic in TOPICS:
        n = truth.count(topic)
        print(f"  {topic:<30} {sum(h for h, t in zip(hits, truth) if t == topic)}/{n}")
    mistakes = Counter(f"{t} -> {p}" for p, t in zip(predicted, truth) if p != t)
    if mistakes:
        print("  mistakes (label -> predicted): " + ", ".join(f"{k} x{v}" for k, v in mistakes.most_common(4)))
    return sum(hits) / len(hits)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/labels/pilot.jsonl")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 penalty on weights, added to the loss")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "models/topic_classifier.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    train_rows, test_rows = split(load_rows(args.data), args.test_fraction, random.Random(args.seed))
    print(f"{len(train_rows)} train / {len(test_rows)} test prompts; train topics "
          f"{dict(Counter(main_topic(r) for r in train_rows))}")

    print(f"Embedding with {ENCODER} on {device} ...")
    x_train = embed([r["prompt"] for r in train_rows], device)
    x_test = embed([r["prompt"] for r in test_rows], device)
    y_train = targets(train_rows)

    model = TopicNet(x_train.shape[1], args.hidden, len(TOPICS), args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
    print(f"Network: {x_train.shape[1]} -> {args.hidden} -> {len(TOPICS)} "
          f"({sum(p.numel() for p in model.parameters()):,} parameters), dropout {args.dropout}, L2 {args.l2}, {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):  # Full batch: the training set is tiny.
        model.train()
        data_loss = -(y_train * F.log_softmax(model(x_train), -1)).sum(-1).mean()
        l2_penalty = args.l2 * sum(w.pow(2).sum() for w in weights)
        loss = data_loss + l2_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in (1, args.epochs) or epoch % 50 == 0:
            print(f"  epoch {epoch:4d}  cross-entropy {data_loss.item():.4f}  L2 penalty {l2_penalty.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_acc = report("Train", model(x_train), train_rows)
        test_acc = report("Test", model(x_test), test_rows)
    majority = Counter(main_topic(r) for r in train_rows).most_common(1)[0][0]
    baseline = sum(main_topic(r) == majority for r in test_rows) / len(test_rows)
    print(f"\nBaseline (always '{majority}'): {baseline:.0%} on test")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": ENCODER, "topics": TOPICS, "hidden": args.hidden, "dropout": args.dropout, "l2": args.l2,
                "state_dict": model.state_dict(), "train_accuracy": train_acc, "test_accuracy": test_acc},
               args.out)
    print(f"Saved network to {args.out}")


if __name__ == "__main__":
    main()
