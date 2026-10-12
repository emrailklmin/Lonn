"""Score labelled prompts with the complexity scorer and compare with the labels.

Usage:
    python3 scripts/evaluate_classifier.py
    python3 scripts/evaluate_classifier.py --data data/labels/pilot.jsonl --out reports/classifier_eval.json

Reports topic accuracy (overall, per topic, and where mistakes go) and how well
complexity ranks the labelled level: Spearman correlation, and AUC for telling
labels 3-5 apart from easier ones (0.5 is chance). Prompt length alone is shown
as a baseline complexity should beat. Labels follow docs/labeling_guide.md.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from lonn.complexity import ComplexityScorer

HARD_LEVEL = 3  # Labels at or above this count as hard for AUC.


def ranks(values):
    """Ranks with ties averaged."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/labels/pilot.jsonl")
    parser.add_argument("--out", type=Path, help="also write per-prompt results as JSON")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    scorer = ComplexityScorer()
    print(f"Loading the classifier and scoring {len(rows)} prompts ...")
    scorer.score_prompt("Warm-up prompt.")

    results, start = [], time.perf_counter()
    for row in rows:
        result = scorer.score_prompt(row["prompt"])
        results.append({
            "label_topic": max(row["topics"], key=row["topics"].get),
            "label_complexity": row["complexity"],
            "topic": max(result.topic_scores, key=result.topic_scores.get).value,
            "complexity": result.complexity,
            "tokens": len(scorer.classifier._tokenizer(row["prompt"])["input_ids"]),
        })
    ms = (time.perf_counter() - start) * 1000 / len(rows)

    truth = [r["label_topic"] for r in results]
    hits = [r["topic"] == r["label_topic"] for r in results]
    level = np.array([r["label_complexity"] for r in results])
    complexity = np.array([r["complexity"] for r in results])
    length = np.array([r["tokens"] for r in results])

    print(f"\nTopic accuracy: {np.mean(hits):.0%} ({sum(hits)}/{len(rows)})")
    for topic in sorted(set(truth)):
        n = truth.count(topic)
        print(f"  {topic:<30} {sum(h for h, t in zip(hits, truth) if t == topic)}/{n}")
    mistakes = Counter(f"{r['label_topic']} -> {r['topic']}" for r in results if r["topic"] != r["label_topic"])
    print("Most common mistakes (label -> predicted):")
    for pair, count in mistakes.most_common(5):
        print(f"  {count:3d}  {pair}")

    hard = level >= HARD_LEVEL
    print(f"\nComplexity vs labelled level ({hard.sum()} prompts labelled {HARD_LEVEL}-5)")
    print(f"  {'':<22} {'Spearman':>9} {'AUC hard':>9}")
    print(f"  {'scorer complexity':<22} {spearman(complexity, level):9.2f} {auc(complexity, hard):9.2f}")
    print(f"  {'prompt length only':<22} {spearman(length, level):9.2f} {auc(length, hard):9.2f}")
    print(f"\n{ms:.0f} ms per prompt on average")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Per-prompt results written to {args.out}")


if __name__ == "__main__":
    main()
