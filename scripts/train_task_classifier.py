"""Train the production task-type classifier and write config/task_classifier.json (#19).

The bake-off winner (docs/classifier_bakeoff.md): frozen BAAI/bge-small-en-v1.5
embeddings plus a linear layer, trained on the train split of the task data
(data/task_data, or LonnAI/lonn-task-data) and local GPQA if present. The
validation split is only scored. The output pins the embedder revision, so
lonn.task_classifier reproduces the embeddings exactly.

    python3 scripts/train_task_classifier.py
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import classifier_bakeoff as bo  # noqa: E402
from lonn.task_classifier import DEFAULT_TASK_CLASSIFIER, Embedder, TaskClassifier  # noqa: E402
from lonn.task_types import load_task_types  # noqa: E402

EMBEDDER = {'repo': 'BAAI/bge-small-en-v1.5', 'revision': '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
            'max_len': 512, 'pooling': 'cls', 'license': 'MIT'}
AGREEMENT_SAMPLE = 200


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=DEFAULT_TASK_CLASSIFIER)
    args = parser.parse_args(argv)

    task_ids = [t.id for t in load_task_types()]
    train, val = bo.load_task_data()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    embedder = Embedder(EMBEDDER['repo'], EMBEDDER['max_len'], EMBEDDER['revision'], device=device)
    name = f"bge-small@{EMBEDDER['revision'][:7]}"
    x_train = bo.cached_embeddings(name, embedder, [r['text'] for r in train], 'train')
    x_val = bo.cached_embeddings(name, embedder, [r['text'] for r in val], 'validation')
    y_train = np.array([task_ids.index(r['task_type']) for r in train])
    weight, bias = bo.fit_linear(x_train, y_train, len(task_ids))

    lengths = [0] * len(val)  # the long-prompt split is reported by the bake-off, not here
    logits = (x_val @ weight.T + bias) * bo.LOGIT_SCALE
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    metrics = bo.evaluate(probs, val, task_ids, lengths)

    manifest = ROOT / 'data' / 'task_data' / 'manifest.json'
    config = {
        'version': 1,
        'generated_by': 'scripts/train_task_classifier.py (do not edit by hand)',
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        'training_data': {'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]
                          if manifest.exists() else None,
                          'train_rows': len(train), 'validation_rows': len(val)},
        'validation': {'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']},
        'embedder': EMBEDDER,
        'task_types': task_ids,
        'logit_scale': bo.LOGIT_SCALE,
        'weight': [[round(float(v), 6) for v in row] for row in weight],
        'bias': [round(float(v), 6) for v in bias],
    }
    args.output.write_text(json.dumps(config) + '\n')

    # The shipped classifier (CPU, pinned revision, rounded weights) must agree with training.
    sample = val[:AGREEMENT_SAMPLE]
    shipped = TaskClassifier(args.output).probabilities([r['text'] for r in sample])
    agree = np.mean([max(p, key=p.get) == task_ids[int(row.argmax())] for p, row in zip(shipped, probs[:len(sample)])])
    print(f"Wrote {args.output}: validation accuracy {metrics['accuracy']:.1%}, macro-F1 {metrics['macro_f1']:.3f}; "
          f"shipped classifier agrees with training on {agree:.1%} of {len(sample)} validation prompts.")
    if agree < 0.98:
        raise SystemExit('Shipped classifier disagrees with training; check the embedder setup.')


if __name__ == '__main__':
    main()
