"""Train the complexity head (#19, docs/oracle_gap.md).

A linear layer on the task classifier's frozen bge-small embeddings learns the LLM
difficulty labels from scripts/complexity_labels.py (easy / medium / hard). 15% of the
labelled prompts are held out and scored. The head is then checked against the
answer bank, which is used only for measuring, never for training: does predicted
complexity follow how many models, and how many cheap models, solved each benchmark
question? It also writes the benchmark signals (task type + complexity) for
eval_task_router.py.

    python3 scripts/train_complexity.py
"""
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
import complexity_labels as cl  # noqa: E402
from lonn.task_classifier import DEFAULT_TASK_CLASSIFIER, Embedder, TaskClassifier  # noqa: E402

LEVELS = ['easy', 'medium', 'hard']
OUT_DIR = ROOT / 'results' / 'classifier_bakeoff' / 'bge-small-linear-wildchat-complexity'
HEAD_OUT = OUT_DIR / 'complexity_head.json'  # an experiment, not loaded in production
SIGNALS_OUT = OUT_DIR / 'benchmark_signals.jsonl'
CHEAP = ('gpt-oss-20b', 'gpt-oss-120b', 'glm-5.3-flash')


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    import pyarrow.parquet as pq
    texts = {r['id']: r['text'] for r in pq.read_table(ROOT / 'data' / 'task_data' / 'train.parquet').to_pylist()
             if r['variant'] == 'as_benchmark'}
    labels = [r for r in cl.read_labels().values() if r.get('difficulty') and r['id'] in texts]
    labels.sort(key=lambda r: r['id'])
    holdout = [hashlib.sha256(r['id'].encode()).digest()[0] < 38 for r in labels]  # about 15%
    config = json.loads(DEFAULT_TASK_CLASSIFIER.read_text())
    e = config['embedder']
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    embedder = Embedder(e['repo'], e['max_len'], e['revision'], device=device)
    x = bo.cached_embeddings(f"bge-small@{e['revision'][:7]}", embedder, [texts[r['id']] for r in labels], 'complexity')
    y = np.array([LEVELS.index(r['difficulty']) for r in labels])
    train = ~np.array(holdout)
    weight, bias = bo.fit_linear(x[train], y[train], len(LEVELS))
    pred = ((x[~train] @ weight.T + bias)).argmax(1)
    held = {'prompts': int((~train).sum()), 'accuracy': round(float((pred == y[~train]).mean()), 4),
            'within_one_level': round(float((abs(pred - y[~train]) <= 1).mean()), 4)}
    distribution = {level: int((y == i).sum()) for i, level in enumerate(LEVELS)}

    head = {'version': 1, 'generated_by': 'scripts/train_complexity.py (do not edit by hand)',
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'git_commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            'labels': {'labeller': cl.LABELLER, 'prompts': len(labels), 'distribution': distribution},
            'holdout': held, 'embedder': e, 'levels': LEVELS, 'logit_scale': bo.LOGIT_SCALE,
            'weight': [[round(float(v), 6) for v in row] for row in weight],
            'bias': [round(float(v), 6) for v in bias]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HEAD_OUT.write_text(json.dumps(head) + '\n')
    print(f'Wrote {HEAD_OUT}: {len(labels)} labels {distribution}; held-out accuracy '
          f"{held['accuracy']:.1%}, within one level {held['within_one_level']:.1%}")

    # Benchmark: signals for eval_task_router.py, and how complexity tracks real outcomes.
    from benchmark import eval_router as ev
    from benchmark.results import ResultBank
    from lonn.model_pool import load_model_pool
    bench = bo.load_benchmark(bo.DEFAULT_DATA_DIR)
    signals = TaskClassifier(complexity_path=HEAD_OUT).signals([b['text'] for b in bench])
    with SIGNALS_OUT.open('w') as handle:
        for b, s in zip(bench, signals):
            scores = {t: round(p, 5) for t, p in s.task_scores.items() if p >= 1e-5}
            total = sum(scores.values())
            handle.write(json.dumps({'prompt_id': b['prompt_id'], 'complexity': round(s.complexity, 4),
                                     'task_scores': {t: p / total for t, p in scores.items()}}) + '\n')
    bank = ResultBank()
    pool = ev.bank_pool(load_model_pool(), bank, None)
    solved, cheap_solved, predicted = [], [], []
    for b, s in zip(bench, signals):
        outcomes = {m.id: ev.outcome(bank.lookup(b['prompt_id'], m.id), m) for m in pool}
        solved.append(sum(1 for o in outcomes.values() if o and o[0]))
        cheap_solved.append(sum(1 for m in pool if m.id.split('/')[-1] in CHEAP and outcomes[m.id] and outcomes[m.id][0]))
        predicted.append(s.complexity)
    print(f'Benchmark ({len(bench)} questions): Spearman of predicted complexity with models solving '
          f'{spearman(predicted, solved):+.3f}, with cheap models solving {spearman(predicted, cheap_solved):+.3f} '
          '(negative = harder predicted, fewer solve)')
    print(f'Wrote {SIGNALS_OUT}')


if __name__ == '__main__':
    main()
