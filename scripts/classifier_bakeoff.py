"""Task-type classifier bake-off, stage A: cheap candidates on a laptop (issue #19).

Candidates (docs/classifier_bakeoff.md):
    laya-zeroshot     stock Laya, one Choice question over the task-type descriptions
    gte-base-linear   frozen Alibaba-NLP/gte-modernbert-base embeddings + a linear layer
    gte-base-knn      the same embeddings, k nearest training prompts vote
    bge-small-linear  frozen BAAI/bge-small-en-v1.5 embeddings + a linear layer

Each candidate is scored on the validation split of the task data
(data/task_data, docs/task_data.md) and writes its predicted task-type
distribution for the 690 benchmark questions, which eval_task_router.py
--signals turns into a routing result. Latency is one prompt at a time on CPU.

    python3 scripts/classifier_bakeoff.py                       # all candidates
    python3 scripts/classifier_bakeoff.py --candidates gte-base-linear gte-base-knn

Results: results/classifier_bakeoff/<candidate>/{metrics.json,benchmark_signals.jsonl}.
New embedding models are downloaded to a temporary folder (--model-cache), not the
Hugging Face cache, and embeddings are cached under the system temp folder.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('USE_TF', '0')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark.build_data import DATASETS, DEFAULT_DATA_DIR  # noqa: E402
from benchmark.templates import user_message  # noqa: E402
from lonn.task_types import load_task_types  # noqa: E402

TASK_DATA = ROOT / 'data' / 'task_data'
OUT_DIR = ROOT / 'results' / 'classifier_bakeoff'
TEMP = Path(tempfile.gettempdir()) / 'lonn-bakeoff'
LONG_TOKENS = 370  # Laya's old prompt budget; prompts above it are reported separately.
EMBEDDERS = {'gte-base': ('Alibaba-NLP/gte-modernbert-base', 512),
             'bge-small': ('BAAI/bge-small-en-v1.5', 512)}
CANDIDATES = ('laya-zeroshot', 'gte-base-linear', 'gte-base-knn', 'bge-small-linear')
LAYA_INSTRUCTIONS = 'Which kind of task is the user asking an AI assistant to do? Pick the single best fit.'
HEAD_SHARE = 0.35


# Data

def load_task_data(include_restricted=True) -> tuple[list, list]:
    import pyarrow.parquet as pq
    rows = []
    for name in ('train.parquet', 'validation.parquet', 'restricted/gpqa.parquet'):
        path = TASK_DATA / name
        if path.exists() and (include_restricted or 'restricted' not in name):
            rows += pq.read_table(path).to_pylist()
    if not rows:
        raise SystemExit('No task data: run scripts/build_task_data.py first (or download LonnAI/lonn-task-data).')
    return [r for r in rows if r['split'] == 'train'], [r for r in rows if r['split'] == 'validation']


def load_benchmark(data_dir) -> list:
    """The benchmark questions with saved answers (everything but BFCL), as the router sees them."""
    out = []
    for name, spec in DATASETS.items():
        if name.startswith('bfcl'):
            continue
        path = Path(data_dir) / 'prompts' / spec.category / f'{name}.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            out.append({'prompt_id': record['prompt_id'], 'text': user_message(record)})
    return out


# Embedding candidates

def head_tail(ids, budget):
    if len(ids) <= budget:
        return ids
    head = int(budget * HEAD_SHARE)
    return ids[:head] + ids[-(budget - head):]


class Embedder:
    def __init__(self, repo, max_len, cache_dir, device):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(repo, cache_dir=cache_dir).to(device).eval()
        self.max_len, self.device = max_len, device

    def encode_ids(self, texts):
        budget = self.max_len - 2
        return [head_tail(self.tok(t, add_special_tokens=False)['input_ids'], budget) for t in texts]

    @torch.no_grad()
    def embed(self, texts, batch_size=32, progress=''):
        ids = self.encode_ids(texts)
        order = np.argsort([len(i) for i in ids])
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        for start in range(0, len(order), batch_size):
            batch = [ids[i] for i in order[start:start + batch_size]]
            width = max(len(b) for b in batch) + 2
            input_ids = torch.full((len(batch), width), pad)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, b in enumerate(batch):
                seq = [cls] + b + [sep]
                input_ids[row, :len(seq)] = torch.tensor(seq)
                mask[row, :len(seq)] = 1
            hidden = self.model(input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
            vectors = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1).cpu().numpy()  # CLS pooling
            out[order[start:start + batch_size]] = vectors
            if progress and start % (batch_size * 100) == 0:
                print(f'    {progress}: {start}/{len(texts)}', flush=True)
        return out


def cached_embeddings(name, embedder, texts, label):
    key = hashlib.sha256(('\x00'.join(texts) + name).encode()).hexdigest()[:16]
    path = TEMP / 'embeddings' / f'{name}-{label}-{key}.npy'
    if path.exists():
        return np.load(path)
    vectors = embedder.embed(texts, progress=f'{name} {label}')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def train_linear(x, y, n_classes, epochs=300, weight_decay=1e-4, lr=0.05, seed=0):
    """Multinomial logistic regression with class weights (thin task types count more)."""
    torch.manual_seed(seed)
    x, y = torch.tensor(x), torch.tensor(y)
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    weights = (counts.sum() / (n_classes * counts))
    layer = torch.nn.Linear(x.shape[1], n_classes)
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(layer(x) * 20.0, y)  # Inputs are unit vectors: scale logits up.
        loss.backward()
        optimizer.step()
    layer.eval()
    return lambda v: torch.softmax(layer(torch.tensor(v)) * 20.0, dim=-1).detach().numpy()


def knn_predictor(x_train, y_train, n_classes, k=15, temperature=0.05):
    xt = torch.tensor(x_train)
    yt = torch.tensor(y_train)

    def predict(v):
        sims = torch.tensor(v) @ xt.T
        top, idx = sims.topk(k, dim=-1)
        votes = torch.zeros((len(v), n_classes))
        votes.scatter_add_(1, yt[idx], torch.softmax(top / temperature, dim=-1))
        return votes.numpy()
    return predict


# Laya candidate

def laya_predictor(task_ids, descriptions):
    from lonn.complexity import ComplexityScorer
    os.environ['LONN_LAYA_MODEL'] = ''  # stock checkpoint
    scorer = ComplexityScorer()
    question = {'task': {'type': 'choice', 'instructions': LAYA_INSTRUCTIONS,
                         'criteria': dict(zip(task_ids, descriptions))}}

    def predict_one(text):
        answer = scorer._predict(text, question)['answers']['task']
        probs = np.array([float(answer['probabilities'].get(t, 0.0)) for t in task_ids])
        return probs / probs.sum() if probs.sum() else np.eye(len(task_ids))[task_ids.index(answer['choice'])]

    def predict(texts, progress=''):
        out = []
        for i, text in enumerate(texts):
            out.append(predict_one(text))
            if progress and i % 500 == 0:
                print(f'    laya {progress}: {i}/{len(texts)}', flush=True)
        return np.array(out)
    return predict, predict_one


# Metrics

def evaluate(probs, rows, task_ids, lengths) -> dict:
    pred = probs.argmax(-1)
    gold = np.array([task_ids.index(r['task_type']) for r in rows])
    per_type = {}
    f1s = []
    for i, t in enumerate(task_ids):
        tp = int(((pred == i) & (gold == i)).sum())
        fp, fn = int(((pred == i) & (gold != i)).sum()), int(((pred != i) & (gold == i)).sum())
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_type[t] = {'n': tp + fn, 'recall': round(recall, 4), 'precision': round(precision, 4), 'f1': round(f1, 4)}
    confusions = Counter((task_ids[g], task_ids[p]) for g, p in zip(gold, pred) if g != p)

    def acc(mask):
        mask = np.array(mask)
        return {'n': int(mask.sum()), 'accuracy': round(float((pred[mask] == gold[mask]).mean()), 4) if mask.any() else None}
    return {'accuracy': round(float((pred == gold).mean()), 4), 'macro_f1': round(float(np.mean(f1s)), 4),
            'by_variant': {v: acc([r['variant'] == v for r in rows]) for v in ('as_benchmark', 'plain')},
            'long_prompts': acc([n > LONG_TOKENS for n in lengths]),
            'per_type': per_type,
            'top_confusions': [{'gold': g, 'predicted': p, 'n': n} for (g, p), n in confusions.most_common(10)]}


def latency_ms(predict_one, texts, n=40) -> float:
    for text in texts[:3]:
        predict_one(text)  # warm-up
    start = time.perf_counter()
    for text in texts[:n]:
        predict_one(text)
    return round((time.perf_counter() - start) / min(n, len(texts)) * 1000, 1)


def save(name, metrics, task_ids, bench, bench_probs):
    folder = OUT_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=1) + '\n')
    with (folder / 'benchmark_signals.jsonl').open('w') as handle:
        for item, p in zip(bench, bench_probs):
            scores = {t: round(float(v), 5) for t, v in zip(task_ids, p) if v >= 1e-5}
            total = sum(scores.values())
            scores = {t: v / total for t, v in scores.items()}
            handle.write(json.dumps({'prompt_id': item['prompt_id'], 'task_scores': scores, 'complexity': None}) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--candidates', nargs='+', choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--model-cache', type=Path, default=TEMP / 'models',
                        help='where new embedding models are downloaded (default: system temp folder)')
    args = parser.parse_args(argv)

    types = load_task_types()
    task_ids, descriptions = [t.id for t in types], [t.description for t in types]
    train, val = load_task_data()
    bench = load_benchmark(args.data_dir)
    y_train = np.array([task_ids.index(r['task_type']) for r in train])
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'{len(train)} train rows, {len(val)} validation rows, {len(bench)} benchmark prompts; device {device}')

    from transformers import AutoTokenizer
    length_tok = AutoTokenizer.from_pretrained(EMBEDDERS['gte-base'][0], cache_dir=args.model_cache)
    val_lengths = [len(length_tok(r['text'], add_special_tokens=False)['input_ids']) for r in val]

    embedded = {}
    for name in args.candidates:
        print(f'\n== {name}', flush=True)
        common = {'candidate': name, 'train_rows': len(train), 'validation_rows': len(val),
                  'task_types': task_ids, 'date': time.strftime('%Y-%m-%d')}
        if name == 'laya-zeroshot':
            predict, predict_one = laya_predictor(task_ids, descriptions)
            val_probs, bench_probs = predict([r['text'] for r in val], 'validation'), predict([b['text'] for b in bench], 'benchmark')
            metrics = {**common, 'model': 'convaiinnovations/laya (stock)', 'trained': False,
                       **evaluate(val_probs, val, task_ids, val_lengths),
                       'latency_ms_per_prompt': latency_ms(predict_one, [r['text'] for r in val]),
                       'latency_device': 'Laya default (MPS on this Mac)'}
        else:
            embedder_name, method = name.rsplit('-', 1)
            repo, max_len = EMBEDDERS[embedder_name]
            if embedder_name not in embedded:
                embedder = Embedder(repo, max_len, args.model_cache, device)
                embedded[embedder_name] = (embedder, cached_embeddings(embedder_name, embedder, [r['text'] for r in train], 'train'),
                                           cached_embeddings(embedder_name, embedder, [r['text'] for r in val], 'validation'),
                                           cached_embeddings(embedder_name, embedder, [b['text'] for b in bench], 'benchmark'))
            embedder, x_train, x_val, x_bench = embedded[embedder_name]
            classify = (train_linear(x_train, y_train, len(task_ids)) if method == 'linear'
                        else knn_predictor(x_train, y_train, len(task_ids)))
            val_probs, bench_probs = classify(x_val), classify(x_bench)
            cpu = Embedder(repo, max_len, args.model_cache, 'cpu')
            metrics = {**common, 'model': repo, 'trained': method,
                       **evaluate(val_probs, val, task_ids, val_lengths),
                       'latency_ms_per_prompt': latency_ms(lambda t: classify(cpu.embed([t])), [r['text'] for r in val]),
                       'latency_device': 'CPU, batch 1'}
        save(name, metrics, task_ids, bench, bench_probs)
        print(f"  accuracy {metrics['accuracy']:.1%}, macro-F1 {metrics['macro_f1']:.3f}, plain "
              f"{metrics['by_variant']['plain']['accuracy']}, long prompts {metrics['long_prompts']['accuracy']}, "
              f"{metrics['latency_ms_per_prompt']} ms/prompt", flush=True)
    print(f'\nSaved to {OUT_DIR}. Next: eval_task_router.py --signals {OUT_DIR}/<candidate>/benchmark_signals.jsonl')


if __name__ == '__main__':
    main()
