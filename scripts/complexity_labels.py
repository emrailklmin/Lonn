"""Difficulty labels for the complexity signal (#19, docs/oracle_gap.md).

An LLM rates training prompts (never the 690 benchmark questions) as easy, medium or
hard in concrete terms: would a small, cheap model get it fully right? A linear layer
on the task classifier's embeddings then learns to predict this from the prompt alone
(scripts/train_complexity.py), with no model call at routing time.

Paid OpenRouter calls: --dry-run first and ask before running.

    python3 scripts/complexity_labels.py --dry-run
    python3 scripts/complexity_labels.py --budget 1.0
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from benchmark.run import CallError, FatalError, OpenRouterClient, call_with_retries  # noqa: E402
from lonn.task_types import load_task_types  # noqa: E402

OUT = ROOT / 'data' / 'task_data' / 'complexity' / 'labels.jsonl'
LABELLER = 'deepseek/deepseek-v4.1-flash'
PRICE_PER_M = (0.15, 0.6)
PER_TYPE = 300
SEED = 20261023
WORKERS = 8
LABEL_CHARS = 3000
SYSTEM = 'You estimate how hard a request is for AI assistants of different sizes.'
PROMPT = '''How hard is this request for an AI assistant to answer fully and correctly?

- easy: a small, fast, cheap model would very likely get it fully right.
- medium: a small model might fail; a capable mid-size model would likely succeed.
- hard: only the strongest models are likely to get it right (expert knowledge,
  long or tricky reasoning, many constraints, long high-quality output).

REQUEST:
<<<
{text}
>>>

Reply with JSON only: {{"difficulty": "easy" | "medium" | "hard"}}'''


def clip(text: str) -> str:
    if len(text) <= LABEL_CHARS:
        return text
    head = int(LABEL_CHARS * 0.6)
    return f'{text[:head]}\n[...]\n{text[-(LABEL_CHARS - head):]}'


def sample_prompts() -> list:
    """PER_TYPE train-split prompts per measured task type, one variant each, deterministic."""
    import pyarrow.parquet as pq
    measured = {t.id for t in load_task_types() if t.measured}
    rows = [r for r in pq.read_table(ROOT / 'data' / 'task_data' / 'train.parquet').to_pylist()
            if r['task_type'] in measured and r['variant'] == 'as_benchmark']
    by_type = {}
    for r in rows:
        by_type.setdefault(r['task_type'], []).append(r)
    chosen = []
    for task, group in sorted(by_type.items()):
        group.sort(key=lambda r: hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest())
        chosen += group[:PER_TYPE]
    return chosen


def read_labels(path: Path = OUT) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            out[row['id']] = row
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--budget', type=float, default=1.0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    prompts = sample_prompts()
    done = read_labels()
    todo = [p for p in prompts if not (done.get(p['id']) or {}).get('difficulty')]
    tokens = sum((len(SYSTEM) + len(PROMPT) + len(clip(p['text']))) / 3.5 for p in todo)
    estimate = (tokens * PRICE_PER_M[0] + len(todo) * 300 * PRICE_PER_M[1]) / 1e6
    print(f'{len(todo)} prompts to label ({len(done)} saved); estimate {estimate:.2f} USD')
    if args.dry_run or not todo:
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    client = OpenRouterClient(os.environ['OPENROUTER_API_KEY'].strip(), timeout=120, deadline=300)

    def one(p):
        body = {'model': LABELLER, 'temperature': 0, 'max_tokens': 1000, 'usage': {'include': True},
                'reasoning': {'effort': 'low', 'exclude': True},
                'messages': [{'role': 'system', 'content': SYSTEM},
                             {'role': 'user', 'content': PROMPT.format(text=clip(p['text']))}]}
        try:
            data, _ = call_with_retries(client, body)
        except CallError as exc:
            return {'id': p['id'], 'difficulty': None, 'error': str(exc)[:200], 'cost_usd': 0.0}
        match = re.search(r'"difficulty"\s*:\s*"(easy|medium|hard)"', data['choices'][0]['message'].get('content') or '')
        return {'id': p['id'], 'task_type': p['task_type'], 'difficulty': match.group(1) if match else None,
                'labeller': LABELLER, 'cost_usd': (data.get('usage') or {}).get('cost') or 0.0, 'error': None}

    spent = sum(r.get('cost_usd') or 0 for r in done.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('a') as handle, ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(todo), WORKERS):
            try:
                rows = list(pool.map(one, todo[start:start + WORKERS]))
            except FatalError as exc:
                raise SystemExit(f'Stopped: {exc}') from None
            for row in rows:
                handle.write(json.dumps(row) + '\n')
                spent += row['cost_usd']
            handle.flush()
            if spent >= args.budget:
                print(f'Budget of {args.budget} USD reached; stopping.')
                break
    print(f'Done: {OUT} ({spent:.3f} USD in total).')


if __name__ == '__main__':
    main()
