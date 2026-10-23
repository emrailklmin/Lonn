"""Everyday prompts for the task classifier: sample from WildChat, label, review (#19).

The classifier was trained on benchmark-style prompts and misreads casual ones
(docs/classifier_bakeoff.md). This script builds two sets of real first user
turns from allenai/WildChat-1M (ODC-BY), from disjoint users:

- eval:  about 300 prompts, labelled by two strong models; where they agree the
         label is gold, where they differ a human decides (review.csv). Never
         trained on.
- train: about 3,000 prompts, labelled by one model, added to the task data.

    python3 scripts/everyday_prompts.py sample                      # free: reads WildChat into memory
    python3 scripts/everyday_prompts.py label --set eval --model anthropic/claude-sonnet-5.5 --dry-run
    python3 scripts/everyday_prompts.py label --set eval --model anthropic/claude-sonnet-5.5 --budget 3
    python3 scripts/everyday_prompts.py status                      # agreement, cost, review list

Labelling makes paid OpenRouter calls: always --dry-run first and ask before running.
Only safe prompts are sampled (English, not toxic, not redacted, not flagged by
moderation), and a reply that looks filtered is never retried, so no provider
flags the account. Files live in data/task_data/everyday/ (gitignored; WildChat
text is never committed). Resumable: labels are appended per prompt.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from benchmark.run import CallError, FatalError, OpenRouterClient, call_with_retries  # noqa: E402
from lonn.task_types import load_task_types  # noqa: E402

OUT = ROOT / 'data' / 'task_data' / 'everyday'
WILDCHAT = ('allenai/WildChat-1M', '7d6490e')
FILES = 14
ROW_GROUPS_PER_FILE = 5
SEED = 20261023
SIZES = {'eval': 300, 'train': 3000}
MAX_PER_USER = 3
MAX_CHARS = 6000          # longer first turns are skipped
LABEL_CHARS = 3000        # the labeller sees the start and end of longer prompts
GOLD_LABELLERS = ('anthropic/claude-sonnet-5.5', 'google/gemini-3.8-flash')
WORKERS = 8
# Output-token guesses per label (low, high) for --dry-run; reasoning models write more.
OUTPUT_GUESS = {'anthropic/claude-sonnet-5.5': (15, 40), 'google/gemini-3.8-flash': (15, 800),
                'deepseek/deepseek-v4.1-flash': (15, 800)}

RULES = """Rules:
- Pick the type for what the user wants done. If several fit, pick the main task.
- Any arithmetic or mathematics problem, however simple, is math_problem. A
  calculation inside a professional field (finance, physics, chemistry,
  engineering) is knowledge_quantitative.
- code_algorithmic: a full program or algorithm with input/output format,
  constraints or efficiency limits (competitive-programming style).
  code_function: write, fix or explain a short piece of code. Other software
  questions (setup, tools, concepts) are code_function too.
- instruction_following only when the request is open writing whose main point
  is explicit, checkable constraints (word counts, formats, required words). A
  summary or translation with a length limit is still summarization or
  translation.
- business_writing: documents for work (reports, proposals, emails, plans,
  marketing copy, cover letters). creative_writing: fiction, poetry, lyrics,
  jokes, role-play stories.
- conversation: chat, advice, opinions, brainstorming, casual questions with no
  single right answer. knowledge_recall: a factual or conceptual question.
- tool_calling only if the prompt itself lists tools or functions to call.
- other: anything else, including classifying or rating given items."""


def system_prompt(types) -> str:
    lines = '\n'.join(f'- {t.id}: {t.description}' for t in types)
    return ('You label user prompts sent to an AI assistant with the single best task type.\n\n'
            f'Task types:\n{lines}\n\n{RULES}\n\n'
            'Reply with JSON only: {"task_type": "<id>"}')


def normalize(text: str) -> str:
    return ' '.join(text.lower().split())


def clip(text: str, limit=LABEL_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    return f'{text[:head]}\n[...]\n{text[-(limit - head):]}'


# Sampling (free)

def safe_first_turn(row) -> str | None:
    if row['language'] != 'English' or row['toxic'] or row['redacted'] or not row['conversation']:
        return None
    first = row['conversation'][0]
    moderation = (row.get('openai_moderation') or [{}])[0] or {}
    if first['role'] != 'user' or first.get('toxic') or first.get('redacted') or moderation.get('flagged'):
        return None
    text = (first['content'] or '').strip()
    if not 15 <= len(text) <= MAX_CHARS:
        return None
    # WildChat's language tag is per conversation; drop turns that are largely non-Latin script.
    letters = [ch for ch in text if ch.isalpha()]
    if not letters or sum(ch.isascii() for ch in letters) / len(letters) < 0.9:
        return None
    return text


def sample(args):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs, rng = HfFileSystem(), random.Random(SEED)
    columns = ['conversation_hash', 'hashed_ip', 'language', 'toxic', 'redacted', 'conversation', 'openai_moderation']
    candidates, seen_prefix = [], set()
    for index in range(FILES):
        path = f'datasets/{WILDCHAT[0]}@{WILDCHAT[1]}/data/train-{index:05d}-of-{FILES:05d}.parquet'
        with fs.open(path, 'rb') as handle:
            parquet = pq.ParquetFile(handle)
            groups = rng.sample(range(parquet.metadata.num_row_groups), ROW_GROUPS_PER_FILE)
            for group in groups:
                for row in parquet.read_row_group(group, columns=columns).to_pylist():
                    text = safe_first_turn(row)
                    if text is None:
                        continue
                    prefix = normalize(text)[:80]  # templated mass prompts share a start
                    if prefix in seen_prefix:
                        continue
                    seen_prefix.add(prefix)
                    candidates.append({'id': row['conversation_hash'], 'user': row['hashed_ip'], 'text': text})
        print(f'  file {index + 1}/{FILES}: {len(candidates)} candidates', flush=True)

    per_user = defaultdict(list)
    for c in candidates:
        per_user[c['user']].append(c)
    users = sorted(per_user)
    rng.shuffle(users)
    sets, need = {'eval': [], 'train': []}, dict(SIZES)
    for user in users:  # whole users go to one set, eval first
        target = 'eval' if len(sets['eval']) < need['eval'] else 'train'
        if target == 'train' and len(sets['train']) >= need['train']:
            break
        sets[target] += per_user[user][:MAX_PER_USER]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        rows = rows[:need[name]]
        (OUT / f'{name}_prompts.jsonl').write_text(''.join(json.dumps({**r, 'user': None}) + '\n' for r in rows))
        print(f'{name}: {len(rows)} prompts from {len({r["id"] for r in rows})} conversations')
    print(f'Wrote {OUT}. Candidates after filters and de-duplication: {len(candidates)}.')


# Labelling (paid)

def load_prompts(name):
    path = OUT / f'{name}_prompts.jsonl'
    if not path.exists():
        raise SystemExit('Run `sample` first.')
    return [json.loads(line) for line in path.read_text().splitlines()]


def label_path(name, model):
    return OUT / 'labels' / f"{name}--{model.replace('/', '__')}.jsonl"


def read_labels(name, model) -> dict:
    path = label_path(name, model)
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        labels[row['id']] = row
    return labels


def parse_label(text: str, known: set) -> str | None:
    match = re.search(r'\{[^{}]*"task_type"[^{}]*\}', text or '')
    try:
        value = json.loads(match.group(0))['task_type'] if match else None
    except (json.JSONDecodeError, KeyError):
        value = None
    return value if value in known else None


def price(model_id):
    """(input, output) USD per million tokens from config/models.yaml."""
    import yaml
    for entry in yaml.safe_load((ROOT / 'config' / 'models.yaml').read_text()):
        if entry['id'] == f'openrouter/{model_id}':
            return entry['cost_per_m_input'], entry['cost_per_m_output']
    raise SystemExit(f'{model_id} is not in config/models.yaml; add its prices first.')


def label(args):
    types = load_task_types()
    known = {t.id for t in types}
    system = system_prompt(types)
    prompts = load_prompts(args.set)[:args.limit] if args.limit else load_prompts(args.set)
    done = read_labels(args.set, args.model)
    # Relabel call errors and unusable replies, but never a reply that looks filtered.
    todo = [p for p in prompts if p['id'] not in done
            or (not done[p['id']].get('task_type') and not done[p['id']].get('filtered'))]
    cin, cout = price(args.model)
    tokens_in = sum((len(system) + len(clip(p['text']))) / 3.5 + 10 for p in todo)
    low, high = OUTPUT_GUESS.get(args.model, (15, 800))
    estimate = (tokens_in * cin + len(todo) * low * cout) / 1e6, (tokens_in * cin + len(todo) * high * cout) / 1e6
    spent = sum(r.get('cost_usd') or 0 for r in done.values())
    print(f'{args.model} on {args.set}: {len(todo)} to label ({len(done)} saved, {spent:.3f} USD spent); '
          f'estimate {estimate[0]:.2f}-{estimate[1]:.2f} USD')
    if args.dry_run or not todo:
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not key:
        raise SystemExit('Set OPENROUTER_API_KEY in the environment or .env')
    client = OpenRouterClient(key, timeout=120, deadline=300)
    path = label_path(args.set, args.model)
    path.parent.mkdir(parents=True, exist_ok=True)

    def one(prompt):
        body = {'model': args.model, 'temperature': 0, 'max_tokens': args.max_tokens, 'usage': {'include': True},
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': f'Prompt to label:\n<<<\n{clip(prompt["text"])}\n>>>'}]}
        if args.reasoning:
            body['reasoning'] = {'effort': args.reasoning, 'exclude': True}
        try:
            data, _ = call_with_retries(client, body)
        except CallError as exc:
            return {'id': prompt['id'], 'error': str(exc)[:300], 'filtered': exc.kind == 'filtered', 'cost_usd': 0.0}
        text = data['choices'][0]['message'].get('content') or ''
        usage = data.get('usage') or {}
        return {'id': prompt['id'], 'task_type': parse_label(text, known), 'raw': text[:200],
                'cost_usd': usage.get('cost') or 0.0, 'input_tokens': usage.get('prompt_tokens'),
                'output_tokens': usage.get('completion_tokens'), 'error': None}

    total, count = spent, 0
    with path.open('a') as handle, ThreadPoolExecutor(WORKERS) as pool:
        # One batch of WORKERS calls at a time, so the budget check can stop the run.
        for start in range(0, len(todo), WORKERS):
            try:
                rows = list(pool.map(one, todo[start:start + WORKERS]))
            except FatalError as exc:
                raise SystemExit(f'Stopped: {exc}') from None
            for row in rows:
                handle.write(json.dumps(row) + '\n')
                total += row['cost_usd']
            handle.flush()
            count += len(rows)
            if count % 96 < WORKERS:
                print(f'  {count}/{len(todo)} labelled, {total:.3f} USD', flush=True)
            if total >= args.budget:
                print(f'Budget of {args.budget} USD reached; stopping.')
                break
    print(f'Done: {path} ({total:.3f} USD in total for this labeller).')


# Status and review

def gold_labels() -> tuple[dict, list]:
    """(prompt id -> gold label, disagreements). A human choice in review.csv wins."""
    prompts = {p['id']: p for p in load_prompts('eval')}
    a, b = (read_labels('eval', m) for m in GOLD_LABELLERS)
    human = {}
    review = OUT / 'review.csv'
    if review.exists():
        with review.open(newline='') as handle:
            human = {r['id']: r['your_label'].strip() for r in csv.DictReader(handle) if r['your_label'].strip()}
    gold, disagreements = {}, []
    for pid in prompts:
        la, lb = a.get(pid, {}).get('task_type'), b.get(pid, {}).get('task_type')
        if pid in human:
            gold[pid] = human[pid]
        elif la and la == lb:
            gold[pid] = la
        elif la and lb:
            disagreements.append((pid, la, lb))
    return gold, disagreements


def status(args):
    known = {t.id for t in load_task_types()}
    prompts = {p['id']: p for p in load_prompts('eval')}
    gold, disagreements = gold_labels()
    print(f'eval: {len(prompts)} prompts, {len(gold)} with a gold label, {len(disagreements)} to review')
    print('  gold distribution:', dict(Counter(gold.values()).most_common()))
    for path in sorted((OUT / 'labels').glob('*.jsonl')):
        name, model = path.stem.split('--')
        rows = read_labels(name, model.replace('__', '/'))
        cost = sum(r.get('cost_usd') or 0 for r in rows.values())
        bad = sum(1 for r in rows.values() if not r.get('task_type'))
        line = f'  {name:5} {model.replace("__", "/"):32} {len(rows)} labels, {bad} unusable, {cost:.3f} USD'
        if name == 'eval' and gold:
            hits = [rows[p]['task_type'] == g for p, g in gold.items() if p in rows]
            line += f', agrees with gold on {sum(hits) / max(len(hits), 1):.1%} of {len(hits)}'
        print(line)
    review = OUT / 'review.csv'
    if disagreements and not review.exists():
        with review.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'prompt', GOLD_LABELLERS[0], GOLD_LABELLERS[1], 'your_label'])
            for pid, la, lb in disagreements:
                writer.writerow([pid, prompts[pid]['text'][:600], la, lb, ''])
        print(f'Wrote {review}: fill in your_label (one of: {", ".join(sorted(known))}).')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('sample')
    lab = sub.add_parser('label')
    lab.add_argument('--set', choices=list(SIZES), required=True)
    lab.add_argument('--model', required=True, help='OpenRouter model ID, e.g. anthropic/claude-sonnet-5.5')
    lab.add_argument('--limit', type=int, help='only the first N prompts (pilot)')
    lab.add_argument('--budget', type=float, default=5.0, help='stop when this labeller has spent this many USD')
    lab.add_argument('--max-tokens', type=int, default=1000)
    lab.add_argument('--reasoning', choices=['minimal', 'low', 'medium'],
                     help='ask reasoning models to think less (OpenRouter reasoning.effort)')
    lab.add_argument('--dry-run', action='store_true')
    sub.add_parser('status')
    args = parser.parse_args(argv)
    {'sample': sample, 'label': label, 'status': status}[args.command](args)


if __name__ == '__main__':
    main()
