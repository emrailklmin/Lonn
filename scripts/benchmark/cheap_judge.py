"""Can a cheap, generic judge tell good long-form writing from bad? (#19, docs/oracle_gap.md)

Production has no per-prompt rubric, so the judge sees only the request and the
response and rates how well the response fulfils it, 1-10. It is run on every saved
WritingBench answer that is not cut off (the truncation check already catches those)
and stored in results/benchmark_checks/writing_judge.jsonl, to compare with the Opus
rubric judge's pass/fail and to replay a judge-based escalation.

Paid OpenRouter calls: --dry-run first and ask before running.

    python3 scripts/benchmark/cheap_judge.py --dry-run
    python3 scripts/benchmark/cheap_judge.py --budget 1.5
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from benchmark.build_data import DATASETS, DEFAULT_DATA_DIR  # noqa: E402
from benchmark.results import ResultBank  # noqa: E402
from benchmark.run import CallError, FatalError, OpenRouterClient, call_with_retries  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'benchmark_checks' / 'writing_judge.jsonl'
DATASET = 'writingbench_business'
JUDGE = 'deepseek/deepseek-v4.1-flash'
PRICE_PER_M = (0.15, 0.6)  # config/models.yaml
JUDGE_VERSION = 'generic-v1'
WORKERS = 8
SYSTEM = ('You review answers written by an AI assistant. Be strict: an answer that is generic, padded, '
          'incomplete, off-topic or ignores explicit requirements of the request deserves a low score.')
PROMPT = '''Rate how well the RESPONSE fulfils the REQUEST, from 1 (fails it) to 10 (excellent: complete,
specific, accurate, follows every explicit requirement, professional quality).

REQUEST:
<<<
{query}
>>>

RESPONSE:
<<<
{response}
>>>

Reply with JSON only: {{"score": <integer 1-10>}}'''


def read_judgments(path: Path = OUT) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            out[(row['prompt_id'], row['model_id'])] = row
    return out


def parse_score(text: str):
    match = re.search(r'"score"\s*:\s*(\d+)', text or '')
    score = int(match.group(1)) if match else None
    return score if score is not None and 1 <= score <= 10 else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--budget', type=float, default=1.5)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    spec = DATASETS[DATASET]
    queries = {}
    for line in (DEFAULT_DATA_DIR / 'prompts' / spec.category / f'{DATASET}.jsonl').read_text().splitlines():
        record = json.loads(line)
        queries[record['prompt_id']] = record['input']['query']
    bank = ResultBank()
    done = read_judgments()
    todo = [(pid, mid, row) for (pid, mid), row in sorted(bank.rows.items())
            if pid in queries and not row.get('error') and row.get('finish_reason') != 'length'
            and (row.get('response') or '').strip() and not (done.get((pid, mid)) or {}).get('score')]
    tokens = sum((len(SYSTEM) + len(PROMPT) + len(queries[p]) + len(r['response'])) / 3.5 for p, _, r in todo)
    estimate = (tokens * PRICE_PER_M[0] + len(todo) * 400 * PRICE_PER_M[1]) / 1e6
    print(f'{len(todo)} answers to judge ({len(done)} saved); estimate {estimate:.2f} USD')
    if args.dry_run or not todo:
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    client = OpenRouterClient(os.environ['OPENROUTER_API_KEY'].strip(), timeout=120, deadline=300)

    def one(job):
        pid, mid, row = job
        body = {'model': JUDGE, 'temperature': 0, 'max_tokens': 1500, 'usage': {'include': True},
                'reasoning': {'effort': 'low', 'exclude': True},
                'messages': [{'role': 'system', 'content': SYSTEM},
                             {'role': 'user', 'content': PROMPT.format(query=queries[pid], response=row['response'])}]}
        try:
            data, _ = call_with_retries(client, body)
        except CallError as exc:
            return {'prompt_id': pid, 'model_id': mid, 'score': None, 'error': str(exc)[:200], 'cost_usd': 0.0}
        usage = data.get('usage') or {}
        return {'prompt_id': pid, 'model_id': mid, 'judge': JUDGE, 'judge_version': JUDGE_VERSION,
                'score': parse_score(data['choices'][0]['message'].get('content')),
                'cost_usd': usage.get('cost') or 0.0, 'error': None}

    spent = sum(r.get('cost_usd') or 0 for r in done.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('a') as handle, ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(todo), WORKERS):
            try:
                rows = list(pool.map(one, todo[start:start + WORKERS]))
            except FatalError as exc:
                raise SystemExit(f'Stopped: {exc}') from None
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
                spent += row['cost_usd']
            handle.flush()
            if spent >= args.budget:
                print(f'Budget of {args.budget} USD reached; stopping.')
                break
    print(f'Done: {OUT} ({spent:.3f} USD in total).')


if __name__ == '__main__':
    main()
