"""Run every active model on every benchmark question and save the graded answers.

Calls OpenRouter directly (not src/lonn/executor.py). Each answer is graded and
appended to results/benchmark/<category>/<dataset>/<model>.jsonl as soon as it
arrives; rerunning skips saved answers and retries saved errors.

    python3 scripts/benchmark/run.py --dry-run                 # plan and cost estimate, no calls
    python3 scripts/benchmark/run.py --limit 10 --budget 3     # pilot: 10 questions per dataset
    python3 scripts/benchmark/run.py --dataset math500 --model gpt-oss-20b
    python3 scripts/benchmark/run.py --regrade                 # regrade saved answers, no calls

--budget is a hard cap on the total recorded spend in the results folder,
including earlier runs. Needs OPENROUTER_API_KEY (environment or .env) and the
prompt files from scripts/benchmark/build_data.py.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from benchmark.build_data import DATASETS, DEFAULT_DATA_DIR, SEED  # noqa: E402
from benchmark.graders import grade  # noqa: E402
from benchmark.results import (RESULTS_DIR, ResultBank, append_row, iter_files, read_rows,  # noqa: E402
                               result_path, rewrite_rows)
from benchmark.templates import FAMILY_OF, TEMPLATE_VERSION, build_request  # noqa: E402

API_URL = 'https://openrouter.ai/api/v1/chat/completions'
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403}  # Bad key, no credits, forbidden: stop the whole run.
BACKOFF = (5, 15, 45)  # Seconds before each retry.
# Categories not run unless explicitly included (decision 2026-10-12: tool calling waits).
DEFERRED_CATEGORIES = {'tool_calling'}
# Rough output-token guesses per answer (low, high) for --dry-run before any data exists.
OUTPUT_GUESS = {'multiple_choice': (300, 2000), 'math': (500, 3000), 'code': (500, 2500),
                'tool_calling': (100, 600), 'instruction_following': (400, 1500), 'writing': (900, 2000)}


class FatalError(Exception):
    """An error that will fail every call (credentials, credits); stops the run."""


class CallError(Exception):
    def __init__(self, message, status=None, retryable=False):
        super().__init__(message)
        self.status, self.retryable = status, retryable


# OpenRouter

class OpenRouterClient:
    def __init__(self, api_key: str, timeout: float):
        import httpx
        self.http = httpx.Client(timeout=timeout, headers={
            'Authorization': f'Bearer {api_key}', 'X-Title': 'Lonn benchmark'})
        self.httpx = httpx

    def complete(self, body: dict) -> dict:
        try:
            response = self.http.post(API_URL, json=body)
        except (self.httpx.TimeoutException, self.httpx.TransportError) as exc:
            raise CallError(f'{type(exc).__name__}: {exc}', retryable=True) from None
        if response.status_code in FATAL_STATUS:
            raise FatalError(f'OpenRouter {response.status_code}: {response.text[:300]}')
        if response.status_code != 200:
            raise CallError(f'HTTP {response.status_code}: {response.text[:300]}',
                            response.status_code, response.status_code in RETRYABLE_STATUS)
        data = response.json()
        error = data.get('error') or (data.get('choices') or [{}])[0].get('error')
        if error:
            code = error.get('code') if isinstance(error, dict) else None
            code = code if isinstance(code, int) else None
            if code in FATAL_STATUS:
                raise FatalError(f'OpenRouter error: {error}')
            raise CallError(f'OpenRouter error: {str(error)[:300]}', code,
                            code is None or code in RETRYABLE_STATUS)
        if not data.get('choices'):
            raise CallError('response has no choices', retryable=True)
        return data


def call_with_retries(client, body, retries=len(BACKOFF)):
    for attempt in range(retries + 1):
        try:
            return client.complete(body), attempt
        except CallError as exc:
            if not exc.retryable or attempt == retries:
                raise
            time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)] * random.uniform(1, 1.5))


def parse_answer(data: dict) -> dict:
    choice = data['choices'][0]
    message = choice.get('message') or {}
    usage = data.get('usage') or {}
    tool_calls = [{'id': call.get('id'), 'name': (call.get('function') or {}).get('name'),
                   'arguments': (call.get('function') or {}).get('arguments')}
                  for call in message.get('tool_calls') or []]
    return {
        'response': message.get('content') or '',
        'reasoning': message.get('reasoning') or None,
        'tool_calls': tool_calls or None,
        'finish_reason': choice.get('finish_reason'),
        'native_finish_reason': choice.get('native_finish_reason'),
        'provider': data.get('provider'),
        'generation_id': data.get('id'),
        'input_tokens': usage.get('prompt_tokens'),
        'output_tokens': usage.get('completion_tokens'),
        'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
        'cost_usd': usage.get('cost'),
    }


# Running

def load_records(data_dir: Path, names) -> dict:
    records = {}
    for name in names:
        path = Path(data_dir) / 'prompts' / DATASETS[name].category / f'{name}.jsonl'
        if not path.exists():
            raise SystemExit(f'Missing {path}; run scripts/benchmark/build_data.py first')
        with path.open(encoding='utf-8') as handle:
            records[name] = [json.loads(line) for line in handle]
    return records


def select(records: list, limit, seed: int) -> list:
    """A seeded subset, the same for every model (for pilots), or everything."""
    if limit is None or limit >= len(records):
        return records
    ordered = sorted(records, key=lambda r: r['prompt_id'])
    return sorted(random.Random(f'{seed}:pilot').sample(ordered, limit), key=lambda r: r['prompt_id'])


def answer_one(client, record, model, settings) -> dict:
    """Call, grade and return one result row; call failures become error rows."""
    request = build_request(record)
    body = {'model': model.id.removeprefix('openrouter/'), 'messages': request['messages'],
            'max_tokens': settings['max_tokens'], 'temperature': settings['temperature'],
            'seed': settings['seed'], 'usage': {'include': True}}
    if request['tools']:
        body['tools'] = request['tools']
    row = {'prompt_id': record['prompt_id'], 'dataset': record['dataset'],
           'category': record['category'], 'difficulty': record['difficulty'], 'model_id': model.id,
           'prompt_template_version': TEMPLATE_VERSION, **{k: settings[k] for k in
           ('max_tokens', 'temperature', 'seed')}}
    started = time.perf_counter()
    try:
        data, retries = call_with_retries(client, body)
    except CallError as exc:
        return {**row, 'error': {'message': str(exc), 'status': exc.status}, 'correct': None,
                'cost_usd': 0.0, 'latency_ms': round((time.perf_counter() - started) * 1000),
                'timestamp': _now()}
    row.update(parse_answer(data))
    row.update(latency_ms=round((time.perf_counter() - started) * 1000), retries=retries,
               error=None, timestamp=_now())
    row['cost_source'] = 'openrouter'
    if row['cost_usd'] is None:  # Fall back to configured prices if OpenRouter omits cost.
        row['cost_usd'] = ((row['input_tokens'] or 0) * model.cost_per_m_input
                           + (row['output_tokens'] or 0) * model.cost_per_m_output) / 1e6
        row['cost_source'] = 'estimate'
    try:
        row.update(grade(record, row))
    except Exception as exc:  # A grader bug must not lose a paid answer.
        row.update(correct=None, grader='error', grade_details=f'{type(exc).__name__}: {exc}'[:300])
    return row


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _short(model_id):
    return model_id.removeprefix('openrouter/')


class Progress:
    def __init__(self, cap, spent, interval=30.0, every=10):
        self.cap, self.spent, self.interval, self.every = cap, spent, interval, every
        self.lock = threading.Lock()
        self.calls = self.errors = 0

    def block(self, model_id, dataset, total, skipped):
        self.model_id, self.dataset, self.total = _short(model_id), dataset, total
        self.done = self.correct = self.graded = self.block_errors = self.truncated = 0
        self.block_cost, self.started, self.last = 0.0, time.time(), 0.0
        print(f'\n[{self.model_id}] {DATASETS[dataset].category}/{dataset}: '
              f'{total} to run, {skipped} already saved', flush=True)

    def add(self, row):
        with self.lock:
            self.done += 1
            self.calls += 1
            self.spent += row.get('cost_usd') or 0.0
            self.block_cost += row.get('cost_usd') or 0.0
            if row.get('error'):
                self.errors += 1
                self.block_errors += 1
                print(f"  ! {row['prompt_id']}: {row['error']['message'][:150]}", flush=True)
            if row.get('correct') is not None:
                self.graded += 1
                self.correct += bool(row['correct'])
            self.truncated += row.get('finish_reason') == 'length'
            now = time.time()
            if self.done == self.total or self.done % self.every == 0 or now - self.last > self.interval:
                self.last = now
                print(f'  [{self.model_id}] {self.dataset} {self.done}/{self.total} | '
                      f'{self._accuracy()} | errors {self.block_errors} | ${self.block_cost:.4f} '
                      f'| total spent ${self.spent:.2f} of ${self.cap:.2f}', flush=True)

    def _accuracy(self):
        return f'acc {self.correct / self.graded:.0%}' if self.graded else 'acc n/a'

    def end_block(self):
        minutes, seconds = divmod(int(time.time() - self.started), 60)
        print(f'  done {self.dataset}: {self._accuracy()}, {self.truncated} truncated, '
              f'{self.block_errors} errors, ${self.block_cost:.4f}, {minutes}m{seconds:02d}s', flush=True)


def plan(models, records, results_dir, limit, seed):
    """[(model, dataset, pending records, saved count)] in run order."""
    jobs = []
    for model in models:
        for name, items in records.items():
            chosen = select(items, limit, seed)
            saved = read_rows(result_path(results_dir, DATASETS[name].category, name, model.id))
            pending = [r for r in chosen if r['prompt_id'] not in saved
                       or saved[r['prompt_id']].get('error')]
            jobs.append((model, name, pending, len(chosen) - len(pending)))
    return jobs


def reserve(models, settings):
    """Worst-case cost of the calls in flight, kept free under the budget."""
    worst = max(model.cost_per_m_output for model in models) * settings['max_tokens'] / 1e6
    return worst * settings['workers']


def run(jobs, client, settings, results_dir, budget, bank_spent):
    progress = Progress(budget, bank_spent)
    margin = reserve([job[0] for job in jobs], settings)
    stopped = None
    with ThreadPoolExecutor(settings['workers']) as pool:
        for model, name, pending, saved in jobs:
            if stopped:
                break
            if not pending:
                print(f'[{_short(model.id)}] {name}: all {saved} saved, skipping')
                continue
            progress.block(model.id, name, len(pending), saved)
            queue, running = list(pending), set()
            path = result_path(results_dir, DATASETS[name].category, name, model.id)
            try:
                while (queue or running) and not stopped:
                    while queue and len(running) < settings['workers']:
                        if progress.spent + margin >= budget:
                            stopped = f'budget: ${progress.spent:.2f} spent, cap ${budget:.2f}'
                            break
                        running.add(pool.submit(answer_one, client, queue.pop(0), model, settings))
                    if not running:
                        break
                    finished, running = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        try:
                            row = future.result()
                        except FatalError as exc:
                            stopped = str(exc)
                            continue
                        append_row(path, row)
                        progress.add(row)
            except KeyboardInterrupt:
                stopped = 'interrupted (Ctrl-C)'
                print('\nInterrupted: saving the calls already in flight...', flush=True)
            if stopped:  # Let calls already sent finish and save them: they are paid for.
                for future in wait(running).done:
                    try:
                        row = future.result()
                    except FatalError:
                        continue
                    append_row(path, row)
                    progress.add(row)
            progress.end_block()
    print(f'\nStopped: {stopped}. Rerun the same command to continue.' if stopped
          else '\nAll selected questions are saved.')
    print(f'This run: {progress.calls} calls, {progress.errors} errors. '
          f'Total recorded spend: ${progress.spent:.2f}')
    return stopped


def dry_run(jobs, results_dir):
    bank = ResultBank(results_dir)
    measured = {}
    for row in bank.answered():
        measured.setdefault((row['model_id'], row['dataset']), []).append(row.get('output_tokens') or 0)
    totals = {}
    for model, name, pending, saved in jobs:
        calls = len(pending)
        input_tokens = sum(len(json.dumps(build_request(r))) / 3.5 for r in pending)
        sample = measured.get((model.id, name))
        low, high = ((sum(sample) / len(sample),) * 2 if sample else OUTPUT_GUESS[FAMILY_OF[name]])
        cost = [(input_tokens * model.cost_per_m_input + calls * out * model.cost_per_m_output) / 1e6
                for out in (low, high)]
        total = totals.setdefault(model.id, [0, 0, 0.0, 0.0, 0])
        total[0] += calls
        total[1] += saved
        total[2] += cost[0]
        total[3] += cost[1]
        total[4] += bool(sample)
    print(f"{'model':42} {'to run':>7} {'saved':>6} {'est. cost (low-high)':>22}")
    for model_id, (calls, saved, low, high, measured_sets) in totals.items():
        print(f'{_short(model_id):42} {calls:>7} {saved:>6}   ${low:>7.2f} - ${high:>7.2f}'
              + ('  (measured)' if measured_sets else ''))
    print(f"{'total':42} {sum(t[0] for t in totals.values()):>7} {sum(t[1] for t in totals.values()):>6}"
          f"   ${sum(t[2] for t in totals.values()):>7.2f} - ${sum(t[3] for t in totals.values()):>7.2f}")
    print(f'Already spent: ${bank.spent():.2f}. Estimates use measured output tokens where saved '
          'answers exist, otherwise rough guesses; no calls were made.')


def regrade(data_dir, results_dir, names):
    records = load_records(data_dir, names)
    by_id = {r['prompt_id']: r for items in records.values() for r in items}
    changed = 0
    for path in iter_files(results_dir):
        if path.parent.name not in names:
            continue
        rows = list(read_rows(path).values())
        for row in rows:
            if row.get('error'):
                continue
            new = grade(by_id[row['prompt_id']], row)
            changed += any(row.get(k) != v for k, v in new.items())
            row.update(new)
        rewrite_rows(path, rows)
    print(f'Regraded saved answers; {changed} changed. No API calls were made.')


def select_datasets(datasets, categories, include_deferred=False) -> list:
    """Dataset names matching the filters; deferred categories need include_deferred."""
    names = [n for n, s in DATASETS.items()
             if (not datasets or n in datasets) and (not categories or s.category in categories)]
    deferred = [n for n in names if DATASETS[n].category in DEFERRED_CATEGORIES]
    if deferred and not include_deferred:
        if datasets or categories:
            raise SystemExit(f"{', '.join(deferred)} {'is' if len(deferred) == 1 else 'are'} deferred; "
                             'add --include-tool-calling to run them')
        names = [n for n in names if n not in deferred]
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', action='append', choices=list(DATASETS), help='repeatable')
    parser.add_argument('--category', action='append',
                        choices=sorted({s.category for s in DATASETS.values()}), help='repeatable')
    parser.add_argument('--model', action='append', help='substring of a model ID; repeatable')
    parser.add_argument('--limit', type=int, help='seeded subset of N questions per dataset (pilot)')
    parser.add_argument('--budget', type=float, default=45.0,
                        help='hard cap on total recorded spend in USD (default 45)')
    parser.add_argument('--workers', type=int, default=8, help='parallel calls (default 8)')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--timeout', type=float, default=300.0, help='seconds per call')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--include-tool-calling', action='store_true',
                        help='also run the deferred tool-calling (BFCL) datasets')
    parser.add_argument('--dry-run', action='store_true', help='show the plan and cost; no calls')
    parser.add_argument('--regrade', action='store_true', help='regrade saved answers; no calls')
    args = parser.parse_args(argv)

    names = select_datasets(args.dataset, args.category, args.include_tool_calling)
    if args.regrade:
        return regrade(args.data_dir, args.results_dir, names)

    from lonn.model_pool import load_model_pool
    models = [m for m in load_model_pool()
              if not args.model or any(part in m.id for part in args.model)]
    if not models or not names:
        raise SystemExit('No models or datasets match the filters')
    jobs = plan(models, load_records(args.data_dir, names), args.results_dir, args.limit, args.seed)
    print(f"{len(models)} models x {len(names)} datasets; "
          f"{sum(len(j[2]) for j in jobs)} calls to make, {sum(j[3] for j in jobs)} already saved")
    if args.dry_run:
        return dry_run(jobs, args.results_dir)

    try:  # IFEval's sentence counting needs NLTK data; fail before paying for answers.
        import nltk
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        raise SystemExit('Missing NLTK data; run: python3 -c "import nltk; nltk.download(\'punkt_tab\')"')
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        raise SystemExit('Set OPENROUTER_API_KEY in the environment or .env')
    settings = {'max_tokens': args.max_tokens, 'temperature': args.temperature,
                'seed': args.seed, 'workers': args.workers}
    spent = ResultBank(args.results_dir).spent()
    print(f'Budget ${args.budget:.2f}, already spent ${spent:.2f}, {args.workers} parallel calls, '
          f'max_tokens {args.max_tokens}, template {TEMPLATE_VERSION}')
    run(jobs, OpenRouterClient(api_key, args.timeout), settings, args.results_dir, args.budget, spent)


if __name__ == '__main__':
    main()
