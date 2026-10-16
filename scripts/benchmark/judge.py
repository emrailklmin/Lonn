"""Grade the saved WritingBench answers with an LLM judge (Claude Opus 5.5 via OpenRouter).

Each answer gets one call that scores all of its prompt's five rubric criteria (1-10),
using WritingBench's own scoring rules and per-criterion bands. WritingBench calls the
judge once per criterion; one call per answer is about a fifth of the cost. The judge
never sees which model wrote the answer. Empty answers get the lowest score without a
call. An answer passes (correct) when its mean score is at least PASS_MEAN.

Judgments are appended to results/judgments/writingbench_business/<model>.jsonl as they
arrive, then copied into the answer rows (correct, score, judge_scores ...). Rerunning
skips judged answers and retries errors; run.py --regrade keeps the judge scores.

    python3 scripts/benchmark/judge.py --dry-run             # plan and cost, no calls
    python3 scripts/benchmark/judge.py --limit 2 --budget 30 # pilot: same 2 prompts per model
    python3 scripts/benchmark/judge.py                       # everything (resumes)
    python3 scripts/benchmark/judge.py --batch               # everything via OpenRouter's batch API (half price)
    python3 scripts/benchmark/judge.py --collect [--wait]    # fetch finished batches and save the judgments

The batch API bills about half price and usually finishes within an hour (window 24 h),
but a submitted batch cannot be stopped, so the budget is checked against its estimate
before it is sent. Open batches are recorded in results/judgments/<dataset>/batches.json.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from benchmark import run  # noqa: E402
from benchmark.build_data import DEFAULT_DATA_DIR, SEED  # noqa: E402
from benchmark.results import (RESULTS_DIR, ResultBank, append_row, model_slug, read_rows,  # noqa: E402
                               rewrite_rows)

ROOT = Path(__file__).resolve().parents[2]
JUDGMENTS_DIR = ROOT / 'results' / 'judgments'
DATASET = 'writingbench_business'
JUDGE_MODEL = 'anthropic/claude-opus-5.5'
JUDGE_VERSION = 'judge-v1'
PRICE_PER_M = (4.0, 20.0)  # Opus 5.5 input/output USD per M tokens, OpenRouter 2026-10-16.
# Opus 5.5 on OpenRouter refuses requests with reasoning disabled (HTTP 400, 2026-10-16),
# so reasoning is capped at Anthropic's minimum budget instead. It is billed as output.
REASONING_TOKENS = 1024
MAX_TOKENS = REASONING_TOKENS + 1500
PASS_MEAN = 7.0  # WritingBench's "above average" band starts at 7.
BATCH_URL = 'https://openrouter.ai/api/v1/batches'
BATCH_PRICE_FACTOR = 0.5  # Measured 2026-10-16: one batch judgment cost half the per-call price.
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}

SYSTEM = 'You are an expert evaluator with extensive experience in evaluating response of given query.'
# Scoring rules and warnings follow WritingBench's evaluate_prompt (X-PLUG/WritingBench prompt.py).
RULES = '''** Scoring Rules **
"1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
"3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
"5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
"7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
"9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."

- Score each criterion separately, using that criterion's own score bands below.
- Give one concrete sentence of reason per criterion, referring to specific parts of the Response.
- Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.
- Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.
- Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.
- Scoring Range: assign an integer score between 1 and 10 for each criterion.'''
TRUNCATED_NOTE = ('- This Response was cut off by a length limit. Judge it exactly as given: '
                  'anything missing counts as missing.')


def build_request(record: dict, response: str, truncated: bool) -> dict:
    criteria = record['reference']['checklist']
    listed = '\n\n'.join(
        f"{number}. {item['name']}: {item['criteria_description']}\n"
        + '\n'.join(f'   {band}: {item[band]}' for band in ('1-2', '3-4', '5-6', '7-8', '9-10'))
        for number, item in enumerate(criteria, 1))
    names = ', '.join(f'"{item["name"]}"' for item in criteria)
    prompt = (f"Evaluate the Response based on the Query and each of the {len(criteria)} Criteria "
              f"below, following the Scoring Rules.\n\n{RULES}\n"
              + (f'{TRUNCATED_NOTE}\n' if truncated else '')
              + f"\n** Criteria **\n{listed}\n\n** Query **\n```\n{record['input']['query']}\n```\n\n"
              f"** Response **\n```\n{response}\n```\n\n** Output format **\n"
              "Only output this JSON, with one entry per criterion in the order given "
              f"({names}), and do not use \" inside reason:\n"
              '{"scores": [{"criterion": "<name>", "score": <integer 1-10>, "reason": "<one sentence>"}]}')
    return {'model': JUDGE_MODEL, 'max_tokens': MAX_TOKENS, 'temperature': 0.0,
            'reasoning': {'max_tokens': REASONING_TOKENS}, 'usage': {'include': True},
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]}


SCORE_ENTRY = re.compile(r'"score"\s*:\s*(\d+(?:\.0)?)\s*,\s*"reason"\s*:\s*"(.*?)"\s*[,}]', re.DOTALL)


def parse_scores(text: str, criteria: list) -> list:
    """[{criterion, score, reason}] in rubric order; ValueError if the output is unusable.

    If the JSON is malformed (seen: a missing '{' between entries), the score/reason pairs
    are read in order instead, but only when there is exactly one per criterion.
    """
    fenced = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text or '', re.DOTALL)
    raw = fenced.group(1) if fenced else (re.search(r'\{.*\}', text or '', re.DOTALL) or [None])[0]
    if raw is None:
        raise ValueError('no JSON object')
    try:
        items = json.loads(raw)['scores']
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        pairs = SCORE_ENTRY.findall(raw)
        if len(pairs) != len(criteria):
            raise ValueError(f'bad JSON: {exc}') from None
        items = [{'score': float(score), 'reason': reason} for score, reason in pairs]
    if not isinstance(items, list) or len(items) != len(criteria):
        raise ValueError(f'expected {len(criteria)} scores')
    scores = []
    for item, criterion in zip(items, criteria):
        score = item.get('score') if isinstance(item, dict) else None
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if type(score) is not int or not 1 <= score <= 10:
            raise ValueError(f'invalid score {score!r}')
        scores.append({'criterion': criterion['name'], 'score': score,
                       'reason': str(item.get('reason', ''))[:500]})
    return scores


def verdict(scores: list) -> dict:
    mean = sum(scores) / len(scores)
    return {'judge_mean': round(mean, 3), 'score': round(mean / 10, 4), 'correct': mean >= PASS_MEAN}


def judgment_path(judgments_dir: Path, model_id: str) -> Path:
    return Path(judgments_dir) / DATASET / f'{model_slug(model_id)}.jsonl'


def read_judgments(judgments_dir: Path) -> dict:
    """Latest judgment per (prompt_id, model_id)."""
    judged = {}
    for path in sorted((Path(judgments_dir) / DATASET).glob('*.jsonl')):
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                judged[(item['prompt_id'], item['model_id'])] = item
    return judged


def apply_judgments(results_dir: Path, judgments_dir: Path) -> int:
    """Copy successful judgments into the answer rows; returns how many rows changed."""
    judged = read_judgments(judgments_dir)
    changed = 0
    for path in sorted((Path(results_dir) / 'writing' / DATASET).glob('*.jsonl')):
        rows = list(read_rows(path).values())
        touched = False
        for row in rows:
            item = judged.get((row['prompt_id'], row['model_id']))
            if not item or item.get('error') or row.get('error'):
                continue
            update = {**verdict([s['score'] for s in item['scores']]),
                      'judge_scores': item['scores'], 'judge_model': item['judge_model'],
                      'judge_version': item['judge_version'], 'judge_cost_usd': item['cost_usd'],
                      'grader': 'llm_judge', 'grader_version': item['judge_version'],
                      'extracted_answer': None,
                      'grade_details': item.get('skipped') or f"mean {item['judge_mean']:.1f}/10"}
            if any(row.get(key) != value for key, value in update.items()):
                row.update(update)
                touched = True
                changed += 1
        if touched:
            rewrite_rows(path, rows)
    return changed


def reserve(workers: int) -> float:
    """Worst-case cost of the judge calls in flight, kept free under the budget."""
    return workers * (12000 * PRICE_PER_M[0] + MAX_TOKENS * PRICE_PER_M[1]) / 1e6


def _base(row: dict) -> dict:
    return {'prompt_id': row['prompt_id'], 'model_id': row['model_id'], 'judge_model': JUDGE_MODEL,
            'judge_version': JUDGE_VERSION, 'answer_truncated': row.get('finish_reason') == 'length',
            'timestamp': run._now()}


def token_cost(usage: dict, factor: float = 1.0) -> float:
    return ((usage.get('prompt_tokens') or 0) * PRICE_PER_M[0]
            + (usage.get('completion_tokens') or 0) * PRICE_PER_M[1]) / 1e6 * factor


def judgment_from_reply(record: dict, row: dict, data: dict, cost: float) -> dict:
    """A judgment from a judge reply; a paid but unparseable reply is an error that keeps its cost."""
    usage = data.get('usage') or {}
    base = {**_base(row), 'cost_usd': cost, 'input_tokens': usage.get('prompt_tokens'),
            'output_tokens': usage.get('completion_tokens'), 'generation_id': data.get('id')}
    text = ((data.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
    try:
        scores = parse_scores(text, record['reference']['checklist'])
    except ValueError as exc:
        return {**base, 'error': f'unparseable judge output: {exc}', 'raw': text[:2000]}
    return {**base, 'scores': scores, 'judge_mean': verdict([s['score'] for s in scores])['judge_mean'],
            'error': None}


def empty_judgment(record: dict, row: dict) -> dict:
    """Lowest score for an empty answer, without a call."""
    return {**_base(row), 'skipped': 'empty response', 'cost_usd': 0.0, 'error': None, 'judge_mean': 1.0,
            'scores': [{'criterion': c['name'], 'score': 1, 'reason': 'empty response'}
                       for c in record['reference']['checklist']]}


def pending_answers(records, results_dir, judgments_dir, limit, seed) -> list:
    """Answer rows still to judge: not errors, in the selected subset, not judged successfully,
    and not waiting in an open batch."""
    chosen = {r['prompt_id'] for r in run.select(records, limit, seed)}
    judged = read_judgments(judgments_dir)
    queued = {tuple(key) for batch in load_batches(judgments_dir) if not batch.get('collected')
              for key in batch['map'].values()}
    pending = []
    for path in sorted((Path(results_dir) / 'writing' / DATASET).glob('*.jsonl')):
        for row in sorted(read_rows(path).values(), key=lambda r: r['prompt_id']):
            key = (row['prompt_id'], row['model_id'])
            done = judged.get(key)
            if (row.get('error') or row['prompt_id'] not in chosen or key in queued
                    or (done and not done.get('error'))):
                continue
            pending.append(row)
    return pending


def judge_one(client, record: dict, row: dict) -> dict:
    """One judgment (worker thread). Errors are saved."""
    body = build_request(record, row['response'], row.get('finish_reason') == 'length')
    try:
        data, _ = run.call_with_retries(client, body)
    except run.CallError as exc:
        return {**_base(row), 'error': f'call failed: {exc}', 'cost_usd': 0.0}
    usage = data.get('usage') or {}
    cost = usage['cost'] if usage.get('cost') is not None else token_cost(usage)
    return judgment_from_reply(record, row, data, cost)


# Batch API (half price, cannot be stopped once submitted)

class BatchAPI:
    def __init__(self, api_key: str, transport=None):
        import httpx
        self.http = httpx.Client(timeout=300, transport=transport, headers={
            'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json',
            'X-Title': 'Lonn judge batch'})

    def _json(self, response):
        if response.status_code >= 300:
            raise SystemExit(f'Batch API HTTP {response.status_code}: {response.text[:400]}')
        return response.json()

    def create(self, payload: dict) -> dict:
        # json.dumps keeps key order: 'requests' must come after endpoint/model (else HTTP 400).
        return self._json(self.http.post(BATCH_URL, content=json.dumps(payload)))

    def get(self, batch_id: str) -> dict:
        return self._json(self.http.get(f'{BATCH_URL}/{batch_id}'))


def batches_path(judgments_dir: Path) -> Path:
    return Path(judgments_dir) / DATASET / 'batches.json'


def load_batches(judgments_dir: Path) -> list:
    path = batches_path(judgments_dir)
    return json.loads(path.read_text()) if path.exists() else []


def save_batches(judgments_dir: Path, batches: list) -> None:
    path = batches_path(judgments_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(batches, indent=2) + '\n')
    temporary.replace(path)


def estimate_batch_cost(bodies: list) -> float:
    """Generous estimate: ~3 characters per input token, full reasoning budget plus scores."""
    return sum((len(json.dumps(body)) / 3.0 * PRICE_PER_M[0]
                + (REASONING_TOKENS + 400) * PRICE_PER_M[1]) / 1e6 for body in bodies) * BATCH_PRICE_FACTOR


def reparse_errors(records, results_dir, judgments_dir) -> int:
    """Parse saved raw replies of 'unparseable' judgments again (after a parser fix); no calls.
    The new judgment keeps the original cost. Returns how many were recovered."""
    by_id = {r['prompt_id']: r for r in records}
    recovered = 0
    for (prompt_id, model_id), item in read_judgments(judgments_dir).items():
        if not (item.get('error') or '').startswith('unparseable') or not item.get('raw'):
            continue
        try:
            scores = parse_scores(item['raw'], by_id[prompt_id]['reference']['checklist'])
        except ValueError:
            continue
        fixed = {k: v for k, v in item.items() if k not in ('error', 'raw')}
        fixed.update(scores=scores, judge_mean=verdict([s['score'] for s in scores])['judge_mean'],
                     error=None, reparsed_at=run._now())
        append_row(judgment_path(judgments_dir, model_id), fixed)
        recovered += 1
    apply_judgments(results_dir, judgments_dir)
    return recovered


def submit_batch(api, records, results_dir, judgments_dir, *, limit, budget, seed):
    """Send all pending answers as one batch; returns its id, or None if nothing was sent."""
    apply_judgments(results_dir, judgments_dir)
    by_id = {r['prompt_id']: r for r in records}
    pending = pending_answers(records, results_dir, judgments_dir, limit, seed)
    calls = []
    for row in pending:
        if (row.get('response') or '').strip():
            calls.append(row)
        else:
            append_row(judgment_path(judgments_dir, row['model_id']), empty_judgment(by_id[row['prompt_id']], row))
    apply_judgments(results_dir, judgments_dir)  # Empty answers count at once.
    if not calls:
        print(f'Nothing to send ({len(pending)} empty answers scored without a call).')
        return None
    bodies = [build_request(by_id[row['prompt_id']], row['response'], row.get('finish_reason') == 'length')
              for row in calls]
    estimate, spent = estimate_batch_cost(bodies), ResultBank(results_dir).spent()
    print(f'{len(calls)} answers to judge in one batch; estimated ${estimate:.2f} '
          f'(already spent ${spent:.2f}, budget ${budget:.2f})')
    if spent + estimate > budget:
        raise SystemExit(f'Not sent: ${spent:.2f} + estimated ${estimate:.2f} exceeds the ${budget:.2f} budget. '
                         'Use --limit or a larger --budget.')
    mapping, requests = {}, []
    for number, (row, body) in enumerate(zip(calls, bodies), 1):
        custom_id = f'j{number:04d}'  # The provider only allows [A-Za-z0-9_-].
        mapping[custom_id] = [row['prompt_id'], row['model_id']]
        requests.append({'custom_id': custom_id, 'body': {k: v for k, v in body.items() if k != 'model'}})
    created = api.create({'endpoint': '/v1/chat/completions', 'model': JUDGE_MODEL,
                          'completion_window': '24h', 'requests': requests})
    batches = load_batches(judgments_dir)
    batches.append({'id': created['id'], 'submitted_at': run._now(), 'status': created.get('status'),
                    'estimate_usd': round(estimate, 4), 'collected': False, 'map': mapping})
    save_batches(judgments_dir, batches)
    print(f"Submitted batch {created['id']} ({created.get('status')}). Fetch it later with --collect.")
    return created['id']


def collect_batches(api, records, results_dir, judgments_dir) -> list:
    """Fetch open batches; save the judgments of finished ones. Returns statuses of open batches."""
    by_id = {r['prompt_id']: r for r in records}
    answers = {}
    for path in sorted((Path(results_dir) / 'writing' / DATASET).glob('*.jsonl')):
        for row in read_rows(path).values():
            answers[(row['prompt_id'], row['model_id'])] = row
    batches, statuses = load_batches(judgments_dir), []
    for batch in batches:
        if batch.get('collected'):
            continue
        data = api.get(batch['id'])
        batch['status'] = data.get('status')
        counts = data.get('request_counts') or {}
        print(f"Batch {batch['id']}: {batch['status']}, {counts.get('completed', 0)}/{counts.get('total', '?')} "
              f"done, {counts.get('failed', 0)} failed")
        if batch['status'] not in TERMINAL:
            statuses.append(batch['status'])
            continue
        items = data.get('results') or []
        # Items carry tokens but no cost; split the batch's billed cost in proportion to tokens.
        estimates = [token_cost((item.get('response') or {}).get('body', {}).get('usage') or {})
                     for item in items]
        billed = (data.get('usage') or {}).get('cost')
        scale = billed / sum(estimates) if billed is not None and sum(estimates) else BATCH_PRICE_FACTOR
        for item, estimate in zip(items, estimates):
            prompt_id, model_id = batch['map'][item['custom_id']]
            row = answers[(prompt_id, model_id)]
            if item.get('error') or (item.get('response') or {}).get('status_code', 200) >= 300:
                judgment = {**_base(row), 'error': f"batch item failed: {str(item.get('error') or item['response'])[:300]}",
                            'cost_usd': 0.0}
            else:
                judgment = judgment_from_reply(by_id[prompt_id], row, item['response']['body'], estimate * scale)
            judgment['batch_id'] = batch['id']
            append_row(judgment_path(judgments_dir, model_id), judgment)
        missing = set(batch['map']) - {item['custom_id'] for item in items}
        for custom_id in sorted(missing):  # Failed or expired batches: retried on the next submit.
            prompt_id, model_id = batch['map'][custom_id]
            append_row(judgment_path(judgments_dir, model_id),
                       {**_base(answers[(prompt_id, model_id)]), 'batch_id': batch['id'], 'cost_usd': 0.0,
                        'error': f"no result (batch {batch['status']})"})
        batch['collected'] = True
        batch['billed_usd'] = billed
        print(f'  saved {len(items)} judgments, billed ${billed if billed is not None else 0:.4f}')
    save_batches(judgments_dir, batches)
    applied = apply_judgments(results_dir, judgments_dir)
    if applied:
        print(f'{applied} answer rows updated.')
    return statuses


def run_judge(client, records: list, results_dir: Path, judgments_dir: Path, *, limit, budget,
              workers, seed):
    """Judge pending answers; returns why it stopped early, or None."""
    apply_judgments(results_dir, judgments_dir)
    by_id = {r['prompt_id']: r for r in records}
    judged = read_judgments(judgments_dir)
    pending = pending_answers(records, results_dir, judgments_dir, limit, seed)
    spent = ResultBank(results_dir).spent()
    print(f'{len(pending)} answers to judge with {JUDGE_MODEL} ({len(judged)} judgments saved). '
          f'Budget ${budget:.2f}, already spent ${spent:.2f}, pass mark {PASS_MEAN}/10')

    counts = {'done': 0, 'passed': 0, 'errors': 0, 'cost': 0.0}

    def save(item):
        append_row(judgment_path(judgments_dir, item['model_id']), item)
        counts['done'] += 1
        counts['cost'] += item.get('cost_usd') or 0.0
        if item.get('error'):
            counts['errors'] += 1
            print(f"  ! {item['prompt_id']} [{run._short(item['model_id'])}]: {item['error'][:150]}")
        else:
            counts['passed'] += item['judge_mean'] >= PASS_MEAN
        if counts['done'] % 10 == 0 or counts['done'] == len(pending):
            graded = counts['done'] - counts['errors']
            rate = f"pass {counts['passed'] / graded:.0%}" if graded else 'pass n/a'
            print(f"  [judge] {counts['done']}/{len(pending)} | {rate} | errors {counts['errors']} "
                  f"| ${counts['cost']:.4f} | total spent ${spent + counts['cost']:.2f} of ${budget:.2f}",
                  flush=True)

    stopped = None
    calls = []
    for row in pending:
        if (row.get('response') or '').strip():
            calls.append(row)
        else:  # Nothing to judge: lowest score, no call.
            save(empty_judgment(by_id[row['prompt_id']], row))
    margin = reserve(workers)
    with ThreadPoolExecutor(workers) as pool:
        queue, running = list(calls), set()
        try:
            while (queue or running) and not stopped:
                while queue and len(running) < workers:
                    if spent + counts['cost'] + margin >= budget:
                        stopped = f"budget: ${spent + counts['cost']:.2f} spent, cap ${budget:.2f}"
                        break
                    row = queue.pop(0)
                    running.add(pool.submit(judge_one, client, by_id[row['prompt_id']], row))
                if not running:
                    break
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    try:
                        save(future.result())
                    except run.FatalError as exc:
                        stopped = str(exc)
        except KeyboardInterrupt:
            stopped = 'interrupted (Ctrl-C)'
        for future in wait(running).done:  # Paid calls in flight: save them.
            try:
                save(future.result())
            except run.FatalError:
                pass
    applied = apply_judgments(results_dir, judgments_dir)
    print(f"\n{'Stopped: ' + stopped + '. Rerun to continue.' if stopped else 'All selected answers judged.'}"
          f" Judge spend this run ${counts['cost']:.2f}; {applied} answer rows updated.")
    return stopped


def dry_run(records, results_dir, judgments_dir, limit, seed):
    by_id = {r['prompt_id']: r for r in records}
    pending = pending_answers(records, results_dir, judgments_dir, limit, seed)
    bodies = [build_request(by_id[row['prompt_id']], row['response'], row.get('finish_reason') == 'length')
              for row in pending if (row.get('response') or '').strip()]
    batch = estimate_batch_cost(bodies)
    print(f'{len(bodies)} judge calls (+{len(pending) - len(bodies)} empty answers scored without a call): '
          f'estimated ${batch / BATCH_PRICE_FACTOR:.2f} one call at a time, ${batch:.2f} with --batch. '
          f'Already spent ${ResultBank(results_dir).spent():.2f}. No calls made.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--limit', type=int, help='seeded subset of N prompts, the same for every model')
    parser.add_argument('--budget', type=float, default=45.0, help='hard cap on total recorded spend (USD)')
    parser.add_argument('--workers', type=int, default=4, help='parallel judge calls (default 4)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--judgments-dir', type=Path, default=JUDGMENTS_DIR)
    parser.add_argument('--dry-run', action='store_true', help='show the plan and cost; no calls')
    parser.add_argument('--batch', action='store_true', help='send pending answers as one half-price batch')
    parser.add_argument('--collect', action='store_true', help='fetch finished batches and save judgments')
    parser.add_argument('--wait', action='store_true', help='with --collect: poll every minute until done')
    parser.add_argument('--reparse', action='store_true',
                        help='parse saved unparseable judge replies again; no calls')
    args = parser.parse_args(argv)
    records = run.load_records(DEFAULT_DATA_DIR, [DATASET])[DATASET]
    if args.dry_run:
        return dry_run(records, args.results_dir, args.judgments_dir, args.limit, args.seed)
    if args.reparse:
        count = reparse_errors(records, args.results_dir, args.judgments_dir)
        return print(f'Recovered {count} judgments from saved replies. No calls were made.')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        raise SystemExit('Set OPENROUTER_API_KEY in the environment or .env')
    if args.batch:
        return submit_batch(BatchAPI(api_key), records, args.results_dir, args.judgments_dir,
                            limit=args.limit, budget=args.budget, seed=args.seed)
    if args.collect:
        import time
        import httpx
        api = BatchAPI(api_key)
        while True:
            try:
                open_batches = collect_batches(api, records, args.results_dir, args.judgments_dir)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if not args.wait:
                    raise
                print(f'Polling failed ({type(exc).__name__}); trying again in a minute.', flush=True)
                open_batches = True
            if not (open_batches and args.wait):
                return
            time.sleep(60)
    run_judge(run.OpenRouterClient(api_key, timeout=300, deadline=900), records, args.results_dir,
              args.judgments_dir, limit=args.limit, budget=args.budget, workers=args.workers, seed=args.seed)


if __name__ == '__main__':
    main()
