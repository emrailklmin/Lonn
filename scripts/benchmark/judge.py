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
MAX_TOKENS = 1500
PASS_MEAN = 7.0  # WritingBench's "above average" band starts at 7.

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
            'reasoning': {'enabled': False}, 'usage': {'include': True},
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]}


def parse_scores(text: str, criteria: list) -> list:
    """[{criterion, score, reason}] in rubric order; ValueError if the output is unusable."""
    fenced = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text or '', re.DOTALL)
    raw = fenced.group(1) if fenced else (re.search(r'\{.*\}', text or '', re.DOTALL) or [None])[0]
    if raw is None:
        raise ValueError('no JSON object')
    try:
        items = json.loads(raw)['scores']
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f'bad JSON: {exc}') from None
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


def judge_one(client, record: dict, row: dict) -> dict:
    """One judgment (worker thread). Errors are saved; a paid but unparseable reply keeps its cost."""
    base = {'prompt_id': row['prompt_id'], 'model_id': row['model_id'], 'judge_model': JUDGE_MODEL,
            'judge_version': JUDGE_VERSION, 'answer_truncated': row.get('finish_reason') == 'length',
            'timestamp': run._now()}
    body = build_request(record, row['response'], base['answer_truncated'])
    try:
        data, _ = run.call_with_retries(client, body)
    except run.CallError as exc:
        return {**base, 'error': f'call failed: {exc}', 'cost_usd': 0.0}
    usage = data.get('usage') or {}
    cost = usage.get('cost')
    if cost is None:
        cost = ((usage.get('prompt_tokens') or 0) * PRICE_PER_M[0]
                + (usage.get('completion_tokens') or 0) * PRICE_PER_M[1]) / 1e6
    base.update(cost_usd=cost, input_tokens=usage.get('prompt_tokens'),
                output_tokens=usage.get('completion_tokens'), generation_id=data.get('id'))
    text = (data['choices'][0].get('message') or {}).get('content') or ''
    try:
        scores = parse_scores(text, record['reference']['checklist'])
    except ValueError as exc:
        return {**base, 'error': f'unparseable judge output: {exc}', 'raw': text[:2000]}
    return {**base, 'scores': scores, 'judge_mean': verdict([s['score'] for s in scores])['judge_mean'],
            'error': None}


def run_judge(client, records: list, results_dir: Path, judgments_dir: Path, *, limit, budget,
              workers, seed):
    """Judge pending answers; returns why it stopped early, or None."""
    apply_judgments(results_dir, judgments_dir)
    by_id = {r['prompt_id']: r for r in records}
    chosen = {r['prompt_id'] for r in run.select(records, limit, seed)}
    judged = read_judgments(judgments_dir)
    pending = []
    for path in sorted((Path(results_dir) / 'writing' / DATASET).glob('*.jsonl')):
        for row in sorted(read_rows(path).values(), key=lambda r: r['prompt_id']):
            done = judged.get((row['prompt_id'], row['model_id']))
            if row.get('error') or row['prompt_id'] not in chosen or (done and not done.get('error')):
                continue
            pending.append(row)
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
            criteria = by_id[row['prompt_id']]['reference']['checklist']
            save({'prompt_id': row['prompt_id'], 'model_id': row['model_id'], 'judge_model': JUDGE_MODEL,
                  'judge_version': JUDGE_VERSION, 'skipped': 'empty response', 'cost_usd': 0.0,
                  'scores': [{'criterion': c['name'], 'score': 1, 'reason': 'empty response'}
                             for c in criteria],
                  'judge_mean': 1.0, 'error': None, 'timestamp': run._now()})
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
    chosen = {r['prompt_id'] for r in run.select(records, limit, seed)}
    judged = read_judgments(judgments_dir)
    calls = empty = 0
    cost = 0.0
    for path in sorted((Path(results_dir) / 'writing' / DATASET).glob('*.jsonl')):
        for row in read_rows(path).values():
            done = judged.get((row['prompt_id'], row['model_id']))
            if row.get('error') or row['prompt_id'] not in chosen or (done and not done.get('error')):
                continue
            if not (row.get('response') or '').strip():
                empty += 1
                continue
            body = build_request(by_id[row['prompt_id']], row['response'], row.get('finish_reason') == 'length')
            calls += 1
            cost += (len(json.dumps(body)) / 3.5 * PRICE_PER_M[0] + 400 * PRICE_PER_M[1]) / 1e6
    print(f'{calls} judge calls (+{empty} empty answers scored without a call), '
          f'estimated ${cost:.2f}. Already spent ${ResultBank(results_dir).spent():.2f}. No calls made.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--limit', type=int, help='seeded subset of N prompts, the same for every model')
    parser.add_argument('--budget', type=float, default=45.0, help='hard cap on total recorded spend (USD)')
    parser.add_argument('--workers', type=int, default=4, help='parallel judge calls (default 4)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--judgments-dir', type=Path, default=JUDGMENTS_DIR)
    parser.add_argument('--dry-run', action='store_true', help='show the plan and cost; no calls')
    args = parser.parse_args(argv)
    records = run.load_records(DEFAULT_DATA_DIR, [DATASET])[DATASET]
    if args.dry_run:
        return dry_run(records, args.results_dir, args.judgments_dir, args.limit, args.seed)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        raise SystemExit('Set OPENROUTER_API_KEY in the environment or .env')
    run_judge(run.OpenRouterClient(api_key, timeout=300, deadline=900), records, args.results_dir,
              args.judgments_dir, limit=args.limit, budget=args.budget, workers=args.workers, seed=args.seed)


if __name__ == '__main__':
    main()
