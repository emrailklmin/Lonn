"""Run lonn.answer_check on every saved answer in the answer bank (#19, docs/oracle_gap.md).

Writes results/benchmark_checks/answer_checks.jsonl: per (question, model), whether
the reply passes the checks production runs before returning it (not truncated, not
empty, and for programming tasks with worked examples in the prompt, code that
reproduces them). eval_task_router.py --cascade replays escalation from this file.
No API calls. Code runs one answer at a time in the sandbox, as graders.py requires.

    python3 scripts/benchmark/answer_checks.py           # resumes; reruns rows from an older CHECK_VERSION
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from benchmark.build_data import DATASETS, DEFAULT_DATA_DIR  # noqa: E402
from benchmark.results import RESULTS_DIR, ResultBank  # noqa: E402
from benchmark.templates import user_message  # noqa: E402
from lonn.answer_check import CHECK_VERSION, check_answer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'benchmark_checks' / 'answer_checks.jsonl'


def read_checks(path: Path = OUT) -> dict:
    checks = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            checks[(row['prompt_id'], row['model_id'])] = row
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args(argv)

    bank = ResultBank(args.results_dir)
    prompts = {}
    for name, spec in DATASETS.items():
        path = Path(args.data_dir) / 'prompts' / spec.category / f'{name}.jsonl'
        if path.exists() and not name.startswith('bfcl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                prompts[record['prompt_id']] = user_message(record)
    done = {k: v for k, v in read_checks(args.output).items() if v.get('check_version') == CHECK_VERSION}
    todo = sorted(k for k, row in bank.rows.items() if k[0] in prompts and not row.get('error') and k not in done)
    print(f'{len(done)} checked, {len(todo)} to check', flush=True)
    rows = list(done.values())
    for i, (prompt_id, model_id) in enumerate(todo, 1):
        answer = bank.rows[(prompt_id, model_id)]
        result = check_answer(prompts[prompt_id], answer.get('response'), answer.get('finish_reason'))
        rows.append({'prompt_id': prompt_id, 'model_id': model_id, 'ok': result.ok, 'reason': result.reason,
                     'check_version': CHECK_VERSION})
        if i % 200 == 0:
            print(f'  {i}/{len(todo)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r['prompt_id'], r['model_id']))
    args.output.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    failed = sum(not r['ok'] for r in rows)
    print(f'Wrote {args.output}: {len(rows)} answers, {failed} fail a check.')


if __name__ == '__main__':
    main()
