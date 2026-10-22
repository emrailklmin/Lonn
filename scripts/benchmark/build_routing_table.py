"""Build config/routing_table.json from the whole answer bank (#19).

The production table uses every graded answer. Evaluation never uses this file:
eval_task_router.py rebuilds the table per fold so no question is scored by a
table that has seen its answers.

    python3 scripts/benchmark/build_routing_table.py
    python3 scripts/benchmark/build_routing_table.py --smoothing 5 --output /tmp/table.json
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from benchmark import eval_router as ev  # noqa: E402
from benchmark.build_data import DEFAULT_DATA_DIR  # noqa: E402
from benchmark.eval_task_router import annotate, table_rows  # noqa: E402
from benchmark.results import RESULTS_DIR, ResultBank  # noqa: E402
from lonn.model_pool import load_model_pool  # noqa: E402
from lonn.routing_table import DEFAULT_ROUTING_TABLE, build_table  # noqa: E402
from lonn.task_types import load_task_types  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoothing', type=float, default=5.0)
    parser.add_argument('--output', type=Path, default=DEFAULT_ROUTING_TABLE)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    bank = ResultBank(args.results_dir)
    pool = ev.bank_pool(load_model_pool(), bank, None)
    questions, _ = ev.load_questions(args.data_dir, bank)
    if not questions:
        raise SystemExit('No questions with saved answers; build the prompt files (build_data.py) first.')
    questions = annotate(questions, load_task_types())
    metadata = {'generated_by': 'scripts/benchmark/build_routing_table.py (do not edit by hand)',
                'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'git_commit': ev._git('rev-parse', '--short', 'HEAD'), 'source': 'results/benchmark',
                'questions': len(questions), 'models': [m.id for m in pool], 'smoothing': args.smoothing,
                'cost': 'mean billed cost per request (OpenRouter), output budget 8192 tokens'}
    table = build_table(table_rows(questions, pool, bank), smoothing=args.smoothing, metadata=metadata)
    table.save(args.output)
    print(f'Wrote {args.output}: {len(table.task_types)} task types, {len(pool)} models, '
          f'{len(questions)} questions.')


if __name__ == '__main__':
    main()
