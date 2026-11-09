"""Always one fixed model (default Claude Sonnet), with and without the answer check (#19).

Without the check this is the fixed-model baseline. With it, a failed check escalates once
to escalation_target(), as in production: the best other model at accuracy - lam * cost, if
that score is positive. The table behind the escalation is cross-fitted like
eval_task_router.py. No API calls.

    python3 scripts/benchmark/eval_fixed_cascade.py --lambdas 0 5 10 15 20
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_task_router as etr  # noqa: E402
from eval_task_router import ev  # noqa: E402
from benchmark.answer_checks import OUT as CHECKS_PATH, read_checks  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='openrouter/anthropic/claude-sonnet-5.5')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[20])
    parser.add_argument('--signals', type=Path,
                        default=etr.ROOT / 'results/classifier_bakeoff/bge-small-linear-wildchat/benchmark_signals.jsonl')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args(argv)

    bank = etr.ResultBank(etr.RESULTS_DIR)
    pool = ev.bank_pool(etr.load_model_pool(), bank, None)
    questions, _ = ev.load_questions(etr.DEFAULT_DATA_DIR, bank)
    questions = etr.annotate(questions, etr.load_task_types())
    signals = etr.load_signals(args.signals)
    checks = read_checks(CHECKS_PATH)
    fixed = next(m for m in pool if m.id == args.model)

    stats = {lam: {'correct': 0, 'cost': 0.0, 'escalated': 0, 'targets': Counter()} for lam in args.lambdas}
    plain = {'correct': 0, 'cost': 0.0}
    for repeat in range(args.repeats):
        assignment = etr.folds_for(questions, args.folds, repeat)
        for fold in range(args.folds):
            train = [q for q, f in zip(questions, assignment) if f != fold]
            test = [q for q, f in zip(questions, assignment) if f == fold]
            table = etr.build_table(etr.table_rows(train, pool, bank), smoothing=5.0)
            for q in test:
                d = etr.decide(q, [fixed], bank)
                plain['correct'] += d['correct']
                plain['cost'] += d['cost_usd']
                if d['chosen'] is None:
                    for lam in args.lambdas:
                        stats[lam]['cost'] += 0.0
                    continue
                signal = etr.signal_for(q, signals)
                for lam in args.lambdas:
                    target = etr.escalation_target(signal, table, pool, lam, fixed)
                    r = etr.cascade(q, d, target, bank, checks)
                    s = stats[lam]
                    s['correct'] += r['correct']
                    s['cost'] += r['cost_usd']
                    if r['escalated']:
                        s['escalated'] += 1
                        s['targets'][r['chosen'].split('/')[-1]] += 1
    n = len(questions) * args.repeats
    print(f"{args.model} without check: {plain['correct'] / n:.1%} at {plain['cost'] / args.repeats:.3f} USD")
    for lam, s in stats.items():
        top = ', '.join(f'{k} {v / max(s["escalated"], 1):.0%}' for k, v in s['targets'].most_common(3))
        print(f"with check, lambda {lam:g}: {s['correct'] / n:.1%} at {s['cost'] / args.repeats:.3f} USD, "
              f"escalated {s['escalated'] / n:.1%} (to {top})")


if __name__ == '__main__':
    main()
