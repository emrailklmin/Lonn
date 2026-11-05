"""Evaluate the task-type router (#19) against the answer bank, cross-fitted.

The routing table is built from the answer bank, so scoring the router on the
same questions would let it see the answers. Every number here is therefore
**cross-fitted**: the questions are split into folds (stratified per dataset),
the table is built from the other folds, and the router is scored on the
held-out fold. This repeats with new random splits and the results are averaged.

The classifier is replaced by the truth (an upper bound for a classifier): each
question's task type comes from config/task_types.yaml, and its complexity from
the dataset's difficulty label (easy 0, medium 0.5, hard 1). The point is to
test the routing logic before training a classifier.

    python3 scripts/benchmark/eval_task_router.py --name task-true
    python3 scripts/benchmark/eval_task_router.py --name task-true-tau --mode threshold --taus 0.8 0.9 0.95
    python3 scripts/benchmark/eval_task_router.py --name task-gte-linear --signals results/classifier_bakeoff/gte-base-linear/benchmark_signals.jsonl

Results go to results/router_eval/<name>/ and a row of results/router_eval/runs.csv,
like eval_router.py. No API calls are made.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from benchmark import eval_router as ev  # noqa: E402
from benchmark.build_data import DATASETS, DEFAULT_DATA_DIR  # noqa: E402
from benchmark.results import RESULTS_DIR, ResultBank  # noqa: E402
from lonn.model_pool import load_model_pool  # noqa: E402
from lonn.routing_table import (DEFAULT_TASK_APTITUDES, add_supplied_aptitudes, build_table,  # noqa: E402
                                load_supplied_aptitudes)
from lonn.schemas import TaskSignal  # noqa: E402
from lonn.task_router import (DEFAULT_LAMBDA, ESCALATION_FACTOR, LEVEL_POSITION, escalation_target,  # noqa: E402
                              rank_models)
from lonn.task_types import difficulty_for_record, load_task_types, task_for_record  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCORER = 'true-task'
DEFAULT_LAMBDAS = [0, 2, 5, 10, 15, 20, 30, 40, 60, 80, 120, 160]


def annotate(questions, types) -> list:
    """Add the true task type and difficulty level to every question."""
    by_id = {t.id: t for t in types}
    out = []
    for q in questions:
        task = task_for_record(q, types)
        if task is None:
            raise ValueError(f"{q['prompt_id']}: dataset {q['dataset']} has no task type")
        out.append({**q, 'task': task, 'level': difficulty_for_record(q, by_id[task])})
    return out


def table_rows(questions, pool, bank) -> list:
    """Graded answers of these questions as routing-table rows (failed calls left out)."""
    rows = []
    for q in questions:
        for m in pool:
            result = ev.outcome(bank.lookup(q['prompt_id'], m.id), m)
            if result is not None:
                rows.append({'task': q['task'], 'difficulty': q['level'], 'model_id': m.id,
                             'correct': result[0], 'cost_usd': result[1]})
    return rows


def load_saved_questions(path: Path, bank) -> list:
    """Recover original question metadata from a committed evaluation and answer bank.

    Prompt text is not needed for replay: classifier signals and answer checks
    were already saved. Sort by the original dataset and prompt-file order so
    the same seeded cross-validation folds are obtained.
    """
    decisions = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    metadata = {}
    for (prompt_id, _), row in bank.rows.items():
        fields = {key: row.get(key) for key in ('prompt_id', 'dataset', 'category', 'difficulty')}
        if prompt_id in metadata and metadata[prompt_id] != fields:
            raise ValueError(f'{prompt_id}: inconsistent question metadata in answer bank')
        metadata[prompt_id] = fields
    known_tasks = {t.id for t in load_task_types()}
    questions, seen = [], set()
    for decision in decisions:
        prompt_id = decision['prompt_id']
        if prompt_id in seen or prompt_id not in metadata:
            raise ValueError(f'{prompt_id}: duplicate question or missing answer-bank metadata')
        seen.add(prompt_id)
        task, level = decision['task'], decision['level']
        if task not in known_tasks or (level is not None and level not in LEVEL_POSITION):
            raise ValueError(f'{prompt_id}: invalid saved task or difficulty')
        questions.append({**metadata[prompt_id], 'task': task, 'level': level})
    order = {name: i for i, name in enumerate(DATASETS)}
    if not questions or any(q['dataset'] not in order for q in questions):
        raise ValueError('Saved questions are empty or include unknown datasets')
    return sorted(questions, key=lambda q: (order[q['dataset']], q['prompt_id']))


def folds_for(questions, folds, seed) -> list:
    """Fold index per question, stratified per dataset."""
    rng, assigned = random.Random(seed), {}
    by_dataset = {}
    for i, q in enumerate(questions):
        by_dataset.setdefault(q['dataset'], []).append(i)
    for indices in by_dataset.values():
        indices = indices[:]
        rng.shuffle(indices)
        offset = rng.randrange(folds)
        for j, i in enumerate(indices):
            assigned[i] = (j + offset) % folds
    return [assigned[i] for i in range(len(questions))]


def signal_for(q, signals=None) -> TaskSignal:
    """A classifier's saved signal for this question, or by default the truth: the task type,
    and complexity from the difficulty label."""
    if signals is not None:
        saved = signals[q['prompt_id']]
        return TaskSignal(dict(saved['task_scores']), saved.get('complexity'))
    return TaskSignal({q['task']: 1.0}, LEVEL_POSITION[q['level']] if q['level'] else None)


def load_signals(path) -> dict:
    """prompt_id -> {task_scores, complexity} from a classifier's JSON Lines file."""
    signals = {}
    for line in Path(path).read_text().splitlines():
        item = json.loads(line)
        total = sum(item['task_scores'].values())
        signals[item['prompt_id']] = {'task_scores': {t: p / total for t, p in item['task_scores'].items()},
                                      'complexity': item.get('complexity')}
    return signals


def decide(q, ranking, bank) -> dict:
    """Production fallback rule: next candidate only when the saved call failed."""
    for fallbacks, m in enumerate(ranking):
        result = ev.outcome(bank.lookup(q['prompt_id'], m.id), m)
        if result is not None:
            correct, cost, latency, _ = result
            return {'chosen': m.id, 'fallbacks': fallbacks, 'correct': correct, 'cost_usd': cost,
                    'latency_ms': latency}
    return {'chosen': None, 'fallbacks': len(ranking), 'correct': False, 'cost_usd': 0.0, 'latency_ms': 0}


def cascade(q, d, target, bank, checks) -> dict:
    """Production's answer check: escalate once when the chosen answer fails it (both are paid)."""
    check = checks.get((q['prompt_id'], d['chosen']))
    if d['chosen'] is None or check is None or check['ok'] or target is None:
        return {**d, 'escalated': False, 'check': None if check is None else check['reason']}
    result = ev.outcome(bank.lookup(q['prompt_id'], target.id), target)
    if result is None:  # the escalation call failed: keep the first answer
        return {**d, 'escalated': False, 'check': check['reason']}
    correct, cost, latency, _ = result
    return {**d, 'chosen': target.id, 'correct': correct, 'cost_usd': d['cost_usd'] + cost,
            'latency_ms': d['latency_ms'] + latency, 'escalated': True, 'check': check['reason']}


def crossfit(questions, pool, bank, settings, folds=5, repeats=10, smoothing=5.0, seed=0, signals=None,
             checks=None, escalation_factor=ESCALATION_FACTOR, supplied_aptitudes=None) -> dict:
    """settings: list of (key, rank_models kwargs). Returns per key the averaged metrics,
    the choice shares and, from the first repeat, the per-question decisions."""
    out = {key: {'items': [], 'choices': Counter(), 'decisions': []} for key, _ in settings}
    for repeat in range(repeats):
        assignment = folds_for(questions, folds, seed + repeat)
        for fold in range(folds):
            train = [q for q, f in zip(questions, assignment) if f != fold]
            test = [q for q, f in zip(questions, assignment) if f == fold]
            table = build_table(table_rows(train, pool, bank), smoothing=smoothing)
            if supplied_aptitudes is not None:
                table = add_supplied_aptitudes(table, supplied_aptitudes)
            for key, kwargs in settings:
                for q in test:
                    signal = signal_for(q, signals)
                    rank_kwargs = {k: v for k, v in kwargs.items() if k != 'cascade'}
                    ranking = rank_models(signal, table, pool, limit=ev.FALLBACK_LIMIT, **rank_kwargs)
                    d = decide(q, ranking, bank)
                    if kwargs.get('cascade') and d['chosen'] is not None:
                        first = next(m for m in pool if m.id == d['chosen'])
                        target = escalation_target(signal, table, pool, rank_kwargs.get('lam', DEFAULT_LAMBDA), first,
                                                   factor=escalation_factor,
                                                   use_complexity=rank_kwargs.get('use_complexity', False))
                        d = cascade(q, d, target, bank, checks or {})
                    out[key]['items'].append({k: q[k] for k in ev.GROUPS} | {'task': q['task']} | d)
                    out[key]['choices'][d['chosen']] += 1
                    if repeat == 0:
                        out[key]['decisions'].append({'prompt_id': q['prompt_id'], 'task': q['task'],
                                                      'level': q['level'], 'fold': fold,
                                                      'signal': {'task_scores': signal.task_scores,
                                                                 'complexity': signal.complexity},
                                                      'ranking': [m.id for m in ranking], **d})
    summary = {}
    for key, data in out.items():
        items, total = data['items'], sum(data['choices'].values())
        m = ev.metrics(items)
        # Each repeat answers every question once: report per-pass-through totals.
        m.update(questions=len(questions), passes=round(m['passes'] / repeats, 4),
                 cost_usd=round(m['cost_usd'] / repeats, 6),
                 escalated_share=round(sum(bool(i.get('escalated')) for i in items) / len(items), 4))
        by_task = {}
        for item in items:
            by_task.setdefault(item['task'], []).append(item)
        summary[key] = {'metrics': m,
                        'choice_share': {k: round(v / total, 4) for k, v in data['choices'].most_common()},
                        'by_task': {t: ev.metrics(v) | {'questions': len(v) // repeats,
                                                        'passes': round(sum(i['correct'] for i in v) / repeats, 4),
                                                        'cost_usd': round(sum(i['cost_usd'] for i in v) / repeats, 6)}
                                    for t, v in sorted(by_task.items())},
                        'decisions': data['decisions']}
    return summary


def baselines(questions, pool, bank) -> dict:
    per_model = {m.id: [] for m in pool}
    oracle = []
    for q in questions:
        options = []
        for m in pool:
            result = ev.outcome(bank.lookup(q['prompt_id'], m.id), m)
            correct, cost, latency = (False, 0.0, 0) if result is None else result[:3]
            per_model[m.id].append({'correct': correct, 'cost_usd': cost, 'latency_ms': latency})
            if result is not None:
                options.append((correct, cost, latency))
        right = [o for o in options if o[0]]
        pick = min(right or options, key=lambda o: o[1])
        oracle.append({'correct': pick[0], 'cost_usd': pick[1], 'latency_ms': pick[2]})
    models = {mid: ev.metrics(items) for mid, items in per_model.items()}
    best = max(pool, key=lambda m: (models[m.id]['passes'], -models[m.id]['cost_usd']))
    cheapest = min(pool, key=lambda m: (m.cost_per_m_input + m.cost_per_m_output, m.id))
    return {'oracle': ev.metrics(oracle), 'best_fixed': {'model': best.id, **models[best.id]},
            'cheapest': {'model': cheapest.id, **models[cheapest.id]}, 'models': models}


def reference_run(eval_dir, name):
    """Pass rate and cost of an earlier eval_router.py run (the current router)."""
    import csv
    path = Path(eval_dir) / 'runs.csv'
    if not path.exists():
        return None
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['run_name'] == name:
                return {'run': name, 'pass_rate': float(row['pass_rate']), 'cost_usd': float(row['total_cost_usd'])}
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True)
    parser.add_argument('--description', default='')
    parser.add_argument('--mode', choices=['utility', 'threshold'], default='utility')
    parser.add_argument('--lam', type=float, default=DEFAULT_LAMBDA, help='lambda for the headline run (utility mode)')
    parser.add_argument('--lambdas', type=float, nargs='+', default=DEFAULT_LAMBDAS, help='lambda sweep')
    parser.add_argument('--tau', type=float, default=0.9, help='tau for the headline run (threshold mode)')
    parser.add_argument('--taus', type=float, nargs='+', default=[0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
    parser.add_argument('--cascade', action='store_true',
                        help='check each answer (results/benchmark_checks, answer_checks.py) and escalate once on failure')
    parser.add_argument('--escalation-factor', type=float, default=ESCALATION_FACTOR,
                        help='with --cascade: escalation ranks at lambda times this')
    parser.add_argument('--levels-from-signals', action='store_true',
                        help="bin each question's predicted complexity (from --signals) into the table's levels")
    parser.add_argument('--use-complexity', action='store_true',
                        help='headline run with complexity (default: the router default, off); '
                             'the sweep always covers both')
    parser.add_argument('--signals', type=Path,
                        help='a classifier\'s benchmark_signals.jsonl instead of the true task type')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--smoothing', type=float, default=5.0)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--reference-run', default='laya-stock-a05',
                        help='eval_router.py run used as "the current router" for the gap closed')
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--eval-dir', type=Path, default=ev.EVAL_DIR)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--saved-questions', type=Path,
                        help='replay question metadata from a previous decisions.jsonl when prompt files are absent')
    parser.add_argument('--legacy-global-fallback', action='store_true',
                        help='reproduce Prototype 3.0 without supplied or derived task scores')
    args = parser.parse_args(argv)

    bank = ResultBank(args.results_dir)
    pool = ev.bank_pool(load_model_pool(), bank, args.models)
    if args.saved_questions:
        questions, excluded = load_saved_questions(args.saved_questions, bank), 0
    else:
        questions, excluded = ev.load_questions(args.data_dir, bank)
        if not questions:
            raise SystemExit('No questions with saved answers; build the prompt files or use --saved-questions.')
        questions = annotate(questions, load_task_types())
    signals = load_signals(args.signals) if args.signals else None
    if args.levels_from_signals:
        # The table's difficulty cells use the same scale as the router: the predicted complexity.
        if signals is None:
            raise SystemExit('--levels-from-signals needs --signals with complexity')
        for q in questions:
            c = signals[q['prompt_id']]['complexity']
            q['level'] = None if c is None else ('easy' if c < 1 / 3 else 'medium' if c < 2 / 3 else 'hard')
    if signals is not None and (missing := [q['prompt_id'] for q in questions if q['prompt_id'] not in signals]):
        raise SystemExit(f'{len(missing)} questions have no signal in {args.signals}, e.g. {missing[0]}')

    if args.mode == 'utility':
        points = [('lam', lam) for lam in sorted(set(args.lambdas) | {args.lam})]
        headline = args.lam
    else:
        points = [('tau', tau) for tau in sorted(set(args.taus) | {args.tau})]
        headline = args.tau
    settings = []
    for use_complexity in (True, False):
        for kind, value in points:
            kwargs = {'mode': args.mode, 'use_complexity': use_complexity, 'cascade': args.cascade,
                      **({'lam': value} if kind == 'lam' else {'tau': value})}
            settings.append(((use_complexity, value), kwargs))
    print(f'Cross-fitting {len(questions)} questions: {args.folds} folds x {args.repeats} repeats, '
          f'{len(settings)} settings...', flush=True)
    checks = None
    if args.cascade:
        from benchmark.answer_checks import OUT as CHECKS_PATH, read_checks
        checks = read_checks(CHECKS_PATH)
        if not checks:
            raise SystemExit('No answer checks: run scripts/benchmark/answer_checks.py first.')
    supplied = None if args.legacy_global_fallback else load_supplied_aptitudes()
    results = crossfit(questions, pool, bank, settings, args.folds, args.repeats, args.smoothing, signals=signals,
                       checks=checks, escalation_factor=args.escalation_factor, supplied_aptitudes=supplied)
    base = baselines(questions, pool, bank)
    reference = reference_run(args.eval_dir, args.reference_run)

    def gap_closed(pass_rate):
        if not reference or base['oracle']['pass_rate'] == reference['pass_rate']:
            return None
        return round((pass_rate - reference['pass_rate']) / (base['oracle']['pass_rate'] - reference['pass_rate']), 4)

    sweep = {}
    for use_complexity in (True, False):
        label = 'with_complexity' if use_complexity else 'without_complexity'
        sweep[label] = []
        for _, value in points:
            r = results[(use_complexity, value)]
            sweep[label].append({'value': value, 'pass_rate': r['metrics']['pass_rate'],
                                 'cost_usd': r['metrics']['cost_usd'],
                                 'gap_closed': gap_closed(r['metrics']['pass_rate']),
                                 'top_model_share': max(r['choice_share'].values()),
                                 'escalated_share': r['metrics']['escalated_share']})
    head = results[(args.use_complexity, headline)]
    router = head['metrics']
    summary = {'policies': {'router': router, **base}, 'reference': reference, 'use_complexity': args.use_complexity,
               'gap_closed': gap_closed(router['pass_rate']), 'choice_share': head['choice_share'],
               'by_task': head['by_task'], 'sweep': sweep,
               'headline_other_complexity_setting': results[(not args.use_complexity, headline)]['metrics'],
               'fallbacks_used': sum(d['fallbacks'] > 0 for d in head['decisions']),
               'unanswered': sum(d['chosen'] is None for d in head['decisions'])}
    scorer = SCORER
    if args.signals:
        path = args.signals.resolve()
        scorer = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    config = ev.run_config(args.name, args.description, scorer, None, pool, 8192, len(questions), excluded)
    config.update(policy=f"task-{args.mode}{'-cascade' if args.cascade else ''}", **{'lambda': headline if args.mode == 'utility' else None},
                  tau=headline if args.mode == 'threshold' else None,
                  crossfit=f'{args.folds}x{args.repeats}', smoothing=args.smoothing,
                  use_complexity=args.use_complexity, levels_from_signals=args.levels_from_signals,
                  escalation_factor=args.escalation_factor,
                  question_source=str(args.saved_questions) if args.saved_questions else str(args.data_dir),
                  aptitude_source=None if supplied is None else str(DEFAULT_TASK_APTITUDES.relative_to(ROOT)),
                  aptitude_sha256=None if supplied is None else hashlib.sha256(DEFAULT_TASK_APTITUDES.read_bytes()).hexdigest())
    folder = ev.save_run(args.eval_dir, config, head['decisions'], summary)
    fixed = {m.removeprefix('openrouter/'): v for m, v in base['models'].items()}
    for label, pts in sweep.items():
        ev.write_sweep_svg([{'alpha': p['value'], **p} for p in pts], fixed,
                           ev.PLOT_DIR / args.name / f'sweep_{label}.svg',
                           label_prefix='l=' if args.mode == 'utility' else 't=')

    print(f"\n{'setting':28} {'pass':>6} {'cost $':>8} {'gap closed':>11} {'top model':>10}")
    for label, pts in sweep.items():
        for p in pts:
            closed = f"{p['gap_closed']:.0%}" if p['gap_closed'] is not None else 'n/a'
            name = f"{label.split('_')[0]} {'l' if args.mode == 'utility' else 't'}={p['value']:g}"
            print(f"{name:28} {p['pass_rate']:6.1%} {p['cost_usd']:8.3f} {closed:>11} {p['top_model_share']:10.0%}")
    if reference:
        print(f"\nCurrent router ({reference['run']}): {reference['pass_rate']:.1%} at {reference['cost_usd']:.3f} USD")
    print(f"Oracle: {base['oracle']['pass_rate']:.1%} at {base['oracle']['cost_usd']:.3f} USD; best fixed "
          f"{base['best_fixed']['model'].removeprefix('openrouter/')}: {base['best_fixed']['pass_rate']:.1%} "
          f"at {base['best_fixed']['cost_usd']:.3f} USD")
    print(f"Headline ({'lambda' if args.mode == 'utility' else 'tau'} {headline:g}, "
          f"{'with' if args.use_complexity else 'without'} complexity): "
          f"{router['pass_rate']:.1%} at {router['cost_usd']:.3f} USD; choices: "
          + ', '.join(f"{k.split('/')[-1]} {v:.0%}" for k, v in head['choice_share'].items()))
    print(f'\nSaved to {folder} and {args.eval_dir / "runs.csv"}. No API calls were made.')


if __name__ == '__main__':
    main()
