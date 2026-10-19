"""Evaluate the router against the answer bank (#18): replay, don't call models.

For every benchmark question this does what production does up to execution, using the
existing code unchanged (docs/Prototype_architecture.md, sections 5 and 7):

    user message -> ComplexityScorer.score_prompt -> rank_models(signal, pool, alpha, limit=5)

and then, instead of executor.execute, looks up the chosen model's saved, graded answer
in the answer bank (results/benchmark/). As in production, the next ranked model is used
only when the saved call failed (an error row), never because an answer was wrong.

The router is compared with an oracle (cheapest correct model per question), the best
fixed model, the cheapest model, random choice and every single model. Each run is saved
under its --name in results/router_eval/<name>/ and as a row of results/router_eval/runs.csv;
earlier runs are kept, and a rerun with the same name replaces it.

    python3 scripts/benchmark/eval_router.py --name stock-a05 --description "stock Laya"
    python3 scripts/benchmark/eval_router.py --name finetuned-a07 --scorer LonnAI/laya-lonn --alpha 0.7
    python3 scripts/benchmark/eval_router.py --name topics --scorer true-topic --alpha-sweep
    python3 scripts/benchmark/eval_router.py --name small-pool --models gpt-oss deepseek

No API calls are made. Laya (about 850 MB) is downloaded on first use unless
--scorer true-topic; its scores are cached per scorer in reports/router_eval_cache/.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from benchmark.build_data import DATASETS, DEFAULT_DATA_DIR  # noqa: E402
from benchmark.results import RESULTS_DIR, ResultBank  # noqa: E402
from benchmark.templates import user_message  # noqa: E402
from lonn.complexity import ComplexityScorer  # noqa: E402
from lonn.model_pool import load_model_pool  # noqa: E402
from lonn.router import rank_models  # noqa: E402
from lonn.schemas import ComplexityResult, TopicCategory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / 'results' / 'router_eval'
CACHE_DIR = ROOT / 'reports' / 'router_eval_cache'
PLOT_DIR = ROOT / 'reports' / 'router_eval'
TRUE_TOPIC = 'true-topic'
FALLBACK_LIMIT = 5  # rank_models(limit=5): production tries up to five candidates.
# --scorer true-topic: a heuristic complexity from the dataset's own difficulty label.
DIFFICULTY_COMPLEXITY = {'easy': 0.25, 'medium': 0.5, 'hard': 0.75}
GROUPS = ('category', 'dataset', 'difficulty')
RUN_COLUMNS = ['run_name', 'description', 'date', 'git_commit', 'git_dirty', 'scorer', 'alpha', 'models',
               'prices_hash', 'max_tokens_equivalent', 'questions', 'excluded', 'pass_rate', 'total_cost_usd',
               'cost_per_pass_usd', 'mean_latency_ms', 'oracle_pass_rate', 'oracle_cost_usd',
               'best_fixed_model', 'best_fixed_pass_rate', 'best_fixed_cost_usd']


# Signals (what the scorer sees is the user message, as a client would send it)

def true_topic_signal(record: dict) -> ComplexityResult:
    """The dataset's own topic label and difficulty instead of Laya (ablation)."""
    return ComplexityResult(DIFFICULTY_COMPLEXITY[record['difficulty']],
                            {TopicCategory(record['topic']): 1.0})


def laya_signal_fn(checkpoint: str):
    """score_prompt with the chosen English checkpoint, via the existing LONN_LAYA_MODEL
    mechanism ('stock' = Laya's default checkpoint)."""
    os.environ['LONN_LAYA_MODEL'] = '' if checkpoint == 'stock' else checkpoint
    scorer = ComplexityScorer()
    return lambda record: scorer.score_prompt(user_message(record))


class CachedSignals:
    """Signals cached per prompt_id in a JSON Lines file, so reruns skip the scorer."""

    def __init__(self, signal_fn, path: Path):
        self.signal_fn, self.path, self.cache = signal_fn, Path(path), {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                item = json.loads(line)
                self.cache[item['prompt_id']] = item

    def __call__(self, record: dict) -> ComplexityResult:
        item = self.cache.get(record['prompt_id'])
        if item is None:
            signal = self.signal_fn(record)
            item = {'prompt_id': record['prompt_id'], 'complexity': signal.complexity,
                    'topic_scores': {topic.value: p for topic, p in signal.topic_scores.items()}}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as handle:
                handle.write(json.dumps(item) + '\n')
            self.cache[record['prompt_id']] = item
        return ComplexityResult(item['complexity'],
                                {TopicCategory(t): p for t, p in item['topic_scores'].items()})


# Outcomes from the answer bank

def outcome(row, model, max_tokens_equivalent=None):
    """(correct, cost, latency, truncated_at_equivalent) of a saved answer, or None for a
    call failure. With max_tokens_equivalent, an answer that needed more output tokens counts
    as cut off (failed) and its cost is scaled down to that budget using list prices."""
    if row is None or row.get('error'):
        return None
    correct, cost = bool(row.get('correct')), row.get('cost_usd') or 0.0
    out_tokens, in_tokens = row.get('output_tokens') or 0, row.get('input_tokens') or 0
    truncated = bool(max_tokens_equivalent and out_tokens > max_tokens_equivalent)
    if truncated:
        listed = in_tokens * model.cost_per_m_input + out_tokens * model.cost_per_m_output
        capped = in_tokens * model.cost_per_m_input + max_tokens_equivalent * model.cost_per_m_output
        cost, correct = (cost * capped / listed if listed else cost), False
    return correct, cost, row.get('latency_ms') or 0, truncated


def decide(record, signal, pool, alpha, bank, max_tokens_equivalent=None) -> dict:
    """The router's decision for one question, with production's fallback rule."""
    ranking = rank_models(signal, pool, alpha=alpha, limit=FALLBACK_LIMIT)
    decision = {key: record[key] for key in ('prompt_id', 'dataset', 'category', 'difficulty')}
    decision.update(signal={'complexity': round(signal.complexity, 4),
                            'topic_scores': {t.value: round(p, 4) for t, p in signal.topic_scores.items()}},
                    ranking=[m.id for m in ranking], alpha=alpha)
    for fallbacks, candidate in enumerate(ranking):
        result = outcome(bank.lookup(record['prompt_id'], candidate.id), candidate, max_tokens_equivalent)
        if result is None:
            continue  # Saved call failed: production tries the next ranked model.
        correct, cost, latency, truncated = result
        decision.update(chosen=candidate.id, fallbacks=fallbacks, correct=correct, cost_usd=cost,
                        latency_ms=latency, truncated_at_equivalent=truncated)
        return decision
    decision.update(chosen=None, fallbacks=len(ranking), correct=False, cost_usd=0.0, latency_ms=0,
                    truncated_at_equivalent=False)
    return decision


def metrics(items: list) -> dict:
    """items: dicts with correct (bool or expected value), cost_usd and latency_ms."""
    passes = sum(item['correct'] for item in items)
    cost = sum(item['cost_usd'] for item in items)
    return {'questions': len(items), 'passes': round(passes, 4),
            'pass_rate': round(passes / len(items), 4) if items else None,
            'cost_usd': round(cost, 6), 'cost_per_pass_usd': round(cost / passes, 6) if passes else None,
            'mean_latency_ms': round(sum(item['latency_ms'] for item in items) / len(items)) if items else None}


def _by_group(per_question: list) -> dict:
    groups = {}
    for group in GROUPS:
        buckets = {}
        for item in per_question:
            buckets.setdefault(item[group], []).append(item)
        groups[group] = {key: metrics(bucket) for key, bucket in sorted(buckets.items())}
    return groups


def evaluate(questions, signal_fn, pool, alpha, bank, max_tokens_equivalent=None):
    """Router decisions plus baselines on the same questions. Returns (decisions, summary)."""
    decisions = [decide(q, signal_fn(q), pool, alpha, bank, max_tokens_equivalent) for q in questions]
    per_model = {m.id: [] for m in pool}
    oracle, random_choice, quality_fallback = [], [], []
    for question, decision in zip(questions, decisions):
        base = {key: question[key] for key in GROUPS}
        options = {}
        for m in pool:
            result = outcome(bank.lookup(question['prompt_id'], m.id), m, max_tokens_equivalent)
            correct, cost, latency = (False, 0.0, 0) if result is None else result[:3]
            per_model[m.id].append({**base, 'correct': correct, 'cost_usd': cost, 'latency_ms': latency})
            if result is not None:
                options[m.id] = (correct, cost, latency)
        right = [option for option in options.values() if option[0]]
        pick = min(right or options.values(), key=lambda o: o[1], default=(False, 0.0, 0))
        oracle.append({**base, 'correct': pick[0], 'cost_usd': pick[1], 'latency_ms': pick[2]})
        n = len(pool)
        random_choice.append({**base, 'correct': sum(i['correct'] for i in (per_model[m.id][-1] for m in pool)) / n,
                              'cost_usd': sum(per_model[m.id][-1]['cost_usd'] for m in pool) / n,
                              'latency_ms': sum(per_model[m.id][-1]['latency_ms'] for m in pool) / n})
        # Hypothetical (not what production does): retry down the ranking until an answer is right.
        spent, latency, solved = 0.0, 0, False
        for model_id in decision['ranking']:
            option = options.get(model_id)
            if option is None:
                continue
            spent, latency = spent + option[1], latency + option[2]
            if option[0]:
                solved = True
                break
        quality_fallback.append({**base, 'correct': solved, 'cost_usd': spent, 'latency_ms': latency})
    router = [{**{k: d[k] for k in GROUPS}, 'correct': d['correct'], 'cost_usd': d['cost_usd'],
               'latency_ms': d['latency_ms']} for d in decisions]
    model_metrics = {model_id: metrics(items) for model_id, items in per_model.items()}
    best = max(pool, key=lambda m: (model_metrics[m.id]['passes'], -model_metrics[m.id]['cost_usd']))
    cheapest = min(pool, key=lambda m: (m.cost_per_m_input + m.cost_per_m_output, m.id))
    policies = {'router': metrics(router), 'oracle': metrics(oracle),
                'best_fixed': {'model': best.id, **model_metrics[best.id]},
                'cheapest': {'model': cheapest.id, **model_metrics[cheapest.id]},
                'random': metrics(random_choice), 'models': model_metrics,
                'quality_fallback_hypothetical': metrics(quality_fallback)}
    by = {'router': router, 'oracle': oracle, 'best_fixed': per_model[best.id],
          'cheapest': per_model[cheapest.id], 'random': random_choice}
    summary = {'policies': policies, 'fallbacks_used': sum(d['fallbacks'] > 0 for d in decisions),
               'unanswered': sum(d['chosen'] is None for d in decisions)}
    for group in GROUPS:
        summary[f'by_{group}'] = {name: _by_group(items)[group] for name, items in by.items()}
    return decisions, summary


def alpha_sweep(questions, signal_fn, pool, bank, alphas=None, max_tokens_equivalent=None) -> list:
    points = []
    for alpha in alphas or [round(a / 10, 1) for a in range(11)]:
        router = evaluate(questions, signal_fn, pool, alpha, bank, max_tokens_equivalent)[1]['policies']['router']
        points.append({'alpha': alpha, 'pass_rate': router['pass_rate'], 'cost_usd': router['cost_usd']})
    return points


# Pool and questions

def bank_pool(models, bank, wanted) -> list:
    """Pool models that have saved answers, optionally narrowed by ID substrings (--models)."""
    in_bank = {model_id for _, model_id in bank.rows}
    pool = [m for m in models if m.id in in_bank]
    if wanted:
        pool = [m for m in pool if any(part in m.id for part in wanted)]
        if not pool:
            raise SystemExit(f'No pool model with saved answers matches {wanted}')
    return pool


def load_questions(data_dir, bank) -> tuple:
    """(questions with saved answers, number of questions skipped for having none)."""
    answered = {prompt_id for prompt_id, _ in bank.rows}
    questions, excluded = [], 0
    for name, spec in DATASETS.items():
        path = Path(data_dir) / 'prompts' / spec.category / f'{name}.jsonl'
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            if record['prompt_id'] in answered:
                questions.append(record)
            elif record['topic'] is not None:
                excluded += 1
    return questions, excluded


# Run records

def _git(*args) -> str:
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def run_config(name, description, scorer, alpha, pool, max_tokens_equivalent, questions, excluded) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name or ''):
        raise ValueError(f'Run name {name!r}: use letters, digits, ".", "_" and "-"')
    prices = {m.id: [m.cost_per_m_input, m.cost_per_m_output] for m in pool}
    return {'run_name': name, 'description': description, 'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'git_commit': _git('rev-parse', '--short', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain', '--untracked-files=no')),
            'scorer': scorer, 'alpha': alpha, 'models': [m.id for m in pool], 'prices': prices,
            'prices_hash': hashlib.sha256(json.dumps(prices, sort_keys=True).encode()).hexdigest()[:12],
            'max_tokens_equivalent': max_tokens_equivalent, 'questions': questions, 'excluded': excluded}


def save_run(eval_dir, config, decisions, summary, sweep=None) -> Path:
    folder = Path(eval_dir) / config['run_name']
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (folder / 'summary.json').write_text(json.dumps({**summary, 'alpha_sweep': sweep}, indent=2) + '\n')
    (folder / 'decisions.jsonl').write_text(''.join(json.dumps(d, sort_keys=True) + '\n' for d in decisions))
    policies = summary['policies']
    row = {key: config[key] for key in RUN_COLUMNS if key in config}
    row.update(models=' '.join(m.removeprefix('openrouter/') for m in config['models']),
               pass_rate=policies['router']['pass_rate'], total_cost_usd=policies['router']['cost_usd'],
               cost_per_pass_usd=policies['router']['cost_per_pass_usd'],
               mean_latency_ms=policies['router']['mean_latency_ms'],
               oracle_pass_rate=policies['oracle']['pass_rate'], oracle_cost_usd=policies['oracle']['cost_usd'],
               best_fixed_model=policies['best_fixed']['model'].removeprefix('openrouter/'),
               best_fixed_pass_rate=policies['best_fixed']['pass_rate'],
               best_fixed_cost_usd=policies['best_fixed']['cost_usd'])
    runs_path = Path(eval_dir) / 'runs.csv'
    rows = []
    if runs_path.exists():
        with runs_path.open(newline='') as handle:
            rows = [r for r in csv.DictReader(handle) if r['run_name'] != config['run_name']]
    rows.append(row)
    with runs_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['date']))
    return folder


def write_sweep_svg(points, fixed, path, width=640, height=420) -> None:
    """Pass rate vs total cost: the router's alpha sweep as a line, fixed models as dots."""
    xs = [p['cost_usd'] for p in points] + [f['cost_usd'] for f in fixed.values()]
    ys = [p['pass_rate'] for p in points] + [f['pass_rate'] for f in fixed.values()]
    x0, x1 = 0.0, max(xs) * 1.05 or 1.0
    y0, y1 = max(0.0, min(ys) - 0.05), min(1.0, max(ys) + 0.05)
    pad = 60

    def px(x, y):
        return (pad + (x - x0) / (x1 - x0) * (width - 2 * pad),
                height - pad - (y - y0) / ((y1 - y0) or 1) * (height - 2 * pad))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="black"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="black"/>',
             f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle">total cost (USD)</text>',
             f'<text x="15" y="{height / 2}" transform="rotate(-90 15 {height / 2})" text-anchor="middle">pass rate</text>']
    for value in (y0, (y0 + y1) / 2, y1):
        _, y = px(x0, value)
        parts.append(f'<text x="{pad - 5}" y="{y + 4}" text-anchor="end">{value:.0%}</text>')
    for value in (x0, x1 / 2, x1):
        x, _ = px(value, y0)
        parts.append(f'<text x="{x}" y="{height - pad + 15}" text-anchor="middle">{value:.2f}</text>')
    line = ' '.join('{:.1f},{:.1f}'.format(*px(p['cost_usd'], p['pass_rate'])) for p in points)
    parts.append(f'<polyline points="{line}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
    for p in points:
        x, y = px(p['cost_usd'], p['pass_rate'])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#1f77b4"/>'
                     f'<text x="{x + 5:.1f}" y="{y - 5:.1f}" fill="#1f77b4">a={p["alpha"]}</text>')
    for name, f in fixed.items():
        x, y = px(f['cost_usd'], f['pass_rate'])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#d62728"/>'
                     f'<text x="{x + 6:.1f}" y="{y + 4:.1f}" fill="#d62728">{name}</text>')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(parts) + '\n</svg>\n')


def print_report(config, summary):
    policies = summary['policies']
    print(f"\nRun {config['run_name']}: scorer {config['scorer']}, alpha {config['alpha']}, "
          f"{len(config['models'])} models, {config['questions']} questions"
          + (f", output budget {config['max_tokens_equivalent']}" if config['max_tokens_equivalent'] else ''))
    print(f"{'policy':32} {'pass':>6} {'cost $':>9} {'$/pass':>9} {'latency ms':>11}")
    rows = [('router', policies['router']), ('oracle (cheapest correct)', policies['oracle']),
            (f"best fixed: {policies['best_fixed']['model'].removeprefix('openrouter/')}", policies['best_fixed']),
            (f"cheapest: {policies['cheapest']['model'].removeprefix('openrouter/')}", policies['cheapest']),
            ('random (expected)', policies['random']),
            ('[hypothetical quality fallback]', policies['quality_fallback_hypothetical'])]
    for label, m in rows:
        per_pass = f"{m['cost_per_pass_usd']:.4f}" if m['cost_per_pass_usd'] is not None else 'n/a'
        print(f"{label[:32]:32} {m['pass_rate']:6.1%} {m['cost_usd']:9.4f} {per_pass:>9} {m['mean_latency_ms']:11.0f}")
    print(f"Router fallbacks used: {summary['fallbacks_used']}; questions with no usable answer: {summary['unanswered']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='run name; a rerun with the same name replaces it')
    parser.add_argument('--description', default='', help='free text describing the router')
    parser.add_argument('--scorer', default='stock',
                        help="'stock' (default Laya), a checkpoint such as LonnAI/laya-lonn or a local "
                             f"directory, or '{TRUE_TOPIC}' (dataset topic labels, no Laya)")
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--alpha-sweep', action='store_true', help='also evaluate alpha 0.0, 0.1 ... 1.0')
    parser.add_argument('--models', nargs='+', help='ID substrings of the models the router may choose from')
    parser.add_argument('--max-tokens-equivalent', type=int, choices=[4096, 8192], default=8192,
                        help='8192 = as generated; 4096 = production executor default (longer answers fail)')
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--eval-dir', type=Path, default=EVAL_DIR)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    bank = ResultBank(args.results_dir)
    pool = bank_pool(load_model_pool(), bank, args.models)
    questions, excluded = load_questions(args.data_dir, bank)
    if not questions:
        raise SystemExit('No questions with saved answers; build the prompt files (build_data.py) first.')
    budget = None if args.max_tokens_equivalent == 8192 else args.max_tokens_equivalent
    config = run_config(args.name, args.description, args.scorer, args.alpha, pool, args.max_tokens_equivalent,
                        len(questions), excluded)
    slug = re.sub(r'[^A-Za-z0-9._-]', '_', args.scorer)
    base = true_topic_signal if args.scorer == TRUE_TOPIC else laya_signal_fn(args.scorer)
    signals = CachedSignals(base, CACHE_DIR / f'signals-{slug}.jsonl')
    print(f'Scoring {len(questions)} questions with {args.scorer} (cached after the first run)...', flush=True)
    decisions, summary = evaluate(questions, signals, pool, args.alpha, bank, budget)
    sweep = None
    if args.alpha_sweep:
        sweep = alpha_sweep(questions, signals, pool, bank, max_tokens_equivalent=budget)
        fixed = {m.removeprefix('openrouter/'): summary['policies']['models'][m] for m in summary['policies']['models']}
        svg = PLOT_DIR / args.name / 'alpha_sweep.svg'
        write_sweep_svg(sweep, fixed, svg)
        print(f'Alpha sweep plot: {svg}')
    folder = save_run(args.eval_dir, config, decisions, summary, sweep)
    print_report(config, summary)
    print(f'\nSaved to {folder} and {args.eval_dir / "runs.csv"}. No API calls were made.')


if __name__ == '__main__':
    main()
