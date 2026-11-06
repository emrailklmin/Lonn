"""Reproducible synthetic traffic routed with the existing task table."""
import json
import random
from datetime import datetime, timedelta, timezone

from lonn.model_pool import load_model_pool
from lonn.routing_table import load_runtime_table
from lonn.schemas import TaskSignal
from lonn.task_router import rank_models
from .baseline import top_model_baseline
from .policy import DEMO_SECURITY
from .pricing import charge


def additional_traffic(row, rng, start, end, count):
    """Create additional synthetic traffic with the same billed usage and nearby times."""
    timestamp = datetime.fromisoformat(row['timestamp'])
    for number in range(1, count):
        shifted = timestamp + timedelta(seconds=rng.randrange(-1800, 1801))
        shifted = max(start, min(end - timedelta(seconds=1), shifted))
        yield {**row, 'id': f"{row['id']}-traffic-{number}", 'timestamp': shifted.isoformat()}


def history(seed, entries, *, now=None):
    rng = random.Random(seed['random_seed'])
    traffic_rng = random.Random(seed['random_seed'] + 1)
    table = load_runtime_table()
    active_ids = {m['id'] for m in entries if m['active'] and m['provider'] == 'openrouter'}
    full_pool = [model for model in load_model_pool() if model.id in active_ids]
    # Historical applications can have narrower model policies. These are
    # synthetic snapshots, not changes to the customer's present settings.
    european_pool = [m for m in full_pool if 'europe' in DEMO_SECURITY[m.id]['regions']
                     and DEMO_SECURITY[m.id]['no_training']]
    open_pool = [m for m in full_pool if m.id.rsplit('/', 1)[-1] in
                 ('glm-5.3-flash', 'gpt-oss-120b', 'gpt-oss-20b')]
    oss_pool = [m for m in open_pool if m.id.rsplit('/', 1)[-1].startswith('gpt-oss-')]
    pools = {'standard': full_pool, 'europe_no_training': european_pool,
             'open_models': open_pool, 'oss_family': oss_pool}
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=seed['window_days'])

    # Sample real hours in the rolling window. Weekdays, office hours and a few
    # occasional spikes are busier; each calendar day also varies independently.
    hours = [start + timedelta(hours=i) for i in range(seed['window_days'] * 24)]
    dates = list(dict.fromkeys(hour.date() for hour in hours))
    spikes = set(rng.sample(dates, min(3, len(dates))))
    daily = {day: rng.uniform(.72, 1.28) * (rng.uniform(1.9, 2.5) if day in spikes else 1)
             for day in dates}
    hour_weights = [(1 if hour.weekday() < 5 else .22) * daily[hour.date()] *
                    (1 if 8 <= hour.hour < 19 else .30 if 19 <= hour.hour < 23 else .10)
                    for hour in hours]
    chosen_hours = rng.choices(hours, weights=hour_weights, k=seed['request_count'])
    timestamps = sorted(hour + timedelta(seconds=rng.randrange(3600)) for hour in chosen_hours)

    lambdas = (0.0, .2, .4, .7, 1.0)
    ranks = {(task, lam, name): rank_models(TaskSignal({task: 1.0}), table, pool, lam=lam)
             for task in seed['tasks'] for lam in lambdas for name, pool in pools.items()}
    baselines = {name: top_model_baseline(pool, table, 0, 0) for name, pool in pools.items()}
    team_weights = {
        'math_problem': (8, 1, 1), 'code_function': (9, 1, 1), 'code_algorithmic': (9, 1, 1),
        'science_expert': (5, 2, 1), 'knowledge_quantitative': (6, 2, 1),
        'instruction_following': (4, 4, 2), 'business_writing': (1, 4, 5),
        'summarization': (2, 5, 3), 'translation': (1, 4, 5),
        'information_extraction': (4, 5, 1), 'creative_writing': (1, 1, 8),
        'knowledge_recall': (3, 5, 2), 'conversation': (1, 7, 2), 'tool_calling': (6, 3, 1),
    }
    centers = {
        'math_problem': (6.5, 6.5), 'code_function': (6.4, 6.3), 'code_algorithmic': (6.5, 6.5),
        'science_expert': (6.3, 6.2), 'knowledge_quantitative': (6.2, 5.8),
        'instruction_following': (6.2, 5.9), 'business_writing': (6.1, 6.3),
        'summarization': (7.6, 5.8), 'translation': (6.8, 6.5),
        'information_extraction': (7.2, 5.7), 'creative_writing': (6.0, 6.9),
        'knowledge_recall': (5.5, 5.7), 'conversation': (6.1, 5.8), 'tool_calling': (6.5, 5.2),
    }
    teams = ('team-engineering', 'team-support', 'team-content')
    members = {'team-engineering': 'user-engineer', 'team-support': 'user-support', 'team-content': 'user-content'}
    for i, timestamp in enumerate(timestamps):
        task = rng.choices(seed['tasks'], weights=seed['task_weights'])[0]
        lam = rng.choices(lambdas, weights=(4, 45, 22, 18, 11))[0]
        pool_name = rng.choices(tuple(pools), weights=(76, 14, 7, 3))[0]
        ranked = ranks[task, lam, pool_name]
        failed = rng.random() < .004
        fallback = rng.choices((0, 1, 2), weights=(97, 2.8, .2))[0]
        attempted = ranked if failed else ranked[:min(fallback + 1, len(ranked))]
        model = attempted[-1]
        input_center, output_center = centers[task]
        multiplier = seed.get('token_multiplier', 1)
        inputs = max(24, min(9000, int(multiplier * rng.lognormvariate(input_center, .7))))
        outputs = max(18, min(4096, int(multiplier * rng.lognormvariate(output_center, .65))))
        latency = round(280 + outputs * rng.uniform(2.2, 11.4) + rng.uniform(90, 1700), 1)
        if failed:
            inputs, outputs, latency = 0, 0, round(rng.uniform(1100, 28000), 1)
        baseline = dict(baselines[pool_name], input_tokens=inputs, output_tokens=outputs)
        billing = charge(inputs, outputs, model.cost_per_m_input, model.cost_per_m_output,
                         model_id=model.id, top_model_id=baseline['model_id'])
        baseline['estimated_cost_usd'] = billing['estimated_top_model_cost_usd']
        team = rng.choices(teams, weights=team_weights[task])[0]
        user = rng.choices((members[team], 'user-admin', None), weights=(55, 15, 30))[0]
        row = dict(id=f'seed-{i:03}', customer_id=seed['id'], team_id=team,
                   project_id='project-' + team, application_id='app-' + team, user_id=user,
                   timestamp=timestamp.isoformat(), source='seeded', model=model.id,
                   status='failed' if failed else 'success', input_tokens=inputs,
                   output_tokens=outputs, cost_usd=billing['total_cost_usd'], latency_ms=latency,
                   details=json.dumps({'synthetic': True, 'cost_basis': 'estimate', 'billing': billing,
                       'task': task, 'signal': {'task_scores': {task: 1.0}, 'complexity': None},
                       'lam': lam, 'attempted_models': [m.id for m in attempted],
                       'usage_complete': len(attempted) == 1 and not failed, 'baseline': baseline,
                       'scenario': pool_name, 'seed_window_end': end.isoformat()}))
        yield row
        yield from additional_traffic(row, traffic_rng, start, end, seed.get('traffic_multiplier', 1))
