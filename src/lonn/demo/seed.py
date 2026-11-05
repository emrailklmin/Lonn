"""Reproducible, synthetic requests routed using the prototype's task table."""
import json
import random
from datetime import datetime, timedelta

from lonn.model_pool import load_model_pool
from lonn.routing_table import RoutingTable
from lonn.schemas import TaskSignal
from lonn.task_router import rank_models
from .baseline import top_model_baseline
from .pricing import charge


def history(seed, entries):
    rng = random.Random(seed['random_seed'])
    table = RoutingTable.load()
    active_ids = {m['id'] for m in entries if m['active'] and m['provider'] == 'openrouter'}
    pool = [model for model in load_model_pool() if model.id in active_ids]
    origin = datetime.fromisoformat(seed['start'])
    timestamps = sorted(origin + timedelta(days=rng.choices(range(10), weights=[3,8,11,5,2,1,12,9,7,6])[0],
                                           hours=rng.triangular(0, 13, 6), seconds=rng.randrange(60))
                        for _ in range(seed['request_count']))
    failures = set(rng.sample(range(seed['request_count']), 4))
    for i, timestamp in enumerate(timestamps):
        task = rng.choices(seed['tasks'], weights=seed['task_weights'])[0]
        signal = TaskSignal({task: 1.0})
        ranked = rank_models(signal, table, pool, lam=0.2)
        failed = i in failures
        # The live API tries ranked candidates on errors and records the last attempt.
        attempted = ranked if failed else ranked[:1]
        model = attempted[-1]
        input_center = {'math_problem': 6.4, 'code_function': 5.5,
                        'creative_writing': 5.7, 'knowledge_recall': 5.0}[task]
        output_center = {'math_problem': 6.4, 'code_function': 5.6,
                         'creative_writing': 6.7, 'knowledge_recall': 5.2}[task]
        inputs = max(24, min(9000, int(rng.lognormvariate(input_center, .8))))
        outputs = max(18, min(4096, int(rng.lognormvariate(output_center, .7))))
        latency = round(280 + outputs * rng.uniform(2.2, 11.4) + rng.uniform(90, 1700), 1)
        if failed:
            inputs, outputs, latency = 0, 0, round(rng.uniform(1100, 28000), 1)
        baseline = top_model_baseline(pool, table, inputs, outputs)
        billing = charge(inputs, outputs, model.cost_per_m_input, model.cost_per_m_output,
                         model_id=model.id, top_model_id=baseline['model_id'] if baseline else None)
        if baseline:
            baseline['estimated_cost_usd'] = billing['estimated_top_model_cost_usd']
        yield dict(id=f'seed-{i:03}', customer_id=seed['id'],
                   team_id={'code_function':'team-engineering','math_problem':'team-engineering',
                            'knowledge_recall':'team-support','creative_writing':'team-content'}[task],
                   project_id='project-' + {'code_function':'team-engineering','math_problem':'team-engineering',
                            'knowledge_recall':'team-support','creative_writing':'team-content'}[task],
                   application_id='app-' + {'code_function':'team-engineering','math_problem':'team-engineering',
                            'knowledge_recall':'team-support','creative_writing':'team-content'}[task],
                   timestamp=timestamp.isoformat(), source='seeded', model=model.id,
                   status='failed' if failed else 'success', input_tokens=inputs,
                   output_tokens=outputs, cost_usd=billing['total_cost_usd'], latency_ms=latency,
                   details=json.dumps({'synthetic': True, 'cost_basis': 'calculated', 'billing': billing,
                       'task': task, 'signal': {'task_scores': signal.task_scores, 'complexity': None},
                       'lam': 0.2, 'attempted_models': [m.id for m in attempted],
                       'usage_complete': not failed,
                       'baseline': baseline,
                       'scenario': 'Synthetic task mix; selected using the existing router at lambda 0.20'}))
