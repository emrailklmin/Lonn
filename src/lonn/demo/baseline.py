"""Counterfactual baseline from the eligible pool's overall benchmark quality."""
from .pricing import BENCHMARK_COSTS


def top_model_baseline(models, table, input_tokens, output_tokens):
    # global_row is the existing equal-weighted mean across measured task types.
    # A per-task λ=0 routing decision can select a different model for every
    # request, so it is not the pool-wide highest-quality model.
    measured = [model for model in models if model.id in table.global_row]
    if not measured:
        return None
    top = min(measured, key=lambda model: (-table.global_row[model.id].accuracy,
                                            table.global_row[model.id].cost_usd,
                                            model.provider, model.id))
    benchmark = BENCHMARK_COSTS.get(top.id)
    return {'model_id': top.id, 'input_rate_per_m': top.cost_per_m_input,
            'output_rate_per_m': top.cost_per_m_output,
            'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'benchmark_cost_per_690': float(benchmark) if benchmark is not None else None,
            'ranking': 'global_benchmark_v1',
            'quality_score': table.global_row[top.id].accuracy,
            'eligible_model_ids': [model.id for model in models],
            'estimated': True}
