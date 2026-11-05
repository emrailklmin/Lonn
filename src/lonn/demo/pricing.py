"""Customer tariff: model token cost plus 20% of positive benchmark savings."""
from decimal import Decimal

SERVICE_FEE_RATE = Decimal('0.20')
FEE_BASIS = 'estimated_savings_v1'
# User-supplied benchmark: total USD for 690 prompts. Unknown models stay unknown.
BENCHMARK_COSTS = {
    'openrouter/anthropic/claude-sonnet-5.5': Decimal('7.57'),
    'openrouter/meta/muse-spark-1.3': Decimal('7.37'),
    'openrouter/google/gemini-3.8-flash': Decimal('4.53'),
    'openrouter/deepseek/deepseek-v4-pro': Decimal('3.98'),
    'openrouter/deepseek/deepseek-v4.1-flash': Decimal('1.17'),
    'openrouter/z-ai/glm-5.3-flash': Decimal('0.38'),
    'openrouter/openai/gpt-oss-120b': Decimal('0.30'),
    'openrouter/openai/gpt-oss-20b': Decimal('0.19'),
}


def token_cost(input_tokens, output_tokens, input_rate, output_rate):
    return (Decimal(input_tokens) * Decimal(str(input_rate)) +
            Decimal(output_tokens) * Decimal(str(output_rate))) / Decimal(1_000_000)


def request_billing(model_costs, top_model_id):
    """Sum completed calls, estimate each relative cost, then clamp the fee once."""
    costs = [(model_id, Decimal(str(cost))) for model_id, cost in model_costs]
    model_cost = sum((cost for _, cost in costs), Decimal(0))
    top = BENCHMARK_COSTS.get(top_model_id)
    comparable = top is not None and all(model_id in BENCHMARK_COSTS for model_id, _ in costs)
    baseline = sum((cost * top / BENCHMARK_COSTS[model_id] for model_id, cost in costs), Decimal(0)) if comparable else None
    savings = baseline - model_cost if baseline is not None else None
    fee = SERVICE_FEE_RATE * max(Decimal(0), savings) if savings is not None else Decimal(0)
    return {'model_cost_usd': float(model_cost), 'service_fee_usd': float(fee),
            'service_fee_rate': float(SERVICE_FEE_RATE), 'service_fee_basis': FEE_BASIS,
            'estimated_top_model_cost_usd': float(baseline) if baseline is not None else None,
            'estimated_savings_usd': float(savings) if savings is not None else None,
            'total_cost_usd': float(model_cost + fee)}


def charge(input_tokens, output_tokens, input_rate, output_rate, *, model_id=None, top_model_id=None):
    return request_billing([(model_id, token_cost(input_tokens, output_tokens, input_rate, output_rate))], top_model_id)
