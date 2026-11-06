"""Customer tariff: model token cost plus 20% of positive benchmark savings."""
from decimal import Decimal
import json
from pathlib import Path

SERVICE_FEE_RATE = Decimal('0.20')
FEE_BASIS = 'estimated_savings_v2'
# Fixed-model totals from the existing 690-question benchmark, not token prices.
BENCHMARK_SUMMARY = Path(__file__).resolve().parents[3] / 'results/router_eval/task-true-cascade/summary.json'
_models = json.loads(BENCHMARK_SUMMARY.read_text(), parse_float=Decimal)['policies']['models']
BENCHMARK_COSTS = {model_id: Decimal(str(result['cost_usd'])) for model_id, result in _models.items()
                   if result['questions'] == 690}
if not BENCHMARK_COSTS or any(not cost.is_finite() or cost <= 0 for cost in BENCHMARK_COSTS.values()):
    raise ValueError('Invalid benchmark cost totals')


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
    savings = max(Decimal(0), baseline - model_cost) if baseline is not None else None
    fee = SERVICE_FEE_RATE * savings if savings is not None else Decimal(0)
    return {'model_cost_usd': float(model_cost), 'service_fee_usd': float(fee),
            'service_fee_rate': float(SERVICE_FEE_RATE), 'service_fee_basis': FEE_BASIS,
            'estimated_top_model_cost_usd': float(baseline) if baseline is not None else None,
            'estimated_savings_usd': float(savings) if savings is not None else None,
            'estimated_customer_savings_usd': float(savings - fee) if savings is not None else None,
            'total_cost_usd': float(model_cost + fee)}


def charge(input_tokens, output_tokens, input_rate, output_rate, *, model_id=None, top_model_id=None):
    return request_billing([(model_id, token_cost(input_tokens, output_tokens, input_rate, output_rate))], top_model_id)
