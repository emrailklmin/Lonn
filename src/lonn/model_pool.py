"""Load curated model metadata; discovery and probing are separate tools."""

from math import isfinite
from pathlib import Path

import yaml

from .schemas import PROVIDER_API_KEYS, Model, TopicCategory

DEFAULT_POOL = Path(__file__).resolve().parents[2] / 'config' / 'models.yaml'


def load_model_pool(path: str | Path = DEFAULT_POOL) -> list[Model]:
    """Load active, fully configured models or fail with actionable errors.

    Unknown prices, context limits, or aptitudes are never replaced by zeros.
    External availability reports do not silently modify the curated pool.
    """
    entries = yaml.safe_load(Path(path).read_text())
    if not isinstance(entries, list):
        raise ValueError('Model pool must be a YAML list')
    models, errors, seen = [], [], set()
    for index, entry in enumerate(entries):
        try:
            if not isinstance(entry, dict):
                raise ValueError('entry must be a mapping')
            if type(entry.get('active')) is not bool:
                raise ValueError('active must be true or false')
            if not entry['active']:
                continue
            model_id = entry.get('id')
            if not isinstance(model_id, str):
                raise ValueError('id must be a string')
            provider = entry.get('provider')
            if not isinstance(provider, str) or provider not in PROVIDER_API_KEYS:
                raise ValueError(f'provider must be one of {", ".join(PROVIDER_API_KEYS)}')
            parts = model_id.split('/', 2)
            if len(parts) != 3 or parts[0] != provider or not all(parts):
                raise ValueError(f'id must be {provider}/<publisher>/<model>')
            if model_id in seen:
                raise ValueError('duplicate model ID')
            seen.add(model_id)
            prices = [entry.get(key) for key in ('cost_per_m_input', 'cost_per_m_output')]
            if any(type(x) not in (int, float) or not isfinite(x) or x < 0 for x in prices):
                raise ValueError('cost_per_m_input/output must be known finite non-negative prices')
            if entry.get('price_basis') not in ('provider', 'estimate', 'free_tier'):
                raise ValueError('price_basis must be provider, estimate, or free_tier')
            context = entry.get('context_window')
            if type(context) is not int or context <= 0:
                raise ValueError('context_window needs a verified positive token limit')
            scores = entry.get('topic_aptitudes')
            if not isinstance(scores, dict) or set(scores) != {t.value for t in TopicCategory}:
                raise ValueError('topic_aptitudes must supply every TopicCategory')
            if any(type(x) not in (int, float) or not isfinite(x) or not 0 <= x <= 1
                   for x in scores.values()):
                raise ValueError('topic_aptitudes need measured or explicitly estimated scores in [0, 1]')
            models.append(Model(model_id, provider, *prices, context,
                                {TopicCategory(k): v for k, v in scores.items()}))
        except ValueError as exc:
            label = entry.get('id', index) if isinstance(entry, dict) else index
            errors.append(f'{label}: {exc}')
    if errors:
        raise ValueError('Invalid active model configuration:\n' + '\n'.join(errors))
    if not models:
        raise ValueError('No active models in pool')
    return models


class ModelPool:
    load_model_pool = staticmethod(load_model_pool)
