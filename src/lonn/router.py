"""Select a model using topic aptitude, prompt complexity, and price."""

from math import fsum, isclose, isfinite

from .schemas import ComplexityResult, Model, TopicCategory


def _validate_scores(scores: dict[TopicCategory, float]) -> None:
    for topic, value in scores.items():
        if topic not in set(TopicCategory):
            raise ValueError(f"Unknown topic: {topic!r}")
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Topic scores and aptitudes must be finite and in [0, 1]")


def weighted_score(quality: float, cost: float, alpha: float = 0.5) -> float:
    """Balance quality and normalized cost, with all inputs in [0, 1]."""
    for name, value in (("quality", quality), ("cost", cost), ("alpha", alpha)):
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    return alpha * quality - (1 - alpha) * cost


def rank_models(
    signal: ComplexityResult, models: list[Model], alpha: float = 0.5,
    limit: int = 5,
) -> list[Model]:
    """Return the best candidates, ordered by score, for execution/fallback.

    Quality is topic-weighted aptitude; missing aptitudes count as zero.
    Complexity scales the quality contribution: easy prompts rely less on
    benchmark-derived aptitude differences, while hard prompts use the full
    quality-versus-cost tradeoff selected by alpha.
    Price is the sum of input/output costs per million tokens, normalized by
    the highest combined price in this pool (zero when all models are free).
    This assumes equal input/output token counts, not actual request cost.
    Changing the pool can therefore change the normalization and selection.

    Score ties prefer lower price, then higher quality, then provider and ID.
    Returns at most ``limit`` models, so the caller can attempt the next-ranked
    candidate when execution fails.
    Alpha defaults to 0.5; 0 prioritizes cost and 1 prioritizes quality for
    the most complex prompt. At complexity 0, the primary score depends only
    on price; aptitude breaks price ties. Context limits cannot be checked
    without request token counts.
    """
    weighted_score(0, 0, alpha)  # Validate even for an empty pool.
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not models:
        raise ValueError("Cannot route with an empty model pool")
    if not isfinite(signal.complexity) or not 0 <= signal.complexity <= 1:
        raise ValueError("Complexity must be finite and in [0, 1]")
    _validate_scores(signal.topic_scores)
    if not isclose(fsum(signal.topic_scores.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Topic probabilities must sum to 1")

    candidates = []
    identities = set()
    for model in models:
        identity = (model.provider, model.id)
        if identity in identities:
            raise ValueError(f"Duplicate model identity: {identity!r}")
        identities.add(identity)
        _validate_scores(model.topic_aptitudes)
        costs = (model.cost_per_m_input, model.cost_per_m_output)
        if any(not isfinite(cost) or cost < 0 for cost in costs):
            raise ValueError("Model costs must be finite and non-negative")
        price = sum(costs)
        if not isfinite(price):
            raise ValueError("Combined model cost must be finite")
        aptitude = fsum(
            probability * model.topic_aptitudes.get(topic, 0.0)
            for topic, probability in signal.topic_scores.items()
        )
        candidates.append((model, aptitude, price))

    max_price = max(item[2] for item in candidates)
    ranked = sorted(
        candidates,
        key=lambda item: (
            -(alpha * signal.complexity * item[1]
              - (1 - alpha) * (item[2] / max_price if max_price else 0)),
            item[2],
            -item[1],
            item[0].provider,
            item[0].id,
        ),
    )
    return [item[0] for item in ranked[:limit]]


def select_model(
    signal: ComplexityResult, models: list[Model], alpha: float = 0.5
) -> Model:
    """Return the top-ranked model (compatibility helper)."""
    return rank_models(signal, models, alpha, limit=1)[0]


class Router:
    """Class interface matching the architecture's Router contract."""

    select_model = staticmethod(select_model)
    rank_models = staticmethod(rank_models)
