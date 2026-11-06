"""Route on task type, complexity, measured or supplied aptitude, and cost.

For each pool model the router estimates the expected accuracy and the expected
cost per request of this prompt from the routing table:

- **Task-type mix:** a probability-weighted mix of the task types' rows, so an
  uncertain classification gives a smooth blend instead of a hard jump.
  Prototype 4.0 provides explicit scores for all 15 known task types. An unknown
  task type falls back to the measured table's ``global`` row.
- **Complexity** (only with ``use_complexity=True``): within a task type, the
  row moves from the ``easy`` cell (complexity 0) through ``medium`` (0.5) to
  ``hard`` (1), interpolating linearly between the levels the task type has.
  ``None`` uses the task type's average. It is off by default because, with
  the datasets' difficulty labels, it has not improved the cross-fitted
  pass-vs-cost curve yet (docs/task_router.md).

When an answer fails a check (lonn.answer_check), ``escalation_target`` picks
the model to try next, if a second call is worth its expected cost.

It then picks a model by one of two rules:

- ``utility`` (default): maximize ``lam * accuracy - (1-lam) * cost / scale``.
  Lambda ranges from cost only (0) to quality only (1); scale is shared by the table.
  Explicit ``cost_weight`` is retained solely for historical benchmark comparisons.
- ``threshold``: the cheapest model whose expected accuracy is at least
  ``tau``. If none qualifies, the most accurate model.

Models missing from the table are not ranked. The router makes no API calls.
"""

from dataclasses import dataclass
from math import fsum, isclose, isfinite

from .routing_table import ALL, Cells, CellStats, RoutingTable
from .schemas import Model, TaskSignal

LEVEL_POSITION = {'easy': 0.0, 'medium': 0.5, 'hard': 1.0}
DEFAULT_LAMBDA = 0.5
LEGACY_COST_WEIGHT = 20.0  # Retained only for historical benchmark replay.
DEFAULT_USE_COMPLEXITY = False
FALLBACK_LIMIT = 5  # Candidates returned, so callers can fall back when a provider call fails.


@dataclass(frozen=True)
class Estimate:
    model: Model
    accuracy: float
    cost_usd: float
    score: float


def validate_signal(signal: TaskSignal) -> None:
    if not signal.task_scores:
        raise ValueError('task_scores must not be empty')
    for task, p in signal.task_scores.items():
        if not isinstance(task, str) or not isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Task scores must map task IDs to probabilities in [0, 1]')
    if not isclose(fsum(signal.task_scores.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError('Task probabilities must sum to 1')
    c = signal.complexity
    if c is not None and (not isfinite(c) or not 0 <= c <= 1):
        raise ValueError('Complexity must be None or in [0, 1]')


def _blend(a: Cells, b: Cells, weight_b: float) -> Cells:
    return {m: CellStats(a[m].accuracy + (b[m].accuracy - a[m].accuracy) * weight_b,
                         a[m].cost_usd + (b[m].cost_usd - a[m].cost_usd) * weight_b, a[m].n)
            for m in a if m in b}


def task_row(table: RoutingTable, task: str, complexity: float | None) -> Cells:
    """The task type's row at this complexity (see the module docstring)."""
    if task not in table.cells:
        return table.global_row
    levels = table.levels(task)
    if complexity is None or not levels:
        return table.cell(task, ALL)
    points = [(LEVEL_POSITION[level], table.cell(task, level)) for level in levels]
    if complexity <= points[0][0]:
        return points[0][1]
    if complexity >= points[-1][0]:
        return points[-1][1]
    for (x0, low), (x1, high) in zip(points, points[1:]):
        if x0 <= complexity <= x1:
            return _blend(low, high, (complexity - x0) / (x1 - x0))
    return table.cell(task, ALL)  # Unreachable: the points cover [first, last].


def cost_scale(table: RoutingTable) -> float:
    """One shared training-table maximum across every model and task average."""
    return max((stat.cost_usd for task in table.task_types
                for stat in table.cell(task).values()), default=0.0) or 1.0


def estimate(signal: TaskSignal, table: RoutingTable, models: list[Model], cost_weight: float | None = None,
             use_complexity: bool = DEFAULT_USE_COMPLEXITY, *,
             lam: float | None = None, normalization: float | None = None) -> list[Estimate]:
    """Expected accuracy, cost and utility score for every model the table covers."""
    validate_signal(signal)
    if cost_weight is not None and (lam is not None or not isfinite(cost_weight) or cost_weight < 0):
        raise ValueError('Use either lam in [0, 1] or a historical non-negative cost_weight')
    lam = DEFAULT_LAMBDA if lam is None else lam
    if not isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError('lam must be in [0, 1]')
    scale = cost_scale(table) if normalization is None else normalization
    if not isfinite(scale) or scale <= 0:
        raise ValueError('Cost normalization must be positive')
    complexity = signal.complexity if use_complexity else None
    rows = [(p, task_row(table, task, complexity)) for task, p in signal.task_scores.items() if p > 0]
    estimates, seen = [], set()
    for model in models:
        if model.id in seen:
            raise ValueError(f'Duplicate model ID: {model.id!r}')
        seen.add(model.id)
        if not all(model.id in row for _, row in rows):
            continue
        accuracy = fsum(p * row[model.id].accuracy for p, row in rows)
        cost = fsum(p * row[model.id].cost_usd for p, row in rows)
        score = (accuracy - cost_weight * cost if cost_weight is not None else
                 lam * accuracy - (1 - lam) * cost / scale)
        estimates.append(Estimate(model, accuracy, cost, score))
    return estimates


def rank_models(signal: TaskSignal, table: RoutingTable, models: list[Model], cost_weight: float | None = None,
                limit: int = FALLBACK_LIMIT, mode: str = 'utility', tau: float | None = None,
                use_complexity: bool = DEFAULT_USE_COMPLEXITY, *,
                lam: float | None = None, normalization: float | None = None) -> list[Model]:
    """Best candidates first, for execution and fallback on provider errors."""
    if type(limit) is not int or limit <= 0:
        raise ValueError('limit must be a positive integer')
    estimates = estimate(signal, table, models, cost_weight, use_complexity, lam=lam, normalization=normalization)
    if not estimates:
        raise ValueError('No pool model has routing-table measurements')

    def tiebreak(e):
        return (e.cost_usd, -e.accuracy, e.model.provider, e.model.id)

    if mode == 'utility':
        ranked = sorted(estimates, key=lambda e: (-e.score, *tiebreak(e)))
    elif mode == 'threshold':
        if tau is None or not isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError('threshold mode needs tau in [0, 1]')
        good = sorted((e for e in estimates if e.accuracy >= tau), key=tiebreak)
        rest = sorted((e for e in estimates if e.accuracy < tau), key=lambda e: (-e.accuracy, *tiebreak(e)))
        ranked = good + rest
    else:
        raise ValueError(f'Unknown mode {mode!r}')
    return [e.model for e in ranked[:limit]]


ESCALATION_FACTOR = 1.0  # Escalation uses the request's own lambda (see escalation_target).


def escalation_target(signal: TaskSignal, table: RoutingTable, models: list[Model], cost_weight: float,
                      first: Model, factor: float = ESCALATION_FACTOR,
                      use_complexity: bool = DEFAULT_USE_COMPLEXITY, *,
                      lam: float | None = None, normalization: float | None = None) -> Model | None:
    """The model to try when ``first``'s answer fails a check (lonn.answer_check).

    A failed answer is almost never right (under 1% on the benchmark), so a second
    call is worth its price when the best other model's own score,
    ``accuracy - cost_weight * factor * cost``, is positive: its expected accuracy is worth
    more than its expected cost at this lambda. Returns that model, or None when no
    second call pays off (then the first answer is returned as it is).
    """
    others = [e for e in estimate(signal, table, models, None if cost_weight is None else cost_weight * factor, use_complexity,
                                           lam=lam, normalization=normalization) if e.model.id != first.id]
    if not others:
        return None
    best = max(others, key=lambda e: (e.score, -e.cost_usd, e.model.id))
    return best.model if best.score > 0 else None
