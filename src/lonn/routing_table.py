"""Measured accuracy and cost per (task type, difficulty, model) (issue #19).

The table is built from graded answers, for example the benchmark answer bank,
passed in as plain rows. It does not read the bank itself and makes no routing
decisions.

- ``all``: every row of the task type.
- ``easy`` / ``medium`` / ``hard``: rows with that difficulty label. Small cells
  are shrunk towards the task type's ``all`` value with ``smoothing``
  pseudo-answers, so a 30-question cell cannot swing the choice on its own.
- ``global``: the equal-weighted mean of every task type's ``all`` cell, used
  for probability mass on unmeasured task types.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from .task_types import DIFFICULTIES

DEFAULT_ROUTING_TABLE = Path(__file__).resolve().parents[2] / 'config' / 'routing_table.json'
ALL = 'all'


@dataclass(frozen=True)
class CellStats:
    accuracy: float
    cost_usd: float  # mean billed cost per request
    n: int


Cells = dict[str, CellStats]  # model ID -> stats


class RoutingTable:
    def __init__(self, cells: dict[str, dict[str, Cells]], global_row: Cells, metadata: dict | None = None):
        self.cells, self.global_row, self.metadata = cells, global_row, metadata or {}

    @property
    def task_types(self) -> list[str]:
        return sorted(self.cells)

    def levels(self, task: str) -> list[str]:
        """Difficulty levels the task type has cells for, easy to hard."""
        return [level for level in DIFFICULTIES if level in self.cells.get(task, {})]

    def cell(self, task: str, level: str = ALL) -> Cells:
        return self.cells[task][level]

    def to_dict(self) -> dict:
        def dump(cells):
            return {m: {'accuracy': round(s.accuracy, 6), 'cost_usd': round(s.cost_usd, 8), 'n': s.n}
                    for m, s in sorted(cells.items())}
        return {'version': 1, 'metadata': self.metadata,
                'cells': {t: {lv: dump(c) for lv, c in levels.items()} for t, levels in sorted(self.cells.items())},
                'global': dump(self.global_row)}

    @classmethod
    def from_dict(cls, data: dict) -> 'RoutingTable':
        def load(cells):
            return {m: CellStats(s['accuracy'], s['cost_usd'], s['n']) for m, s in cells.items()}
        return cls({t: {lv: load(c) for lv, c in levels.items()} for t, levels in data['cells'].items()},
                   load(data['global']), data.get('metadata') or {})

    def save(self, path: str | Path = DEFAULT_ROUTING_TABLE) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + '\n')

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ROUTING_TABLE) -> 'RoutingTable':
        return cls.from_dict(json.loads(Path(path).read_text()))


def build_table(rows, smoothing: float = 5.0, metadata: dict | None = None) -> RoutingTable:
    """Build the table from rows with ``task``, ``difficulty`` (or None), ``model_id``,
    ``correct`` (bool) and ``cost_usd``. Failed calls should be left out by the caller."""
    if smoothing < 0:
        raise ValueError('smoothing must be non-negative')
    sums: dict[tuple[str, str, str], list] = {}
    for row in rows:
        levels = [ALL] + ([row['difficulty']] if row.get('difficulty') else [])
        for level in levels:
            if level != ALL and level not in DIFFICULTIES:
                raise ValueError(f"Unknown difficulty {level!r}")
            total = sums.setdefault((row['task'], level, row['model_id']), [0, 0, 0.0])
            total[0] += bool(row['correct'])
            total[1] += 1
            total[2] += float(row['cost_usd'] or 0.0)

    cells: dict[str, dict[str, Cells]] = {}
    for (task, level, model_id), (passes, n, cost) in sums.items():
        if level == ALL:
            cells.setdefault(task, {}).setdefault(ALL, {})[model_id] = CellStats(passes / n, cost / n, n)
    for (task, level, model_id), (passes, n, cost) in sums.items():
        if level == ALL:
            continue
        parent = cells[task][ALL][model_id]
        k = smoothing
        cells[task].setdefault(level, {})[model_id] = CellStats(
            (passes + k * parent.accuracy) / (n + k), (cost + k * parent.cost_usd) / (n + k), n)

    global_row: Cells = {}
    models = {m for levels in cells.values() for m in levels[ALL]}
    for model_id in sorted(models):
        stats = [levels[ALL][model_id] for levels in cells.values() if model_id in levels[ALL]]
        global_row[model_id] = CellStats(sum(s.accuracy for s in stats) / len(stats),
                                         sum(s.cost_usd for s in stats) / len(stats),
                                         sum(s.n for s in stats))
    return RoutingTable(cells, global_row, metadata)
