"""Task aptitudes and costs per (task type, difficulty, model).

The table is built from graded answers, for example the benchmark answer bank,
passed in as plain rows. It does not read the bank itself and makes no routing
decisions.

- ``all``: every row of the task type.
- ``easy`` / ``medium`` / ``hard``: rows with that difficulty label. Small cells
  are shrunk towards the task type's ``all`` value with ``smoothing``
  pseudo-answers, so a 30-question cell cannot swing the choice on its own.
- ``global``: the equal-weighted mean of the measured task types' ``all`` cells.
  Prototype 4.0 uses its cost as a fallback for supplied and derived aptitudes.
"""

from dataclasses import dataclass
import json
from math import fsum, isfinite
from pathlib import Path

import yaml

from .schemas import TaskAptitude
from .task_types import DIFFICULTIES, load_task_types

DEFAULT_ROUTING_TABLE = Path(__file__).resolve().parents[2] / 'config' / 'routing_table.json'
DEFAULT_TASK_APTITUDES = Path(__file__).resolve().parents[2] / 'config' / 'task_aptitudes.yaml'
ALL = 'all'


@dataclass(frozen=True)
class CellStats:
    accuracy: float
    cost_usd: float  # mean billed cost per request
    n: int
    source: str = 'measured'
    cost_source: str = 'task_benchmark'


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

    def model_aptitudes(self, model_id: str) -> dict[str, TaskAptitude]:
        """The same task scores used for routing, with their provenance."""
        scores = {}
        for task in self.task_types:
            stat = self.cell(task).get(model_id)
            if stat is not None:
                scores[task] = TaskAptitude(
                    stat.accuracy, stat.source, stat.n if stat.source == 'measured' else None,
                    stat.cost_usd, stat.cost_source)
        return scores

    def to_dict(self) -> dict:
        def dump_stat(s):
            out = {'accuracy': round(s.accuracy, 6), 'cost_usd': round(s.cost_usd, 8), 'n': s.n}
            if s.source != 'measured':
                out['source'] = s.source
            if s.cost_source != 'task_benchmark':
                out['cost_source'] = s.cost_source
            return out

        def dump(cells):
            return {m: dump_stat(s) for m, s in sorted(cells.items())}
        return {'version': 1, 'metadata': self.metadata,
                'cells': {t: {lv: dump(c) for lv, c in levels.items()} for t, levels in sorted(self.cells.items())},
                'global': dump(self.global_row)}

    @classmethod
    def from_dict(cls, data: dict) -> 'RoutingTable':
        def load(cells):
            return {m: CellStats(s['accuracy'], s['cost_usd'], s['n'], s.get('source', 'measured'),
                                 s.get('cost_source', 'task_benchmark')) for m, s in cells.items()}
        return cls({t: {lv: load(c) for lv, c in levels.items()} for t, levels in data['cells'].items()},
                   load(data['global']), data.get('metadata') or {})

    def save(self, path: str | Path = DEFAULT_ROUTING_TABLE) -> None:
        if Path(path).resolve() == DEFAULT_ROUTING_TABLE.resolve() and any(
            stat.source != 'measured'
            for levels in self.cells.values() for cells in levels.values() for stat in cells.values()
        ):
            raise ValueError('Save only the measured table to the default path; supplied scores have their own config')
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + '\n')

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ROUTING_TABLE) -> 'RoutingTable':
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_supplied_aptitudes(path: str | Path = DEFAULT_TASK_APTITUDES) -> dict[str, dict[str, float]]:
    """Read the one authoritative set of user-supplied task estimates."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or data.get('version') != 1 or data.get('source') != 'user_supplied':
        raise ValueError('Invalid supplied task aptitude metadata')
    models = data.get('models')
    if not isinstance(models, dict) or not models:
        raise ValueError('Supplied task aptitudes must contain model scores')
    tasks = {t.id for t in load_task_types() if not t.measured and t.id != 'other'}
    for model_id, scores in models.items():
        if not isinstance(model_id, str) or not isinstance(scores, dict) or set(scores) != tasks:
            raise ValueError(f'{model_id}: supplied aptitudes must cover exactly {sorted(tasks)}')
        if any(type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError(f'{model_id}: supplied aptitudes must be finite numbers in [0, 1]')
    return models


def add_supplied_aptitudes(table: RoutingTable, supplied: dict[str, dict[str, float]]) -> RoutingTable:
    """Overlay fixed estimates and derive `other` from this table's measured rows.

    The original measured rows and global benchmark cost remain unchanged. This
    works on both the production table and a training-fold table during replay.
    """
    types = load_task_types()
    measured = [t.id for t in types if t.measured]
    estimated = [t.id for t in types if not t.measured and t.id != 'other']
    if len(measured) != 8 or len(estimated) != 6 or 'other' not in {t.id for t in types}:
        raise ValueError('Prototype 4.0 requires eight measured, six supplied and one other task type')
    missing_models = set(table.global_row) - set(supplied)
    if missing_models:
        raise ValueError(f'Missing supplied aptitudes for models: {sorted(missing_models)}')
    if set(estimated + ['other']) & set(table.cells):
        raise ValueError('Supplied aptitudes must not overwrite existing task measurements')
    cells = {task: dict(levels) for task, levels in table.cells.items()}
    for task in estimated + ['other']:
        cells[task] = {ALL: {}}
    for model_id, global_stat in table.global_row.items():
        scores = supplied[model_id]
        if not isinstance(scores, dict) or set(scores) != set(estimated) or any(
            type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= 1 for v in scores.values()
        ):
            raise ValueError(f'{model_id}: invalid supplied task aptitudes')
        try:
            measured_scores = [table.cell(task)[model_id].accuracy for task in measured]
        except KeyError as exc:
            raise ValueError(f'{model_id}: missing measured aptitude for {exc.args[0]}') from exc
        for task in estimated:
            cells[task][ALL][model_id] = CellStats(scores[task], global_stat.cost_usd, 0,
                                                  'supplied', 'global_benchmark')
        other = fsum(measured_scores + [scores[task] for task in estimated]) / (len(measured) + len(estimated))
        cells['other'][ALL][model_id] = CellStats(other, global_stat.cost_usd, 0, 'derived', 'global_benchmark')
    return RoutingTable(cells, table.global_row, {**table.metadata, 'task_aptitudes': 'user_supplied'})


def load_runtime_table() -> RoutingTable:
    """Load measured scores and overlay the configured Prototype 4.0 estimates."""
    return add_supplied_aptitudes(RoutingTable.load(), load_supplied_aptitudes())


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
