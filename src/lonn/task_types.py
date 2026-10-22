"""Load the task-type taxonomy from config/task_types.yaml (issue #19).

The loader makes no routing decisions. It also resolves benchmark records to
their task type and difficulty, which the routing-table builder and the
evaluation need.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_TASK_TYPES = Path(__file__).resolve().parents[2] / 'config' / 'task_types.yaml'
DIFFICULTIES = ('easy', 'medium', 'hard')


@dataclass(frozen=True)
class Anchor:
    dataset: str
    subjects: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TaskType:
    id: str
    measured: bool
    description: str
    anchors: tuple[Anchor, ...] = ()
    difficulty: dict = field(default_factory=dict, hash=False, compare=False)


def load_task_types(path: str | Path = DEFAULT_TASK_TYPES) -> list[TaskType]:
    config = yaml.safe_load(Path(path).read_text())
    types = []
    for entry in config['task_types']:
        anchors = tuple(Anchor(a['dataset'], tuple(a['subjects']) if a.get('subjects') else None)
                        for a in entry.get('anchors') or ())
        types.append(TaskType(entry['id'], bool(entry['measured']), entry['description'].strip(),
                              anchors, entry.get('difficulty') or {}))
    ids = [t.id for t in types]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task-type IDs in task_types config')
    return types


def task_for_record(record: dict, types: list[TaskType]) -> str | None:
    """The measured task type a benchmark record is anchored to, or None."""
    for task in types:
        for anchor in task.anchors:
            if anchor.dataset != record['dataset']:
                continue
            if anchor.subjects is None or record.get('subject') in anchor.subjects:
                return task.id
    return None


def difficulty_for_record(record: dict, task: TaskType) -> str | None:
    """The record's difficulty label under the task type's rule, or None."""
    rule = task.difficulty
    if 'map' in rule:
        return rule['map'].get(record.get(rule['field']))
    if 'by_dataset' in rule:
        return rule['by_dataset'].get(record['dataset'])
    return None
