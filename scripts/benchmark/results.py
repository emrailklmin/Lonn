"""The answer bank: one JSON Lines file per (dataset, model), committed to git.

results/benchmark/<category>/<dataset>/<model>.jsonl gets one line per answer,
appended as soon as it arrives. A later line for the same (prompt, model)
replaces an earlier one, so a retried error is simply appended again.

    from benchmark.results import ResultBank
    bank = ResultBank()
    row = bank.lookup('math500-017', 'openrouter/openai/gpt-oss-20b')
    row['correct'], row['cost_usd'], row['response']
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / 'results' / 'benchmark'


def model_slug(model_id: str) -> str:
    """'openrouter/openai/gpt-oss-20b' -> 'openai__gpt-oss-20b' (safe as a file name)."""
    return model_id.removeprefix('openrouter/').replace('/', '__')


def result_path(results_dir: Path, category: str, dataset: str, model_id: str) -> Path:
    return Path(results_dir) / category / dataset / f'{model_slug(model_id)}.jsonl'


_locks = defaultdict(threading.Lock)


def append_row(path: Path, row: dict) -> None:
    """Append one answer and flush it to disk immediately (safe across threads)."""
    path = Path(path)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
    with _locks[str(path)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def read_rows(path: Path) -> dict:
    """Latest row per prompt_id in one file. A half-written last line (crash) is skipped."""
    rows = {}
    path = Path(path)
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row['prompt_id']] = row
    return rows


def rewrite_rows(path: Path, rows: list) -> None:
    """Replace a file's contents atomically (used by regrading)."""
    path = Path(path)
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                                 for row in sorted(rows, key=lambda r: r['prompt_id'])),
                         encoding='utf-8')
    temporary.replace(path)


def iter_files(results_dir: Path = RESULTS_DIR):
    return sorted(Path(results_dir).glob('*/*/*.jsonl'))


class ResultBank:
    """All saved answers, keyed by (prompt_id, model_id)."""

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.rows = {}
        for path in iter_files(results_dir):
            for row in read_rows(path).values():
                self.rows[(row['prompt_id'], row['model_id'])] = row

    def lookup(self, prompt_id: str, model_id: str):
        return self.rows.get((prompt_id, model_id))

    def answered(self):
        """Rows with a model answer (no call error)."""
        return [row for row in self.rows.values() if not row.get('error')]

    def spent(self) -> float:
        """Model calls plus LLM-judge calls recorded in the answer rows."""
        return sum((row.get('cost_usd') or 0.0) + (row.get('judge_cost_usd') or 0.0)
                   for row in self.rows.values())

    def summary(self, by=('model_id', 'dataset')) -> list:
        """Accuracy, cost and tokens grouped by row fields (e.g. model_id, category, difficulty)."""
        groups = defaultdict(list)
        for row in self.rows.values():
            groups[tuple(row.get(key) for key in by)].append(row)
        table = []
        for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
            graded = [row for row in rows if row.get('correct') is not None]
            table.append({
                **dict(zip(by, key)), 'answers': len(rows),
                'errors': sum(bool(row.get('error')) for row in rows),
                'graded': len(graded),
                'accuracy': (sum(row['correct'] for row in graded) / len(graded)) if graded else None,
                'truncated': sum(row.get('finish_reason') == 'length' for row in rows),
                'cost_usd': sum(row.get('cost_usd') or 0.0 for row in rows),
                'output_tokens': sum(row.get('output_tokens') or 0 for row in rows),
            })
        return table
