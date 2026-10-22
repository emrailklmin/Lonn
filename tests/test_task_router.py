"""Routing table and task-type router (#19). Pure functions: no bank, no network."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lonn.routing_table import ALL, RoutingTable, build_table  # noqa: E402
from lonn.schemas import Model, TaskSignal, TopicCategory as T  # noqa: E402
from lonn import task_router as tr  # noqa: E402
from lonn.task_types import Anchor, TaskType, difficulty_for_record, load_task_types, task_for_record  # noqa: E402


def model(name):
    return Model(f'openrouter/test/{name}', 'openrouter', 1.0, 1.0, 100000, {t: 0.5 for t in T})


CHEAP, STRONG = model('cheap'), model('strong')
POOL = [CHEAP, STRONG]


def rows(task, difficulty, model_id, passes, n, cost):
    return [{'task': task, 'difficulty': difficulty, 'model_id': model_id, 'correct': i < passes,
             'cost_usd': cost} for i in range(n)]


def fixture_table(smoothing=0.0):
    """code: cheap solves easy problems, only strong solves hard ones.
    writing: no difficulty labels; strong is better."""
    data = (rows('code', 'easy', CHEAP.id, 10, 10, 0.001) + rows('code', 'hard', CHEAP.id, 2, 10, 0.002)
            + rows('code', 'easy', STRONG.id, 10, 10, 0.005) + rows('code', 'hard', STRONG.id, 9, 10, 0.01)
            + rows('writing', None, CHEAP.id, 1, 10, 0.001) + rows('writing', None, STRONG.id, 8, 10, 0.05))
    return build_table(data, smoothing=smoothing)


class RoutingTableTests(unittest.TestCase):
    def test_cells_and_levels(self):
        table = fixture_table()
        self.assertEqual(table.task_types, ['code', 'writing'])
        self.assertEqual(table.levels('code'), ['easy', 'hard'])
        self.assertEqual(table.levels('writing'), [])
        self.assertAlmostEqual(table.cell('code', ALL)[CHEAP.id].accuracy, 0.6)
        self.assertAlmostEqual(table.cell('code', 'hard')[CHEAP.id].accuracy, 0.2)
        self.assertAlmostEqual(table.cell('code', 'hard')[STRONG.id].cost_usd, 0.01)

    def test_smoothing_shrinks_levels_towards_the_task_average(self):
        table = fixture_table(smoothing=10.0)
        # (2 + 10 * 0.6) / (10 + 10)
        self.assertAlmostEqual(table.cell('code', 'hard')[CHEAP.id].accuracy, 0.4)
        self.assertAlmostEqual(table.cell('code', ALL)[CHEAP.id].accuracy, 0.6)

    def test_global_row_is_the_equal_weighted_task_mean(self):
        table = fixture_table()
        self.assertAlmostEqual(table.global_row[CHEAP.id].accuracy, (0.6 + 0.1) / 2)

    def test_round_trip(self):
        table = fixture_table(smoothing=5.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'table.json'
            table.save(path)
            loaded = RoutingTable.load(path)
        self.assertEqual(loaded.to_dict(), table.to_dict())

    def test_rejects_unknown_difficulty_and_negative_smoothing(self):
        with self.assertRaises(ValueError):
            build_table(rows('code', 'extreme', CHEAP.id, 1, 1, 0.0))
        with self.assertRaises(ValueError):
            build_table([], smoothing=-1)


class TaskRouterTests(unittest.TestCase):
    def setUp(self):
        self.table = fixture_table()

    def pick(self, scores, complexity=None, **kwargs):
        kwargs.setdefault('use_complexity', True)
        return tr.rank_models(TaskSignal(scores, complexity), self.table, POOL, **kwargs)[0]

    def test_easy_prompt_goes_cheap_and_hard_prompt_goes_strong(self):
        self.assertEqual(self.pick({'code': 1.0}, 0.0, lam=20), CHEAP)
        self.assertEqual(self.pick({'code': 1.0}, 1.0, lam=20), STRONG)

    def test_complexity_is_off_by_default(self):
        self.assertFalse(tr.DEFAULT_USE_COMPLEXITY)
        # Lambda 70: cheap wins on the code average (0.495 vs 0.425), strong on hard code (0.2 vs 0.06).
        hard = TaskSignal({'code': 1.0}, 1.0)
        self.assertEqual(tr.rank_models(hard, self.table, POOL, lam=70)[0], CHEAP)
        self.assertEqual(tr.rank_models(hard, self.table, POOL, lam=70, use_complexity=True)[0], STRONG)

    def test_complexity_interpolates_between_levels(self):
        row = tr.task_row(self.table, 'code', 0.5)
        self.assertAlmostEqual(row[CHEAP.id].accuracy, 0.6)
        self.assertAlmostEqual(row[STRONG.id].cost_usd, 0.0075)

    def test_no_complexity_uses_the_task_average(self):
        self.assertIs(tr.task_row(self.table, 'code', None), self.table.cell('code', ALL))
        self.assertIs(tr.task_row(self.table, 'writing', 0.9), self.table.cell('writing', ALL))

    def test_lambda_trades_quality_for_cost(self):
        self.assertEqual(self.pick({'writing': 1.0}, lam=0), STRONG)
        self.assertEqual(self.pick({'writing': 1.0}, lam=1000), CHEAP)

    def test_task_mix_is_probability_weighted(self):
        [cheap] = [e for e in tr.estimate(TaskSignal({'code': 0.5, 'writing': 0.5}), self.table, POOL)
                   if e.model == CHEAP]
        self.assertAlmostEqual(cheap.accuracy, 0.5 * 0.6 + 0.5 * 0.1)

    def test_unmeasured_task_uses_the_global_row(self):
        [cheap] = [e for e in tr.estimate(TaskSignal({'summarization': 1.0}), self.table, POOL)
                   if e.model == CHEAP]
        self.assertAlmostEqual(cheap.accuracy, self.table.global_row[CHEAP.id].accuracy)

    def test_threshold_mode_picks_the_cheapest_good_enough_model(self):
        self.assertEqual(self.pick({'code': 1.0}, 0.0, mode='threshold', tau=0.9), CHEAP)
        self.assertEqual(self.pick({'code': 1.0}, 1.0, mode='threshold', tau=0.9), STRONG)
        # Nobody reaches tau: the most accurate model.
        self.assertEqual(self.pick({'writing': 1.0}, mode='threshold', tau=0.99), STRONG)

    def test_models_missing_from_the_table_are_not_ranked(self):
        ranking = tr.rank_models(TaskSignal({'code': 1.0}), self.table, POOL + [model('new')])
        self.assertEqual({m.id for m in ranking}, {CHEAP.id, STRONG.id})
        with self.assertRaises(ValueError):
            tr.rank_models(TaskSignal({'code': 1.0}), self.table, [model('new')])

    def test_invalid_input(self):
        for signal in (TaskSignal({}), TaskSignal({'code': 0.5}), TaskSignal({'code': 1.0}, 1.5)):
            with self.assertRaises(ValueError):
                tr.rank_models(signal, self.table, POOL)
        with self.assertRaises(ValueError):
            tr.rank_models(TaskSignal({'code': 1.0}), self.table, POOL, lam=-1)
        with self.assertRaises(ValueError):
            tr.rank_models(TaskSignal({'code': 1.0}), self.table, POOL, mode='threshold')
        with self.assertRaises(ValueError):
            tr.rank_models(TaskSignal({'code': 1.0}), self.table, POOL, mode='best')
        with self.assertRaises(ValueError):
            tr.rank_models(TaskSignal({'code': 1.0}), self.table, [CHEAP, CHEAP])


class TaskTypeLoaderTests(unittest.TestCase):
    def test_config_loads(self):
        types = load_task_types()
        self.assertIn('math_problem', {t.id for t in types})
        self.assertTrue(all(t.description for t in types))

    def test_record_resolution(self):
        types = load_task_types()
        by_id = {t.id: t for t in types}
        record = {'dataset': 'mmlu_pro', 'subject': 'law'}
        self.assertEqual(task_for_record(record, types), 'knowledge_recall')
        self.assertEqual(difficulty_for_record(record, by_id['knowledge_recall']), 'medium')
        record = {'dataset': 'math500', 'level_label': 'Level 4'}
        self.assertEqual(difficulty_for_record(record, by_id['math_problem']), 'hard')
        self.assertIsNone(task_for_record({'dataset': 'bfcl_simple'}, types))
        self.assertIsNone(difficulty_for_record({'dataset': 'ifeval'}, TaskType('x', True, 'x', (Anchor('ifeval'),))))


if __name__ == '__main__':
    unittest.main()
