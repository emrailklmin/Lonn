"""Prototype 4.0 aptitude sources, routing, and fold-local derivation."""
from math import fsum
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastapi.testclient import TestClient  # noqa: E402
from lonn.api import create_app  # noqa: E402
from lonn.model_pool import load_model_pool  # noqa: E402
from lonn.routing_table import (RoutingTable, add_supplied_aptitudes, build_table,  # noqa: E402
                                load_supplied_aptitudes)
from lonn.schemas import TaskSignal  # noqa: E402
from lonn.task_router import rank_models  # noqa: E402
from lonn.task_types import load_task_types  # noqa: E402


class TaskAptitudeTests(unittest.TestCase):
    def setUp(self):
        self.original = RoutingTable.load()
        self.supplied = load_supplied_aptitudes()
        self.table = add_supplied_aptitudes(self.original, self.supplied)
        self.types = load_task_types()

    def test_measured_cells_are_unchanged_and_all_types_are_present(self):
        self.assertEqual(set(self.table.task_types), {t.id for t in self.types})
        self.assertEqual(len(self.table.task_types), 15)
        self.assertEqual(self.table.global_row, self.original.global_row)
        for task in self.original.task_types:
            self.assertEqual(self.table.cells[task], self.original.cells[task])
        self.assertEqual(len(self.supplied), 8)

    def test_supplied_scores_fallback_costs_and_other_mean_for_every_model(self):
        self.assertEqual(self.supplied['openrouter/meta/muse-spark-1.3']['information_extraction'], 0.957)
        self.assertEqual(self.supplied['openrouter/openai/gpt-oss-20b']['tool_calling'], 0.548)
        for model_id, values in self.supplied.items():
            profile = self.table.model_aptitudes(model_id)
            self.assertEqual(len(profile), 15)
            for task, expected in values.items():
                self.assertEqual(profile[task].score, expected)
                self.assertEqual(profile[task].source, 'supplied')
                self.assertIsNone(profile[task].sample_size)
                self.assertEqual(profile[task].cost_usd, self.original.global_row[model_id].cost_usd)
                self.assertEqual(profile[task].cost_source, 'global_benchmark')
            expected_other = fsum(p.score for task, p in profile.items() if task != 'other') / 14
            self.assertAlmostEqual(profile['other'].score, expected_other)
            self.assertEqual(profile['other'].source, 'derived')
            self.assertEqual(profile['other'].cost_usd, self.original.global_row[model_id].cost_usd)
            self.assertEqual(profile['math_problem'].source, 'measured')
            self.assertGreater(profile['math_problem'].sample_size, 0)

    def test_other_changes_with_measured_rows(self):
        model_id = 'openrouter/test/model'
        measured = [t.id for t in self.types if t.measured]
        supplied = {model_id: {t.id: 0.8 for t in self.types if not t.measured and t.id != 'other'}}
        rows = [{'task': task, 'model_id': model_id, 'correct': True, 'cost_usd': 0.001}
                for task in measured]
        first = add_supplied_aptitudes(build_table(rows), supplied)
        changed = [{**r, 'correct': False} if r['task'] == 'math_problem' else r for r in rows]
        second = add_supplied_aptitudes(build_table(changed), supplied)
        self.assertAlmostEqual(first.cell('other')[model_id].accuracy - second.cell('other')[model_id].accuracy,
                               1 / 14)
        self.assertEqual(first.cell('summarization')[model_id].accuracy, second.cell('summarization')[model_id].accuracy)

    def test_supplied_scores_change_ranking_without_overwriting_measured_types(self):
        models = load_model_pool()
        signal = TaskSignal({'translation': 1.0})
        old = rank_models(signal, self.original, models, lam=0, limit=1)[0]
        new = rank_models(signal, self.table, models, lam=0, limit=1)[0]
        self.assertEqual(old.id, 'openrouter/anthropic/claude-sonnet-5.5')
        self.assertEqual(new.id, 'openrouter/google/gemini-3.8-flash')

    def test_rejects_incomplete_or_invalid_supplied_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.yaml'
            for bad in ({'summarization': 0.5}, {**next(iter(self.supplied.values())), 'translation': 1.1}):
                path.write_text(yaml.safe_dump({'version': 1, 'source': 'user_supplied', 'models': {'x': bad}}))
                with self.subTest(scores=bad), self.assertRaises(ValueError):
                    load_supplied_aptitudes(path)
        with self.assertRaisesRegex(ValueError, 'Missing supplied aptitudes'):
            add_supplied_aptitudes(self.original, {})

    def test_expanded_table_cannot_overwrite_the_measured_source(self):
        with self.assertRaisesRegex(ValueError, 'Save only the measured table'):
            self.table.save()

    def test_api_exposes_the_same_fifteen_scores_and_sources(self):
        with TestClient(create_app()) as client:
            models = client.get('/models').json()
            model = next(m for m in models if m['id'] == 'openrouter/google/gemini-3.8-flash')
            scores = model['task_aptitudes']
            self.assertEqual(len(scores), 15)
            self.assertEqual(scores['translation']['score'], 1.0)
            self.assertEqual(scores['translation']['source'], 'supplied')
            self.assertEqual(scores['other']['source'], 'derived')
            self.assertEqual(scores['math_problem']['source'], 'measured')
            routed = client.post('/route', json={'prompt': 'Translate this', 'lam': 0,
                                                 'signal': {'task_scores': {'translation': 1.0}}})
            self.assertEqual(routed.status_code, 200, routed.text)
            self.assertEqual(routed.json()['models'][0]['id'], model['id'])


if __name__ == '__main__':
    unittest.main()
