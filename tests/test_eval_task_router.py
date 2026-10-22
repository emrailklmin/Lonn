"""Cross-fitted task-router evaluation (#19). Fake bank: no Laya, no network."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from benchmark import eval_task_router as et, results  # noqa: E402
from lonn.schemas import Model, TopicCategory as T  # noqa: E402
from lonn.task_types import load_task_types  # noqa: E402

CHEAP = Model('openrouter/test/cheap', 'openrouter', 0.1, 0.1, 100000, {t: 0.5 for t in T})
STRONG = Model('openrouter/test/strong', 'openrouter', 10.0, 10.0, 100000, {t: 0.5 for t in T})
POOL = [CHEAP, STRONG]


def question(dataset, n, level=None):
    record = {'prompt_id': f'{dataset}-{n:03d}', 'dataset': dataset, 'category': 'x', 'difficulty': 'medium',
              'topic': 'coding'}
    if level:
        record['level_label'] = level
    return record


class EvalTaskRouterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        bank_dir = Path(directory.name) / 'bank'
        raw = [question('humaneval_plus', n) for n in range(10)]
        raw += [question('math500', n, 'Level 1' if n < 5 else 'Level 5') for n in range(10)]
        self.questions = et.annotate(raw, load_task_types())
        for q in self.questions:
            hard = q.get('level_label') == 'Level 5'
            for m, correct, cost in ((CHEAP, not hard, 0.001), (STRONG, True, 0.002)):
                row = {'prompt_id': q['prompt_id'], 'model_id': m.id, 'dataset': q['dataset'],
                       'error': None, 'correct': correct, 'cost_usd': cost, 'latency_ms': 100,
                       'output_tokens': 10, 'input_tokens': 10}
                results.append_row(results.result_path(bank_dir, 'x', q['dataset'], m.id), row)
        self.bank = results.ResultBank(bank_dir)

    def test_annotate(self):
        tasks = {q['prompt_id']: (q['task'], q['level']) for q in self.questions}
        self.assertEqual(tasks['humaneval_plus-000'], ('code_function', None))
        self.assertEqual(tasks['math500-000'], ('math_problem', 'easy'))
        self.assertEqual(tasks['math500-009'], ('math_problem', 'hard'))
        with self.assertRaises(ValueError):
            et.annotate([question('bfcl_simple', 1)], load_task_types())

    def test_folds_are_stratified_and_cover_every_question(self):
        folds = et.folds_for(self.questions, 5, seed=3)
        self.assertEqual(len(folds), len(self.questions))
        for dataset in ('humaneval_plus', 'math500'):
            counts = [sum(1 for q, f in zip(self.questions, folds) if q['dataset'] == dataset and f == k)
                      for k in range(5)]
            self.assertEqual(counts, [2] * 5)

    def test_a_question_is_never_scored_by_a_table_built_from_it(self):
        seen = []
        original = et.table_rows

        def spy(train, pool, bank):
            seen.append({q['prompt_id'] for q in train})
            return original(train, pool, bank)

        with patch.object(et, 'table_rows', spy):
            summary = et.crossfit(self.questions, POOL, self.bank, [('k', {'lam': 20})], folds=5, repeats=1)
        decisions = summary['k']['decisions']
        self.assertEqual(len(decisions), len(self.questions))
        for d in decisions:
            self.assertNotIn(d['prompt_id'], seen[d['fold']])

    def test_saved_classifier_signals_replace_the_truth(self):
        signals = {q['prompt_id']: {'task_scores': {'summarization': 1.0}, 'complexity': None} for q in self.questions}
        summary = et.crossfit(self.questions, POOL, self.bank, [('k', {'lam': 20})], folds=5, repeats=1,
                              signals=signals)
        self.assertTrue(all(d['signal']['task_scores'] == {'summarization': 1.0}
                            for d in summary['k']['decisions']))

    def test_load_signals_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 's.jsonl'
            path.write_text('{"prompt_id": "a", "task_scores": {"x": 2, "y": 2}, "complexity": null}\n')
            self.assertEqual(et.load_signals(path)['a']['task_scores'], {'x': 0.5, 'y': 0.5})

    def test_complexity_routes_hard_math_to_the_strong_model(self):
        # No smoothing: the fixture's cells hold only 4 questions per level.
        summary = et.crossfit(self.questions, POOL, self.bank, [('k', {'lam': 20, 'use_complexity': True})],
                              folds=5, repeats=2, smoothing=0.0)
        chosen = {d['prompt_id']: d['chosen'] for d in summary['k']['decisions']}
        self.assertEqual(chosen['math500-000'], CHEAP.id)
        self.assertEqual(chosen['math500-009'], STRONG.id)
        self.assertEqual(summary['k']['metrics']['pass_rate'], 1.0)
        self.assertEqual(summary['k']['metrics']['questions'], len(self.questions))


if __name__ == '__main__':
    unittest.main()
