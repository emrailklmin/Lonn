"""Task classifier (#19): the linear head with a fake embedder, and the shipped config. No downloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lonn.task_classifier import DEFAULT_TASK_CLASSIFIER, TaskClassifier, head_tail  # noqa: E402
from lonn.task_types import load_task_types  # noqa: E402

CONFIG = {'task_types': ['math_problem', 'code_function'], 'logit_scale': 20.0,
          'weight': [[1.0, 0.0], [0.0, 1.0]], 'bias': [0.0, 0.0],
          'embedder': {'repo': 'unused', 'revision': 'unused', 'max_len': 512}}


def fake_embed(texts):
    return [[1.0, 0.0] if 'integral' in t else [0.0, 1.0] for t in texts]


class TaskClassifierTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'classifier.json'
        self.path.write_text(json.dumps(CONFIG))

    def test_classify_returns_a_distribution_without_complexity(self):
        signal = TaskClassifier(self.path, embed=fake_embed).classify('Compute the integral of x^2')
        self.assertAlmostEqual(sum(signal.task_scores.values()), 1.0)
        self.assertGreater(signal.task_scores['math_problem'], 0.99)
        self.assertIsNone(signal.complexity)

    def test_batch_probabilities(self):
        probs = TaskClassifier(self.path, embed=fake_embed).probabilities(['an integral', 'reverse a string'])
        self.assertEqual([max(p, key=p.get) for p in probs], ['math_problem', 'code_function'])

    def test_blank_prompt_and_bad_config(self):
        with self.assertRaises(ValueError):
            TaskClassifier(self.path, embed=fake_embed).classify('  ')
        self.path.write_text(json.dumps({**CONFIG, 'bias': [0.0]}))
        with self.assertRaises(ValueError):
            TaskClassifier(self.path, embed=fake_embed)

    def test_head_tail_keeps_start_and_end(self):
        self.assertEqual(head_tail(list(range(5)), 10), list(range(5)))
        kept = head_tail(list(range(100)), 20)
        self.assertEqual(len(kept), 20)
        self.assertEqual((kept[0], kept[-1]), (0, 99))

    def test_shipped_classifier_matches_the_taxonomy(self):
        config = json.loads(DEFAULT_TASK_CLASSIFIER.read_text())
        self.assertEqual(config['task_types'], [t.id for t in load_task_types()])
        self.assertEqual(len(config['weight']), len(config['task_types']))
        self.assertEqual({len(row) for row in config['weight']}, {384})  # bge-small hidden size
        self.assertEqual(len(config['embedder']['revision']), 40)  # pinned commit


if __name__ == '__main__':
    unittest.main()
