import importlib.util
import sys
import unittest
import warnings
from math import fsum
from pathlib import Path

# Allow direct execution from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lonn.complexity import ComplexityScorer
from lonn.router import select_model
from lonn.schemas import ComplexityResult, Model, TopicCategory as Topic


class FakeLaya:
    """Stands in for laya.Router, returning canned answers per question."""

    def __init__(self, category_answer, complexity_score):
        self.category_answer = category_answer
        self.complexity_score = complexity_score

    def predict(self, text, question):
        if "category" in question:
            return {"answers": {"category": self.category_answer}}
        return {"answers": {"complexity": {"score": self.complexity_score}}}


def probs(**values):
    return {t.value: values.get(t.value, 0.0) for t in Topic}


class ComplexityScorerTests(unittest.TestCase):
    def score(self, category_answer, complexity_score=2.0):
        laya = FakeLaya(category_answer, complexity_score)
        return ComplexityScorer(laya).score_prompt("prompt")

    def test_returns_schema_result(self):
        result = self.score({"choice": "coding",
                             "probabilities": probs(coding=0.7, math_reasoning=0.2, creative_writing=0.1)})
        self.assertIsInstance(result, ComplexityResult)
        self.assertEqual(set(result.topic_scores), set(Topic))
        self.assertTrue(all(isinstance(t, Topic) for t in result.topic_scores))
        self.assertAlmostEqual(result.topic_scores[Topic.CODING], 0.7)
        self.assertAlmostEqual(result.complexity, 0.5)

    def test_distribution_is_normalized(self):
        result = self.score({"choice": "math_reasoning", "probabilities": probs(math_reasoning=3, coding=1)})
        self.assertAlmostEqual(fsum(result.topic_scores.values()), 1.0, delta=1e-9)
        self.assertAlmostEqual(result.topic_scores[Topic.MATH_REASONING], 0.75)

    def test_distribution_formats(self):
        names = [t.value for t in Topic]
        answers = [
            {"choice": "coding", "scores": [0.7, 0.1, 0.1, 0.1]},
            {"choice": "coding", "options": [
                {"label": n, "probability": 0.7 if n == "coding" else 0.1} for n in names]},
        ]
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertAlmostEqual(self.score(answer).topic_scores[Topic.CODING], 0.7)

    def test_missing_distribution_falls_back_to_choice(self):
        for answer in ({"choice": "math_reasoning"},
                       {"choice": "math_reasoning", "probabilities": probs()},
                       {"choice": "math_reasoning", "probabilities": probs(coding=float("nan"))}):
            with self.subTest(answer=answer), warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = self.score(answer)
                self.assertEqual(len(w), 1)
                self.assertEqual(result.topic_scores[Topic.MATH_REASONING], 1.0)
                self.assertEqual(fsum(result.topic_scores.values()), 1.0)

    def test_complexity_is_mapped_and_clamped(self):
        answer = {"choice": "coding", "probabilities": probs(coding=1)}
        for level, expected in [(0, 0.0), (1, 0.25), (4, 1.0), (-1, 0.0), (7, 1.0)]:
            with self.subTest(level=level):
                self.assertEqual(self.score(answer, level).complexity, expected)
        with self.assertRaises(ValueError):
            self.score(answer, float("nan"))

    def test_output_is_accepted_by_router(self):
        # Awkward probabilities that would not sum to 1 if rounded.
        answer = {"choice": "coding", "probabilities": probs(
            coding=1 / 3, math_reasoning=1 / 7, creative_writing=1 / 11,
            general_knowledge_extraction=1 / 13)}
        model = Model("m", "example", 1, 1, 128000, {Topic.CODING: 0.5})
        self.assertIs(select_model(self.score(answer), [model]), model)


@unittest.skipUnless(importlib.util.find_spec("laya"), "laya is not installed")
class LiveLayaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = ComplexityScorer()

    def test_coding_prompt(self):
        result = self.scorer.score_prompt("Write a Python function that sorts a list")
        self.assertEqual(max(result.topic_scores, key=result.topic_scores.get), Topic.CODING)
        self.assertAlmostEqual(fsum(result.topic_scores.values()), 1.0, delta=1e-6)
        self.assertTrue(0 <= result.complexity <= 1)


if __name__ == "__main__":
    unittest.main()
