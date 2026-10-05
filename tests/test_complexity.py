import importlib.util
import os
import sys
import types
import unittest
import warnings
from math import fsum
from pathlib import Path

# Allow direct execution from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lonn.complexity import ComplexityScorer
from lonn.router import select_model
from lonn.schemas import ComplexityResult, Model, TopicCategory as Topic


class WordTokenizer:
    """One token per word, standing in for a Laya checkpoint's tokenizer."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}

    def decode(self, ids, clean_up_tokenization_spaces=True):
        return " ".join(ids)


class FakeAgent:
    tok = WordTokenizer()
    cfg = {"max_len": 30}


class FakeLaya:
    """Stands in for laya.Router, answering every question it is asked.

    Each question takes 10 tokens of a 30-token window, leaving 12 prompt
    tokens after the scorer's margin.
    """

    def __init__(self, category_answer, complexity_probability):
        self.category_answer = category_answer
        self.complexity_probability = complexity_probability
        self.seen = []

    def route(self, text, questions):
        return types.SimpleNamespace(model="english")

    def load(self, name):
        return FakeAgent()

    def predict(self, text, questions, model=None):
        self.seen.append(text)
        state_tokens = len(text.split())
        usage = {"input_tokens": min(30, 10 + state_tokens), "state_tokens": state_tokens}
        answers = {}
        if "category" in questions:
            answers["category"] = self.category_answer
        if "complexity" in questions:
            answers["complexity"] = {"noul": self.complexity_probability}
        return {"answers": answers, "usage": usage}


def probs(**values):
    return {t.value: values.get(t.value, 0.0) for t in Topic}


class ComplexityScorerTests(unittest.TestCase):
    def score(self, category_answer, complexity_probability=0.5, prompt="prompt"):
        self.laya = FakeLaya(category_answer, complexity_probability)
        return ComplexityScorer(self.laya).score_prompt(prompt)

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

    def test_complexity_is_hard_probability_clamped(self):
        answer = {"choice": "coding", "probabilities": probs(coding=1)}
        for probability, expected in [(0.0, 0.0), (0.3, 0.3), (1.0, 1.0), (-0.2, 0.0), (1.5, 1.0)]:
            with self.subTest(probability=probability):
                self.assertEqual(self.score(answer, probability).complexity, expected)
        with self.assertRaises(ValueError):
            self.score(answer, float("nan"))

    def test_both_questions_share_one_call(self):
        self.score({"choice": "coding", "probabilities": probs(coding=1)})
        self.assertEqual([t for t in self.laya.seen if t != "Hi"], ["prompt"])

    def test_short_prompt_is_sent_unchanged(self):
        prompt = " ".join(f"w{i}" for i in range(12))
        self.score({"choice": "coding", "probabilities": probs(coding=1)}, prompt=prompt)
        self.assertEqual([t for t in self.laya.seen if t != "Hi"], [prompt])

    def test_long_prompt_keeps_start_and_end(self):
        prompt = " ".join(f"w{i}" for i in range(100))
        self.score({"choice": "coding", "probabilities": probs(coding=1)}, prompt=prompt)
        # 12-token budget: first int(12 * 0.35) = 4 words, last 8 words.
        expected = "w0 w1 w2 w3 ... " + " ".join(f"w{i}" for i in range(92, 100))
        self.assertEqual([t for t in self.laya.seen if t != "Hi"], [expected])

    def test_output_is_accepted_by_router(self):
        # Awkward probabilities that would not sum to 1 if rounded.
        answer = {"choice": "coding", "probabilities": probs(
            coding=1 / 3, math_reasoning=1 / 7, creative_writing=1 / 11,
            general_knowledge_extraction=1 / 13)}
        model = Model("m", "example", 1, 1, 128000, {Topic.CODING: 0.5})
        self.assertIs(select_model(self.score(answer), [model]), model)


@unittest.skipUnless(importlib.util.find_spec("laya"), "laya is not installed")
class LayaModelSelectionTests(unittest.TestCase):
    def load(self, value):
        from unittest import mock

        from lonn import complexity

        env = {} if value is None else {"LONN_LAYA_MODEL": value}
        with mock.patch.dict("os.environ", env, clear=False), \
                mock.patch("dotenv.load_dotenv"), mock.patch("laya.Router") as router:
            if value is None:
                os.environ.pop("LONN_LAYA_MODEL", None)
            complexity._load_laya_router()
        return router.call_args.kwargs["models"]

    def test_default_uses_stock_checkpoints(self):
        self.assertIsNone(self.load(None))

    def test_env_replaces_english_checkpoint(self):
        self.assertEqual(self.load(" someone/laya-lonn "), {"english": "someone/laya-lonn"})


@unittest.skipUnless(importlib.util.find_spec("laya"), "laya is not installed")
class LiveLayaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = ComplexityScorer()

    def test_request_after_long_context_is_seen(self):
        notes = "Meeting notes: revenue rose, the warehouse is full, carriers raised prices. " * 60
        result = self.scorer.score_prompt(
            notes + "Task: turn these notes into a humorous short story told by the warehouse.")
        self.assertEqual(max(result.topic_scores, key=result.topic_scores.get),
                         Topic.CREATIVE_WRITING)

    def test_coding_prompt(self):
        result = self.scorer.score_prompt("Write a Python function that sorts a list")
        self.assertEqual(max(result.topic_scores, key=result.topic_scores.get), Topic.CODING)
        self.assertAlmostEqual(fsum(result.topic_scores.values()), 1.0, delta=1e-6)
        self.assertTrue(0 <= result.complexity <= 1)

    def test_hard_prompt_scores_above_easy_prompt(self):
        easy = self.scorer.score_prompt("What is the capital of France?")
        hard = self.scorer.score_prompt(
            "Prove that every planar graph is 5-colourable, then design and analyse an "
            "algorithm that finds such a colouring in linear time.")
        self.assertGreater(hard.complexity, easy.complexity)


if __name__ == "__main__":
    unittest.main()
