import sys
import unittest
from itertools import permutations
from pathlib import Path

# Allow direct execution from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lonn.router import Router, rank_models, select_model, weighted_score
from lonn.schemas import ComplexityResult, Model, TopicCategory as Topic


def model(name, aptitude, price=1.0):
    return Model(name, "example", price, price, 128000, {Topic.CODING: aptitude})


class RouterTests(unittest.TestCase):
    def test_alpha_changes_selection(self):
        cheap, strong = model("cheap", 0.4), model("strong", 0.9, 3)
        for alpha, expected in [(0, cheap), (0.5, cheap), (0.8, strong), (1, strong)]:
            with self.subTest(alpha=alpha):
                signal = ComplexityResult(0.7, {Topic.CODING: 1.0})
                self.assertIs(Router().select_model(signal, [strong, cheap], alpha), expected)

    def test_rank_models_returns_top_five_in_selection_order(self):
        candidates = [model(f"model-{score}", score / 10, 10 - score) for score in range(1, 8)]
        signal = ComplexityResult(1, {Topic.CODING: 1})
        ranked = rank_models(signal, candidates, alpha=1)
        self.assertEqual([item.id for item in ranked],
                         ["model-7", "model-6", "model-5", "model-4", "model-3"])
        self.assertEqual(Router.rank_models(signal, candidates, alpha=1, limit=2), ranked[:2])

    def test_rank_models_rejects_invalid_limit(self):
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                rank_models(ComplexityResult(.5, {Topic.CODING: 1}),
                            [model("valid", .5)], limit=limit)

    def test_uses_full_topic_distribution(self):
        coding = model("coding", 0.9)
        mixed = model("mixed", 0.6, 2)
        mixed.topic_aptitudes[Topic.MATH_REASONING] = 0.8
        signal = ComplexityResult(0.6, {Topic.CODING: 0.5, Topic.MATH_REASONING: 0.5})
        self.assertIs(select_model(signal, [coding, mixed], alpha=1), mixed)

    def test_ties_are_independent_of_pool_order(self):
        candidates = [model("b", 0.8), model("a", 0.8), model("weak", 0.3)]
        for complexity in (0.2, 1.0):
            for pool in permutations(candidates):
                signal = ComplexityResult(complexity, {Topic.CODING: 1})
                self.assertIs(select_model(signal, list(pool)), candidates[1])

    def test_quality_ties_prefer_lower_price(self):
        cheap, expensive = model("cheap", 0.8), model("expensive", 0.8, 2)
        self.assertIs(select_model(ComplexityResult(1, {Topic.CODING: 1}),
                                   [expensive, cheap], alpha=1), cheap)

    def test_weighted_score(self):
        self.assertAlmostEqual(weighted_score(0.8, 0.2, 0.75), 0.55)
        self.assertEqual(weighted_score(0.8, 0.2, 0), -0.2)
        self.assertEqual(weighted_score(0.8, 0.2, 1), 0.8)

    def test_free_pool(self):
        weak, strong = model("weak", 0.3, 0), model("strong", 0.9, 0)
        signal = ComplexityResult(0.5, {Topic.CODING: 1})
        self.assertIs(select_model(signal, [weak, strong]), strong)

    def test_cost_normalization_is_invariant_to_price_units(self):
        cheap, strong = model("cheap", 0.4), model("strong", 0.9, 3)
        signal = ComplexityResult(0.5, {Topic.CODING: 1})
        before = select_model(signal, [cheap, strong], alpha=0.8)
        for candidate in (cheap, strong):
            candidate.cost_per_m_input *= 1000
            candidate.cost_per_m_output *= 1000
        self.assertIs(select_model(signal, [cheap, strong], alpha=0.8), before)

    def test_invalid_weights(self):
        for alpha in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                select_model(ComplexityResult(0.5, {Topic.CODING: 1}),
                             [model("valid", 0.5)], alpha=alpha)

    def test_invalid_signal(self):
        signals = [
            ComplexityResult(float("nan"), {Topic.CODING: 1}),
            ComplexityResult(1.1, {Topic.CODING: 1}),
            ComplexityResult(0.5, {}),
            ComplexityResult(0.5, {Topic.CODING: 0.3}),
            ComplexityResult(0.5, {Topic.CODING: -1, Topic.MATH_REASONING: 2}),
            ComplexityResult(0.5, {"unknown": 1}),
        ]
        for signal in signals:
            with self.subTest(signal=signal), self.assertRaises(ValueError):
                select_model(signal, [model("valid", 0.5)])

    def test_invalid_pool(self):
        duplicate = model("same", 0.5)
        pools = [[], [duplicate, duplicate], [model("invalid", float("nan"))],
                 [model("invalid", 1.1)], [model("invalid", 0.5, -1)],
                 [model("invalid", 0.5, float("inf"))]]
        for pool in pools:
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                select_model(ComplexityResult(0.5, {Topic.CODING: 1}), pool)


if __name__ == "__main__":
    unittest.main()
