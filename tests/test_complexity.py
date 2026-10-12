import importlib.util
import sys
import unittest
from math import fsum
from pathlib import Path

# Allow direct execution from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lonn.complexity import COMPLEXITY_WEIGHTS, ComplexityScorer
from lonn.nvidia_classifier import REPO, PromptProfile
from lonn.router import select_model
from lonn.schemas import ComplexityResult, Model, TopicCategory as Topic


def profile(tasks, creativity=0.0, reasoning=0.0, contextual_knowledge=0.0,
            few_shots=0.0, domain_knowledge=0.0, constraints=0.0):
    """A classifier result with the given task-type probabilities and dimensions."""
    return PromptProfile(
        task_type=max(tasks, key=tasks.get),
        task_probabilities=tasks,
        creativity=creativity,
        reasoning=reasoning,
        contextual_knowledge=contextual_knowledge,
        few_shots=few_shots,
        domain_knowledge=domain_knowledge,
        constraints=constraints,
        complexity=0.0,  # NVIDIA's own overall score; the scorer does not use it.
    )


class FakeClassifier:
    """Stands in for NvidiaPromptClassifier, returning one canned profile."""

    def __init__(self, result):
        self.result = result
        self.seen = []

    def classify(self, prompt):
        self.seen.append(prompt)
        return self.result


class ComplexityScorerTests(unittest.TestCase):
    def score(self, result):
        self.classifier = FakeClassifier(result)
        return ComplexityScorer(self.classifier).score_prompt("prompt")

    def test_returns_schema_result(self):
        result = self.score(profile({"Code Generation": 0.9, "Open QA": 0.1}, reasoning=0.5))
        self.assertIsInstance(result, ComplexityResult)
        self.assertEqual(set(result.topic_scores), set(Topic))
        self.assertTrue(all(isinstance(t, Topic) for t in result.topic_scores))
        self.assertAlmostEqual(fsum(result.topic_scores.values()), 1.0, delta=1e-9)
        self.assertTrue(0 <= result.complexity <= 1)

    def test_task_types_map_to_topics(self):
        cases = [
            ("Code Generation", Topic.CODING),
            ("Text Generation", Topic.CREATIVE_WRITING),
            ("Rewrite", Topic.CREATIVE_WRITING),
            ("Brainstorming", Topic.CREATIVE_WRITING),
            ("Chatbot", Topic.CREATIVE_WRITING),
            ("Summarization", Topic.GENERAL_KNOWLEDGE_EXTRACTION),
        ]
        for task, topic in cases:
            with self.subTest(task=task):
                self.assertEqual(self.score(profile({task: 1.0})).topic_scores[topic], 1.0)

    def test_reasoning_splits_questions_into_math_and_knowledge(self):
        scores = self.score(profile({"Open QA": 0.8, "Code Generation": 0.2}, reasoning=0.75)).topic_scores
        self.assertAlmostEqual(scores[Topic.MATH_REASONING], 0.6)
        self.assertAlmostEqual(scores[Topic.GENERAL_KNOWLEDGE_EXTRACTION], 0.2)
        self.assertAlmostEqual(scores[Topic.CODING], 0.2)

    def test_distribution_is_normalized(self):
        scores = self.score(profile({"Code Generation": 3, "Rewrite": 1})).topic_scores
        self.assertAlmostEqual(scores[Topic.CODING], 0.75)
        self.assertAlmostEqual(fsum(scores.values()), 1.0, delta=1e-9)

    def test_invalid_task_probabilities_raise(self):
        for tasks in ({"Open QA": float("nan")}, {"Open QA": -0.1}, {"Open QA": 0.0}):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                self.score(profile(tasks))

    def test_complexity_ignores_creativity(self):
        plain = self.score(profile({"Open QA": 1.0}, reasoning=0.4)).complexity
        creative = self.score(profile({"Open QA": 1.0}, reasoning=0.4, creativity=1.0)).complexity
        self.assertEqual(plain, creative)
        self.assertAlmostEqual(plain, 0.4 * COMPLEXITY_WEIGHTS["reasoning"])

    def test_complexity_weights_sum_to_one_and_clamp(self):
        self.assertAlmostEqual(fsum(COMPLEXITY_WEIGHTS.values()), 1.0)
        everything = dict(reasoning=1, contextual_knowledge=1, domain_knowledge=1, constraints=1)
        self.assertEqual(self.score(profile({"Open QA": 1.0}, **everything, few_shots=5)).complexity, 1.0)
        with self.assertRaises(ValueError):
            self.score(profile({"Open QA": 1.0}, constraints=float("nan")))

    def test_one_classifier_call_per_prompt(self):
        self.score(profile({"Open QA": 1.0}))
        self.assertEqual(self.classifier.seen, ["prompt"])

    def test_output_is_accepted_by_router(self):
        # Awkward probabilities that would not sum to 1 if rounded.
        result = self.score(profile({"Code Generation": 1 / 3, "Open QA": 1 / 7, "Rewrite": 1 / 11}, reasoning=1 / 13))
        model = Model("m", "example", 1, 1, 128000, {Topic.CODING: 0.5})
        self.assertIs(select_model(result, [model]), model)


def _model_is_cached():
    if not importlib.util.find_spec("transformers"):
        return False
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(REPO, "model.safetensors"), str)


@unittest.skipUnless(_model_is_cached(), "NVIDIA classifier is not downloaded")
class LiveClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = ComplexityScorer()

    def top_topic(self, prompt):
        scores = self.scorer.score_prompt(prompt).topic_scores
        return max(scores, key=scores.get)

    def test_topics(self):
        self.assertEqual(self.top_topic("Write a Python function that sorts a list"), Topic.CODING)
        self.assertEqual(self.top_topic("Write a short poem about autumn leaves."), Topic.CREATIVE_WRITING)
        self.assertEqual(self.top_topic("What is the capital of France?"), Topic.GENERAL_KNOWLEDGE_EXTRACTION)

    def test_request_after_long_context_is_seen(self):
        notes = "Meeting notes: revenue rose, the warehouse is full, carriers raised prices. " * 60
        self.assertEqual(
            self.top_topic(notes + "Task: turn these notes into a humorous short story told by the warehouse."),
            Topic.CREATIVE_WRITING)

    def test_hard_prompt_scores_above_easy_prompt(self):
        easy = self.scorer.score_prompt("What is the capital of France?")
        hard = self.scorer.score_prompt(
            "Prove that every planar graph is 5-colourable, then design and analyse an "
            "algorithm that finds such a colouring in linear time.")
        self.assertGreater(hard.complexity, easy.complexity)


if __name__ == "__main__":
    unittest.main()
