"""Score a prompt's topic distribution and complexity with Laya; no routing.

Two Laya rounds per prompt:
  Round 1: a "choice" question giving a probability per TopicCategory.
  Round 2: a "score" question giving an expected difficulty level, mapped to [0, 1].
"""

import json
import logging
import os
import warnings
from math import fsum, isfinite

from .schemas import ComplexityResult, TopicCategory

logger = logging.getLogger(__name__)

TOPIC_CRITERIA = {
    TopicCategory.CODING: "writing, debugging, explaining or reviewing source code, scripts, SQL, regex, APIs, or software engineering questions",
    TopicCategory.MATH_REASONING: "mathematics, logic puzzles, quantitative or multi-step analytical reasoning, proofs, calculations, statistics",
    TopicCategory.CREATIVE_WRITING: "stories, poetry, marketing copy, tone-sensitive emails, rewriting or editing prose where style and nuance matter",
    TopicCategory.GENERAL_KNOWLEDGE_EXTRACTION: "factual questions, summarizing, explaining concepts, extracting or looking up information from text or general knowledge",
}

COMPLEXITY_LEVELS = [
    "trivial: a single simple step, short answer",
    "simple: straightforward, one clear task",
    "moderate: several steps or some domain knowledge needed",
    "hard: multi-step, needs careful reasoning or expertise",
    "very hard: expert-level, long or highly constrained multi-part task",
]

CATEGORY_QUESTION = {
    "category": {
        "type": "choice",
        "instructions": "Which kind of task is the user asking an AI assistant to do? Pick the single best fit.",
        "criteria": {topic.value: text for topic, text in TOPIC_CRITERIA.items()},
    }
}

COMPLEXITY_QUESTION = {
    "complexity": {
        "type": "score",
        "instructions": "How difficult is this task for an AI assistant to do well?",
        "criteria": COMPLEXITY_LEVELS,
    }
}

NAMES = [topic.value for topic in TopicCategory]


def _find_distribution(answer: dict) -> dict[str, float] | None:
    """Find the per-category probabilities inside Laya's 'choice' answer."""
    for value in answer.values():
        if isinstance(value, dict) and set(NAMES) <= set(value):
            return {c: float(value[c]) for c in NAMES}
        if isinstance(value, (list, tuple)) and len(value) == len(NAMES):
            if all(isinstance(v, (int, float)) for v in value):
                return dict(zip(NAMES, map(float, value)))
            if all(isinstance(v, dict) for v in value):
                out = {}
                for v in value:
                    label = v.get("label") or v.get("option") or v.get("name")
                    prob = v.get("probability", v.get("prob", v.get("p")))
                    if label in NAMES and isinstance(prob, (int, float)):
                        out[label] = float(prob)
                if set(out) == set(NAMES):
                    return out
    return None


def _load_laya_router():
    os.environ.setdefault("USE_TF", "0")  # Avoids a known TensorFlow import hang.
    from laya import Router
    
    return Router(preload=False, default="english", max_loaded=1) #Router(preload=True)


class ComplexityScorer:
    """Produce a ComplexityResult for a prompt.

    Laya is loaded lazily on first use, so importing this module does not
    require Laya. Pass `laya_router` to reuse an instance or inject a fake.
    """

    def __init__(self, laya_router=None):
        self._laya = laya_router

    @property
    def laya(self):
        if self._laya is None:
            self._laya = _load_laya_router()
        return self._laya

    def topic_scores(self, prompt: str) -> dict[TopicCategory, float]:
        res = self.laya.predict(prompt, CATEGORY_QUESTION)
        logger.debug("category raw: %s", json.dumps(res, indent=2, default=str))
        answer = res["answers"]["category"]

        dist = _find_distribution(answer)
        if dist is not None and not all(isfinite(p) and p >= 0 for p in dist.values()):
            dist = None
        if dist is None or fsum(dist.values()) == 0:
            warnings.warn("Laya returned no usable category distribution; "
                          "using its top choice only.")
            return {t: 1.0 if t.value == answer["choice"] else 0.0 for t in TopicCategory}

        total = fsum(dist.values())
        return {TopicCategory(name): p / total for name, p in dist.items()}

    def complexity(self, prompt: str) -> float:
        res = self.laya.predict(prompt, COMPLEXITY_QUESTION)
        logger.debug("complexity raw: %s", json.dumps(res, indent=2, default=str))
        level = float(res["answers"]["complexity"]["score"])  # Expected level, 0..4.
        if not isfinite(level):
            raise ValueError(f"Laya returned a non-finite complexity score: {level!r}")
        return min(max(level / (len(COMPLEXITY_LEVELS) - 1), 0.0), 1.0)

    def score_prompt(self, prompt: str) -> ComplexityResult:
        return ComplexityResult(self.complexity(prompt), self.topic_scores(prompt))


def score_prompt(prompt: str) -> ComplexityResult:
    """Convenience wrapper using a shared, lazily loaded scorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = ComplexityScorer()
    return _default_scorer.score_prompt(prompt)


_default_scorer: ComplexityScorer | None = None
