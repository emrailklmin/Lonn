"""Score a prompt's topic distribution and complexity with Laya; no routing.

Both outputs are produced in a single Laya inference:

- Topic classification:
    probability distribution across TopicCategory values.

- Complexity:
    expected difficulty level mapped to [0, 1].

The Laya router is loaded lazily and reused across requests.
"""

import json
import logging
import os
import warnings
from math import fsum, isfinite

from .schemas import ComplexityResult, TopicCategory


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topic definitions
# ---------------------------------------------------------------------------

TOPIC_CRITERIA = {
    TopicCategory.CODING: (
        "writing, debugging, explaining or reviewing source code, scripts, "
        "SQL, regex, APIs, or software engineering questions"
    ),
    TopicCategory.MATH_REASONING: (
        "mathematics, logic puzzles, quantitative or multi-step analytical "
        "reasoning, proofs, calculations, statistics"
    ),
    TopicCategory.CREATIVE_WRITING: (
        "stories, poetry, marketing copy, tone-sensitive emails, rewriting "
        "or editing prose where style and nuance matter"
    ),
    TopicCategory.GENERAL_KNOWLEDGE_EXTRACTION: (
        "factual questions, summarizing, explaining concepts, extracting "
        "or looking up information from text or general knowledge"
    ),
}


# ---------------------------------------------------------------------------
# Complexity definitions
# ---------------------------------------------------------------------------

COMPLEXITY_LEVELS = [
    "trivial: a single simple step, short answer",
    "simple: straightforward, one clear task",
    "moderate: several steps or some domain knowledge needed",
    "hard: multi-step, needs careful reasoning or expertise",
    "very hard: expert-level, long or highly constrained multi-part task",
]


# ---------------------------------------------------------------------------
# Laya questions
# ---------------------------------------------------------------------------

CATEGORY_QUESTION = {
    "category": {
        "type": "choice",
        "instructions": (
            "Which kind of task is the user asking an AI assistant to do? "
            "Pick the single best fit."
        ),
        "criteria": {
            topic.value: description
            for topic, description in TOPIC_CRITERIA.items()
        },
    }
}


COMPLEXITY_QUESTION = {
    "complexity": {
        "type": "score",
        "instructions": (
            "How difficult is this task for an AI assistant to do well?"
        ),
        "criteria": COMPLEXITY_LEVELS,
    }
}


# IMPORTANT:
# Laya can answer several questions in a single forward pass.
# Sending both questions together avoids running the model twice.
QUESTIONS = {
    **CATEGORY_QUESTION,
    **COMPLEXITY_QUESTION,
}


TOPIC_NAMES = [topic.value for topic in TopicCategory]


# ---------------------------------------------------------------------------
# Laya output parsing
# ---------------------------------------------------------------------------

def _find_distribution(answer: dict) -> dict[str, float] | None:
    """Find per-category probabilities inside Laya's choice answer."""

    for value in answer.values():

        # Format:
        # {
        #     "coding": 0.7,
        #     "math_reasoning": 0.1,
        #     ...
        # }
        if isinstance(value, dict) and set(TOPIC_NAMES) <= set(value):
            return {
                name: float(value[name])
                for name in TOPIC_NAMES
            }

        # Format:
        # [0.7, 0.1, 0.1, 0.1]
        if (
            isinstance(value, (list, tuple))
            and len(value) == len(TOPIC_NAMES)
        ):
            if all(isinstance(v, (int, float)) for v in value):
                return dict(
                    zip(
                        TOPIC_NAMES,
                        map(float, value),
                    )
                )

            # Format:
            # [
            #     {"label": "coding", "probability": 0.7},
            #     ...
            # ]
            if all(isinstance(v, dict) for v in value):
                distribution = {}

                for item in value:
                    label = (
                        item.get("label")
                        or item.get("option")
                        or item.get("name")
                    )

                    probability = item.get(
                        "probability",
                        item.get(
                            "prob",
                            item.get("p"),
                        ),
                    )

                    if (
                        label in TOPIC_NAMES
                        and isinstance(probability, (int, float))
                    ):
                        distribution[label] = float(probability)

                if set(distribution) == set(TOPIC_NAMES):
                    return distribution

    return None


def _parse_topic_scores(
    answer: dict,
) -> dict[TopicCategory, float]:
    """Convert Laya's category answer into normalized topic probabilities."""

    distribution = _find_distribution(answer)

    # Reject invalid distributions.
    if distribution is not None:
        valid = all(
            isfinite(probability) and probability >= 0
            for probability in distribution.values()
        )

        if not valid:
            distribution = None

    # Fall back to Laya's top choice if probabilities are unavailable.
    if distribution is None or fsum(distribution.values()) == 0:

        warnings.warn(
            "Laya returned no usable category distribution; "
            "using its top choice only."
        )

        choice = answer["choice"]

        return {
            topic: 1.0 if topic.value == choice else 0.0
            for topic in TopicCategory
        }

    # Normalize distribution to sum to 1.
    total = fsum(distribution.values())

    return {
        TopicCategory(name): probability / total
        for name, probability in distribution.items()
    }


def _parse_complexity(answer: dict) -> float:
    """Convert Laya's expected complexity level to [0, 1]."""

    level = float(answer["score"])

    if not isfinite(level):
        raise ValueError(
            f"Laya returned a non-finite complexity score: {level!r}"
        )

    max_level = len(COMPLEXITY_LEVELS) - 1

    normalized = level / max_level

    return min(
        max(normalized, 0.0),
        1.0,
    )


# ---------------------------------------------------------------------------
# Laya router
# ---------------------------------------------------------------------------

def _load_laya_router():
    """Create the shared Laya router.

    The router itself handles device selection automatically.

    max_loaded=2 allows both the English and multilingual checkpoints to
    remain resident instead of constantly evicting/reloading each other.
    """

    # Must be set before importing Laya / transformers.
    # Avoids a known TensorFlow import hang.
    os.environ.setdefault("USE_TF", "0")

    from laya import Router

    return Router(
        preload=False,
        default="english",
        max_loaded=2,
    )


# ---------------------------------------------------------------------------
# Complexity scorer
# ---------------------------------------------------------------------------

class ComplexityScorer:
    """Score prompt complexity and topic using one Laya inference.

    Laya is loaded lazily on first use.

    Pass ``laya_router`` manually to:
    - reuse an existing router
    - inject a fake router for tests
    - control preloading/device behaviour externally
    """

    def __init__(self, laya_router=None):
        self._laya = laya_router

    @property
    def laya(self):
        """Return the shared Laya router, loading it if necessary."""

        if self._laya is None:
            self._laya = _load_laya_router()

        return self._laya

    def score_prompt(self, prompt: str) -> ComplexityResult:
        """Score topic distribution and complexity in one forward pass."""

        result = self.laya.predict(
            prompt,
            QUESTIONS,
        )

        logger.debug(
            "Laya raw result: %s",
            json.dumps(
                result,
                indent=2,
                default=str,
            ),
        )

        answers = result["answers"]

        topic_scores = _parse_topic_scores(
            answers["category"]
        )

        complexity = _parse_complexity(
            answers["complexity"]
        )

        return ComplexityResult(
            complexity,
            topic_scores,
        )

    def topic_scores(
        self,
        prompt: str,
    ) -> dict[TopicCategory, float]:
        """Score only topic distribution.

        Prefer ``score_prompt`` when both topic and complexity are needed,
        because calling this and ``complexity`` separately requires two
        Laya inference calls.
        """

        result = self.laya.predict(
            prompt,
            CATEGORY_QUESTION,
        )

        logger.debug(
            "Laya category raw result: %s",
            json.dumps(
                result,
                indent=2,
                default=str,
            ),
        )

        return _parse_topic_scores(
            result["answers"]["category"]
        )

    def complexity(
        self,
        prompt: str,
    ) -> float:
        """Score only complexity.

        Prefer ``score_prompt`` when both topic and complexity are needed.
        """

        result = self.laya.predict(
            prompt,
            COMPLEXITY_QUESTION,
        )

        logger.debug(
            "Laya complexity raw result: %s",
            json.dumps(
                result,
                indent=2,
                default=str,
            ),
        )

        return _parse_complexity(
            result["answers"]["complexity"]
        )


# ---------------------------------------------------------------------------
# Shared convenience scorer
# ---------------------------------------------------------------------------

_default_scorer: ComplexityScorer | None = None


def score_prompt(prompt: str) -> ComplexityResult:
    """Score a prompt using a shared, lazily loaded scorer."""

    global _default_scorer

    if _default_scorer is None:
        _default_scorer = ComplexityScorer()

    return _default_scorer.score_prompt(prompt)