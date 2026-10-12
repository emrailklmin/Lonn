"""Score a prompt's topic distribution and complexity; no routing.

Both outputs come from one local inference of NVIDIA's prompt task and
complexity classifier (see `nvidia_classifier.py`):

- Topic classification:
    NVIDIA's task-type distribution mapped onto TopicCategory values.

- Complexity:
    weighted sum of NVIDIA's reasoning, constraint, domain-knowledge,
    context and few-shot dimensions, in [0, 1].

The classifier is loaded lazily and reused across requests.
"""

import logging
from math import fsum, isfinite

from .nvidia_classifier import OVERALL_WEIGHTS, NvidiaPromptClassifier, PromptProfile
from .schemas import ComplexityResult, TopicCategory


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topic definitions
# ---------------------------------------------------------------------------

# Topics follow NVIDIA's view of the task (docs/labeling_guide.md):
#
# - coding: the user wants code written, completed or fixed.
# - creative_writing: text generation, rewriting, brainstorming, conversation.
# - math_reasoning: a question whose answer must be worked out step by step.
# - general_knowledge_extraction: every other question or text-processing task
#   (open or closed QA, summarizing, extraction, classification), including
#   explaining how code or a library works.
TASK_TOPICS = {
    "Code Generation": TopicCategory.CODING,
    "Text Generation": TopicCategory.CREATIVE_WRITING,
    "Rewrite": TopicCategory.CREATIVE_WRITING,
    "Brainstorming": TopicCategory.CREATIVE_WRITING,
    "Chatbot": TopicCategory.CREATIVE_WRITING,
}

# NVIDIA has no maths task type, but its reasoning dimension separates the
# pilot maths prompts from all others almost perfectly (AUC 0.96). Probability
# on any other task type is therefore split by it: reasoning goes to
# math_reasoning, the rest to general_knowledge_extraction.


# ---------------------------------------------------------------------------
# Complexity definitions
# ---------------------------------------------------------------------------

# On the 228 pilot prompts this ranks the labelled difficulty better than Laya
# did (Spearman 0.53 vs 0.34; AUC 0.86 vs 0.79 for labels 3-5).
#
# NVIDIA's own overall score gives creativity 35% weight, but creativity does
# not track difficulty (AUC 0.51), so a short poem outranked a proof. These are
# NVIDIA's remaining weights, rescaled to sum to 1.
_NVIDIA_WEIGHTS = {
    name: weight
    for name, weight in OVERALL_WEIGHTS.items()
    if name != "creativity"
}

COMPLEXITY_WEIGHTS = {
    name: weight / fsum(_NVIDIA_WEIGHTS.values())
    for name, weight in _NVIDIA_WEIGHTS.items()
}


# ---------------------------------------------------------------------------
# Classifier output parsing
# ---------------------------------------------------------------------------

def _topic_scores(profile: PromptProfile) -> dict[TopicCategory, float]:
    """Map NVIDIA's task-type distribution onto normalized topic probabilities."""

    reasoning = min(max(float(profile.reasoning), 0.0), 1.0)
    scores = dict.fromkeys(TopicCategory, 0.0)

    for task, probability in profile.task_probabilities.items():
        probability = float(probability)

        if not isfinite(probability) or probability < 0:
            raise ValueError(
                f"NVIDIA classifier returned an invalid probability for {task!r}: {probability!r}"
            )

        topic = TASK_TOPICS.get(task)

        if topic is not None:
            scores[topic] += probability
        else:
            scores[TopicCategory.MATH_REASONING] += probability * reasoning
            scores[TopicCategory.GENERAL_KNOWLEDGE_EXTRACTION] += probability * (1 - reasoning)

    total = fsum(scores.values())

    if total <= 0:
        raise ValueError("NVIDIA classifier returned no task-type probability")

    return {
        topic: score / total
        for topic, score in scores.items()
    }


def _complexity(profile: PromptProfile) -> float:
    """Combine NVIDIA's complexity dimensions into one score in [0, 1]."""

    score = fsum(
        weight * float(getattr(profile, name))
        for name, weight in COMPLEXITY_WEIGHTS.items()
    )

    if not isfinite(score):
        raise ValueError(
            f"NVIDIA classifier returned a non-finite complexity: {score!r}"
        )

    # Few-shot count runs 0-5, so a prompt full of examples could exceed 1.
    return min(
        max(score, 0.0),
        1.0,
    )


# ---------------------------------------------------------------------------
# Complexity scorer
# ---------------------------------------------------------------------------

class ComplexityScorer:
    """Score prompt topic and complexity with one classifier inference.

    The classifier is loaded lazily on first use.

    Pass ``classifier`` manually to:
    - reuse an existing instance
    - inject a fake classifier for tests
    - control the device externally
    """

    def __init__(self, classifier=None):
        self._classifier = classifier

    @property
    def classifier(self):
        """Return the shared NVIDIA classifier, creating it if necessary."""

        if self._classifier is None:
            self._classifier = NvidiaPromptClassifier()

        return self._classifier

    def profile(self, prompt: str) -> PromptProfile:
        """Return the classifier's full view of the prompt."""

        profile = self.classifier.classify(prompt)

        logger.debug(
            "NVIDIA classifier profile: %s",
            profile,
        )

        return profile

    def score_prompt(self, prompt: str) -> ComplexityResult:
        """Score topic distribution and complexity in one inference."""

        profile = self.profile(prompt)

        return ComplexityResult(
            _complexity(profile),
            _topic_scores(profile),
        )

    def topic_scores(
        self,
        prompt: str,
    ) -> dict[TopicCategory, float]:
        """Score only the topic distribution.

        Prefer ``score_prompt`` when both are needed; each call runs the model.
        """

        return _topic_scores(self.profile(prompt))

    def complexity(
        self,
        prompt: str,
    ) -> float:
        """Score only complexity.

        Prefer ``score_prompt`` when both are needed; each call runs the model.
        """

        return _complexity(self.profile(prompt))


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
