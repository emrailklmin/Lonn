"""Score a prompt's topic distribution and complexity with Laya; no routing.

Both outputs are produced in a single Laya inference:

- Topic classification:
    probability distribution across TopicCategory values.

- Complexity:
    Laya's probability that the task is hard, in [0, 1].

Prompts longer than Laya's window keep their start and end; the middle is
dropped, because requests often follow long pasted context.

The Laya router is loaded lazily and reused across requests. Set
LONN_LAYA_MODEL (environment or repository .env) to a Hugging Face repo id or
local directory to replace the English checkpoint with a fine-tuned one; see
legacy/scripts/finetune_laya.py.
"""

import json
import logging
import os
import warnings
from math import fsum, isfinite
from pathlib import Path

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

# A yes/no question rather than a 0-4 difficulty scale: Laya's expected level
# stayed between levels 1 and 2 for almost every prompt, while this wording's
# probability spreads out and ranks the pilot labels better (Spearman 0.37 vs
# 0.27; AUC 0.81 vs 0.72 for labels 3-5). Negative wordings ("can a small
# model do it?", "is it simple?") came out inverted, so keep it positive.
COMPLEXITY_CRITERIA = {
    "true": (
        "hard: needs expert knowledge, careful multi-step reasoning, "
        "or many constraints"
    ),
    "false": "easy: a simple, common request any assistant handles",
}


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
        "type": "noul",
        "instructions": (
            "Is this task hard for an AI assistant to do well?"
        ),
        "criteria": COMPLEXITY_CRITERIA,
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


# Share of the token budget kept from the start of an over-long prompt; the
# rest comes from its end. The margin covers the " ... " joint and tokenizer
# boundary effects when the two parts are re-encoded.
HEAD_SHARE = 0.35
HEAD_TAIL_MARGIN = 8


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
    """Return Laya's probability that the task is hard, clamped to [0, 1]."""

    probability = float(answer["noul"])

    if not isfinite(probability):
        raise ValueError(
            f"Laya returned a non-finite complexity probability: {probability!r}"
        )

    return min(
        max(probability, 0.0),
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

    LONN_LAYA_MODEL replaces only the English checkpoint; non-English
    prompts still use Laya's multilingual one.
    """

    # Must be set before importing Laya / transformers.
    # Avoids a known TensorFlow import hang.
    os.environ.setdefault("USE_TF", "0")

    from dotenv import load_dotenv
    from laya import Router

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    english = os.environ.get("LONN_LAYA_MODEL", "").strip()

    return Router(
        models={"english": english} if english else None,
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
        self._budgets: dict[tuple[str, tuple[str, ...]], int] = {}

    @property
    def laya(self):
        """Return the shared Laya router, loading it if necessary."""

        if self._laya is None:
            self._laya = _load_laya_router()

        return self._laya

    def _budget(self, model: str, questions: dict) -> int:
        """Prompt tokens that fit beside every question in one `model` window.

        Each question is its own row sharing the prompt, so the question with
        the longest instructions and options sets the budget.
        """

        key = (model, tuple(questions))

        if key not in self._budgets:
            max_len = self.laya.load(model).cfg.get("max_len", 512)
            question_tokens = 0

            for qid, question in questions.items():
                usage = self.laya.predict("Hi", {qid: question}, model=model)["usage"]
                question_tokens = max(
                    question_tokens,
                    usage["input_tokens"] - usage["state_tokens"],
                )

            self._budgets[key] = max_len - question_tokens - HEAD_TAIL_MARGIN

        return self._budgets[key]

    def _fit(self, prompt: str, model: str, questions: dict) -> str:
        """Shorten a prompt that would be truncated, keeping its start and end.

        Laya only reads the first window of a prompt, but requests often come
        after long pasted context, so the middle is dropped instead of the end.
        """

        tok = self.laya.load(model).tok
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        budget = self._budget(model, questions)

        if len(ids) <= budget:
            return prompt

        head = int(budget * HEAD_SHARE)
        tail = budget - head

        logger.info(
            "prompt of %d tokens shortened to its first %d and last %d tokens",
            len(ids), head, tail,
        )

        decode = lambda part: tok.decode(part, clean_up_tokenization_spaces=False)

        return f"{decode(ids[:head])} ... {decode(ids[-tail:])}"

    def _predict(self, prompt: str, questions: dict) -> dict:
        """Route the prompt, fit it to the chosen checkpoint, and ask Laya."""

        model = self.laya.route(prompt, questions).model

        return self.laya.predict(
            self._fit(prompt, model, questions),
            questions,
            model=model,
        )

    def score_prompt(self, prompt: str) -> ComplexityResult:
        """Score topic distribution and complexity in one forward pass."""

        result = self._predict(
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

        result = self._predict(
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

        result = self._predict(
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