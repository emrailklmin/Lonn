"""Shared data contracts for prompt analysis, routing, and execution."""

from dataclasses import dataclass
from enum import Enum

# Supported LiteLLM providers and the environment variable holding each key.
# A model ID starts with its provider: "<provider>/<publisher>/<model>".
PROVIDER_API_KEYS = {
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class TopicCategory(str, Enum):
    CODING = "coding"
    MATH_REASONING = "math_reasoning"
    CREATIVE_WRITING = "creative_writing"
    GENERAL_KNOWLEDGE_EXTRACTION = "general_knowledge_extraction"


@dataclass
class ComplexityResult:
    complexity: float
    topic_scores: dict[TopicCategory, float]


@dataclass
class TaskSignal:
    """Classifier output for the task-type router (issue #19).

    ``task_scores`` is a probability distribution over task-type IDs from
    ``config/task_types.yaml``. ``complexity`` is in [0, 1], where 0, 0.5 and
    1 mean the easy, medium and hard end of the task type; ``None`` means
    unknown, so the router uses the task type's average.
    """

    task_scores: dict[str, float]
    complexity: float | None = None


@dataclass
class Model:
    id: str
    provider: str
    cost_per_m_input: float
    cost_per_m_output: float
    context_window: int
    topic_aptitudes: dict[TopicCategory, float]


@dataclass
class TokenUsage:
    """Reported token counts; total_cost uses provider billing when available.

    input_cost/output_cost remain configured-price estimates; cost_basis describes
    total_cost, which may differ from their sum due to caching or provider fees."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    cost_basis: str = "estimate"
    service_fee: float = 0.0
    provider_cost: float | None = None


@dataclass
class ModelResponse:
    """Text output and execution metadata returned to the application."""

    content: str
    model: Model
    usage: TokenUsage
    latency_ms: float
    finish_reason: str | None = None
