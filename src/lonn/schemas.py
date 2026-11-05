"""Shared data contracts for prompt analysis, routing, and execution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

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
class TaskAptitude:
    """A task score and its provenance, as used by the task router."""

    score: float
    source: Literal['measured', 'supplied', 'derived']
    sample_size: int | None
    cost_usd: float
    cost_source: Literal['task_benchmark', 'global_benchmark']


@dataclass
class Model:
    id: str
    provider: str
    cost_per_m_input: float
    cost_per_m_output: float
    context_window: int
    # Legacy four-topic scores are kept for the original router and SDK clients.
    topic_aptitudes: dict[TopicCategory, float]
    # Populated from the effective routing table; never a second definition.
    task_aptitudes: dict[str, TaskAptitude] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Reported token counts and estimated costs using configured prices."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


@dataclass
class ModelResponse:
    """Text output and execution metadata returned to the application."""

    content: str
    model: Model
    usage: TokenUsage
    latency_ms: float
    finish_reason: str | None = None
