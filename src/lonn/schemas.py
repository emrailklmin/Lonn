"""Shared data contracts for prompt analysis, routing, and execution."""

from dataclasses import dataclass
from enum import Enum


class TopicCategory(str, Enum):
    CODING = "coding"
    MATH = "math"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    GENERAL_KNOWLEDGE_EXTRACTION = "general_knowledge_extraction"


@dataclass
class ComplexityResult:
    complexity: float
    topic_scores: dict[TopicCategory, float]


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
