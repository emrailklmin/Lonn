"""Shared data contracts for prompt analysis and model routing."""

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
