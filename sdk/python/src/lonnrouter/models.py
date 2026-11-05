"""Typed representations of the Lonn HTTP response contracts."""

from typing import Literal

from pydantic import BaseModel, Field


class Signal(BaseModel):
    """Task-type probabilities (summing to 1) and an optional complexity in [0, 1]."""

    task_scores: dict[str, float]
    complexity: float | None = None


class CandidateEstimate(BaseModel):
    """One ranked model's quality and cost estimates, using the effective routing table."""

    model_id: str
    expected_accuracy: float
    expected_cost_usd: float
    score: float


class TaskAptitude(BaseModel):
    """A model's task score, its provenance, and the cost used when routing."""

    score: float
    source: Literal['measured', 'supplied', 'derived']
    sample_size: int | None
    cost_usd: float
    cost_source: Literal['task_benchmark', 'global_benchmark']


class Model(BaseModel):
    id: str
    provider: str
    cost_per_m_input: float
    cost_per_m_output: float
    context_window: int
    # The older four-topic scores are retained for backwards compatibility.
    topic_aptitudes: dict[str, float]
    task_aptitudes: dict[str, TaskAptitude] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


class ModelResponse(BaseModel):
    content: str
    model: Model
    usage: TokenUsage
    latency_ms: float
    finish_reason: str | None = None


class Attempt(BaseModel):
    """One model call the server made for a generate request."""

    model_id: str
    check: str  # "ok", "truncated", "empty", "failed_examples", "not_checked" or "error"
    cost_usd: float
    latency_ms: float


class RouteResult(BaseModel):
    models: list[Model]
    estimates: list[CandidateEstimate]
    signal: Signal
    lam: float
    analysis_source: Literal["classifier", "provided"]
    routing_latency_ms: float

    @property
    def model(self) -> Model:
        """The highest-ranked model, retained for simple routing callers."""
        return self.models[0]


class GenerateResult(BaseModel):
    model: Model
    estimates: list[CandidateEstimate]
    signal: Signal
    lam: float
    analysis_source: Literal["classifier", "provided"]
    routing_latency_ms: float
    response: ModelResponse
    attempts: list[Attempt]
    escalated: bool
    total_cost_usd: float
    total_latency_ms: float

    @property
    def content(self) -> str:
        return self.response.content

    @property
    def usage(self) -> TokenUsage:
        return self.response.usage

    @property
    def finish_reason(self) -> str | None:
        return self.response.finish_reason


class Health(BaseModel):
    status: str
