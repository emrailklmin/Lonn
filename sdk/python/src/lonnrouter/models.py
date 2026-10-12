"""Typed representations of the Lonn HTTP response contracts."""

from typing import Literal

from pydantic import BaseModel


class Signal(BaseModel):
    complexity: float
    topic_scores: dict[str, float]


class Model(BaseModel):
    id: str
    provider: str
    cost_per_m_input: float
    cost_per_m_output: float
    context_window: int
    topic_aptitudes: dict[str, float]


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


class RouteResult(BaseModel):
    model: Model
    signal: Signal
    alpha: float
    analysis_source: Literal["laya", "provided"]
    routing_latency_ms: float


class GenerateResult(RouteResult):
    response: ModelResponse
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
