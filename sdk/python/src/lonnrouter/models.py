"""Typed representations of the Lonn HTTP response contracts."""

from typing import Literal

from pydantic import BaseModel


class Signal(BaseModel):
    """Task-type probabilities (summing to 1) and an optional complexity in [0, 1]."""

    task_scores: dict[str, float]
    complexity: float | None = None


class CandidateEstimate(BaseModel):
    """The router's expectation for one ranked model, from the measured routing table."""

    model_id: str
    expected_accuracy: float
    expected_cost_usd: float
    score: float


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
    cost_basis: str = "estimate"
    service_fee: float = 0.0
    provider_cost: float | None = None


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
    cost_basis: str = "estimate"


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


class Billing(BaseModel):
    """Provider/model token cost, savings-based Lönn fee, and customer total."""
    model_cost_usd: float
    service_fee_usd: float
    total_cost_usd: float
    estimated_top_model_cost_usd: float | None = None
    estimated_savings_usd: float | None = None
    estimated_customer_savings_usd: float | None = None
    service_fee_rate: float
    service_fee_basis: str


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
    cost_basis: str = "estimate"
    total_latency_ms: float
    billing: Billing | None = None
    lambda_source: str | None = None
    lam_locked: bool | None = None

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
