"""HTTP interface for routing and end-to-end benchmarks.

Run: python3 -m uvicorn lonn.api:app --app-dir src --host 127.0.0.1
"""

from contextlib import asynccontextmanager
import logging
from math import fsum, isclose
from threading import Lock
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .complexity import ComplexityScorer
from .executor import execute
from .model_pool import load_model_pool
from .router import select_model
from .schemas import ComplexityResult, Model, ModelResponse, TopicCategory


logger = logging.getLogger(__name__)


UnitScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]


class SignalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    complexity: UnitScore
    topic_scores: dict[TopicCategory, UnitScore]

    @field_validator("topic_scores")
    @classmethod
    def probabilities_sum_to_one(cls, scores):
        if not isclose(fsum(scores.values()), 1, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Topic probabilities must sum to 1")
        return scores


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, strict=True)
    alpha: UnitScore = 0.5
    signal: SignalInput | None = None

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, prompt):
        if not prompt.strip():
            raise ValueError("Prompt must not be blank")
        return prompt


class GenerateRequest(RouteRequest):
    max_tokens: int = Field(default=4096, gt=0, strict=True)
    timeout: float = Field(default=120, gt=0, le=600, allow_inf_nan=False, strict=True)


class RouteResponse(BaseModel):
    model: Model
    signal: ComplexityResult
    alpha: float
    analysis_source: str
    routing_latency_ms: float


class GenerateResponse(RouteResponse):
    response: ModelResponse
    total_latency_ms: float


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        # Fail startup on invalid configuration; freeze the pool for this run.
        app.state.models = load_model_pool()
        app.state.scorer = ComplexityScorer()
        app.state.scorer_lock = Lock()
        yield

    app = FastAPI(title="Lonn Router", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        """Local service health; does not probe the classifier or any provider."""
        return {"status": "ok"}

    @app.get("/models", response_model=list[Model])
    def models():
        return app.state.models

    def route_request(request: RouteRequest) -> RouteResponse:
        started = perf_counter()
        try:
            if request.signal is not None:
                signal = ComplexityResult(request.signal.complexity, request.signal.topic_scores)
            else:
                # Protect lazy model initialization and inference across worker threads.
                with app.state.scorer_lock:
                    signal = app.state.scorer.score_prompt(request.prompt)
            model = select_model(signal, app.state.models, alpha=request.alpha)
        except Exception as exc:
            logger.warning(
                "Prompt analysis or routing failed (source=%s error=%s)",
                "provided" if request.signal is not None else "laya",
                type(exc).__name__,
            )
            raise HTTPException(503, "Prompt analysis or routing failed. Check server dependencies and model configuration.") from None
        return RouteResponse(
            model=model, signal=signal, alpha=request.alpha,
            analysis_source="provided" if request.signal is not None else "laya",
            routing_latency_ms=(perf_counter() - started) * 1000,
        )

    @app.post("/route", response_model=RouteResponse)
    def route(request: RouteRequest):
        """Analyze and select a model without calling NVIDIA."""
        return route_request(request)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        """Route and execute exactly once, without automatic retries or fallback."""
        started = perf_counter()
        routed = route_request(request)
        if request.max_tokens >= routed.model.context_window:
            raise HTTPException(422, "max_tokens must leave context space for the prompt")
        try:
            response = execute(request.prompt, routed.model,
                               timeout=request.timeout, max_tokens=request.max_tokens)
        except Exception as exc:
            # Raw provider errors can contain credentials or benchmark prompts.
            code = getattr(exc, "status_code", None)
            logger.warning(
                "Provider execution failed (provider=%s model=%s error=%s status=%s)",
                routed.model.provider, routed.model.id, type(exc).__name__, code,
            )
            if code == 429:
                raise HTTPException(429, "Provider rate limit reached") from None
            if code in (408, 504) or isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
                raise HTTPException(504, "Provider request timed out") from None
            if code in (401, 403):
                raise HTTPException(502, "Provider rejected server credentials or access") from None
            raise HTTPException(502, "Model execution failed. Check server credentials, dependencies, and provider availability.") from None
        return GenerateResponse(
            **routed.model_dump(), response=response,
            total_latency_ms=(perf_counter() - started) * 1000,
        )

    return app


app = create_app()
