"""HTTP interface for routing and end-to-end benchmarks.

Routing uses the task-type router (issue #19, docs/task_router.md): the task
classifier gives a probability per task type, and the router picks the model
with the highest estimated quality minus ``lam`` times its expected cost per
request. The routing table combines measured and user-supplied task scores.

`/generate` checks the answer before returning it (lonn.answer_check: cut off,
empty, or code that fails the prompt's worked examples). If it fails and a second
model is worth its expected cost, that model answers instead (one escalation).

Run: python3 -m uvicorn lonn.api:app --app-dir src --host 127.0.0.1
"""

from contextlib import asynccontextmanager
import logging
from math import fsum, isclose
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .answer_check import check_answer
from .executor import execute
from .model_pool import load_model_pool
from .routing_table import load_runtime_table
from .schemas import Model, ModelResponse, TaskSignal
from .task_classifier import TaskClassifier
from .task_router import DEFAULT_LAMBDA, FALLBACK_LIMIT, escalation_target, estimate, rank_models
from .task_types import load_task_types


logger = logging.getLogger(__name__)


UnitScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
Lambda = Annotated[float, Field(ge=0, le=100000, allow_inf_nan=False, strict=True)]
KNOWN_TASK_TYPES = frozenset(t.id for t in load_task_types())


class SignalInput(BaseModel):
    """A caller-supplied classifier result, which skips the task classifier."""

    model_config = ConfigDict(extra="forbid")
    task_scores: dict[str, UnitScore]
    complexity: UnitScore | None = None

    @field_validator("task_scores")
    @classmethod
    def known_types_summing_to_one(cls, scores):
        unknown = set(scores) - KNOWN_TASK_TYPES
        if unknown:
            raise ValueError(f"Unknown task types: {sorted(unknown)}")
        if not isclose(fsum(scores.values()), 1, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Task probabilities must sum to 1")
        return scores


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, strict=True)
    lam: Lambda = DEFAULT_LAMBDA
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
    verify: bool = Field(default=True, strict=True)  # check the answer and escalate once if it fails


class CandidateEstimate(BaseModel):
    model_id: str
    expected_accuracy: float = Field(description="Probability-weighted task quality; may include supplied estimates.")
    expected_cost_usd: float
    score: float


class Attempt(BaseModel):
    """One model call made for a /generate request."""

    model_id: str
    check: str  # "ok", "truncated", "empty", "failed_examples", "not_checked" or "error"
    cost_usd: float
    latency_ms: float


class RouteResponse(BaseModel):
    models: list[Model]
    estimates: list[CandidateEstimate]
    signal: TaskSignal
    lam: float
    analysis_source: Literal["classifier", "provided"]
    routing_latency_ms: float


class GenerateResponse(BaseModel):
    model: Model
    estimates: list[CandidateEstimate]
    signal: TaskSignal
    lam: float
    analysis_source: Literal["classifier", "provided"]
    routing_latency_ms: float
    response: ModelResponse
    attempts: list[Attempt]
    escalated: bool
    total_cost_usd: float
    total_latency_ms: float


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        # Fail startup on invalid configuration; freeze the pool and table for this run.
        app.state.models = load_model_pool()
        app.state.table = load_runtime_table()
        for model in app.state.models:
            model.task_aptitudes = app.state.table.model_aptitudes(model.id)
        app.state.classifier = TaskClassifier()  # The embedder loads on first use.
        app.state.classifier_lock = Lock()
        yield

    app = FastAPI(title="Lonn Router", version="2.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        """Local service health; does not load the classifier or call providers."""
        return {"status": "ok"}

    @app.get("/models", response_model=list[Model])
    def models():
        return app.state.models

    def route_request(request: RouteRequest) -> RouteResponse:
        started = perf_counter()
        source = "provided" if request.signal is not None else "classifier"
        try:
            if request.signal is not None:
                signal = TaskSignal(dict(request.signal.task_scores), request.signal.complexity)
            else:
                # Protect lazy model loading and inference across worker threads.
                with app.state.classifier_lock:
                    signal = app.state.classifier.classify(request.prompt)
            ranked = rank_models(signal, app.state.table, app.state.models, lam=request.lam,
                                 limit=FALLBACK_LIMIT)
            by_id = {e.model.id: e for e in estimate(signal, app.state.table, app.state.models, request.lam)}
        except Exception as exc:
            logger.warning("Prompt analysis or routing failed (source=%s error=%s)", source, type(exc).__name__)
            raise HTTPException(503, "Prompt analysis or routing failed. Check server dependencies and "
                                     "the routing configuration.") from None
        return RouteResponse(
            models=ranked,
            estimates=[CandidateEstimate(model_id=m.id, expected_accuracy=round(by_id[m.id].accuracy, 4),
                                         expected_cost_usd=round(by_id[m.id].cost_usd, 8),
                                         score=round(by_id[m.id].score, 4)) for m in ranked],
            signal=signal, lam=request.lam, analysis_source=source,
            routing_latency_ms=(perf_counter() - started) * 1000,
        )

    @app.post("/route", response_model=RouteResponse)
    def route(request: RouteRequest):
        """Classify the prompt and rank models without calling a provider."""
        return route_request(request)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        """Try up to five ranked models, falling back when execution fails."""
        started = perf_counter()
        routed = route_request(request)
        eligible_models = [
            model for model in routed.models
            if request.max_tokens < model.context_window
        ]
        if not eligible_models:
            raise HTTPException(422, "max_tokens must leave context space for the prompt")
        response = None
        selected_model = None
        last_error = None
        for model in eligible_models:
            try:
                response = execute(request.prompt, model,
                                   timeout=request.timeout, max_tokens=request.max_tokens)
                selected_model = model
                break
            except Exception as exc:
                last_error = exc
                # Raw provider errors can contain credentials or benchmark prompts.
                code = getattr(exc, "status_code", None)
                logger.warning(
                    "Provider execution failed (provider=%s model=%s error=%s status=%s)",
                    model.provider, model.id, type(exc).__name__, code,
                )
        if response is None:
            # Preserve the last provider failure's public status when all candidates fail.
            code = getattr(last_error, "status_code", None)
            if code == 429:
                raise HTTPException(429, "Provider rate limit reached") from None
            if code in (408, 504) or isinstance(last_error, TimeoutError) or "timeout" in type(last_error).__name__.lower():
                raise HTTPException(504, "Provider request timed out") from None
            if code in (401, 403):
                raise HTTPException(502, "Provider rejected server credentials or access") from None
            raise HTTPException(502, "Model execution failed. Check server credentials, dependencies, and provider availability.") from None
        attempts = [Attempt(model_id=selected_model.id, check="not_checked", cost_usd=response.usage.total_cost,
                            latency_ms=response.latency_ms)]
        escalated = False
        if request.verify:
            result = check_answer(request.prompt, response.content, response.finish_reason)
            attempts[0].check = "ok" if result.ok else result.reason
            target = None if result.ok else escalation_target(
                routed.signal, app.state.table, app.state.models, request.lam, selected_model)
            if target is not None and request.max_tokens < target.context_window:
                try:
                    second = execute(request.prompt, target, timeout=request.timeout, max_tokens=request.max_tokens)
                    attempts.append(Attempt(model_id=target.id, check="not_checked", cost_usd=second.usage.total_cost,
                                            latency_ms=second.latency_ms))
                    response, selected_model, escalated = second, target, True
                except Exception as exc:  # Keep the first answer; the escalation is best effort.
                    attempts.append(Attempt(model_id=target.id, check="error", cost_usd=0.0, latency_ms=0.0))
                    logger.warning("Escalation failed (provider=%s model=%s error=%s status=%s)", target.provider,
                                   target.id, type(exc).__name__, getattr(exc, "status_code", None))
        return GenerateResponse(
            model=selected_model, estimates=routed.estimates, signal=routed.signal, lam=routed.lam,
            analysis_source=routed.analysis_source,
            routing_latency_ms=routed.routing_latency_ms, response=response, attempts=attempts,
            escalated=escalated, total_cost_usd=sum(a.cost_usd for a in attempts),
            total_latency_ms=(perf_counter() - started) * 1000,
        )

    return app


app = create_app()
