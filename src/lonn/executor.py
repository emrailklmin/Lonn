"""Execute a selected NVIDIA NIM or OpenRouter model through LiteLLM.

Requires ``litellm`` and ``python-dotenv``. The caller supplies a schemas.Model
whose ID is the full ``<provider>/<publisher>/<model>`` ID from
config/models.yaml. Loading the pool, filtering active models, and routing
remain the caller's job.

Costs are computed from configured prices, not from provider billing data.
"""

import os
from math import isfinite
from pathlib import Path
from time import perf_counter

from .schemas import PROVIDER_API_KEYS, Model, ModelResponse, TokenUsage


def execute(
    prompt: str,
    model: Model,
    *,
    timeout: float = 120.0,
    max_tokens: int = 4096,
) -> ModelResponse:
    """Send one non-streaming text request to the selected model.

    Loads the repository .env without overriding existing environment values.
    Provider errors propagate to the caller; no model fallback is performed.
    Missing usage or non-text responses raise ValueError instead of reporting
    invented token counts. A length-limited response is returned with its
    finish_reason so the UI can indicate truncation.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    if model.provider not in PROVIDER_API_KEYS:
        raise ValueError(f"Model provider must be one of {', '.join(PROVIDER_API_KEYS)}")
    parts = model.id.split("/", 2)
    if len(parts) != 3 or parts[0] != model.provider or not all(parts):
        raise ValueError(f"Model ID must be a full {model.provider}/<publisher>/<model> ID")
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if type(model.context_window) is not int or model.context_window <= 0:
        raise ValueError("Model context_window must be a positive integer")
    if max_tokens >= model.context_window:
        raise ValueError("max_tokens must leave context space for the prompt")
    for price in (model.cost_per_m_input, model.cost_per_m_output):
        if not isfinite(price) or price < 0:
            raise ValueError("Model prices must be finite and non-negative")

    try:
        from dotenv import load_dotenv
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError(
            "Executor dependencies are missing. Install with: "
            "python3 -m pip install litellm python-dotenv"
        ) from exc

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    key_name = PROVIDER_API_KEYS[model.provider]
    api_key = os.environ.get(key_name, "").strip()
    if not api_key or api_key.endswith("-your-key-here"):
        raise ValueError(f"Set {key_name} in the environment or repository .env")

    started = perf_counter()
    response = completion(
        model=model.id,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        stream=False,
        num_retries=0,
        max_retries=0,
    )
    latency_ms = (perf_counter() - started) * 1000

    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Provider returned no response choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    finish_reason = getattr(choices[0], "finish_reason", None)
    # Reasoning can exhaust the output budget before any final text is emitted.
    # Preserve usage and finish_reason so the caller can explain the outcome.
    if message is not None and content is None and finish_reason == "length":
        content = ""
    if not isinstance(content, str) or getattr(message, "tool_calls", None):
        raise ValueError("Executor expected a text response, not tool calls or null content")
    if not content.strip() and finish_reason != "length":
        raise ValueError("Provider returned no final text")

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if any(type(count) is not int or count < 0 for count in (prompt_tokens, completion_tokens)):
        raise ValueError("Provider returned missing or invalid token usage")
    input_cost = prompt_tokens / 1_000_000 * model.cost_per_m_input
    output_cost = completion_tokens / 1_000_000 * model.cost_per_m_output
    return ModelResponse(
        content=content,
        model=model,
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        ),
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )


class ModelExecutor:
    """Class interface matching the architecture's executor contract."""

    execute = staticmethod(execute)
