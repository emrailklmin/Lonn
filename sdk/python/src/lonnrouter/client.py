"""Reusable sync and async clients; no inference dependencies or retries."""

import os
from typing import TypeVar

import httpx
from pydantic import TypeAdapter, ValidationError

from .errors import LonnAPIError, LonnConnectionError, LonnResponseError, LonnTimeoutError
from .models import GenerateResult, Health, Model, RouteResult, Signal


T = TypeVar("T")


def _base_url(value: str | None) -> str:
    value = value if value is not None else os.getenv("LONN_BASE_URL", "http://127.0.0.1:8000")
    url = httpx.URL(value)
    if url.scheme not in ("http", "https") or not url.host or url.query or url.fragment:
        raise ValueError("base_url must be an HTTP(S) server URL without a query or fragment")
    return str(url).rstrip("/")


DEFAULT_LAMBDA = 20.0  # Accuracy points one USD per request is worth; higher picks cheaper models.


def _payload(prompt: str, lam: float, signal: Signal | dict | None) -> dict:
    payload = {"prompt": prompt, "lam": lam}
    if signal is not None:
        payload["signal"] = signal.model_dump(exclude_none=True) if isinstance(signal, Signal) else signal
    return payload


def _parse(response: httpx.Response, result_type: type[T]) -> T:
    if not response.is_success:
        raise LonnAPIError(response.status_code)
    try:
        return TypeAdapter(result_type).validate_json(response.content)
    except (ValidationError, ValueError):
        raise LonnResponseError("Lonn returned an unexpected response format") from None


class LonnRouter:
    """Blocking SDK. Reuse one instance, then call close() or use `with`.

    `request_timeout` controls HTTP network timeouts, not provider execution.
    An injected `http_client` remains owned by the caller.
    """

    def __init__(self, base_url: str | None = None, *, request_timeout: float = 300,
                 http_client: httpx.Client | None = None):
        self.base_url = _base_url(base_url)
        self.request_timeout = request_timeout
        self._owns_client = http_client is None
        self._client = http_client if http_client is not None else httpx.Client()

    def _request(self, method: str, path: str, result_type: type[T], **kwargs) -> T:
        try:
            response = self._client.request(method, f"{self.base_url}/{path}",
                                            timeout=self.request_timeout, **kwargs)
        except httpx.TimeoutException:
            raise LonnTimeoutError("Lonn request timed out; server work may still be running") from None
        except httpx.RequestError:
            raise LonnConnectionError("Could not reach Lonn; check base_url and server connectivity") from None
        return _parse(response, result_type)

    def route(self, prompt: str, *, lam: float = DEFAULT_LAMBDA,
              signal: Signal | dict | None = None) -> RouteResult:
        return self._request("POST", "route", RouteResult, json=_payload(prompt, lam, signal))

    def generate(self, prompt: str, *, lam: float = DEFAULT_LAMBDA,
                 signal: Signal | dict | None = None, max_tokens: int = 4096,
                 provider_timeout: float = 120) -> GenerateResult:
        payload = _payload(prompt, lam, signal)
        payload.update(max_tokens=max_tokens, timeout=provider_timeout)
        return self._request("POST", "generate", GenerateResult, json=payload)

    def models(self) -> list[Model]:
        return self._request("GET", "models", list[Model])

    def health(self) -> Health:
        return self._request("GET", "health", Health)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncLonnRouter:
    """Async SDK for apps; use `async with` or await aclose()."""

    def __init__(self, base_url: str | None = None, *, request_timeout: float = 300,
                 http_client: httpx.AsyncClient | None = None):
        self.base_url = _base_url(base_url)
        self.request_timeout = request_timeout
        self._owns_client = http_client is None
        self._client = http_client if http_client is not None else httpx.AsyncClient()

    async def _request(self, method: str, path: str, result_type: type[T], **kwargs) -> T:
        try:
            response = await self._client.request(method, f"{self.base_url}/{path}",
                                                  timeout=self.request_timeout, **kwargs)
        except httpx.TimeoutException:
            raise LonnTimeoutError("Lonn request timed out; server work may still be running") from None
        except httpx.RequestError:
            raise LonnConnectionError("Could not reach Lonn; check base_url and server connectivity") from None
        return _parse(response, result_type)

    async def route(self, prompt: str, *, lam: float = DEFAULT_LAMBDA,
                    signal: Signal | dict | None = None) -> RouteResult:
        return await self._request("POST", "route", RouteResult, json=_payload(prompt, lam, signal))

    async def generate(self, prompt: str, *, lam: float = DEFAULT_LAMBDA,
                       signal: Signal | dict | None = None, max_tokens: int = 4096,
                       provider_timeout: float = 120) -> GenerateResult:
        payload = _payload(prompt, lam, signal)
        payload.update(max_tokens=max_tokens, timeout=provider_timeout)
        return await self._request("POST", "generate", GenerateResult, json=payload)

    async def models(self) -> list[Model]:
        return await self._request("GET", "models", list[Model])

    async def health(self) -> Health:
        return await self._request("GET", "health", Health)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
