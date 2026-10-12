"""Exercise the installable SDK against the real API with mocked inference."""

from contextlib import ExitStack
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "sdk/python/src")]

import httpx
from fastapi.testclient import TestClient
from lonn.api import create_app
from lonn.schemas import ComplexityResult, Model, ModelResponse, TokenUsage, TopicCategory
from lonnrouter import (AsyncLonnRouter, LonnAPIError, LonnConnectionError,
                        LonnResponseError, LonnRouter, LonnTimeoutError, Signal)


def fake_inference(stack):
    model = Model("nvidia_nim/test/model", "nvidia_nim", 1, 2, 8192,
                  {TopicCategory.CODING: 1.0})
    stack.enter_context(patch("lonn.api.load_model_pool", return_value=[model]))
    score = stack.enter_context(patch("lonn.api.ComplexityScorer.score_prompt",
                                     return_value=ComplexityResult(.5, {TopicCategory.CODING: 1.0})))
    execute = stack.enter_context(patch("lonn.api.execute", return_value=ModelResponse(
        "Hello from Lonn", model, TokenUsage(10, 20, 30, .00001, .00004, .00005), 12, "length")))
    return model, score, execute


class SDKTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model, self.score, self.execute = fake_inference(self.stack)
        self.http = self.stack.enter_context(TestClient(create_app()))
        self.client = self.stack.enter_context(LonnRouter("http://testserver/", http_client=self.http))

    def test_generate_round_trip(self):
        result = self.client.generate("  Hello  ", alpha=.7, max_tokens=256, provider_timeout=45)
        self.assertEqual(result.content, "Hello from Lonn")
        self.assertEqual(result.model.id, self.model.id)
        self.assertEqual(result.usage.total_tokens, 30)
        self.assertEqual(result.usage.total_cost, .00005)
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.model_dump()["response"]["content"], result.content)
        self.assertGreaterEqual(result.total_latency_ms, result.routing_latency_ms)
        self.score.assert_called_once_with("  Hello  ")
        self.execute.assert_called_once_with("  Hello  ", self.model, timeout=45, max_tokens=256)

    def test_route_signal_and_inspection(self):
        for signal in [Signal(complexity=.5, topic_scores={"coding": 1}),
                       {"complexity": .5, "topic_scores": {"coding": 1}}]:
            result = self.client.route("code", signal=signal)
            self.assertEqual(result.analysis_source, "provided")
        self.assertEqual(self.client.health().status, "ok")
        self.assertEqual(self.client.models()[0].id, self.model.id)
        self.score.assert_not_called()
        self.execute.assert_not_called()

    def test_server_validation_is_typed_error(self):
        with self.assertRaises(LonnAPIError) as caught:
            self.client.generate("private prompt", alpha=2)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertNotIn("private prompt", str(caught.exception))
        self.execute.assert_not_called()

    def test_rate_limit_not_retried(self):
        error = RuntimeError("private provider data")
        error.status_code = 429
        self.execute.side_effect = error
        with self.assertRaises(LonnAPIError) as caught:
            self.client.generate("code")
        self.assertEqual(caught.exception.status_code, 429)
        self.execute.assert_called_once()

    def test_url_prefix_environment_and_network_timeout(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"status": "ok"})
        with patch.dict(os.environ, {"LONN_BASE_URL": "https://example.test/lonn/"}):
            with httpx.Client(transport=httpx.MockTransport(handler)) as http:
                with LonnRouter(http_client=http, request_timeout=42) as router:
                    router.health()
                self.assertFalse(http.is_closed)
        self.assertEqual(str(requests[0].url), "https://example.test/lonn/health")
        self.assertEqual(requests[0].extensions["timeout"]["read"], 42)

    def test_transport_and_invalid_response_errors(self):
        cases = [(httpx.ReadTimeout("secret"), LonnTimeoutError),
                 (httpx.ConnectError("secret"), LonnConnectionError),
                 (httpx.Response(200, text="secret"), LonnResponseError),
                 (httpx.Response(200, json={"wrong": "secret"}), LonnResponseError),
                 (httpx.Response(502, text="secret"), LonnAPIError)]
        for value, expected in cases:
            with self.subTest(expected=expected):
                def handler(request):
                    if isinstance(value, Exception):
                        raise value
                    return value
                with httpx.Client(transport=httpx.MockTransport(handler)) as http:
                    with LonnRouter(http_client=http) as router:
                        with self.assertRaises(expected) as caught:
                            router.health()
                        self.assertNotIn("secret", str(caught.exception))

    def test_owned_client_closed(self):
        with LonnRouter() as router:
            self.assertFalse(router._client.is_closed)
        self.assertTrue(router._client.is_closed)

    def test_invalid_base_url(self):
        for url in ["localhost:8000", "file:///tmp/socket", "https://example.test?x=1"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                LonnRouter(url)


class AsyncSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_api_round_trip(self):
        with ExitStack() as stack:
            model, score, execute = fake_inference(stack)
            app = create_app()
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                    async with AsyncLonnRouter("http://testserver", http_client=http) as router:
                        result = await router.generate("code", max_tokens=256, provider_timeout=30)
                        self.assertEqual(result.content, "Hello from Lonn")
                        execute.assert_called_once_with("code", model, timeout=30, max_tokens=256)
                        selection = await router.route("code", signal=Signal(complexity=.5, topic_scores={"coding": 1}))
                        self.assertEqual(selection.analysis_source, "provided")
                        score.assert_called_once()
                        self.assertEqual((await router.health()).status, "ok")
                        self.assertEqual((await router.models())[0].id, model.id)
                        with self.assertRaises(LonnAPIError):
                            await router.generate(" ")
                    self.assertFalse(http.is_closed)

    async def test_async_transport_failure_and_close(self):
        def fail(request):
            raise httpx.ReadTimeout("secret")
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
            async with AsyncLonnRouter(http_client=http) as router:
                with self.assertRaises(LonnTimeoutError):
                    await router.health()
        async with AsyncLonnRouter() as router:
            self.assertFalse(router._client.is_closed)
        self.assertTrue(router._client.is_closed)


if __name__ == "__main__":
    unittest.main()
