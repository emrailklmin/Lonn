"""HTTP contract tests; no classifier downloads or provider requests."""

from pathlib import Path
from contextlib import ExitStack
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from lonn.api import create_app
from lonn.schemas import ComplexityResult, Model, ModelResponse, TokenUsage, TopicCategory


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model = Model("nvidia_nim/test/strong", "nvidia_nim", 1, 2, 8192,
                           {TopicCategory.CODING: 0.9})
        self.cheap = Model("nvidia_nim/test/cheap", "nvidia_nim", 0, 0, 8192,
                           {TopicCategory.CODING: 0.3})
        self.signal = ComplexityResult(0.7, {TopicCategory.CODING: 1.0})
        self.stack.enter_context(patch("lonn.api.load_model_pool", return_value=[self.model, self.cheap]))
        self.score = self.stack.enter_context(patch("lonn.api.ComplexityScorer.score_prompt", return_value=self.signal))
        self.execute = self.stack.enter_context(patch("lonn.api.execute", return_value=ModelResponse(
            "Answer", self.model, TokenUsage(10, 20, 30, .00001, .00004, .00005), 15, "stop")))
        self.client = self.stack.enter_context(TestClient(create_app()))

    def test_health_models_and_openapi_do_not_load_classifier(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(len(self.client.get("/models").json()), 2)
        self.assertIn("/generate", self.client.get("/openapi.json").json()["paths"])
        self.score.assert_not_called()
        self.execute.assert_not_called()

    def test_automatic_route_and_alpha(self):
        for alpha, model in [(0, self.cheap), (1, self.model)]:
            response = self.client.post("/route", json={"prompt": "  Write code  ", "alpha": alpha})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["model"]["id"], model.id)
            self.assertEqual(result["analysis_source"], "laya")
            self.assertGreaterEqual(result["routing_latency_ms"], 0)
        self.score.assert_called_with("  Write code  ")
        self.execute.assert_not_called()

    def test_supplied_signal_skips_classifier(self):
        response = self.client.post("/route", json={"prompt": "code", "signal": {
            "complexity": .7, "topic_scores": {"coding": 1}}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["analysis_source"], "provided")
        self.score.assert_not_called()
        self.execute.assert_not_called()

    def test_generate_returns_benchmark_metadata_and_forwards_controls(self):
        self.execute.return_value.finish_reason = "length"
        response = self.client.post("/generate", json={"prompt": "code", "alpha": 1,
                                                      "max_tokens": 100, "timeout": 30})
        self.assertEqual(response.status_code, 200, response.text)
        self.execute.assert_called_once_with("code", self.model, timeout=30, max_tokens=100)
        result = response.json()
        self.assertEqual(result["response"]["content"], "Answer")
        self.assertEqual(result["response"]["usage"]["total_tokens"], 30)
        self.assertEqual(result["response"]["usage"]["total_cost"], .00005)
        self.assertEqual(result["response"]["finish_reason"], "length")
        self.assertEqual(result["response"]["latency_ms"], 15)
        self.assertGreaterEqual(result["total_latency_ms"], result["routing_latency_ms"])

    def test_invalid_inputs_rejected_before_work(self):
        for update in [{"prompt": " "}, {"alpha": -1}, {"alpha": "NaN"}, {"alpha": True},
                       {"max_tokens": 0}, {"max_tokens": 1.5}, {"max_tokens": True},
                       {"timeout": 0}, {"timeout": "Infinity"}, {"unknown": 1},
                       {"signal": {"complexity": .5, "topic_scores": {"coding": .2}}},
                       {"signal": {"complexity": .5, "topic_scores": {"unknown": 1}}}]:
            with self.subTest(update=update):
                response = self.client.post("/generate", json={"prompt": "code", **update})
                self.assertEqual(response.status_code, 422, response.text)
        self.score.assert_not_called()
        self.execute.assert_not_called()

    def test_context_budget_rejected_before_provider_call(self):
        response = self.client.post("/generate", json={"prompt": "code", "max_tokens": 8192})
        self.assertEqual(response.status_code, 422)
        self.execute.assert_not_called()

    def test_provider_errors_sanitized_without_retries(self):
        for code, expected in [(429, 429), (401, 502), (403, 502), (504, 504), (500, 502)]:
            with self.subTest(code=code):
                exc = RuntimeError("secret-key and private prompt")
                exc.status_code = code
                self.execute.side_effect = exc
                self.execute.reset_mock()
                response = self.client.post("/generate", json={"prompt": "code"})
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("secret-key", response.text)
                self.assertNotIn("private prompt", response.text)
                self.execute.assert_called_once()

    def test_analysis_failure_does_not_execute(self):
        self.score.side_effect = RuntimeError("private prompt")
        response = self.client.post("/generate", json={"prompt": "code"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private prompt", response.text)
        self.execute.assert_not_called()

    def test_invalid_pool_fails_startup(self):
        with patch("lonn.api.load_model_pool", side_effect=ValueError("Invalid pool")):
            with self.assertRaisesRegex(ValueError, "Invalid pool"):
                with TestClient(create_app()):
                    pass


if __name__ == "__main__":
    unittest.main()
