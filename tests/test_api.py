"""HTTP contract tests; no classifier downloads or provider requests."""

from pathlib import Path
from contextlib import ExitStack
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from lonn.api import create_app
from lonn.routing_table import build_table
from lonn.schemas import Model, ModelResponse, TaskSignal, TokenUsage, TopicCategory


def rows(model_id, passes, n, cost):
    return [{"task": "code_function", "difficulty": None, "model_id": model_id,
             "correct": i < passes, "cost_usd": cost} for i in range(n)]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model = Model("nvidia_nim/test/strong", "nvidia_nim", 1, 2, 8192,
                           {TopicCategory.CODING: 0.9})
        self.cheap = Model("nvidia_nim/test/cheap", "nvidia_nim", 0, 0, 8192,
                           {TopicCategory.CODING: 0.3})
        # Strong: 90% at 0.01 USD per request; cheap: 30% at 0.0001 USD.
        table = build_table(rows(self.model.id, 9, 10, 0.01) + rows(self.cheap.id, 3, 10, 0.0001))
        self.signal = TaskSignal({"code_function": 1.0})
        self.stack.enter_context(patch("lonn.api.load_model_pool", return_value=[self.model, self.cheap]))
        self.stack.enter_context(patch("lonn.api.RoutingTable.load", return_value=table))
        self.classify = self.stack.enter_context(patch("lonn.api.TaskClassifier.classify", return_value=self.signal))
        self.execute = self.stack.enter_context(patch("lonn.api.execute", return_value=ModelResponse(
            "Answer", self.model, TokenUsage(10, 20, 30, .00001, .00004, .00005), 15, "stop")))
        self.client = self.stack.enter_context(TestClient(create_app()))

    def test_health_models_and_openapi_do_not_load_classifier(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(len(self.client.get("/models").json()), 2)
        self.assertIn("/generate", self.client.get("/openapi.json").json()["paths"])
        self.classify.assert_not_called()
        self.execute.assert_not_called()

    def test_automatic_route_and_lambda(self):
        for lam, model in [(0, self.model), (1000, self.cheap)]:
            response = self.client.post("/route", json={"prompt": "  Write code  ", "lam": lam})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["models"][0]["id"], model.id)
            self.assertEqual(len(result["models"]), 2)
            self.assertEqual(result["analysis_source"], "classifier")
            self.assertEqual(result["signal"], {"task_scores": {"code_function": 1.0}, "complexity": None})
            self.assertGreaterEqual(result["routing_latency_ms"], 0)
        self.classify.assert_called_with("  Write code  ")
        self.execute.assert_not_called()

    def test_route_reports_estimates_in_ranking_order(self):
        result = self.client.post("/route", json={"prompt": "code"}).json()
        self.assertEqual(result["lam"], 20.0)
        self.assertEqual([e["model_id"] for e in result["estimates"]], [m["id"] for m in result["models"]])
        strong = result["estimates"][0]
        self.assertEqual((strong["expected_accuracy"], strong["expected_cost_usd"]), (0.9, 0.01))
        self.assertAlmostEqual(strong["score"], 0.9 - 20 * 0.01)

    def test_supplied_signal_skips_classifier(self):
        response = self.client.post("/route", json={"prompt": "code", "signal": {
            "task_scores": {"code_function": 0.5, "summarization": 0.5}}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["analysis_source"], "provided")
        self.classify.assert_not_called()
        self.execute.assert_not_called()

    def test_generate_returns_benchmark_metadata_and_forwards_controls(self):
        self.execute.return_value.finish_reason = "length"
        response = self.client.post("/generate", json={"prompt": "code", "lam": 0, "verify": False,
                                                      "max_tokens": 100, "timeout": 30})
        self.assertEqual(response.status_code, 200, response.text)
        self.execute.assert_called_once_with("code", self.model, timeout=30, max_tokens=100)
        result = response.json()
        self.assertEqual(result["response"]["content"], "Answer")
        self.assertEqual(result["response"]["usage"]["total_tokens"], 30)
        self.assertEqual(result["response"]["usage"]["total_cost"], .00005)
        self.assertEqual(result["response"]["finish_reason"], "length")
        self.assertEqual(result["response"]["latency_ms"], 15)
        self.assertEqual(len(result["estimates"]), 2)
        self.assertGreaterEqual(result["total_latency_ms"], result["routing_latency_ms"])

    def test_failed_check_escalates_once_and_reports_both_attempts(self):
        cut_off = ModelResponse("Partial", self.model, TokenUsage(10, 4096, 4106, 0, .04, .04), 900, "length")
        second = ModelResponse("Full answer", self.cheap, TokenUsage(10, 50, 60, 0, .0001, .0001), 100, "stop")
        self.execute.side_effect = [cut_off, second]
        result = self.client.post("/generate", json={"prompt": "code", "lam": 20}).json()
        self.assertTrue(result["escalated"])
        self.assertEqual(result["model"]["id"], self.cheap.id)
        self.assertEqual(result["response"]["content"], "Full answer")
        self.assertEqual([(a["model_id"], a["check"]) for a in result["attempts"]],
                         [(self.model.id, "truncated"), (self.cheap.id, "not_checked")])
        self.assertAlmostEqual(result["total_cost_usd"], .0401)

    def test_passing_check_or_verify_off_does_not_escalate(self):
        self.execute.return_value.finish_reason = "length"
        result = self.client.post("/generate", json={"prompt": "code", "verify": False}).json()
        self.assertFalse(result["escalated"])
        self.assertEqual(result["attempts"][0]["check"], "not_checked")
        self.execute.return_value.finish_reason = "stop"
        result = self.client.post("/generate", json={"prompt": "code"}).json()
        self.assertEqual((result["escalated"], result["attempts"][0]["check"]), (False, "ok"))
        self.assertEqual(self.execute.call_count, 2)

    def test_failed_escalation_keeps_the_first_answer(self):
        cut_off = ModelResponse("Partial", self.model, TokenUsage(10, 4096, 4106, 0, .04, .04), 900, "length")
        self.execute.side_effect = [cut_off, RuntimeError("down")]
        result = self.client.post("/generate", json={"prompt": "code"}).json()
        self.assertFalse(result["escalated"])
        self.assertEqual(result["response"]["content"], "Partial")
        self.assertEqual(result["attempts"][1]["check"], "error")

    def test_generate_falls_back_to_next_ranked_model(self):
        fallback_response = ModelResponse(
            "Fallback answer", self.cheap,
            TokenUsage(10, 12, 22, 0, 0, 0), 18, "stop")
        self.execute.side_effect = [RuntimeError("provider unavailable"), fallback_response]
        response = self.client.post("/generate", json={"prompt": "code", "lam": 0})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"]["id"], self.cheap.id)
        self.assertEqual(response.json()["response"]["content"], "Fallback answer")
        self.assertEqual([call.args[1] for call in self.execute.call_args_list],
                         [self.model, self.cheap])

    def test_invalid_inputs_rejected_before_work(self):
        for update in [{"prompt": " "}, {"lam": -1}, {"lam": "NaN"}, {"lam": True}, {"alpha": 0.5},
                       {"max_tokens": 0}, {"max_tokens": 1.5}, {"max_tokens": True},
                       {"timeout": 0}, {"timeout": "Infinity"}, {"unknown": 1}, {"verify": "yes"},
                       {"signal": {"task_scores": {"code_function": .2}}},
                       {"signal": {"task_scores": {"unknown": 1}}},
                       {"signal": {"task_scores": {"code_function": 1}, "complexity": 2}},
                       {"signal": {"complexity": .5, "topic_scores": {"coding": 1}}}]:
            with self.subTest(update=update):
                response = self.client.post("/generate", json={"prompt": "code", **update})
                self.assertEqual(response.status_code, 422, response.text)
        self.classify.assert_not_called()
        self.execute.assert_not_called()

    def test_context_budget_rejected_before_provider_call(self):
        response = self.client.post("/generate", json={"prompt": "code", "max_tokens": 8192})
        self.assertEqual(response.status_code, 422)
        self.execute.assert_not_called()

    def test_provider_errors_sanitized_after_all_fallbacks_fail(self):
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
                self.assertEqual(self.execute.call_count, 2)

    def test_analysis_failure_does_not_execute(self):
        self.classify.side_effect = RuntimeError("private prompt")
        response = self.client.post("/generate", json={"prompt": "code"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private prompt", response.text)
        self.execute.assert_not_called()

    def test_invalid_configuration_fails_startup(self):
        with patch("lonn.api.load_model_pool", side_effect=ValueError("Invalid pool")):
            with self.assertRaisesRegex(ValueError, "Invalid pool"):
                with TestClient(create_app()):
                    pass
        with patch("lonn.api.RoutingTable.load", side_effect=FileNotFoundError("routing_table.json")):
            with self.assertRaises(FileNotFoundError):
                with TestClient(create_app()):
                    pass


if __name__ == "__main__":
    unittest.main()
