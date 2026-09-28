import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lonn.executor import ModelExecutor, execute
from lonn.schemas import Model, ModelResponse, TokenUsage


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.model = Model("nvidia_nim/openai/gpt-oss-20b", "nvidia_nim",
                           0.03, 0.14, 8192, {})
        self.response = NS(
            choices=[NS(message=NS(content="Hello", tool_calls=None), finish_reason="stop")],
            usage=NS(prompt_tokens=100, completion_tokens=50),
        )
        self.completion = Mock(return_value=self.response)
        self.load_dotenv = Mock()
        modules = patch.dict(sys.modules, {
            "litellm": NS(completion=self.completion),
            "dotenv": NS(load_dotenv=self.load_dotenv),
        })
        modules.start()
        self.addCleanup(modules.stop)
        environment = patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": "test-key"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_request_and_shared_response_contract(self):
        with patch("lonn.executor.perf_counter", side_effect=[1.0, 1.25]):
            result = ModelExecutor().execute("Hi", self.model, timeout=30, max_tokens=256)
        self.assertIsInstance(result, ModelResponse)
        self.assertIsInstance(result.usage, TokenUsage)
        self.assertIs(result.model, self.model)
        self.assertEqual(result.content, "Hello")
        self.assertEqual(result.latency_ms, 250)
        self.assertEqual(result.usage.total_tokens, 150)
        self.assertAlmostEqual(result.usage.input_cost, 0.000003)
        self.assertAlmostEqual(result.usage.output_cost, 0.000007)
        self.assertAlmostEqual(result.usage.total_cost, 0.000010)
        self.completion.assert_called_once_with(
            model=self.model.id, messages=[{"role": "user", "content": "Hi"}],
            api_key="test-key", timeout=30, max_tokens=256,
            stream=False, num_retries=0, max_retries=0,
        )
        self.load_dotenv.assert_called_once_with(
            Path(__file__).resolve().parents[1] / ".env", override=False
        )

    def test_truncation_is_preserved(self):
        self.response.choices[0].finish_reason = "length"
        self.assertEqual(execute("Hi", self.model).finish_reason, "length")

    def test_free_model(self):
        result = execute("Hi", replace(self.model, cost_per_m_input=0, cost_per_m_output=0))
        self.assertEqual(result.usage.total_cost, 0)

    def test_invalid_inputs_do_not_call_provider(self):
        for changes in ({"prompt": " "}, {"timeout": 0}, {"timeout": float("nan")},
                        {"max_tokens": True}, {"max_tokens": 0}, {"max_tokens": 8192},
                        {"model": replace(self.model, id="nvidia_nim/")},
                        {"model": replace(self.model, context_window=0)},
                        {"model": replace(self.model, cost_per_m_input=-1)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                execute(**{"prompt": "Hi", "model": self.model, **changes})
        self.completion.assert_not_called()

    def test_missing_key(self):
        os.environ["NVIDIA_NIM_API_KEY"] = ""
        with self.assertRaisesRegex(ValueError, "NVIDIA_NIM_API_KEY"):
            execute("Hi", self.model)
        self.completion.assert_not_called()

    def test_malformed_responses(self):
        for response in (
            NS(choices=[]), NS(choices=[NS()]),
            NS(choices=[NS(message=NS(content=None))]),
            NS(choices=[NS(message=NS(content="", tool_calls=[NS()]))]),
            NS(choices=self.response.choices, usage=None),
            NS(choices=self.response.choices, usage=NS(prompt_tokens=-1, completion_tokens=2)),
        ):
            self.completion.return_value = response
            with self.subTest(response=response), self.assertRaises(ValueError):
                execute("Hi", self.model)

    def test_provider_errors_propagate_without_retry(self):
        error = TimeoutError("test timeout")
        self.completion.side_effect = error
        with self.assertRaises(TimeoutError) as raised:
            execute("Hi", self.model)
        self.assertIs(raised.exception, error)
        self.completion.assert_called_once()

    def test_missing_dependency_has_setup_message(self):
        with patch.dict(sys.modules, {"litellm": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install"):
                execute("Hi", self.model)


if __name__ == "__main__":
    unittest.main()
