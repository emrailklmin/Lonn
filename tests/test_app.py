"""UI integration tests; no provider requests or classifier downloads."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from streamlit.testing.v1 import AppTest
from lonn.schemas import ModelResponse, TaskSignal, TokenUsage

APP = Path(__file__).resolve().parents[1] / 'examples/app.py'


class AppTests(unittest.TestCase):
    def test_manual_analysis_does_not_call_provider(self):
        with patch('lonn.executor.execute') as execute:
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.radio[0].set_value('Manual task type').run()
            app.text_area[0].set_value('Write a sorting function')
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.success)
            execute.assert_not_called()

    def test_automatic_flow_calls_selected_model_once(self):
        signal = TaskSignal({'code_function': 0.9, 'code_algorithmic': 0.1})
        def reply(prompt, model, **kwargs):
            return ModelResponse('Test answer', model, TokenUsage(10, 20, 30, 0, 0, 0), 150, 'stop')
        with patch('lonn.task_classifier.TaskClassifier.classify', return_value=signal) as score, \
             patch('lonn.executor.execute', side_effect=reply) as execute:
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.text_area[0].set_value('Write a sorting function')
            app.button[1].click().run()
            self.assertFalse(app.exception)
            score.assert_called_once_with('Write a sorting function')
            execute.assert_called_once()
            self.assertIn('Test answer', [text.value for text in app.text])
            # Changing controls rerenders the last result, without resending it.
            app.slider[0].set_value(40.0).run()
            execute.assert_called_once()

    def test_manual_flow_falls_back_to_next_ranked_model(self):
        attempts = []
        def reply(prompt, model, **kwargs):
            attempts.append(model)
            if len(attempts) == 1:
                raise RuntimeError('unavailable')
            return ModelResponse('Fallback answer', model,
                                 TokenUsage(10, 12, 22, 0, 0, 0), 150, 'stop')
        with patch('lonn.executor.execute', side_effect=reply) as execute:
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.radio[0].set_value('Manual task type').run()
            app.text_area[0].set_value('Write a sorting function')
            app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(execute.call_count, 2)
            self.assertIn('Fallback answer', [text.value for text in app.text])

    def test_error_does_not_expose_provider_details(self):
        with patch('lonn.executor.execute', side_effect=RuntimeError('secret-key')):
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.radio[0].set_value('Manual task type').run()
            app.text_area[0].set_value('Hello')
            app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertNotIn('secret-key', app.error[0].value)

    def test_blank_prompt_does_not_execute(self):
        with patch('lonn.executor.execute') as execute:
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.button[1].click().run()
            self.assertTrue(app.warning)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
