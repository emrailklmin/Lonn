import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from lonn.model_pool import load_model_pool
from lonn.schemas import TopicCategory


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_script('pool_probe', 'scripts/check_models.py')
discovery = load_script('pool_discovery', 'scripts/discover_models.py')


class PoolTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'pool.yaml'
        self.entry = dict(id='nvidia_nim/openai/gpt-oss-20b', active=True, provider='nvidia_nim',
                          cost_per_m_input=0.03, cost_per_m_output=0.14, price_basis='estimate',
                          context_window=8192,
                          topic_aptitudes={topic.value: 0.5 for topic in TopicCategory})

    def load(self, entries):
        self.path.write_text(yaml.safe_dump(entries))
        return load_model_pool(self.path)

    def test_mapping_and_inactive_filter(self):
        model, = self.load([self.entry, {'active': False}])
        self.assertEqual(model.id, self.entry['id'])
        self.assertEqual(model.provider, 'nvidia_nim')
        self.assertEqual(model.cost_per_m_input, 0.03)
        self.assertEqual(model.cost_per_m_output, 0.14)
        self.assertEqual(set(model.topic_aptitudes), set(TopicCategory))

    def test_openrouter_provider(self):
        entry = {**self.entry, 'id': 'openrouter/openai/gpt-oss-20b', 'provider': 'openrouter',
                 'price_basis': 'provider'}
        nim, openrouter = self.load([self.entry, entry])
        self.assertEqual(nim.provider, 'nvidia_nim')
        self.assertEqual(openrouter.provider, 'openrouter')
        self.assertEqual(openrouter.id, 'openrouter/openai/gpt-oss-20b')

    def test_provider_must_match_id(self):
        for changes in ({'provider': 'openrouter'}, {'provider': 'together_ai'},
                        {'provider': None}, {'id': 'openai/gpt-oss-20b'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load([{**self.entry, **changes}])

    def test_missing_metadata_rejected(self):
        for field in ('context_window', 'topic_aptitudes', 'cost_per_m_input', 'price_basis'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load([{**self.entry, field: None}])

    def test_duplicate_and_empty_pool(self):
        for entries in ([], [self.entry, self.entry]):
            with self.assertRaises(ValueError):
                self.load(entries)

    def test_unknown_aptitude_rejected(self):
        scores = {**self.entry['topic_aptitudes'], 'coding': None}
        with self.assertRaisesRegex(ValueError, 'topic_aptitudes'):
            self.load([{**self.entry, 'topic_aptitudes': scores}])

    def test_discovery_keeps_unpriced_and_multimodal(self):
        inventory = discovery.build_inventory({'data': [
            {'id': 'example/unpriced'}, {'id': 'example/vision-model'}]})
        self.assertEqual(len(inventory['models']), 2)
        self.assertTrue(all(m['status'] == 'untested' for m in inventory['models']))

    def test_text_probe_does_not_require_tools(self):
        call = Mock(return_value=NS(choices=[NS(message=NS(content='OK'), finish_reason='stop')]))
        result = probe.probe_model(self.entry['id'], call)
        self.assertEqual(result['status'], 'ok')
        self.assertNotIn('tools', call.call_args.kwargs)

    def test_timeout_can_recover(self):
        call = Mock(side_effect=[TimeoutError(), NS(choices=[NS(message=NS(content='OK'))])])
        with patch.object(probe, 'sleep'):
            result = probe.probe_model(self.entry['id'], call)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['attempts'][0]['status'], 'timeout')

    def test_error_classification(self):
        for status, expected in [(404, 'not_found'), (403, 'access_denied'),
                                 (429, 'rate_limited'), (408, 'timeout'), (503, 'server_error')]:
            error = RuntimeError('secret must not be recorded')
            error.status_code = status
            with self.subTest(status=status), patch.object(probe, 'sleep'):
                result = probe.probe_model(self.entry['id'], Mock(side_effect=error))
                self.assertEqual(result['status'], expected)
                self.assertNotIn('secret', str(result))


if __name__ == '__main__':
    unittest.main()
