"""Everyday prompt sampling and labelling helpers (#19). No network, no paid calls."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('everyday_prompts', ROOT / 'scripts' / 'everyday_prompts.py')
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)


def row(text, **overrides):
    base = {'language': 'English', 'toxic': False, 'redacted': False,
            'conversation': [{'role': 'user', 'content': text, 'toxic': False, 'redacted': False}],
            'openai_moderation': [{'flagged': False}]}
    return {**base, **overrides}


class EverydayPromptTests(unittest.TestCase):
    def test_only_safe_english_first_turns_are_kept(self):
        self.assertEqual(ep.safe_first_turn(row('  What is the capital of France?  ')), 'What is the capital of France?')
        for bad in (row('What is the capital of France?', language='German'),
                    row('What is the capital of France?', toxic=True),
                    row('What is the capital of France?', redacted=True),
                    row('What is the capital of France?', openai_moderation=[{'flagged': True}]),
                    row('Hi'),  # too short
                    row('Оптимизируй этот код метода, пожалуйста')):  # non-Latin script
            self.assertIsNone(ep.safe_first_turn(bad))

    def test_parse_label_accepts_only_known_types(self):
        known = {'math_problem', 'conversation'}
        self.assertEqual(ep.parse_label('{"task_type": "math_problem"}', known), 'math_problem')
        self.assertEqual(ep.parse_label('Sure.\n```json\n{"task_type": "conversation"}\n```', known), 'conversation')
        self.assertIsNone(ep.parse_label('{"task_type": "poetry"}', known))
        self.assertIsNone(ep.parse_label('math_problem', known))

    def test_clip_keeps_start_and_end(self):
        text = 'a' * 5000 + 'END'
        clipped = ep.clip(text, 1000)
        self.assertLess(len(clipped), 1100)
        self.assertTrue(clipped.startswith('a') and clipped.endswith('END'))
        self.assertEqual(ep.clip('short'), 'short')

    def test_system_prompt_lists_every_task_type(self):
        from lonn.task_types import load_task_types
        types = load_task_types()
        prompt = ep.system_prompt(types)
        self.assertTrue(all(f'- {t.id}:' in prompt for t in types))


if __name__ == '__main__':
    unittest.main()
