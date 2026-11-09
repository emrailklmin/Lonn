"""Task-type training data builder (#19): exclusion, dedup, caps and splits. No network."""
from collections import Counter
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_task_data', ROOT / 'scripts' / 'build_task_data.py')
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)


def it(n, task='math_problem', text=None, group=None, **variants):
    return td.item('src', n, task, None, variants or {'as_benchmark': text or f'Problem number {n}'}, group)


class TaskDataTests(unittest.TestCase):
    def test_identical_variants_collapse(self):
        x = td.item('s', 1, 't', None, {'as_benchmark': 'Same text', 'plain': ' Same text '})
        self.assertEqual(list(x['variants']), ['as_benchmark'])

    def test_fingerprint_ignores_case_and_whitespace(self):
        self.assertEqual(td.fingerprint('What  is\n2+2?'), td.fingerprint('what is 2+2?'))

    def test_benchmark_questions_are_excluded_by_id_and_by_text(self):
        items = [it(1), it(2), it(3, text='The sampled question')]
        rows, stats = td.finalize(items, {'src:1'}, {td.fingerprint('the sampled  QUESTION')}, cap=10, seed=0)
        self.assertEqual({r['id'] for r in rows}, {'src:2'})
        self.assertEqual(stats['excluded_benchmark'], 2)

    def test_duplicates_are_dropped(self):
        items = [it(1, text='Same prompt'), it(2, text='same   prompt'), it(1, text='Other')]
        rows, stats = td.finalize(items, set(), set(), cap=10, seed=0)
        self.assertEqual([r['id'] for r in rows], ['src:1'])
        self.assertEqual(stats['excluded_duplicate'], 2)

    def test_cap_is_per_task_type_and_deterministic(self):
        items = [it(n) for n in range(20)] + [it(100 + n, task='translation') for n in range(3)]
        rows, stats = td.finalize(items, set(), set(), cap=5, seed=7)
        by_type = {t: {r['id'] for r in rows if r['task_type'] == t} for t in ('math_problem', 'translation')}
        self.assertEqual((len(by_type['math_problem']), len(by_type['translation'])), (5, 3))
        self.assertEqual(stats['capped_math_problem'], 15)
        again, _ = td.finalize(items, set(), set(), cap=5, seed=7)
        self.assertEqual(rows, again)

    def test_uncapped_sources_keep_every_prompt(self):
        items = [it(n) for n in range(20)]
        items += [td.item('wildchat', n, 'math_problem', None, {'as_benchmark': f'real prompt {n}'}) for n in range(8)]
        rows, _ = td.finalize(items, set(), set(), cap=5, seed=0, uncapped={'wildchat'})
        sources = Counter(r['source'] for r in rows)
        self.assertEqual(sources, {'src': 5, 'wildchat': 8})

    def test_variants_and_groups_share_a_split(self):
        items = [it(n, as_benchmark=f'Question {n}\n\nOptions: A) x', plain=f'Question {n}') for n in range(200)]
        items += [it(1000 + n, text=f'Translate {n}', group=f'src:{n}') for n in range(200)]
        rows, _ = td.finalize(items, set(), set(), cap=1000, seed=0)
        split_of = {}
        for r in rows:
            self.assertEqual(split_of.setdefault(r['group'], r['split']), r['split'])
        share = sum(s == 'validation' for s in split_of.values()) / len(split_of)
        self.assertTrue(0.03 < share < 0.2, share)

    def test_dolly_categories_map_to_known_task_types(self):
        from lonn.task_types import load_task_types
        known = {t.id for t in load_task_types()}
        self.assertTrue(set(td.DOLLY_TYPES.values()) <= known)


if __name__ == '__main__':
    unittest.main()
