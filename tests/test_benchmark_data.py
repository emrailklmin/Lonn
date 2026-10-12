import base64
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lonn.schemas import TopicCategory


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bd = load_script('benchmark_build_data', 'scripts/benchmark/build_data.py')

RECORD_KEYS = {'prompt_id', 'dataset', 'category', 'source_dataset', 'source_revision', 'source_id',
               'source_split', 'task_family', 'topic', 'level_label', 'difficulty', 'subject',
               'has_tools', 'tools', 'input', 'reference', 'seed', 'manifest_version'}


def lcb_private(tests):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


def fixture_rows():
    """Small rows in each source's real format, keyed like DATASETS[name].files."""
    bfcl = lambda prefix, n: [{'id': f'{prefix}_{i}',
                               'question': [[{'role': 'user', 'content': f'q{i}'}]],
                               'function': [{'name': 'f', 'parameters': {'type': 'dict'}}]}
                              for i in range(n)]
    answers = lambda prefix, n: [{'id': f'{prefix}_{i}', 'ground_truth': [{'f': {'x': [i]}}]}
                                 for i in range(n)]
    return {
        'bfcl_simple': {'questions': bfcl('simple', 50), 'answers': answers('simple', 50)},
        'bfcl_multiple': {'questions': bfcl('multiple', 50), 'answers': answers('multiple', 50)},
        'bfcl_parallel': {'questions': bfcl('parallel', 50), 'answers': answers('parallel', 50)},
        'bfcl_irrelevance': {'questions': bfcl('irrelevance', 50)},
        'humaneval_plus': {'rows': [{'task_id': f'HumanEval/{i}', 'prompt': 'def f():\n',
                                     'canonical_solution': '  pass', 'entry_point': 'f',
                                     'test': 'def check(c): pass'} for i in range(50)]},
        'mbpp_plus': {'rows': [{'task_id': i, 'prompt': 'Write f.', 'code': 'def f(): pass',
                                'source_file': 'x', 'test_imports': [],
                                'test_list': ['assert f() is None'], 'test': 'assert True'}
                               for i in range(50)]},
        'livecodebench': {'rows': [{'question_title': 't', 'question_content': 'c',
                                    'platform': 'atcoder', 'question_id': f'{level}{i}',
                                    'contest_id': 'abc', 'contest_date': '2025-02-01T00:00:00',
                                    'starter_code': '', 'difficulty': level,
                                    'public_test_cases': json.dumps([{'input': '1', 'output': '2',
                                                                      'testtype': 'stdin'}]),
                                    'private_test_cases': lcb_private([{'input': '3', 'output': '4',
                                                                        'testtype': 'stdin'}]),
                                    'metadata': '{}'}
                                   for level, count in (('easy', 35), ('medium', 45), ('hard', 55))
                                   for i in range(count)]},
        'mmlu_pro': {'rows': [{'question_id': i, 'question': 'q', 'options': ['a', 'b', 'c'],
                               'answer': 'B', 'answer_index': 1, 'cot_content': '',
                               'category': f'cat{i % 14}', 'src': 's'} for i in range(14 * 12)]},
        'gpqa_diamond': {'rows': [{'Record ID': f'rec{i}', 'Question': ' What? ',
                                   'Correct Answer': ' right ', 'Incorrect Answer 1': 'w1',
                                   'Incorrect Answer 2': 'w2', 'Incorrect Answer 3': 'w3',
                                   'High-level domain': 'Physics', 'Subdomain': 'Optics'}
                                  for i in range(60)]},
        'arc_challenge': {'rows': [{'id': f'arc{i}', 'question': 'q',
                                    'choices': {'text': ['x', 'y'], 'label': ['A', 'B']},
                                    'answerKey': 'B'} for i in range(40)]},
        'math500': {'rows': [{'problem': 'p', 'solution': 's', 'answer': '42',
                              'subject': 'Algebra', 'level': level,
                              'unique_id': f'test/algebra/{level}_{i}.json'}
                             for level in range(1, 6) for i in range(45)]},
        'ifeval': {'rows': [{'key': 1000 + i, 'prompt': 'Write.',
                             'instruction_id_list': ['punctuation:no_comma'], 'kwargs': [{}]}
                            for i in range(90)]},
        'writingbench_business': {'rows': [
            {'index': str(i), 'domain1': domain, 'domain2': 'Memo', 'lang': lang, 'query': 'Write a memo.',
             'checklist': [{'name': 'Clarity', 'criteria_description': 'Is it clear?'}]}
            for i, (domain, lang) in enumerate(
                [('Finance & Business', 'en')] * 35 + [('Advertising & Marketing', 'en')] * 35
                + [('Finance & Business', 'zh')] * 5 + [('Education', 'en')] * 5)]},
    }


class AllocationTests(unittest.TestCase):
    def test_equal_split_with_seeded_remainder(self):
        sizes = {f'g{i}': 100 for i in range(14)}
        quotas = bd.allocate(120, sizes, 'seed')
        self.assertEqual(sum(quotas.values()), 120)
        self.assertEqual(set(quotas.values()), {8, 9})
        self.assertEqual(quotas, bd.allocate(120, sizes, 'seed'))

    def test_small_group_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, 'g1'):
            bd.allocate(10, {'g0': 100, 'g1': 2}, 'seed')

    def test_too_few_records_fails(self):
        with self.assertRaises(ValueError):
            bd.sample_records([{'source_id': '1'}], 2, 'seed')


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.records = [{'source_id': str(i), 'level': i % 3} for i in range(100)]

    def ids(self, records):
        return [record['source_id'] for record in records]

    def test_same_seed_same_sample_regardless_of_input_order(self):
        first = bd.sample_records(self.records, 10, 'a')
        second = bd.sample_records(list(reversed(self.records)), 10, 'a')
        self.assertEqual(self.ids(first), self.ids(second))
        self.assertNotEqual(set(self.ids(first)), set(self.ids(bd.sample_records(self.records, 10, 'b'))))

    def test_stratified_quota_per_group(self):
        sample = bd.sample_records(self.records, 30, 'a', strata=lambda r: r['level'])
        self.assertEqual([sum(r['level'] == level for r in sample) for level in range(3)], [10, 10, 10])

    def test_take_all(self):
        self.assertEqual(len(bd.sample_records(self.records, 100, 'a')), 100)

    def test_explicit_quotas(self):
        sample = bd.sample_records(self.records, 12, 'a', strata=lambda r: r['level'],
                                   quotas={0: 2, 1: 4, 2: 6})
        self.assertEqual([sum(r['level'] == level for r in sample) for level in range(3)], [2, 4, 6])

    def test_quotas_must_match_n_and_strata(self):
        strata = lambda r: r['level']
        with self.assertRaises(ValueError):
            bd.sample_records(self.records, 10, 'a', strata=strata, quotas={0: 2, 1: 4, 2: 6})
        with self.assertRaises(ValueError):
            bd.sample_records(self.records, 12, 'a', strata=strata, quotas={0: 6, 9: 6})


class NormalizerTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()

    def build(self, name):
        return bd.build_dataset(bd.DATASETS[name], self.rows[name], seed=7)

    def test_every_dataset_builds_expected_count_and_shape(self):
        topics = {topic.value for topic in TopicCategory} | {None}
        for name, spec in bd.DATASETS.items():
            with self.subTest(name):
                records = self.build(name)
                self.assertEqual(len(records), spec.n)
                self.assertEqual(len({r['prompt_id'] for r in records}), spec.n)
                for record in records:
                    self.assertEqual(set(record), RECORD_KEYS)
                    self.assertIn(record['topic'], topics)
                    self.assertIn(record['difficulty'], bd.DIFFICULTIES)
                    self.assertEqual(record['has_tools'], record['tools'] is not None)
                    self.assertEqual(record['seed'], 7)
                    json.dumps(record)  # Serializable without custom encoders.

    def test_categories_group_datasets(self):
        groups = {}
        for name, spec in bd.DATASETS.items():
            groups.setdefault(spec.category, []).append(name)
        self.assertEqual(set(groups), set(bd.CATEGORIES))
        self.assertEqual({c: sum(bd.DATASETS[n].n for n in names) for c, names in groups.items()},
                         {'tool_calling': 160, 'coding': 200, 'general_knowledge': 200,
                          'math': 150, 'writing': 140})
        for name, spec in bd.DATASETS.items():
            self.assertEqual({r['category'] for r in self.build(name)}, {spec.category})

    def test_total_is_850(self):
        self.assertEqual(sum(spec.n for spec in bd.DATASETS.values()), 850)

    def test_bfcl_joins_answers_and_marks_irrelevance(self):
        simple = self.build('bfcl_simple')[0]
        self.assertTrue(simple['has_tools'])
        self.assertIsNone(simple['topic'])
        number = int(simple['source_id'].split('_')[1])
        self.assertEqual(simple['reference'], {'ground_truth': [{'f': {'x': [number]}}]})
        self.assertEqual(simple['input']['messages'], [{'role': 'user', 'content': f'q{number}'}])
        self.assertEqual(self.build('bfcl_irrelevance')[0]['reference'], {'expect_no_call': True})

    def test_livecodebench_strata_and_decoded_private_tests(self):
        records = self.build('livecodebench')
        self.assertEqual(sorted(r['difficulty'] for r in records),
                         ['easy'] * 30 + ['hard'] * 50 + ['medium'] * 40)
        self.assertEqual(records[0]['reference']['private_test_cases'],
                         [{'input': '3', 'output': '4', 'testtype': 'stdin'}])

    def test_lcb_decoder_rejects_pickled_objects(self):
        payload = base64.b64encode(zlib.compress(pickle.dumps(ValueError('x')))).decode()
        with self.assertRaises(pickle.UnpicklingError):
            bd.decode_lcb_tests(payload)

    def test_mmlu_pro_stratified_by_category(self):
        records = self.build('mmlu_pro')
        counts = {}
        for record in records:
            counts[record['subject']] = counts.get(record['subject'], 0) + 1
        self.assertEqual(len(counts), 14)
        self.assertEqual(set(counts.values()), {8, 9})

    def test_gpqa_shuffles_choices_reproducibly(self):
        records = self.build('gpqa_diamond')
        for record in records:
            choices, letter = record['input']['choices'], record['reference']['answer']
            self.assertEqual(sorted(choices), ['right', 'w1', 'w2', 'w3'])
            self.assertEqual(choices['ABCD'.index(letter)], 'right')
            self.assertEqual(record['input']['question'], 'What?')
        self.assertEqual(records, self.build('gpqa_diamond'))
        self.assertGreater(len({r['reference']['answer'] for r in records}), 1)

    def test_math500_levels_and_difficulty(self):
        records = self.build('math500')
        self.assertEqual([sum(r['level_label'] == f'Level {i}' for r in records) for i in range(1, 6)],
                         [20, 20, 30, 40, 40])
        self.assertEqual({r['difficulty'] for r in records if r['level_label'] == 'Level 5'}, {'hard'})

    def test_writingbench_english_business_only_with_rubric(self):
        records = self.build('writingbench_business')
        self.assertEqual(sorted({r['subject'] for r in records}),
                         ['Advertising & Marketing', 'Finance & Business'])
        self.assertEqual(sum(r['subject'] == 'Finance & Business' for r in records), 30)
        self.assertEqual(records[0]['reference']['checklist'][0]['name'], 'Clarity')
        self.assertEqual(records[0]['topic'], TopicCategory.CREATIVE_WRITING.value)


class OutputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def test_jsonl_is_byte_identical_and_sorted(self):
        records = bd.build_dataset(bd.DATASETS['arc_challenge'], fixture_rows()['arc_challenge'], seed=1)
        first, second = self.dir / 'a.jsonl', self.dir / 'b.jsonl'
        bd.write_jsonl(first, records)
        bd.write_jsonl(second, list(reversed(records)))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        ids = [json.loads(line)['prompt_id'] for line in first.read_text().splitlines()]
        self.assertEqual(ids, sorted(ids))

    def test_readers_by_format(self):
        (self.dir / 'r.jsonl').write_text('{"a": 1}\n\n{"a": 2}\n')
        (self.dir / 'r.json').write_text('{"a": 1}\n')  # BFCL ships JSON Lines as .json.
        (self.dir / 'r.csv').write_text('a,b\n1,2\n')
        self.assertEqual(bd.read_rows(self.dir / 'r.jsonl'), [{'a': 1}, {'a': 2}])
        self.assertEqual(bd.read_rows(self.dir / 'r.json'), [{'a': 1}])
        self.assertEqual(bd.read_rows(self.dir / 'r.csv'), [{'a': '1', 'b': '2'}])

    def write_raw(self, name):
        """Write fixture rows as raw files the way download() would return them."""
        paths = {}
        for logical, rows in fixture_rows()[name].items():
            path = self.dir / 'raw' / name / f'{logical}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            paths[logical] = path
        return paths

    def test_run_writes_prompts_and_manifest_then_check_passes(self):
        paths = self.write_raw('arc_challenge')
        out = self.dir / 'out'
        with patch.object(bd, 'download', return_value=paths) as download:
            manifest = bd.run(['arc_challenge'], out, seed=3)
            self.assertEqual(download.call_count, 1)
            entry = manifest['datasets']['arc_challenge']
            self.assertEqual(entry['n'], 30)
            self.assertEqual(entry['revision'], bd.DATASETS['arc_challenge'].revision)
            self.assertEqual(len(entry['source_ids']), 30)
            self.assertEqual(entry['output_sha256'], bd.sha256(out / 'prompts' / 'general_knowledge' / 'arc_challenge.jsonl'))
            self.assertEqual(set(entry['raw_files']), {'rows'})
            self.assertEqual(entry['category'], 'general_knowledge')
            self.assertEqual(entry['output_file'], 'prompts/general_knowledge/arc_challenge.jsonl')
            saved = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(saved, manifest)
            self.assertEqual(bd.run(['arc_challenge'], out, seed=3, check=True), manifest)

    def test_check_detects_changed_raw_file(self):
        paths = self.write_raw('arc_challenge')
        out = self.dir / 'out'
        with patch.object(bd, 'download', return_value=paths):
            bd.run(['arc_challenge'], out, seed=3)
            paths['rows'].write_text(paths['rows'].read_text().replace('"answerKey": "B"', '"answerKey": "C"'))
            with self.assertRaisesRegex(ValueError, 'arc_challenge'):
                bd.run(['arc_challenge'], out, seed=3, check=True)

    def test_run_keeps_other_manifest_entries(self):
        out = self.dir / 'out'
        for name in ('arc_challenge', 'math500'):
            with patch.object(bd, 'download', return_value=self.write_raw(name)):
                bd.run([name], out, seed=3)
        saved = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(sorted(saved['datasets']), ['arc_challenge', 'math500'])

    def test_run_drops_datasets_no_longer_defined(self):
        out = self.dir / 'out'
        out.mkdir()
        (out / 'manifest.json').write_text(json.dumps(
            {'manifest_version': 1, 'datasets': {'aime_2026': {}}, 'skipped': {'aime_2026': 'x'}}))
        with patch.object(bd, 'download', return_value=self.write_raw('arc_challenge')):
            manifest = bd.run(['arc_challenge'], out, seed=3)
        self.assertEqual(list(manifest['datasets']), ['arc_challenge'])
        self.assertEqual(manifest['skipped'], {})

    def test_gated_download_is_reported_not_fatal(self):
        out = self.dir / 'out'
        paths = self.write_raw('arc_challenge')

        def fake_download(spec, raw_dir):
            if spec.name == 'gpqa_diamond':
                raise bd.ManualStepRequired('accept the terms')
            return paths

        with patch.object(bd, 'download', side_effect=fake_download):
            manifest = bd.run(['gpqa_diamond', 'arc_challenge'], out, seed=3)
        self.assertEqual(list(manifest['datasets']), ['arc_challenge'])
        self.assertEqual(manifest['skipped'], {'gpqa_diamond': 'accept the terms'})


if __name__ == '__main__':
    unittest.main()
