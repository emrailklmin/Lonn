"""WritingBench LLM judge: prompt, parsing, skipping, budget and applying scores. No network."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from benchmark import judge, results, run  # noqa: E402

MODEL_ID = 'openrouter/test/writer'
CRITERIA = [{'name': f'Criterion {i}', 'criteria_description': f'Is aspect {i} good?',
             **{band: f'band {band}' for band in ('1-2', '3-4', '5-6', '7-8', '9-10')}}
            for i in range(1, 6)]


def record(i):
    return {'prompt_id': f'writingbench_business-{i:03d}', 'dataset': 'writingbench_business',
            'category': 'writing', 'difficulty': 'medium', 'input': {'query': f'Write memo {i}.'},
            'reference': {'checklist': CRITERIA}}


def answer_row(i, response='A fine memo.', finish='stop', model_id=MODEL_ID):
    return {'prompt_id': f'writingbench_business-{i:03d}', 'dataset': 'writingbench_business',
            'category': 'writing', 'difficulty': 'medium', 'model_id': model_id,
            'response': response, 'finish_reason': finish, 'error': None, 'correct': None,
            'grader': 'deferred', 'cost_usd': 0.01, 'output_tokens': 100}


def judge_reply(scores, cost=0.02):
    text = json.dumps({'scores': [{'criterion': c['name'], 'score': s, 'reason': 'Because.'}
                                  for c, s in zip(CRITERIA, scores)]})
    return {'id': 'gen-j', 'choices': [{'message': {'content': f'```json\n{text}\n```'},
                                        'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 3000, 'completion_tokens': 200, 'cost': cost}}


class FakeClient:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def complete(self, body):
        self.bodies.append(body)
        reply = self.replies.pop(0) if len(self.replies) > 1 or not isinstance(self.replies[0], Exception) \
            else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


class PromptAndParsingTests(unittest.TestCase):
    def test_request_is_blind_strict_and_cheap(self):
        body = judge.build_request(record(1), 'My memo text.', truncated=True)
        text = '\n'.join(m['content'] for m in body['messages'])
        for needle in ('Criterion 1', 'Is aspect 5 good?', 'band 9-10', 'Write memo 1.', 'My memo text.',
                       'STRICT', 'cut off'):
            self.assertIn(needle, text)
        self.assertNotIn('test/writer', text)  # The judge never sees which model wrote it.
        self.assertEqual(body['model'], judge.JUDGE_MODEL)
        self.assertEqual((body['temperature'], body['reasoning']), (0.0, {'enabled': False}))
        self.assertNotIn('cut off', '\n'.join(m['content'] for m in
                                              judge.build_request(record(1), 'x', False)['messages']))

    def test_parse_valid_scores_in_order(self):
        reply = judge_reply([7, 8, 9, 6, 10])['choices'][0]['message']['content']
        scores = judge.parse_scores(reply, CRITERIA)
        self.assertEqual([s['score'] for s in scores], [7, 8, 9, 6, 10])
        self.assertEqual(scores[0]['criterion'], 'Criterion 1')

    def test_parse_rejects_bad_output(self):
        bad = ['no json here', json.dumps({'scores': [{'score': 5}] * 4}),
               json.dumps({'scores': [{'score': 11}] * 5}), json.dumps({'scores': [{'score': 'high'}] * 5})]
        for text in bad:
            with self.assertRaises(ValueError, msg=text):
                judge.parse_scores(text, CRITERIA)

    def test_pass_threshold(self):
        self.assertTrue(judge.verdict([7, 7, 7, 7, 7])['correct'])
        self.assertFalse(judge.verdict([7, 7, 7, 7, 6])['correct'])
        self.assertEqual(judge.verdict([10] * 5)['score'], 1.0)


class JudgeRunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.results_dir = Path(directory.name) / 'benchmark'
        self.judgments_dir = Path(directory.name) / 'judgments'
        self.path = results.result_path(self.results_dir, 'writing', 'writingbench_business', MODEL_ID)
        for i, response in ((1, 'Memo one.'), (2, ''), (3, 'Memo three.')):
            results.append_row(self.path, answer_row(i, response))
        self.records = [record(i) for i in (1, 2, 3)]
        for target in ('builtins.print', 'benchmark.run.time.sleep'):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_judge(self, client, **kwargs):
        options = dict(limit=None, budget=45.0, workers=1, seed=1)
        options.update(kwargs)
        return judge.run_judge(client, self.records, self.results_dir, self.judgments_dir, **options)

    def rows(self):
        return results.read_rows(self.path)

    def test_judges_answers_skips_empty_and_applies_scores(self):
        client = FakeClient([judge_reply([8, 8, 8, 8, 8]), judge_reply([5, 6, 5, 6, 5])])
        self.run_judge(client)
        self.assertEqual(len(client.bodies), 2)  # The empty answer is not sent.
        rows = self.rows()
        self.assertEqual([rows[f'writingbench_business-00{i}']['correct'] for i in (1, 2, 3)],
                         [True, False, False])
        empty = rows['writingbench_business-002']
        self.assertEqual((empty['judge_mean'], empty['grade_details']), (1.0, 'empty response'))
        first = rows['writingbench_business-001']
        self.assertEqual((first['grader'], first['judge_model'], first['judge_cost_usd']),
                         ('llm_judge', judge.JUDGE_MODEL, 0.02))
        self.assertEqual(len(first['judge_scores']), 5)
        self.assertAlmostEqual(results.ResultBank(self.results_dir).spent(), 0.03 + 0.04)

    def test_resume_skips_judged_and_retries_errors(self):
        flaky = run.CallError('HTTP 503', 503, retryable=True)
        self.run_judge(FakeClient([judge_reply([9] * 5), flaky]))
        client = FakeClient([judge_reply([9] * 5)])
        self.run_judge(client)
        self.assertEqual(len(client.bodies), 1)
        self.assertTrue(all(row['correct'] is not None for row in self.rows().values()))

    def test_unparseable_judgment_is_an_error_not_a_score(self):
        broken = judge_reply([8] * 5)
        broken['choices'][0]['message']['content'] = 'I think it is good.'
        self.run_judge(FakeClient([broken, judge_reply([8] * 5)]))
        self.assertIsNone(self.rows()['writingbench_business-001']['correct'])
        saved = judge.read_judgments(self.judgments_dir)
        self.assertIn('unparseable', saved[('writingbench_business-001', MODEL_ID)]['error'])

    def test_budget_stops_before_overspending(self):
        client = FakeClient([judge_reply([8] * 5, cost=0.02)] * 2)
        stopped = self.run_judge(client, budget=0.03 + judge.reserve(1), workers=1)
        self.assertIn('budget', stopped)
        self.assertEqual(len(client.bodies), 0)

    def test_limit_judges_the_same_seeded_subset(self):
        client = FakeClient([judge_reply([8] * 5)] * 3)
        self.run_judge(client, limit=1)
        self.assertLessEqual(len(client.bodies), 1)

    def test_regrade_keeps_judge_scores(self):
        self.run_judge(FakeClient([judge_reply([8] * 5), judge_reply([8] * 5)]))
        with patch.object(run, 'load_records', return_value={'writingbench_business': self.records}), \
                patch.object(judge, 'JUDGMENTS_DIR', self.judgments_dir):
            run.regrade(None, self.results_dir, ['writingbench_business'])
        self.assertTrue(self.rows()['writingbench_business-001']['correct'])
        self.assertEqual(self.rows()['writingbench_business-001']['grader'], 'llm_judge')


if __name__ == '__main__':
    unittest.main()
