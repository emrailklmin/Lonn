"""WritingBench LLM judge: prompt, parsing, skipping, budget and applying scores. No network."""
import json
from pathlib import Path
import re
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
        self.assertEqual((body['temperature'], body['reasoning']), (0.0, {'max_tokens': judge.REASONING_TOKENS}))
        self.assertGreater(body['max_tokens'], judge.REASONING_TOKENS)
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



class FakeBatchAPI:
    def __init__(self):
        self.created, self.status, self.results, self.billed = [], 'in_progress', None, None

    def create(self, payload):
        self.created.append(payload)
        return {'id': f'batch-{len(self.created)}', 'status': 'validating'}

    def get(self, batch_id):
        return {'id': batch_id, 'status': self.status, 'results': self.results,
                'request_counts': {'total': 2, 'completed': 2 if self.results else 0, 'failed': 0},
                'usage': {'cost': self.billed}}


def batch_item(custom_id, scores, prompt_tokens=3000, completion_tokens=1000):
    reply = judge_reply(scores)
    reply['usage'] = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}
    return {'id': 'r', 'custom_id': custom_id, 'error': None,
            'response': {'status_code': 200, 'request_id': 'q', 'body': reply}}


class BatchTests(JudgeRunTests):
    def submit(self, api, **kwargs):
        options = dict(limit=None, budget=45.0, seed=1)
        options.update(kwargs)
        return judge.submit_batch(api, self.records, self.results_dir, self.judgments_dir, **options)

    def collect(self, api):
        return judge.collect_batches(api, self.records, self.results_dir, self.judgments_dir)

    def test_submit_sends_one_batch_with_safe_ids_and_scores_empty_answers(self):
        api = FakeBatchAPI()
        self.assertEqual(self.submit(api), 'batch-1')
        payload = api.created[0]
        self.assertEqual(list(payload)[-1], 'requests')  # The API rejects 'requests' first.
        self.assertEqual((payload['model'], payload['endpoint']), (judge.JUDGE_MODEL, '/v1/chat/completions'))
        ids = [r['custom_id'] for r in payload['requests']]
        self.assertEqual(ids, ['j0001', 'j0002'])  # The empty answer (002) is not sent.
        self.assertTrue(all(re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids))
        self.assertNotIn('model', payload['requests'][0]['body'])
        self.assertEqual(self.rows()['writingbench_business-002']['judge_mean'], 1.0)
        # Answers waiting in the open batch are not sent again.
        self.assertIsNone(self.submit(api))
        self.assertEqual(len(api.created), 1)

    def test_collect_waits_then_saves_judgments_with_billed_cost(self):
        api = FakeBatchAPI()
        self.submit(api)
        self.assertEqual(self.collect(api), ['in_progress'])
        self.assertIsNone(self.rows()['writingbench_business-001']['correct'])
        api.status, api.billed = 'completed', 0.05
        api.results = [batch_item('j0001', [8] * 5, 3000, 1000), batch_item('j0002', [4] * 5, 1000, 500)]
        self.assertEqual(self.collect(api), [])
        rows = self.rows()
        self.assertEqual((rows['writingbench_business-001']['correct'], rows['writingbench_business-003']['correct']),
                         (True, False))
        costs = [rows[f'writingbench_business-00{i}']['judge_cost_usd'] for i in (1, 3)]
        self.assertAlmostEqual(sum(costs), 0.05)  # The billed total, split by tokens.
        self.assertGreater(costs[0], costs[1])
        self.assertEqual(self.collect(api), [])  # Collected batches are not saved twice.
        self.assertEqual(len(judge.read_judgments(self.judgments_dir)), 3)

    def test_failed_items_and_failed_batches_are_retried_later(self):
        api = FakeBatchAPI()
        self.submit(api)
        api.status, api.billed = 'completed', 0.02
        api.results = [batch_item('j0001', [8] * 5),
                       {'id': 'r', 'custom_id': 'j0002', 'error': {'message': 'overloaded'}, 'response': None}]
        self.collect(api)
        self.assertIsNone(self.rows()['writingbench_business-003']['correct'])
        self.submit(api)
        self.assertEqual([r['custom_id'] for r in api.created[1]['requests']], ['j0001'])
        api.status, api.results = 'expired', None
        self.collect(api)
        saved = judge.read_judgments(self.judgments_dir)
        self.assertIn('expired', saved[('writingbench_business-003', MODEL_ID)]['error'])

    def test_budget_refuses_a_batch_that_could_overspend(self):
        api = FakeBatchAPI()
        with self.assertRaises(SystemExit):
            self.submit(api, budget=0.031)
        self.assertEqual(api.created, [])


if __name__ == '__main__':
    unittest.main()
