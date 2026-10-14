"""Benchmark pipeline: templates, graders, answer bank and runner. No network, no cost."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from benchmark import graders, results, run, templates  # noqa: E402
from lonn.schemas import Model, TopicCategory  # noqa: E402

MODEL = Model('openrouter/test/cheap', 'openrouter', 1.0, 2.0, 100000, {TopicCategory.CODING: 1.0})
SETTINGS = {'max_tokens': 8192, 'temperature': 0.0, 'seed': 1, 'workers': 2}


def rec(dataset, prompt_id='x-001', **fields):
    category = {'mmlu_pro': 'general_knowledge', 'arc_challenge': 'general_knowledge',
                'math500': 'math', 'humaneval_plus': 'coding', 'mbpp_plus': 'coding',
                'livecodebench': 'coding', 'ifeval': 'writing', 'writingbench_business': 'writing'
                }.get(dataset, 'tool_calling')
    return {'prompt_id': prompt_id, 'dataset': dataset, 'category': category,
            'difficulty': 'easy', 'tools': None, **fields}


def mc(answer='B', **extra):
    return rec('mmlu_pro', input={'question': 'Pick one', 'choices': ['a', 'b', 'c'], **extra},
               reference={'answer': answer})


def bfcl(ground_truth=None, functions=None, dataset='bfcl_simple'):
    reference = {'ground_truth': ground_truth} if ground_truth is not None else {'expect_no_call': True}
    return rec(dataset, input={'messages': [{'role': 'user', 'content': 'q'}]},
               tools=functions or [{'name': 'math.area', 'description': 'd',
                                    'parameters': {'type': 'dict', 'properties': {
                                        'base': {'type': 'float'}, 'unit': {'type': 'string'},
                                        'extra': {'type': 'any'}}, 'required': ['base']}}],
               reference=reference)


def call(name, **arguments):
    return {'name': name, 'arguments': json.dumps(arguments)}


def code(body):
    return f'```python\n{body}\n```'


class TemplateTests(unittest.TestCase):
    def test_every_dataset_has_a_family_and_system_prompt(self):
        from benchmark.build_data import DATASETS
        self.assertEqual(set(templates.FAMILY_OF), set(DATASETS))
        self.assertEqual(set(templates.SYSTEM_PROMPTS), set(templates.FAMILIES))

    def test_multiple_choice_message_and_shared_system_prompt(self):
        request = templates.build_request(mc())
        self.assertIn('Answer: X', request['messages'][0]['content'])
        self.assertEqual(request['messages'][1]['content'], 'Question: Pick one\n\nOptions:\nA) a\nB) b\nC) c')
        self.assertIsNone(request['tools'])
        gpqa = dict(mc(), dataset='gpqa_diamond')
        self.assertEqual(templates.build_request(gpqa)['messages'][0], request['messages'][0])

    def test_arc_keeps_its_own_labels(self):
        arc = rec('arc_challenge', input={'question': 'q', 'choices': ['x', 'y'], 'labels': ['1', '2']})
        self.assertIn('1) x\n2) y', templates.build_request(arc)['messages'][1]['content'])

    def test_bfcl_tools_are_json_schema_with_safe_names(self):
        request = templates.build_request(bfcl([]))
        tool = request['tools'][0]['function']
        self.assertEqual(tool['name'], 'math_area')
        self.assertEqual(tool['parameters']['type'], 'object')
        self.assertEqual(tool['parameters']['properties']['base'], {'type': 'number'})
        self.assertEqual(tool['parameters']['properties']['extra'], {})
        self.assertEqual(request['messages'][1], {'role': 'user', 'content': 'q'})

    def test_livecodebench_stdin_and_functional_instructions(self):
        base = {'title': 'T', 'content': 'C', 'platform': 'p', 'contest_date': 'd'}
        stdin = rec('livecodebench', input={**base, 'starter_code': ''})
        functional = rec('livecodebench', input={**base, 'starter_code': 'class Solution:\n    def f(self):'})
        self.assertIn('standard input', templates.build_request(stdin)['messages'][1]['content'])
        self.assertIn('class Solution', templates.build_request(functional)['messages'][1]['content'])


class MultipleChoiceAndMathTests(unittest.TestCase):
    def grade(self, record, text):
        return graders.grade(record, {'response': text})

    def test_answer_line_variants(self):
        self.assertTrue(self.grade(mc(), 'Reasoning...\nAnswer: B')['correct'])
        self.assertTrue(self.grade(mc('C'), '**Answer:** (C)')['correct'])
        self.assertTrue(self.grade(mc(), 'Answer: A\nWait, no.\nAnswer: B')['correct'])
        missing = self.grade(mc(), 'I think it is B.')
        self.assertFalse(missing['correct'])
        self.assertIsNone(missing['extracted_answer'])

    def test_last_boxed_handles_nested_braces(self):
        self.assertEqual(graders.last_boxed('x \\boxed{1} then \\boxed{\\frac{1}{2}}'), '\\frac{1}{2}')
        self.assertIsNone(graders.last_boxed('no box'))

    def test_math_equivalence(self):
        record = rec('math500', input={'problem': 'p'}, reference={'answer': '\\frac{1}{2}'})
        self.assertTrue(self.grade(record, 'so \\boxed{0.5}')['correct'])
        self.assertTrue(self.grade(record, '\\boxed{\\dfrac12}')['correct'])
        self.assertFalse(self.grade(record, '\\boxed{3}')['correct'])
        self.assertFalse(self.grade(record, 'the answer is 1/2')['correct'])


class CodeTests(unittest.TestCase):
    humaneval = rec('humaneval_plus', input={'prompt': 'def add(a, b):\n', 'entry_point': 'add'},
                    reference={'entry_point': 'add', 'test': 'def check(c):\n    assert c(2, 3) == 5\n'})

    def grade(self, record, body):
        return graders.grade(record, {'response': code(body)})

    def test_humaneval_pass_and_fail(self):
        self.assertTrue(self.grade(self.humaneval, 'def add(a, b):\n    return a + b')['correct'])
        failed = self.grade(self.humaneval, 'def add(a, b):\n    return a - b')
        self.assertFalse(failed['correct'])
        self.assertIn('AssertionError', failed['grade_details'])
        self.assertFalse(graders.grade(self.humaneval, {'response': 'return a + b'})['correct'])

    def test_sandbox_blocks_network_home_reads_and_outside_writes(self):
        probe = ('import socket, pathlib\n'
                 'for action in (lambda: socket.create_connection(("1.1.1.1", 80), timeout=2),\n'
                 f'               lambda: open({str(ROOT / "README.md")!r}).read(1),\n'
                 '               lambda: open("/private/var/tmp/lonn-escape.txt", "w")):\n'
                 '    try:\n        action(); print("ALLOWED")\n'
                 '    except OSError:\n        print("blocked")\n')
        with tempfile.TemporaryDirectory() as workdir:
            code_rc, out, err = graders.run_sandboxed(probe, workdir, timeout=20)
        self.assertEqual((code_rc, out.split()), (0, ['blocked'] * 3), err)
        self.assertFalse(Path('/private/var/tmp/lonn-escape.txt').exists())

    def test_sandbox_timeout_kills_the_program(self):
        with tempfile.TemporaryDirectory() as workdir:
            self.assertEqual(graders.run_sandboxed('while True: pass', workdir, timeout=1)[0], None)

    def test_livecodebench_stdin(self):
        record = rec('livecodebench', input={'starter_code': ''}, reference={
            'public_test_cases': [{'input': '2 3\n', 'output': '5\n', 'testtype': 'stdin'}],
            'private_test_cases': [{'input': '1.5 1\n', 'output': '2.5000001', 'testtype': 'stdin'}],
            'metadata': {}})
        self.assertTrue(self.grade(record, 'a, b = map(float, input().split())\n'
                                           'r = a + b\nprint(int(r) if r == int(r) else r)')['correct'])
        wrong = self.grade(record, 'print(0)')
        self.assertEqual((wrong['correct'], wrong['grade_details']), (False, 'test 1: wrong output'))

    def test_livecodebench_many_fast_tests_fit_the_total_limit(self):
        # 45 tests x the 6 s allowance exceeds 120 s; only real run time may count.
        tests = [{'input': f'{i}\n', 'output': f'{i}\n', 'testtype': 'stdin'} for i in range(45)]
        record = rec('livecodebench', input={'starter_code': ''}, reference={
            'public_test_cases': tests, 'private_test_cases': [], 'metadata': {}})
        graded = self.grade(record, 'print(input())')
        self.assertTrue(graded['correct'], graded['grade_details'])

    def test_livecodebench_functional(self):
        record = rec('livecodebench', input={'starter_code': 'class Solution:'}, reference={
            'public_test_cases': [{'input': '[1, 2, 3]\n2', 'output': '[2, 4, 6]', 'testtype': 'functional'}],
            'private_test_cases': [], 'metadata': {'func_name': 'scale'}})
        body = 'class Solution:\n    def scale(self, nums: List[int], k: int) -> List[int]:\n        return {}'
        self.assertTrue(self.grade(record, body.format('[n * k for n in nums]'))['correct'])
        self.assertFalse(self.grade(record, body.format('nums'))['correct'])


class ToolCallingTests(unittest.TestCase):
    truth = [{'math.area': {'base': [10, 10.0], 'unit': ['units', ''], 'extra': ['']}}]

    def grade(self, record, calls):
        return graders.grade(record, {'response': '', 'tool_calls': calls})

    def test_match_with_normalized_strings_and_optional_params(self):
        self.assertTrue(self.grade(bfcl(self.truth), [call('math_area', base=10, unit='Units')])['correct'])
        self.assertTrue(self.grade(bfcl(self.truth), [call('math_area', base=10.0)])['correct'])

    def test_wrong_value_extra_param_or_wrong_count(self):
        for calls in ([call('math_area', base=11)], [call('math_area', base=10, color='red')],
                      [], [call('math_area', base=10)] * 2, [call('other', base=10)]):
            self.assertFalse(self.grade(bfcl(self.truth), calls)['correct'], calls)

    def test_parallel_calls_match_in_any_order(self):
        truth = [{'f': {'city': ['Oslo']}}, {'f': {'city': ['Rome']}}]
        record = bfcl(truth, functions=[{'name': 'f', 'parameters': {'type': 'dict', 'properties': {}}}],
                      dataset='bfcl_parallel')
        self.assertTrue(self.grade(record, [call('f', city='Rome'), call('f', city='oslo')])['correct'])
        self.assertFalse(self.grade(record, [call('f', city='Rome'), call('f', city='Rome')])['correct'])

    def test_irrelevance_wants_no_call(self):
        record = bfcl(dataset='bfcl_irrelevance')
        self.assertTrue(self.grade(record, None)['correct'])
        self.assertFalse(self.grade(record, [call('math_area', base=1)])['correct'])

    def test_unparseable_arguments_fail(self):
        bad = [{'name': 'math_area', 'arguments': '{base: 10'}]
        self.assertFalse(self.grade(bfcl(self.truth), bad)['correct'])


class InstructionAndWritingTests(unittest.TestCase):
    def test_ifeval_strict_prompt_level(self):
        record = rec('ifeval', input={'prompt': 'Write without commas.'}, reference={
            'instruction_id_list': ['punctuation:no_comma'], 'kwargs': [{}]})
        passed = graders.grade(record, {'response': 'No commas here.'})
        self.assertEqual((passed['correct'], passed['extracted_answer']), (True, {'punctuation:no_comma': True}))
        self.assertFalse(graders.grade(record, {'response': 'Well, no.'})['correct'])
        self.assertFalse(graders.grade(record, {'response': '  '})['correct'])

    def test_writing_is_deferred(self):
        graded = graders.grade(rec('writingbench_business', input={'query': 'q'}, reference={}),
                               {'response': 'text'})
        self.assertEqual((graded['correct'], graded['grader']), (None, 'deferred'))


class GraderEdgeCaseTests(unittest.TestCase):
    """Cases found in the grader review (2026-10-12)."""
    humaneval = CodeTests.humaneval
    stdin_record = rec('livecodebench', input={'starter_code': ''}, reference={
        'public_test_cases': [{'input': '1\n', 'output': '1000000\n', 'testtype': 'stdin'}],
        'private_test_cases': [], 'metadata': {}})

    def grade(self, record, response, **answer):
        return graders.grade(record, {'response': response, **answer})

    # False passes: wrong answers must not be graded correct.
    def test_exit_zero_before_tests_is_not_a_pass(self):
        for body in ('import sys\nsys.exit(0)', 'import os\nos._exit(0)', 'exit()', 'raise SystemExit'):
            with self.subTest(body):
                self.assertFalse(self.grade(self.humaneval, code(f'def add(a, b):\n    return 0\n{body}'))['correct'])
        mbpp = rec('mbpp_plus', input={}, reference={'test_imports': [], 'test': 'assert add(2, 3) == 5'})
        self.assertFalse(self.grade(mbpp, code('def add(a, b):\n    return 0\nexit(0)'))['correct'])
        functional = rec('livecodebench', input={'starter_code': 'class Solution:'}, reference={
            'public_test_cases': [{'input': '1', 'output': '2', 'testtype': 'functional'}],
            'private_test_cases': [], 'metadata': {'func_name': 'f'}})
        body = 'class Solution:\n    def f(self, x):\n        return 0\nimport sys\nsys.exit(0)'
        self.assertFalse(self.grade(functional, code(body))['correct'])

    def test_integer_outputs_need_exact_match(self):
        self.assertFalse(self.grade(self.stdin_record, code('print(1000001)'))['correct'])
        self.assertFalse(graders._outputs_match('123456789012345678', '123456789012345679'))
        self.assertTrue(graders._outputs_match('0.3333333', '0.33333333'))
        self.assertFalse(graders._outputs_match('nan', 'nan1'))

    def test_parallel_matching_is_not_greedy(self):
        truth = [{'f': {'x': [1, 2]}}, {'f': {'x': [1]}}]
        record = bfcl(truth, functions=[{'name': 'f', 'parameters': {'type': 'dict', 'properties': {}}}],
                      dataset='bfcl_parallel')
        self.assertTrue(self.grade(record, '', tool_calls=[call('f', x=1), call('f', x=2)])['correct'])

    # False failures: right answers in reasonable shapes.
    def test_think_tags_in_the_response_are_ignored(self):
        ifeval = rec('ifeval', input={'prompt': 'p'}, reference={
            'instruction_id_list': ['punctuation:no_comma'], 'kwargs': [{}]})
        self.assertTrue(self.grade(ifeval, '<think>Well, hmm, ok.</think>\nNo commas here.')['correct'])
        self.assertTrue(self.grade(mc(), '<think>Answer: A</think>Answer: B')['correct'])
        self.assertFalse(self.grade(mc(), '<think>Answer: B')['correct'])  # Unfinished thinking.

    def test_final_answer_and_list_prefixes(self):
        for text in ('Final Answer: B', 'The final answer: B', '- Answer: B', '**Final answer:** B'):
            self.assertTrue(self.grade(mc(), f'Reasoning\n{text}')['correct'], text)
        self.assertFalse(self.grade(mc(), 'The answer: B is wrong, Answer: C')['correct'])

    def test_main_guard_and_body_only_humaneval(self):
        guarded = 'def add(a, b):\n    return a + b\nif __name__ == "__main__":\n    print(add(int(input()), 1))'
        self.assertTrue(self.grade(self.humaneval, code(guarded))['correct'])
        self.assertTrue(self.grade(self.humaneval, code('    return a + b'))['correct'])

    def test_functional_inputs_with_trailing_newline(self):
        record = rec('livecodebench', input={'starter_code': 'class Solution:'}, reference={
            'public_test_cases': [{'input': '[1, 2]\n3\n', 'output': '6', 'testtype': 'functional'}],
            'private_test_cases': [], 'metadata': {'func_name': 'f'}})
        body = 'class Solution:\n    def f(self, nums, k):\n        return sum(nums) * 2'
        self.assertTrue(self.grade(record, code(body))['correct'])

    def test_tool_arguments_as_dict_or_non_object(self):
        truth = [{'math.area': {'base': [10]}}]
        as_dict = [{'name': 'math_area', 'arguments': {'base': 10}}]
        self.assertTrue(self.grade(bfcl(truth), '', tool_calls=as_dict)['correct'])
        as_list = [{'name': 'math_area', 'arguments': '[10]'}]
        self.assertFalse(self.grade(bfcl(truth), '', tool_calls=as_list)['correct'])

    # Robustness of the sandbox.
    def test_output_flood_is_cut_off(self):
        with tempfile.TemporaryDirectory() as workdir:
            code_rc, out, _ = graders.run_sandboxed('while True: print("x" * 10000)', workdir, timeout=10)
        self.assertNotEqual(code_rc, 0)
        self.assertLessEqual(len(out), graders.MAX_OUTPUT_BYTES)

    def test_fork_bomb_is_contained(self):
        with tempfile.TemporaryDirectory() as workdir:
            code_rc, out, _ = graders.run_sandboxed(
                'import os\ntry:\n    os.fork(); print("forked")\nexcept OSError:\n    print("blocked")',
                workdir, timeout=5)
        self.assertEqual(out.split(), ['blocked'])

    def test_math_number_with_thousands_separator(self):
        record = rec('math500', input={'problem': 'p'}, reference={'answer': '10,\\!080'})
        for prediction in ('10080', '10,080', '10{,}080'):
            self.assertTrue(self.grade(record, f'\\boxed{{{prediction}}}')['correct'], prediction)


class ResultBankTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def test_append_latest_wins_and_partial_line_skipped(self):
        path = results.result_path(self.dir, 'math', 'math500', 'openrouter/openai/gpt-oss-20b')
        self.assertEqual(path.name, 'openai__gpt-oss-20b.jsonl')
        base = {'prompt_id': 'math500-001', 'model_id': 'openrouter/openai/gpt-oss-20b',
                'dataset': 'math500', 'category': 'math'}
        results.append_row(path, {**base, 'error': {'message': 'boom'}, 'cost_usd': 0.0})
        results.append_row(path, {**base, 'error': None, 'correct': True, 'cost_usd': 0.25})
        with path.open('a') as handle:
            handle.write('{"prompt_id": "math500-0')  # Crash mid-write.
        bank = results.ResultBank(self.dir)
        self.assertTrue(bank.lookup('math500-001', 'openrouter/openai/gpt-oss-20b')['correct'])
        self.assertEqual(bank.spent(), 0.25)
        summary = bank.summary(by=('model_id',))
        self.assertEqual((summary[0]['answers'], summary[0]['accuracy']), (1, 1.0))


def openrouter_reply(content='Answer: B', cost=0.001, tool_calls=None, finish='stop'):
    message = {'role': 'assistant', 'content': content, 'reasoning': 'thinking...'}
    if tool_calls:
        message['tool_calls'] = [{'id': 'c1', 'type': 'function',
                                  'function': {'name': n, 'arguments': a}} for n, a in tool_calls]
    return {'id': 'gen-1', 'provider': 'Prov', 'choices': [
        {'message': message, 'finish_reason': finish, 'native_finish_reason': finish}],
        'usage': {'prompt_tokens': 50, 'completion_tokens': 20, 'cost': cost,
                  'completion_tokens_details': {'reasoning_tokens': 5}}}


class OpenRouterClientTests(unittest.TestCase):
    def client(self, handler, deadline=900.0):
        import httpx
        return run.OpenRouterClient('key', timeout=5, deadline=deadline,
                                    transport=httpx.MockTransport(handler))

    def test_empty_reply_without_usage_is_not_retryable(self):
        import httpx
        reply = openrouter_reply(content='')
        del reply['usage']
        client = self.client(lambda request: httpx.Response(200, content=json.dumps(reply).encode()))
        with self.assertRaises(run.CallError) as caught:
            client.complete({'messages': [], 'tools': None})
        self.assertEqual((caught.exception.retryable, caught.exception.kind), (False, 'filtered'))

    def test_ok_reply_with_keepalive_whitespace(self):
        import httpx
        body = ('\n\n' + json.dumps(openrouter_reply())).encode()
        data = self.client(lambda request: httpx.Response(200, content=body)).complete(
            {'messages': [], 'tools': None})
        self.assertEqual(data['choices'][0]['message']['content'], 'Answer: B')

    def test_status_codes(self):
        import httpx
        for status, error in ((429, run.CallError), (500, run.CallError), (402, run.FatalError)):
            client = self.client(lambda request, s=status: httpx.Response(s, content=b'{}'))
            with self.assertRaises(error):
                client.complete({'messages': []})
        bad = self.client(lambda request: httpx.Response(400, content=b'bad'))
        with self.assertRaises(run.CallError) as caught:
            bad.complete({'messages': []})
        self.assertFalse(caught.exception.retryable)

    def test_whole_call_deadline(self):
        import httpx

        def slow_stream(request):
            def chunks():
                for _ in range(3):
                    yield b' '  # Keepalive bytes: no read timeout ever fires.
                yield json.dumps(openrouter_reply()).encode()
            return httpx.Response(200, content=chunks())

        clock = iter(range(0, 10000, 400))  # Each check of the clock moves 400 s on.
        with patch.object(run.time, 'monotonic', side_effect=lambda: next(clock)):
            with self.assertRaises(run.CallError) as caught:
                self.client(slow_stream, deadline=900).complete({'messages': []})
        self.assertTrue(caught.exception.retryable)
        self.assertIn('900', str(caught.exception))


class FakeClient:
    def __init__(self, replies=None, fail=()):
        self.replies, self.fail, self.bodies = replies or {}, dict(fail), []

    def complete(self, body):
        self.bodies.append(body)
        question = body['messages'][-1]['content']
        if question in self.fail:
            raise self.fail[question]
        return self.replies.get(question, openrouter_reply())


class RunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.records = [dict(mc(), prompt_id=f'mmlu_pro-{i:03d}',
                             input={'question': f'Q{i}', 'choices': ['a', 'b']}) for i in range(1, 6)]
        sleep = patch.object(run.time, 'sleep')
        sleep.start()
        self.addCleanup(sleep.stop)
        quiet = patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)

    def jobs(self, limit=None):
        return run.plan([MODEL], {'mmlu_pro': self.records}, self.dir, limit, seed=1)

    def rows(self):
        return results.read_rows(results.result_path(self.dir, 'general_knowledge', 'mmlu_pro', MODEL.id))

    def question(self, i):
        return templates.user_message(self.records[i - 1])

    def test_tool_calling_is_deferred_unless_included(self):
        default = run.select_datasets(None, None)
        self.assertNotIn('bfcl_simple', default)
        self.assertIn('math500', default)
        self.assertEqual(len(default), 9)
        for request in ((['bfcl_simple'], None), (None, ['tool_calling'])):
            with self.assertRaises(SystemExit):
                run.select_datasets(*request)
        self.assertEqual(run.select_datasets(None, ['tool_calling'], include_deferred=True),
                         ['bfcl_simple', 'bfcl_multiple', 'bfcl_parallel', 'bfcl_irrelevance'])

    def test_parse_answer_fields(self):
        parsed = run.parse_answer(openrouter_reply(tool_calls=[('f', '{"a": 1}')]))
        self.assertEqual(parsed['tool_calls'], [{'id': 'c1', 'name': 'f', 'arguments': '{"a": 1}'}])
        self.assertEqual((parsed['cost_usd'], parsed['reasoning_tokens'], parsed['provider']),
                         (0.001, 5, 'Prov'))
        self.assertEqual(parsed['reasoning'], 'thinking...')

    def test_run_saves_graded_rows_and_resume_skips_them(self):
        client = FakeClient(replies={self.question(2): openrouter_reply('Answer: A')})
        run.run(self.jobs(), client, SETTINGS, self.dir, budget=10, bank_spent=0)
        rows = self.rows()
        self.assertEqual(len(rows), 5)
        self.assertEqual([rows[f'mmlu_pro-{i:03d}']['correct'] for i in range(1, 6)],
                         [True, False, True, True, True])
        row = rows['mmlu_pro-001']
        self.assertEqual((row['prompt_template_version'], row['cost_usd'], row['max_tokens']),
                         (templates.TEMPLATE_VERSION, 0.001, 8192))
        self.assertNotIn('messages', row)
        self.assertEqual(client.bodies[0]['model'], 'test/cheap')
        self.assertEqual(sum(len(job[2]) for job in self.jobs()), 0)

    def test_grading_runs_in_the_main_thread(self):
        import threading
        threads = []

        def spy(record, answer):
            threads.append(threading.current_thread() is threading.main_thread())
            return graders.grade(record, answer)

        with patch.object(run, 'grade', side_effect=spy):
            run.run(self.jobs(), FakeClient(), SETTINGS, self.dir, budget=10, bank_spent=0)
        self.assertEqual(threads, [True] * 5)

    def test_grader_crash_keeps_the_paid_answer(self):
        with patch.object(run, 'grade', side_effect=RuntimeError('bug')):
            run.run(self.jobs(), FakeClient(), SETTINGS, self.dir, budget=10, bank_spent=0)
        row = self.rows()['mmlu_pro-001']
        self.assertEqual((row['grader'], row['correct'], row['response']), ('error', None, 'Answer: B'))

    def test_broken_provider_responses_are_errors_not_answers(self):
        body = {'messages': [], 'tools': None}
        malformed = openrouter_reply(content=None, finish='error')
        malformed['choices'][0]['native_finish_reason'] = 'MALFORMED_FUNCTION_CALL'
        stray_tool = openrouter_reply(content='', tool_calls=[('f', '{}')], finish='tool_calls')
        no_usage = openrouter_reply(content='')
        del no_usage['usage']
        # Provider glitches: retry. An empty reply without usage may be a safety filter:
        # never retry it automatically (repeats got an account restricted, 2026-10-13).
        self.assertEqual(run.broken_response(malformed, body)[1], True)
        self.assertEqual(run.broken_response(stray_tool, body)[1], True)
        self.assertEqual(run.broken_response(no_usage, body), ('no token usage reported', False))
        self.assertIsNone(run.broken_response(stray_tool, {'tools': [{}]}))
        self.assertIsNone(run.broken_response(openrouter_reply(), body))

    def test_saved_broken_rows_are_run_again(self):
        path = results.result_path(self.dir, 'general_knowledge', 'mmlu_pro', MODEL.id)
        base = {'model_id': MODEL.id, 'dataset': 'mmlu_pro', 'output_tokens': 10, 'error': None}
        results.append_row(path, {**base, 'prompt_id': 'mmlu_pro-001', 'finish_reason': 'error'})
        results.append_row(path, {**base, 'prompt_id': 'mmlu_pro-002', 'output_tokens': None})
        results.append_row(path, {**base, 'prompt_id': 'mmlu_pro-003', 'tool_calls': [{'name': 'f'}]})
        results.append_row(path, {**base, 'prompt_id': 'mmlu_pro-004', 'finish_reason': 'stop'})
        pending = [r['prompt_id'] for r in self.jobs()[0][2]]
        self.assertEqual(pending, ['mmlu_pro-001', 'mmlu_pro-003', 'mmlu_pro-005'])
        forced = run.plan([MODEL], {'mmlu_pro': self.records}, self.dir, None, seed=1, retry_filtered=True)
        self.assertIn('mmlu_pro-002', [r['prompt_id'] for r in forced[0][2]])

    def test_filtered_reply_is_sent_once_and_not_rerun(self):
        filtered = run.CallError('provider returned a broken response: no token usage reported',
                                 retryable=False, kind='filtered')
        client = FakeClient(fail={self.question(2): filtered})
        run.run(self.jobs(), client, SETTINGS, self.dir, budget=10, bank_spent=0)
        sent = [b for b in client.bodies if b['messages'][-1]['content'] == self.question(2)]
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.rows()['mmlu_pro-002']['error']['kind'], 'filtered')
        self.assertEqual(self.jobs()[0][2], [])
        # Rows saved before the 'kind' field existed are recognised by their message.
        old = {'error': {'message': 'provider returned a broken response: no token usage reported',
                         'status': None}, 'dataset': 'mmlu_pro'}
        self.assertFalse(run.needs_rerun(old))
        self.assertTrue(run.needs_rerun(old, retry_filtered=True))

    def test_errors_are_recorded_then_retried_on_the_next_run(self):
        flaky = run.CallError('HTTP 503', 503, retryable=True)
        client = FakeClient(fail={self.question(3): flaky})
        run.run(self.jobs(), client, SETTINGS, self.dir, budget=10, bank_spent=0)
        self.assertEqual(self.rows()['mmlu_pro-003']['error']['status'], 503)
        self.assertEqual(len([b for b in client.bodies if b['messages'][-1]['content'] == self.question(3)]),
                         len(run.BACKOFF) + 1)
        pending = self.jobs()[0][2]
        self.assertEqual([r['prompt_id'] for r in pending], ['mmlu_pro-003'])
        run.run(self.jobs(), FakeClient(), SETTINGS, self.dir, budget=10, bank_spent=0)
        self.assertIsNone(self.rows()['mmlu_pro-003']['error'])

    def test_fatal_error_stops_the_run(self):
        client = FakeClient(fail={self.question(1): run.FatalError('402 no credits')})
        stopped = run.run(self.jobs(), client, dict(SETTINGS, workers=1), self.dir, budget=10, bank_spent=0)
        self.assertIn('402', stopped)
        self.assertEqual(len(client.bodies), 1)

    def test_budget_cap_stops_before_overspending(self):
        settings = dict(SETTINGS, workers=1, max_tokens=1000)  # Reserve: 1000 * $2/M = $0.002.
        client = FakeClient(replies={self.question(i): openrouter_reply(cost=0.01) for i in range(1, 6)})
        stopped = run.run(self.jobs(), client, settings, self.dir, budget=0.025, bank_spent=0)
        self.assertIn('budget', stopped)
        self.assertEqual(len(self.rows()), 3)

    def test_limit_picks_the_same_seeded_subset(self):
        first = [r['prompt_id'] for r in self.jobs(limit=2)[0][2]]
        self.assertEqual(len(first), 2)
        self.assertEqual(first, [r['prompt_id'] for r in self.jobs(limit=2)[0][2]])

    def test_dry_run_and_regrade_make_no_calls(self):
        run.run(self.jobs(), FakeClient(), SETTINGS, self.dir, budget=10, bank_spent=0)
        with patch.object(run, 'OpenRouterClient') as client:
            run.dry_run(self.jobs(), self.dir)
            path = results.result_path(self.dir, 'general_knowledge', 'mmlu_pro', MODEL.id)
            rows = list(results.read_rows(path).values())
            rows[0]['correct'] = None
            results.rewrite_rows(path, rows)
            with patch.object(run, 'load_records', return_value={'mmlu_pro': self.records}):
                run.regrade(self.dir, self.dir, ['mmlu_pro'])
            client.assert_not_called()
        self.assertTrue(all(row['correct'] for row in self.rows().values()))

    def test_summary_prints_without_calls(self):
        run.run(self.jobs(), FakeClient(), SETTINGS, self.dir, budget=10, bank_spent=0)
        with patch.object(run, 'OpenRouterClient') as client:
            run.print_summary(self.dir, ('model_id', 'dataset'))
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
