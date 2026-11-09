"""Answer checks and escalation (#19). Code examples run in the macOS sandbox; skipped elsewhere."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lonn.answer_check import check_answer, extract_examples  # noqa: E402
from lonn.code_sandbox import sandbox_available  # noqa: E402
from lonn.routing_table import build_table  # noqa: E402
from lonn.schemas import Model, TaskSignal, TopicCategory as T  # noqa: E402
from lonn.task_router import escalation_target  # noqa: E402

STDIN_PROMPT = """Print A+B.

Sample Input 1

2 3\r

Sample Output 1

5

Sample Input 2

10 -4

Sample Output 2

6
"""
FUNCTIONAL_PROMPT = """Return the sum of nums plus k.

Example 1:

Input: nums = [1,2,3], k = 4
Output: 10
Explanation: 1 + 2 + 3 + 4.

Example 2:

Input: nums = [], k = 0
Output: 0

class Solution:
    def total(self, nums: List[int], k: int) -> int:
"""


def code(body):
    return f'Here you go:\n```python\n{body}\n```'


class ExtractTests(unittest.TestCase):
    def test_stdin_examples(self):
        examples, method = extract_examples(STDIN_PROMPT)
        self.assertIsNone(method)
        self.assertEqual([(e.kind, e.input, e.output) for e in examples],
                         [('stdin', '2 3\n', '5'), ('stdin', '10 -4\n', '6')])

    def test_functional_examples(self):
        examples, method = extract_examples(FUNCTIONAL_PROMPT)
        self.assertEqual(method, 'total')
        self.assertEqual([(e.input, e.output) for e in examples], [('[1, 2, 3]\n4', '10'), ('[]\n0', '0')])

    def test_no_examples(self):
        self.assertEqual(extract_examples('Write a haiku about rain.'), ([], None))
        # Examples without a Solution starter are not runnable calls.
        self.assertEqual(extract_examples('Example 1:\nInput: x = 1\nOutput: 2'), ([], None))


class CheckTests(unittest.TestCase):
    def test_truncated_and_empty(self):
        self.assertEqual(check_answer('hi', 'partial', 'length').reason, 'truncated')
        self.assertEqual(check_answer('hi', '  ', 'stop').reason, 'empty')
        self.assertTrue(check_answer('hi', 'Hello!', 'stop').ok)
        self.assertTrue(check_answer(STDIN_PROMPT, 'no code here', 'stop', run_code=False).ok)

    @unittest.skipUnless(sandbox_available(), 'needs the macOS sandbox')
    def test_stdin_code(self):
        self.assertTrue(check_answer(STDIN_PROMPT, code('a, b = map(int, input().split())\nprint(a + b)'), 'stop').ok)
        result = check_answer(STDIN_PROMPT, code('a, b = map(int, input().split())\nprint(a * b)'), 'stop')
        self.assertEqual(result.reason, 'failed_examples')
        self.assertEqual(check_answer(STDIN_PROMPT, 'The answer is a+b.', 'stop').detail, 'no Python code block')

    @unittest.skipUnless(sandbox_available(), 'needs the macOS sandbox')
    def test_functional_code(self):
        right = 'class Solution:\n    def total(self, nums, k):\n        return sum(nums) + k'
        wrong = 'class Solution:\n    def total(self, nums, k):\n        return sum(nums)'
        self.assertTrue(check_answer(FUNCTIONAL_PROMPT, code(right), 'stop').ok)
        self.assertFalse(check_answer(FUNCTIONAL_PROMPT, code(wrong), 'stop').ok)


def model(name):
    return Model(f'openrouter/test/{name}', 'openrouter', 1.0, 1.0, 100000, {t: 0.5 for t in T})


class EscalationTests(unittest.TestCase):
    def setUp(self):
        self.cheap, self.mid, self.strong = model('cheap'), model('mid'), model('strong')
        rows = []
        for m, passes, cost in ((self.cheap, 4, 0.0001), (self.mid, 7, 0.002), (self.strong, 9, 0.02)):
            rows += [{'task': 'code', 'difficulty': None, 'model_id': m.id, 'correct': i < passes, 'cost_usd': cost}
                     for i in range(10)]
        self.table = build_table(rows)
        self.pool = [self.cheap, self.mid, self.strong]
        self.signal = TaskSignal({'code': 1.0})

    def target(self, lam, first):
        return escalation_target(self.signal, self.table, self.pool, lam, first)

    def test_escalates_to_the_best_value_other_model(self):
        self.assertEqual(self.target(0, self.cheap), self.strong)
        self.assertEqual(self.target(20, self.cheap), self.mid)    # 0.7 - 0.04 beats 0.9 - 0.4
        self.assertEqual(self.target(20, self.mid), self.strong)   # never the same model again

    def test_no_escalation_when_a_second_call_does_not_pay(self):
        self.assertIsNone(self.target(1000, self.cheap))


if __name__ == '__main__':
    unittest.main()
