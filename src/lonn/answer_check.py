"""Check a model's answer before returning it, so the caller can escalate (issue #19).

Checks, in order (docs/oracle_gap.md):

1. ``truncated``: the reply stopped at the token limit (finish_reason "length").
2. ``empty``: no final text.
3. ``failed_examples``: the prompt shows worked examples of a programming task
   (AtCoder-style "Sample Input / Sample Output", or LeetCode-style
   "Example: Input ... Output ..." with a ``class Solution`` starter), and the
   last Python code block in the reply does not reproduce them in the sandbox.
   Runs only where the sandbox exists (macOS); elsewhere it is skipped.

On the benchmark, code that passes the examples is correct 95% of the time, and
the examples catch 91% of wrong LiveCodeBench answers.

The checker makes no model calls and chooses no models.
"""

from dataclasses import dataclass
import ast
import json
import re
import tempfile

from .code_sandbox import TEST_TIMEOUT, extract_code, outputs_match, run_sandboxed, sandbox_available

CHECK_VERSION = '1'
MAX_EXAMPLES = 5
TOTAL_TIMEOUT = 30.0  # Seconds for all examples of one answer together.

STDIN_EXAMPLE = re.compile(
    r'Sample Input\s*(\d*)\s*\n(.*?)\n\s*Sample Output\s*\1\s*\n(.*?)(?=\n\s*\n|\Z)', re.DOTALL)
FUNCTIONAL_EXAMPLE = re.compile(r'Example\s*\d*\s*:\s*\n+\s*Input:\s*(.+?)\n\s*Output:\s*(.+?)\s*(?:\n|\Z)')
STARTER_METHOD = re.compile(r'class Solution\b.*?def\s+(\w+)\s*\(\s*self', re.DOTALL)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str | None = None  # truncated, empty or failed_examples when not ok
    detail: str | None = None


@dataclass(frozen=True)
class Example:
    kind: str  # "stdin" or "functional"
    input: str  # stdin text, or the call's arguments as JSON lines
    output: str


def _split_arguments(text: str) -> list[str] | None:
    """'nums = [1,2], k = 3' -> ['[1,2]', '3']; None if it is not name = value pairs."""
    parts, depth, current = [], 0, ''
    for char in text:
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        if char == ',' and depth == 0:
            parts.append(current)
            current = ''
        else:
            current += char
    parts.append(current)
    values = []
    for part in parts:
        name, sep, value = part.partition('=')
        if not sep or not name.strip().isidentifier():
            return None
        values.append(value.strip())
    return values


def _as_json(value: str) -> str | None:
    """A literal from a problem statement as JSON text, or None if it is not a literal."""
    for parse in (json.loads, lambda v: ast.literal_eval(v.replace('true', 'True').replace('false', 'False'))):
        try:
            return json.dumps(parse(value))
        except (ValueError, SyntaxError, TypeError):
            continue
    return None


def extract_examples(prompt: str) -> tuple[list[Example], str | None]:
    """(worked examples found in the prompt, the Solution method name for functional ones)."""
    prompt = (prompt or '').replace('\r\n', '\n')
    examples = [Example('stdin', given.strip() + '\n', expected.strip())
                for _, given, expected in STDIN_EXAMPLE.findall(prompt)]
    if examples:
        return examples[:MAX_EXAMPLES], None
    method = STARTER_METHOD.search(prompt)
    if not method:
        return [], None
    for given, expected in FUNCTIONAL_EXAMPLE.findall(prompt):
        arguments = _split_arguments(given.strip())
        values = [_as_json(a) for a in arguments] if arguments else None
        result = _as_json(expected.strip())
        if values and all(v is not None for v in values) and result is not None:
            examples.append(Example('functional', '\n'.join(values), result))
    return examples[:MAX_EXAMPLES], method.group(1)


FUNCTIONAL_PRELUDE = ('from typing import *\nimport math, collections, itertools, heapq, bisect, '
                      'functools, string, re\nfrom collections import *\nfrom itertools import *\n'
                      'from functools import *\nfrom heapq import *\nfrom bisect import *\n')


def _functional_harness(code: str, examples: list[Example], method: str) -> str:
    cases = json.dumps([{'input': e.input, 'output': e.output} for e in examples])
    return f'''{FUNCTIONAL_PRELUDE}
{code}

import json as _json, sys as _sys
for _number, _case in enumerate(_json.loads({cases!r}), 1):
    _args = [_json.loads(line) for line in _case["input"].split("\\n")]
    _got = getattr(Solution(), {method!r})(*_args)
    if _json.loads(_json.dumps(_got)) != _json.loads(_case["output"]):
        print(f"example {{_number}}: wrong output", file=_sys.stderr)
        _sys.exit(1)
'''


def check_examples(prompt: str, content: str) -> CheckResult:
    """Run the reply's code on the prompt's worked examples. ok=True when there is nothing to check."""
    examples, method = extract_examples(prompt)
    if not examples or not sandbox_available():
        return CheckResult(True)
    code = extract_code(content)
    if code is None:
        return CheckResult(False, 'failed_examples', 'no Python code block')
    with tempfile.TemporaryDirectory(prefix='lonn-check-') as workdir:
        if examples[0].kind == 'functional':
            rc, _, err = run_sandboxed(_functional_harness(code, examples, method), workdir,
                                       timeout=min(TOTAL_TIMEOUT, TEST_TIMEOUT * len(examples)))
            if rc != 0:
                return CheckResult(False, 'failed_examples', 'timeout' if rc is None else (err.strip()[-200:] or 'error'))
            return CheckResult(True)
        for number, example in enumerate(examples, 1):
            rc, out, err = run_sandboxed(code, workdir, stdin=example.input, timeout=TEST_TIMEOUT)
            if rc is None:
                return CheckResult(False, 'failed_examples', f'example {number}: timeout')
            if rc != 0 or not outputs_match(out, example.output):
                return CheckResult(False, 'failed_examples', f'example {number}: wrong output')
    return CheckResult(True)


def check_answer(prompt: str, content: str | None, finish_reason: str | None,
                 run_code: bool = True) -> CheckResult:
    """Is this reply worth returning, or should the caller try a stronger model?"""
    if finish_reason == 'length':
        return CheckResult(False, 'truncated')
    if not (content or '').strip():
        return CheckResult(False, 'empty')
    if run_code:
        return check_examples(prompt, content)
    return CheckResult(True)
