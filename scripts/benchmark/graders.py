"""Grade a saved model answer against a benchmark prompt record.

grade(record, answer) never calls a model, so saved answers can be regraded
at any time. answer holds 'response' (final text) and 'tool_calls'. Generated
code runs only inside the macOS sandbox in run_sandboxed(). Grade from the
main thread: math-verify's timeouts only work there, and grading one answer at
a time keeps sandboxed tests from slowing each other into false timeouts.
"""
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from lonn.code_sandbox import MAX_OUTPUT_BYTES, TEST_TIMEOUT  # noqa: E402,F401

from .templates import FAMILY_OF, tool_name  # noqa: E402

GRADER_VERSION = '3'
ROOT = Path(__file__).resolve().parents[2]
TOTAL_TIMEOUT = 120.0  # Seconds for all of one answer's tests together.
MATH_TIMEOUT = 5  # Seconds for math-verify parsing and comparison (main thread only).
FUNCTIONAL_PRELUDE = ('from typing import *\nimport math, collections, itertools, heapq, bisect, '
                      'functools, re, string\nfrom collections import *\n')

THINK_BLOCK = re.compile(r'<(think|thinking)>.*?</\1>', re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r'<(think|thinking)>.*', re.DOTALL | re.IGNORECASE)


def final_text(response) -> str:
    """The answer without inline <think> blocks; unfinished thinking counts as no answer."""
    return UNCLOSED_THINK.sub('', THINK_BLOCK.sub('', response or '')).strip()


def grade(record: dict, answer: dict) -> dict:
    family = FAMILY_OF[record['dataset']]
    answer = {**answer, 'response': final_text(answer.get('response'))}
    extracted, correct, details = GRADERS[family](record, answer)
    return {'extracted_answer': extracted, 'correct': correct, 'grade_details': details,
            'grader': family if correct is not None else 'deferred', 'grader_version': GRADER_VERSION}


# Multiple choice

# "Answer: B", also "Final answer: B", "The answer: (B)", "- **Answer:** B". The label must
# end there or be followed by a delimiter, so "Answer: The answer is B" is not read as "THE".
ANSWER_LINE = re.compile(
    r'^[ \t>*_#\-]*(?:the[ \t]+)?(?:final[ \t]+)?answer[ \t*_]*[:：][ \t*_]*\(?[ \t]*'
    r'([A-Za-z0-9]+)[ \t*_]*(?:$|[).:,\-–])', re.IGNORECASE | re.MULTILINE)


def grade_multiple_choice(record, answer):
    matches = ANSWER_LINE.findall(answer.get('response') or '')
    if not matches:
        return None, False, 'no "Answer: X" line'
    choice = matches[-1].upper()
    return choice, choice == str(record['reference']['answer']).upper(), None


# Math

def last_boxed(text: str):
    """Content of the last \\boxed{...} (or \\fbox{...}), with nested braces."""
    start = max(text.rfind('\\boxed'), text.rfind('\\fbox'))
    if start < 0:
        return None
    brace = text.find('{', start)
    if brace < 0:
        return None
    depth = 0
    for index in range(brace, len(text)):
        depth += {'{': 1, '}': -1}.get(text[index], 0)
        if depth == 0:
            return text[brace + 1:index].strip()
    return None


def grade_math(record, answer):
    import logging
    from math_verify import parse, verify
    logging.getLogger('math_verify').setLevel(logging.ERROR)  # Mutes the per-call timeout warning.
    boxed = last_boxed(answer.get('response') or '')
    if boxed is None:
        return None, False, 'no \\boxed{} answer'
    # math-verify's timeouts use signals, which only work in the main thread.
    timeout = MATH_TIMEOUT if threading.current_thread() is threading.main_thread() else None
    gold = parse(f"\\boxed{{{record['reference']['answer']}}}", parsing_timeout=timeout)
    predicted = parse(f'\\boxed{{{boxed}}}', parsing_timeout=timeout)
    return boxed, bool(verify(gold, predicted, timeout_seconds=timeout)), None


# Code

# The sandbox and output matching live in lonn.code_sandbox, shared with lonn.answer_check.
from lonn.code_sandbox import (CODE_BLOCK, NUMBER, _close_decimals, _outputs_match,  # noqa: E402,F401
                               _sandbox_profile, extract_code, run_sandboxed)


def _failure(stderr: str) -> str:
    return (stderr.strip().splitlines() or ['failed'])[-1][:300]


# Loads the model's code from solution.py with all its names, including _helpers. As an
# imported module, its `if __name__ == "__main__":` block does not run.
IMPORT_SOLUTION = ('import sys as _sys, os as _os\n'
                   '_sys.path.insert(0, _os.getcwd())  # python -I leaves the script folder off the path.\n'
                   'import solution as _solution\n'
                   'globals().update({k: v for k, v in vars(_solution).items() if not k.startswith("__")})\n')


def _run_harness(code: str, harness: str, workdir: str, timeout: float):
    """Run harness against solution.py. A pass needs exit code 0 *and* a fresh nonce printed
    after the last test, so code that exits early (exit(0), os._exit) never passes.
    """
    nonce = uuid.uuid4().hex
    (Path(workdir) / 'solution.py').write_text(FUNCTIONAL_PRELUDE + code)
    code_rc, out, err = run_sandboxed(f'{IMPORT_SOLUTION}{harness}\nprint({nonce!r})\n',
                                      workdir, timeout=timeout)
    if code_rc is None:
        return False, 'timeout'
    if code_rc == 0 and nonce in out:
        return True, None
    return False, _failure(err) if code_rc != 0 else 'exited before the tests finished'


def grade_code(record, answer):
    code = extract_code(answer.get('response'))
    if code is None:
        return None, False, 'no ```python code block'
    reference = record['reference']
    with tempfile.TemporaryDirectory(prefix='lonn-grade-') as workdir:
        if record['dataset'] == 'humaneval_plus':
            entry = reference['entry_point']
            if not re.search(rf'^\s*def\s+{re.escape(entry)}\s*\(', code, re.MULTILINE):
                code = record['input']['prompt'] + code  # Body only: add the given signature.
            harness = f"{reference['test']}\n\ncheck({entry})\n"
            return code, *_run_harness(code, harness, workdir, TOTAL_TIMEOUT)
        if record['dataset'] == 'mbpp_plus':
            harness = '\n'.join(reference['test_imports']) + f"\n{reference['test']}\n"
            return code, *_run_harness(code, harness, workdir, TOTAL_TIMEOUT)
        tests = reference['public_test_cases'] + reference['private_test_cases']
        if record['input']['starter_code']:
            return code, *_livecodebench_functional(code, tests, reference['metadata'], workdir)
        return code, *_livecodebench_stdin(code, tests, workdir)


def _livecodebench_stdin(code, tests, workdir):
    # A full program: it runs as __main__ and must print the expected output for each input.
    remaining = TOTAL_TIMEOUT
    for number, test in enumerate(tests, 1):
        if remaining <= 0:
            return False, 'total time limit exceeded'
        started = time.perf_counter()
        code_rc, out, err = run_sandboxed(code, workdir, stdin=test['input'],
                                          timeout=min(TEST_TIMEOUT, remaining))
        remaining -= time.perf_counter() - started  # Real run time, not the per-test allowance.
        if code_rc is None:
            return False, f'test {number}: timeout'
        if code_rc != 0:
            return False, f'test {number}: {_failure(err)}'
        if not _outputs_match(out, test['output']):
            return False, f'test {number}: wrong output'
    return True, None


def _livecodebench_functional(code, tests, metadata, workdir):
    (Path(workdir) / 'tests.json').write_text(json.dumps(tests))
    harness = f'''
import json as _json, sys as _sys
for _number, _test in enumerate(_json.load(open("tests.json")), 1):
    _args = [_json.loads(line) for line in _test["input"].strip().split("\\n")]
    _got = getattr(Solution(), {metadata["func_name"]!r})(*_args)
    _expected = _json.loads(_test["output"])
    if _json.loads(_json.dumps(_got)) != _expected:
        print(f"test {{_number}}: wrong output", file=_sys.stderr)
        _sys.exit(1)
'''
    return _run_harness(code, harness, workdir, min(TOTAL_TIMEOUT, TEST_TIMEOUT * len(tests)))


# Tool calling (BFCL-style AST check)

def _standardize(text: str) -> str:
    # BFCL compares strings case-insensitively, ignoring spaces and some punctuation.
    return re.sub(r"[ ,./\-_*^'\"]", '', text).lower()


def _value_matches(value, allowed) -> bool:
    if isinstance(allowed, str):
        return isinstance(value, str) and _standardize(value) == _standardize(allowed)
    if isinstance(allowed, bool) or isinstance(value, bool):
        return value is allowed
    if isinstance(allowed, (int, float)):
        return isinstance(value, (int, float)) and math.isclose(value, allowed, rel_tol=1e-9)
    if isinstance(allowed, list):
        return (isinstance(value, list) and len(value) == len(allowed)
                and all(_value_matches(v, a) for v, a in zip(value, allowed)))
    if isinstance(allowed, dict):
        return isinstance(value, dict) and _arguments_match(value, allowed)
    return value == allowed


def _arguments_match(arguments: dict, expected: dict) -> bool:
    """expected maps each parameter to its list of allowed values; '' marks optional."""
    if set(arguments) - set(expected):
        return False
    for name, options in expected.items():
        options = options if isinstance(options, list) else [options]
        if name not in arguments:
            if '' not in options:
                return False
            continue
        if not any(_value_matches(arguments[name], option) for option in options if option != ''):
            return False
    return True


def _call_matches(call, expected: dict) -> bool:
    (name, params), = expected.items()
    return call['name'] == tool_name(name) and _arguments_match(call['arguments'], params)


def grade_tool_calling(record, answer):
    calls = []
    for call in answer.get('tool_calls') or []:
        arguments = call.get('arguments') or '{}'
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None, False, f"unparseable arguments for {call.get('name')}"
        if not isinstance(arguments, dict):
            return None, False, f"arguments for {call.get('name')} are not an object"
        calls.append({'name': call.get('name'), 'arguments': arguments})
    extracted = calls or None
    if record['reference'].get('expect_no_call'):
        return extracted, not calls, None if not calls else 'called a tool for an irrelevant request'
    expected = record['reference']['ground_truth']
    if len(calls) != len(expected):
        return extracted, False, f'{len(calls)} calls, expected {len(expected)}'
    # Parallel calls may come in any order: accept if some pairing matches every call.
    if any(all(_call_matches(call, item) for call, item in zip(calls, order))
           for order in itertools.permutations(expected)):
        return extracted, True, None
    return extracted, False, 'calls do not match the expected calls'


# IFEval

def grade_instruction_following(record, answer):
    from .ifeval import instructions_registry
    import langdetect
    langdetect.DetectorFactory.seed = 0  # Deterministic language detection.
    response = answer.get('response') or ''
    prompt = record['input']['prompt']
    results = []
    for instruction_id, kwargs in zip(record['reference']['instruction_id_list'],
                                      record['reference']['kwargs']):
        checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        checker.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
        if 'prompt' in (checker.get_instruction_args() or {}):
            checker.build_description(prompt=prompt)
        results.append(bool(response.strip()) and bool(checker.check_following(response)))
    followed = dict(zip(record['reference']['instruction_id_list'], results))
    return followed, all(results), None


def grade_writing(record, answer):
    return None, None, 'needs an LLM judge (deferred)'


GRADERS = {'multiple_choice': grade_multiple_choice, 'math': grade_math, 'code': grade_code,
           'tool_calling': grade_tool_calling, 'instruction_following': grade_instruction_following,
           'writing': grade_writing}
