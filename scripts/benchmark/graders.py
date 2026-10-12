"""Grade a saved model answer against a benchmark prompt record.

grade(record, answer) never calls a model, so saved answers can be regraded
at any time. answer holds 'response' (final text) and 'tool_calls'. Generated
code runs only inside the macOS sandbox in run_sandboxed().
"""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

from .templates import FAMILY_OF, tool_name

GRADER_VERSION = '1'
ROOT = Path(__file__).resolve().parents[2]
TEST_TIMEOUT = 6.0  # Seconds per test case (LiveCodeBench's own default).
TOTAL_TIMEOUT = 120.0  # Seconds for all of one answer's tests together.
FUNCTIONAL_PRELUDE = ('from typing import *\nimport math, collections, itertools, heapq, bisect, '
                      'functools, re, string\nfrom collections import *\n')


def grade(record: dict, answer: dict) -> dict:
    family = FAMILY_OF[record['dataset']]
    extracted, correct, details = GRADERS[family](record, answer)
    return {'extracted_answer': extracted, 'correct': correct, 'grade_details': details,
            'grader': family if correct is not None else 'deferred', 'grader_version': GRADER_VERSION}


# Multiple choice

ANSWER_LINE = re.compile(r'^[ \t>*_#]*answer[ \t*_]*[:：][ \t*_]*\(?[ \t]*([A-Za-z0-9]+)[ \t]*\)?',
                         re.IGNORECASE | re.MULTILINE)


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
    # Timeouts are signal based and fail outside the main thread, so they are off.
    gold = parse(f"\\boxed{{{record['reference']['answer']}}}", parsing_timeout=None)
    predicted = parse(f'\\boxed{{{boxed}}}', parsing_timeout=None)
    return boxed, bool(verify(gold, predicted, timeout_seconds=None)), None


# Code

CODE_BLOCK = re.compile(r'```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)```', re.DOTALL | re.IGNORECASE)


def extract_code(text: str):
    blocks = CODE_BLOCK.findall(text or '')
    return blocks[-1] if blocks else None


def _sandbox_profile(workdir: str) -> str:
    home = str(Path.home())
    return '\n'.join([
        '(version 1)', '(allow default)', '(deny network*)', '(deny file-write*)',
        f'(allow file-write* (subpath "{workdir}") (literal "/dev/null"))',
        # No reading the home folder (API keys, tokens), except the venv's libraries.
        f'(deny file-read* (subpath "{home}"))',
        f'(allow file-read-metadata (subpath "{home}"))',
        f'(allow file-read* (subpath "{sys.prefix}"))',
    ])


def run_sandboxed(source: str, workdir: str, stdin: str = '', timeout: float = TEST_TIMEOUT):
    """Run Python source with no network, no writes outside workdir, a clean
    environment, and CPU/time/file-size limits. Returns (returncode, stdout, stderr);
    returncode is None on timeout. Memory cannot be capped reliably on macOS.
    """
    script = Path(workdir) / 'main.py'
    script.write_text(source)
    profile = Path(workdir) / 'sandbox.sb'
    profile.write_text(_sandbox_profile(os.path.realpath(workdir)))

    def limits():
        import resource
        cpu = math.ceil(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))

    process = subprocess.Popen(
        ['/usr/bin/sandbox-exec', '-f', str(profile), sys.executable, '-I', '-B', str(script)],
        cwd=workdir, env={'PATH': '/usr/bin:/bin', 'HOME': workdir, 'LANG': 'C.UTF-8'},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=limits, start_new_session=True)
    try:
        out, err = process.communicate(stdin.encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return None, '', 'timeout'
    return process.returncode, out.decode(errors='replace'), err.decode(errors='replace')


def _outputs_match(got: str, expected: str) -> bool:
    got_lines = [line.strip() for line in got.strip().splitlines()]
    expected_lines = [line.strip() for line in expected.strip().splitlines()]
    if got_lines == expected_lines:
        return True
    if len(got_lines) != len(expected_lines):
        return False
    for got_line, expected_line in zip(got_lines, expected_lines):
        got_tokens, expected_tokens = got_line.split(), expected_line.split()
        if got_tokens == expected_tokens:
            continue
        if len(got_tokens) != len(expected_tokens):
            return False
        try:
            if not all(math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
                       for a, b in zip(got_tokens, expected_tokens)):
                return False
        except ValueError:
            return False
    return True


def _failure(stderr: str) -> str:
    return (stderr.strip().splitlines() or ['failed'])[-1][:300]


def grade_code(record, answer):
    code = extract_code(answer.get('response'))
    if code is None:
        return None, False, 'no ```python code block'
    reference = record['reference']
    with tempfile.TemporaryDirectory(prefix='lonn-grade-') as workdir:
        if record['dataset'] == 'humaneval_plus':
            program = f"{FUNCTIONAL_PRELUDE}{code}\n\n{reference['test']}\n\ncheck({reference['entry_point']})\n"
            code_rc, _, err = run_sandboxed(program, workdir, timeout=TOTAL_TIMEOUT)
            return code, code_rc == 0, None if code_rc == 0 else _failure(err)
        if record['dataset'] == 'mbpp_plus':
            imports = '\n'.join(reference['test_imports'])
            program = f"{FUNCTIONAL_PRELUDE}{imports}\n{code}\n\n{reference['test']}\n"
            code_rc, _, err = run_sandboxed(program, workdir, timeout=TOTAL_TIMEOUT)
            return code, code_rc == 0, None if code_rc == 0 else _failure(err)
        tests = reference['public_test_cases'] + reference['private_test_cases']
        if record['input']['starter_code']:
            return code, *_livecodebench_functional(code, tests, reference['metadata'], workdir)
        return code, *_livecodebench_stdin(code, tests, workdir)


def _livecodebench_stdin(code, tests, workdir):
    remaining = TOTAL_TIMEOUT
    for number, test in enumerate(tests, 1):
        if remaining <= 0:
            return False, 'total time limit exceeded'
        timeout = min(TEST_TIMEOUT, remaining)
        code_rc, out, err = run_sandboxed(code, workdir, stdin=test['input'], timeout=timeout)
        remaining -= timeout
        if code_rc is None:
            return False, f'test {number}: timeout'
        if code_rc != 0:
            return False, f'test {number}: {_failure(err)}'
        if not _outputs_match(out, test['output']):
            return False, f'test {number}: wrong output'
    return True, None


def _livecodebench_functional(code, tests, metadata, workdir):
    (Path(workdir) / 'tests.json').write_text(json.dumps(tests))
    harness = f'''{FUNCTIONAL_PRELUDE}
{code}

import json as _json, sys as _sys
for _number, _test in enumerate(_json.load(open("tests.json")), 1):
    _args = [_json.loads(line) for line in _test["input"].split("\\n")]
    _got = getattr(Solution(), {metadata["func_name"]!r})(*_args)
    _expected = _json.loads(_test["output"])
    if _json.loads(_json.dumps(_got)) != _expected:
        print(f"test {{_number}}: wrong output", file=_sys.stderr)
        _sys.exit(1)
'''
    timeout = min(TOTAL_TIMEOUT, TEST_TIMEOUT * len(tests))
    code_rc, _, err = run_sandboxed(harness, workdir, timeout=timeout)
    if code_rc is None:
        return False, 'timeout'
    return code_rc == 0, None if code_rc == 0 else _failure(err)


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
        try:
            arguments = json.loads(call.get('arguments') or '{}')
        except json.JSONDecodeError:
            return None, False, f"unparseable arguments for {call.get('name')}"
        calls.append({'name': call.get('name'), 'arguments': arguments})
    extracted = calls or None
    if record['reference'].get('expect_no_call'):
        return extracted, not calls, None if not calls else 'called a tool for an irrelevant request'
    expected = list(record['reference']['ground_truth'])
    if len(calls) != len(expected):
        return extracted, False, f'{len(calls)} calls, expected {len(expected)}'
    # Order-free matching for parallel calls.
    for call in calls:
        match = next((item for item in expected if _call_matches(call, item)), None)
        if match is None:
            return extracted, False, f"no match for call {call['name']}"
        expected.remove(match)
    return extracted, True, None


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
