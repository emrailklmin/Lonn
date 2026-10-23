"""Run generated Python code in a locked-down macOS sandbox (shared by the benchmark
graders and the answer checks in lonn.answer_check).

run_sandboxed() blocks network access, writes outside a temporary folder and reading
the home folder (API keys, tokens) except the virtualenv, uses a clean environment and
caps CPU time, forking and output size. Memory cannot be capped reliably on macOS.
It needs /usr/bin/sandbox-exec, so it only works on macOS (see sandbox_available()).
"""

import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

TEST_TIMEOUT = 6.0  # Seconds per test case (LiveCodeBench's own default).
MAX_OUTPUT_BYTES = 64 << 20  # Per output file; larger output kills the program.
SANDBOX_EXEC = '/usr/bin/sandbox-exec'


def sandbox_available() -> bool:
    return Path(SANDBOX_EXEC).exists()


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
    """Run Python source with no network, no writes outside workdir, no forking, a clean
    environment, and CPU/time/output-size limits. Returns (returncode, stdout, stderr);
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
        # Output goes to files, so this also caps stdout and stderr.
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # No forking (fork bombs).

    out_path, err_path = Path(workdir) / 'stdout.txt', Path(workdir) / 'stderr.txt'
    with out_path.open('wb') as out_file, err_path.open('wb') as err_file:
        process = subprocess.Popen(
            [SANDBOX_EXEC, '-f', str(profile), sys.executable, '-I', '-B', str(script)],
            cwd=workdir, env={'PATH': '/usr/bin:/bin', 'HOME': workdir, 'LANG': 'C.UTF-8'},
            stdin=subprocess.PIPE, stdout=out_file, stderr=err_file,
            preexec_fn=limits, start_new_session=True)
        try:
            process.communicate(stdin.encode(), timeout=timeout)
            code_rc = process.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:  # It finished just now.
                pass
            process.communicate()
            code_rc = None

    def read(path):
        with path.open('rb') as handle:
            return handle.read(MAX_OUTPUT_BYTES).decode(errors='replace')
    if code_rc is None:
        return None, read(out_path), 'timeout'
    return code_rc, read(out_path), read(err_path)


NUMBER = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')


def _close_decimals(a: str, b: str) -> bool:
    """Numeric tolerance, only when one side is written as a decimal (never for integers)."""
    if not (NUMBER.fullmatch(a) and NUMBER.fullmatch(b)):
        return False
    if not any(char in token for token in (a, b) for char in '.eE'):
        return False
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)


def _outputs_match(got: str, expected: str) -> bool:
    got_lines = [line.strip() for line in got.strip().splitlines()]
    expected_lines = [line.strip() for line in expected.strip().splitlines()]
    if got_lines == expected_lines:
        return True
    if len(got_lines) != len(expected_lines):
        return False
    for got_line, expected_line in zip(got_lines, expected_lines):
        got_tokens, expected_tokens = got_line.split(), expected_line.split()
        if len(got_tokens) != len(expected_tokens):
            return False
        if not all(a == b or _close_decimals(a, b) for a, b in zip(got_tokens, expected_tokens)):
            return False
    return True


# Public names for callers outside the benchmark graders.
outputs_match = _outputs_match
