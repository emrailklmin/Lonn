"""Sanity-check the graders on real prompt files: each benchmark's own reference
answer must be graded correct. No API calls. LiveCodeBench (no reference
solutions), IFEval and WritingBench are not covered.

    python3 scripts/benchmark/check_graders.py
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.build_data import DATASETS, DEFAULT_DATA_DIR  # noqa: E402
from benchmark.graders import grade  # noqa: E402
from benchmark.templates import FAMILY_OF, tool_name  # noqa: E402


def _first(options):
    """First allowed value; nested dicts list allowed values per field too."""
    value = options[0] if isinstance(options, list) else options
    if isinstance(value, dict):
        return {key: _first(inner) for key, inner in value.items() if _first(inner) != ''}
    return value


def reference_answer(record):
    """A model-style answer built from the gold reference, or None if there is none."""
    family, reference = FAMILY_OF[record['dataset']], record['reference']
    if family == 'multiple_choice':
        return {'response': f"Reasoning.\nAnswer: {reference['answer']}"}
    if family == 'math':
        return {'response': f"So the answer is \\boxed{{{reference['answer']}}}."}
    if record['dataset'] == 'humaneval_plus':
        return {'response': f"```python\n{record['input']['prompt']}{reference['canonical_solution']}\n```"}
    if record['dataset'] == 'mbpp_plus':
        return {'response': f"```python\n{reference['canonical_solution']}\n```"}
    if family == 'tool_calling':
        if reference.get('expect_no_call'):
            return {'response': 'The available functions cannot help.', 'tool_calls': None}
        calls = []
        for expected in reference['ground_truth']:
            (name, params), = expected.items()
            arguments = {key: _first(options) for key, options in params.items() if options[0] != ''}
            # Models see and call the sanitized tool names (no dots).
            calls.append({'name': tool_name(name), 'arguments': json.dumps(arguments)})
        return {'response': '', 'tool_calls': calls}
    return None


def main():
    failures = 0
    for name, spec in DATASETS.items():
        path = DEFAULT_DATA_DIR / 'prompts' / spec.category / f'{name}.jsonl'
        records = [json.loads(line) for line in path.open(encoding='utf-8')]
        results = []
        for record in records:
            answer = reference_answer(record)
            if answer is None:
                break
            graded = grade(record, answer)
            results.append(graded['correct'])
            if not graded['correct']:
                failures += 1
                print(f"  FAIL {record['prompt_id']}: {graded['grade_details'] or graded['extracted_answer']}")
        if results:
            print(f'{name:24} {sum(results)}/{len(results)} reference answers graded correct')
        else:
            print(f'{name:24} skipped (no reference answer to check)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
