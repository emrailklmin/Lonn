"""Turn benchmark prompt records into OpenRouter chat requests.

Every model gets the same system prompt and user message for a question; the
system prompt states the exact output format the grader expects. Bump
TEMPLATE_VERSION whenever any wording changes: results store the version,
not the request text, so a request can be rebuilt from the prompt files.
"""
import copy
import re

TEMPLATE_VERSION = 'v1'

FAMILIES = {
    'multiple_choice': ('mmlu_pro', 'gpqa_diamond', 'arc_challenge'),
    'math': ('math500',),
    'code': ('humaneval_plus', 'mbpp_plus', 'livecodebench'),
    'tool_calling': ('bfcl_simple', 'bfcl_multiple', 'bfcl_parallel', 'bfcl_irrelevance'),
    'instruction_following': ('ifeval',),
    'writing': ('writingbench_business',),
}
FAMILY_OF = {dataset: family for family, datasets in FAMILIES.items() for dataset in datasets}

SYSTEM_PROMPTS = {
    'multiple_choice': (
        'You are answering a multiple-choice question. You may reason step by step first. '
        'End your response with a final line of exactly this form:\n'
        'Answer: X\n'
        'where X is the label of the single best option (for example A). '
        'Write nothing after that line.'),
    'math': (
        'You are solving a math problem. You may reason step by step first. '
        'Give the final answer exactly once, at the end of your response, inside \\boxed{}, '
        'for example \\boxed{42}. Simplify it fully and use exact values (fractions, radicals, '
        '\\pi) rather than decimal approximations unless the problem asks otherwise. '
        'Do not include units in the box.'),
    'code': (
        'You are an expert Python programmer. Solve the task with correct, efficient Python 3 code. '
        'Reply with exactly one Markdown code block that starts with ```python and contains the '
        'complete solution, including every import it needs. Do not write anything outside '
        'the code block.'),
    'tool_calling': (
        'You are a function-calling assistant with access to the functions provided as tools. '
        'If the user\'s request can be fulfilled with the available functions, call the right '
        'function or functions with correct arguments, using one tool call per function '
        'invocation (make several calls if the request needs several). Use only argument values '
        'stated in or directly implied by the request. If none of the functions fits the request, '
        'do not call any function; instead reply briefly that the available functions cannot help.'),
    'instruction_following': (
        'You are a helpful assistant. Follow every instruction in the user\'s request exactly, '
        'including all constraints on format, length, wording, punctuation and language.'),
    'writing': (
        'You are an expert professional writer. Complete the user\'s writing task in full, '
        'following all of its requirements on content, structure, tone, style and length.'),
}

# BFCL uses Python-flavoured schema types; OpenRouter tools expect JSON Schema.
BFCL_TYPES = {'dict': 'object', 'float': 'number', 'tuple': 'array', 'list': 'array',
              'int': 'integer', 'str': 'string', 'bool': 'boolean'}


def user_message(record: dict) -> str:
    family, data = FAMILY_OF[record['dataset']], record['input']
    if family == 'multiple_choice':
        labels = data.get('labels') or [chr(ord('A') + i) for i in range(len(data['choices']))]
        options = '\n'.join(f'{label}) {text}' for label, text in zip(labels, data['choices']))
        return f"Question: {data['question']}\n\nOptions:\n{options}"
    if family == 'math':
        return data['problem']
    if record['dataset'] == 'humaneval_plus':
        return ('Complete the following Python function. Return the complete function definition, '
                f"including its signature and any imports it needs.\n\n```python\n{data['prompt']}```")
    if record['dataset'] == 'mbpp_plus':
        return (f"{data['prompt']}\n\nYour function must pass this example test:\n"
                f"```python\n{data['example_test']}\n```")
    if record['dataset'] == 'livecodebench':
        task = f"{data['title']}\n\n{data['content']}"
        if data['starter_code']:
            return (f'{task}\n\nComplete the following starter code and return the complete class:\n'
                    f"```python\n{data['starter_code']}\n```")
        return (f'{task}\n\nWrite a complete Python 3 program that reads the input from standard '
                'input and prints the answer to standard output.')
    if family == 'instruction_following':
        return data['prompt']
    if family == 'writing':
        return data['query']
    raise ValueError(f"No user message for {record['dataset']}")


def tool_name(name: str) -> str:
    """Tool names may only use letters, digits, '_' and '-' (BFCL uses dots)."""
    return re.sub(r'[^A-Za-z0-9_-]', '_', name)


def _json_schema(schema):
    if isinstance(schema, list):
        return [_json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == 'type' and isinstance(value, str):
            if value == 'any':
                continue  # No JSON Schema type means any type.
            out[key] = BFCL_TYPES.get(value, value)
        elif key == 'properties' and isinstance(value, dict):
            out[key] = {name: _json_schema(prop) for name, prop in value.items()}
        else:
            out[key] = _json_schema(value)
    return out


def bfcl_tools(functions: list) -> list:
    tools = []
    for function in functions:
        function = copy.deepcopy(function)
        tools.append({'type': 'function', 'function': {
            'name': tool_name(function['name']),
            'description': function.get('description', ''),
            'parameters': _json_schema(function.get('parameters', {'type': 'dict', 'properties': {}})),
        }})
    names = [tool['function']['name'] for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError(f'Tool names collide after sanitizing: {names}')
    return tools


def build_request(record: dict) -> dict:
    """Return {'messages': [...], 'tools': [...] or None} for one prompt record."""
    family = FAMILY_OF[record['dataset']]
    system = {'role': 'system', 'content': SYSTEM_PROMPTS[family]}
    if family == 'tool_calling':
        # BFCL questions are already chat messages; keep them as given.
        return {'messages': [system, *record['input']['messages']],
                'tools': bfcl_tools(record['tools'])}
    return {'messages': [system, {'role': 'user', 'content': user_message(record)}], 'tools': None}
