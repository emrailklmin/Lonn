"""Probe text completion availability without changing routing configuration.

Run explicitly: this sends a small synthetic prompt to each selected NVIDIA ID.
Timeouts/rate limits are transient results, not permanent exclusions. Tool-call
support and task quality must be evaluated separately.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter, sleep

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'Reply with the word OK.'


def classify_error(error):
    status = getattr(error, 'status_code', None)
    if status in (401, 403):
        return 'access_denied'
    if status == 404:
        return 'not_found'
    if status == 429:
        return 'rate_limited'
    if status in (408, 504) or isinstance(error, TimeoutError) or 'timeout' in type(error).__name__.lower():
        return 'timeout'
    if isinstance(status, int) and status >= 500:
        return 'server_error'
    return 'request_error'


def probe_model(model_id, completion, *, timeout=120, max_tokens=256, attempts=2):
    history = []
    for attempt in range(attempts):
        start = perf_counter()
        try:
            response = completion(model=model_id, messages=[{'role': 'user', 'content': PROMPT}],
                                  timeout=timeout, max_tokens=max_tokens, stream=False,
                                  num_retries=0, max_retries=0)
            choices = getattr(response, 'choices', None)
            message = getattr(choices[0], 'message', None) if choices else None
            content = getattr(message, 'content', None)
            valid = isinstance(content, str) and bool(content.strip()) and not getattr(message, 'tool_calls', None)
            item = {'status': 'ok' if valid else 'no_text',
                    'finish_reason': getattr(choices[0], 'finish_reason', None) if choices else None}
        except Exception as exc:
            # Do not persist raw SDK messages: they can contain credentials or request data.
            item = {'status': classify_error(exc), 'http_status': getattr(exc, 'status_code', None),
                    'error_type': type(exc).__name__}
        item['latency_ms'] = round((perf_counter() - start) * 1000, 2)
        history.append(item)
        if item['status'] not in ('timeout', 'rate_limited', 'server_error') or attempt + 1 == attempts:
            break
        sleep(min(2 ** attempt, 8))
    return {'id': model_id, 'checked_at': datetime.now(timezone.utc).isoformat(),
            'status': history[-1]['status'], 'attempts': history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=ROOT / 'config/models.yaml')
    parser.add_argument('--catalog', type=Path, help='Probe a discovered JSON inventory instead of the pool')
    parser.add_argument('--model', action='append', help='Full NVIDIA LiteLLM ID; repeat to select models')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/model_availability.json')
    args = parser.parse_args()
    from dotenv import load_dotenv
    from litellm import completion
    load_dotenv(ROOT / '.env', override=False)
    key = os.environ.get('NVIDIA_NIM_API_KEY', '').strip()
    if not key or key == 'nvapi-your-key-here':
        parser.error('Set NVIDIA_NIM_API_KEY in .env or the environment')
    if args.model:
        ids = args.model
    elif args.catalog:
        ids = [m['id'] for m in json.loads(args.catalog.read_text())['models']]
    else:
        ids = [m['id'] for m in yaml.safe_load(args.pool.read_text()) if m.get('active')]
    if any(not model_id.startswith('nvidia_nim/') for model_id in ids):
        parser.error('All model IDs must use the nvidia_nim/ prefix')
    report = {'prompt': PROMPT, 'timeout_seconds': 120, 'max_tokens': 256, 'results': []}
    def call(**kwargs):
        return completion(api_key=key, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model_id in dict.fromkeys(ids):
        result = probe_model(model_id, call)
        report['results'].append(result)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(model_id, result['status'], flush=True)
        if result['status'] == 'access_denied':
            break


if __name__ == '__main__':
    main()
