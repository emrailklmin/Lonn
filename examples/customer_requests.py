"""Run four requests through the authenticated Lönn API (never OpenRouter directly)."""
import argparse
import os
from lonnrouter import LonnRouter

EXAMPLES = [
    ('math_problem', 'Solve 3x + 17 = 59. Explain each step.'),
    ('code_function', 'Write a Python function that removes duplicates from a list while preserving order.'),
    ('creative_writing', 'Write a short, atmospheric story about a lighthouse keeper in winter.'),
    ('knowledge_recall', 'Why do the seasons change on Earth?'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual-tasks', action='store_true', help='Supply task signals to avoid the classifier download')
    args = parser.parse_args()
    if not os.environ.get('LONN_API_KEY'):
        parser.error('Set LONN_API_KEY to the customer key shown in Settings')
    with LonnRouter(api_key=os.environ['LONN_API_KEY'], application=os.environ.get('LONN_APPLICATION')) as client:
        for task, prompt in EXAMPLES:
            print(f'\n{task}: {prompt}')
            result = client.generate(prompt, signal={'task_scores': {task: 1.0}} if args.manual_tasks else None)
            print(result.content)
            print(f'Model: {result.model.id} | cost: ${result.total_cost_usd:.6f}')


if __name__ == '__main__':
    main()
