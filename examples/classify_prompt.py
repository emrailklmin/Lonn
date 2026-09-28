"""Try the complexity scorer by hand.

Usage:
    python examples/classify_prompt.py "Write a Python function that sorts a list"
    python examples/classify_prompt.py            # interactive, Ctrl+C to quit
    python examples/classify_prompt.py "..." --debug   # also log Laya's raw output
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow direct execution from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lonn.complexity import ComplexityScorer


def show(result):
    print("\nTopics:")
    for topic, p in sorted(result.topic_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {topic.value:<32} {p:.2f}")
    print(f"\nComplexity = {result.complexity:.2f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", help="the prompt to classify")
    ap.add_argument("--debug", action="store_true", help="log raw Laya output")
    args = ap.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="[%(name)s] %(message)s")

    print("Loading Laya...")
    scorer = ComplexityScorer()
    scorer.laya  # Load up front rather than on the first prompt.

    if args.prompt:
        show(scorer.score_prompt(args.prompt))
        return

    print("Type a prompt and press Enter (Ctrl+C to quit).")
    try:
        while True:
            text = input("\n> ").strip()
            if text:
                show(scorer.score_prompt(text))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
