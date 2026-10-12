"""Try the complexity scorer by hand, the way a running server would use it.

NVIDIA's classifier is loaded and warmed up once at startup; after that each prompt is scored
with the model already in memory, and the time per prompt is shown.

Usage:
    python examples/scorer_demo.py            # interactive, Ctrl+C to quit
    python examples/scorer_demo.py "Write a Python function that sorts a list"
    python examples/scorer_demo.py "..." --debug   # also log the classifier's full profile
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow direct execution from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from lonn.complexity import ComplexityScorer


def show(result, elapsed_ms):
    print("\nTopics:")
    for topic, p in sorted(result.topic_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {topic.value:<32} {p:.2f}  {'#' * round(p * 30)}")
    print(f"\nComplexity: {result.complexity:.2f}")
    print(f"Scored in {elapsed_ms:.0f} ms")


def timed_score(scorer, text):
    start = time.perf_counter()
    result = scorer.score_prompt(text)
    return result, (time.perf_counter() - start) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", help="the prompt to classify")
    ap.add_argument("--debug", action="store_true", help="log the classifier's full profile")
    args = ap.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="[%(name)s] %(message)s")

    print("Starting up: loading the classifier and warming up...")
    start = time.perf_counter()
    scorer = ComplexityScorer()
    timed_score(scorer, "Warm-up prompt.")  # The first call is slow; pay it now.
    print(f"Ready in {time.perf_counter() - start:.1f} s")

    if args.prompt:
        show(*timed_score(scorer, args.prompt))
        return

    print("Type a prompt and press Enter (Ctrl+C to quit).")
    try:
        while True:
            text = input("\n> ").strip()
            if text:
                show(*timed_score(scorer, text))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
