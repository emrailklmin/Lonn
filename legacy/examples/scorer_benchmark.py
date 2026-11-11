"""Measure how fast the Laya complexity scorer runs on this machine.

Times the model load, the first (cold) call, and repeated warm calls for each
Laya round separately and for a full score_prompt (both rounds).

Usage:
    python legacy/examples/scorer_benchmark.py
    python legacy/examples/scorer_benchmark.py --runs 50
"""

import argparse
import os
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

# Allow direct execution from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

os.environ.setdefault("USE_TF", "0")  # Avoids a known TensorFlow import hang.

from lonn.complexity import ComplexityScorer

PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that sorts a list",
    "Solve for x: 3x + 7 = 22, and explain each step.",
    "Write a short, warm email declining a job offer while keeping the door open.",
    "Summarize the main causes of the First World War in a few bullet points.",
    "Design a rate limiter for a distributed API gateway. Compare token bucket and "
    "sliding window approaches, discuss consistency across nodes, and give a Go "
    "implementation with tests.",
]


def timed(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - start) * 1000


def summarize(label, times_ms):
    times = sorted(times_ms)
    p90 = times[min(len(times) - 1, int(0.9 * len(times)))]
    print(f"  {label:<22} median {statistics.median(times):7.1f} ms   "
          f"mean {statistics.fmean(times):7.1f} ms   p90 {p90:7.1f} ms   "
          f"min {times[0]:7.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="warm runs per prompt round")
    args = ap.parse_args()

    import torch
    from laya import Router

    print(f"Python {platform.python_version()}, torch {torch.__version__}, "
          f"{platform.machine()} {platform.system()}")

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Laya's checkpoint calibration notice.
        laya = Router(preload=True)
    load_ms = (time.perf_counter() - start) * 1000
    devices = {str(a.device) for a in getattr(laya, "_agents", {}).values()
               if hasattr(a, "device")}
    print(f"Device: {', '.join(sorted(devices)) or 'unknown'}")
    print(f"Model load: {load_ms / 1000:.1f} s\n")

    scorer = ComplexityScorer(laya)
    cold_ms = timed(scorer.score_prompt, PROMPTS[0])

    rounds = {"round 1 (topics)": scorer.topic_scores,
              "round 2 (complexity)": scorer.complexity,
              "score_prompt (both)": scorer.score_prompt}
    results = {label: [] for label in rounds}
    for _ in range(args.runs):
        for prompt in PROMPTS:
            for label, fn in rounds.items():
                results[label].append(timed(fn, prompt))

    print(f"First call (cold): {cold_ms:.1f} ms")
    print(f"Warm calls ({args.runs} runs x {len(PROMPTS)} prompts):")
    for label, times in results.items():
        summarize(label, times)


if __name__ == "__main__":
    main()
