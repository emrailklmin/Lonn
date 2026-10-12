"""Build extra topic-labelled prompts from public datasets (experiment).

Usage:
    python3 scripts/build_external_topics.py
    python3 scripts/build_external_topics.py --arena-parts 7 --per-topic 3000

Writes data/external/topics.jsonl (gitignored) in the pilot label format,
{"prompt", "topics", "source"}, for training with --extra. Sources:

- lmarena-ai/arena-human-preference-140k (CC BY 4.0): real Chatbot Arena user
  prompts. Arena's own LLM-generated tags give the topic: is_code -> coding,
  math -> math_reasoning, creative_writing -> creative_writing, none of these
  -> general_knowledge_extraction. Only English, text-only, first user turns.
- databricks/databricks-dolly-15k (CC BY-SA 3.0): prompts written and
  categorised by people. creative_writing and brainstorming ->
  creative_writing; open_qa, general_qa, closed_qa, classification,
  information_extraction and summarization -> general_knowledge_extraction.

Prompts that also appear in data/labels/pilot.jsonl are dropped, so the pilot
test split stays unseen. Labels from both sources are noisier than the pilot
labels; measure their effect on the pilot test split.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
ARENA = "lmarena-ai/arena-human-preference-140k"
DOLLY = "databricks/databricks-dolly-15k"
DOLLY_TOPICS = {
    "creative_writing": "creative_writing",
    "brainstorming": "creative_writing",
    "open_qa": "general_knowledge_extraction",
    "general_qa": "general_knowledge_extraction",
    "closed_qa": "general_knowledge_extraction",
    "classification": "general_knowledge_extraction",
    "information_extraction": "general_knowledge_extraction",
    "summarization": "general_knowledge_extraction",
}


def normalise(text):
    return " ".join(text.lower().split())


def arena_topic(row):
    tags = row["category_tag"] or {}
    if row["is_code"]:
        return "coding"
    if (tags.get("math_v0.1") or {}).get("math"):
        return "math_reasoning"
    if (tags.get("creative_writing_v0.1") or {}).get("creative_writing"):
        return "creative_writing"
    return "general_knowledge_extraction"


def arena_prompts(parts):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for part in range(parts):
        path = hf_hub_download(ARENA, f"data/train-{part:05d}-of-00007.parquet", repo_type="dataset")
        for row in pq.read_table(path, columns=["conversation_a", "category_tag", "language", "is_code"]).to_pylist():
            if row["language"] != "en":
                continue
            turn = next((t for t in row["conversation_a"] if t["role"] == "user"), None)
            if turn is None or any(c.get("type") != "text" or c.get("image") for c in turn["content"]):
                continue
            text = " ".join(c["text"] for c in turn["content"] if c.get("text")).strip()
            yield text, arena_topic(row), "arena"


def dolly_prompts():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DOLLY, "databricks-dolly-15k.jsonl", repo_type="dataset")
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["category"] in DOLLY_TOPICS:
            text = row["instruction"] if not row["context"] else f"{row['instruction']}\n\n{row['context']}"
            yield text.strip(), DOLLY_TOPICS[row["category"]], "dolly"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arena-parts", type=int, default=1, help="Arena parquet files to use (1-7, ~10k prompts each)")
    parser.add_argument("--per-topic", type=int, default=1500, help="maximum prompts per topic and source")
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data/external/topics.jsonl")
    args = parser.parse_args()

    pilot = {normalise(json.loads(line)["prompt"]) for line in (ROOT / "data/labels/pilot.jsonl").read_text().splitlines()}
    groups, seen = defaultdict(list), set(pilot)
    for source in (arena_prompts(args.arena_parts), dolly_prompts()):
        for text, topic, name in source:
            key = normalise(text)
            if key in seen or not args.min_chars <= len(text) <= args.max_chars:
                continue
            seen.add(key)
            groups[(name, topic)].append(text)

    rng = random.Random(args.seed)
    rows = []
    for (name, topic), texts in sorted(groups.items()):
        rng.shuffle(texts)
        rows += [{"prompt": t, "topics": {topic: 1.0}, "source": name} for t in texts[:args.per_topic]]
    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = Counter((r["source"], next(iter(r["topics"]))) for r in rows)
    for (name, topic), n in sorted(counts.items()):
        print(f"  {name:<6} {topic:<30} {n}")
    print(f"Wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
