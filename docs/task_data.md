# Task-type training data

Issue #19. Training data for the task-type classifier: prompts labelled with a task type from `config/task_types.yaml` (`docs/task_types.md`) and, where the source has one, a difficulty. `scripts/build_task_data.py` builds it; the result is shared as the private Hugging Face dataset [LonnAI/lonn-task-data](https://huggingface.co/datasets/LonnAI/lonn-task-data). Only `data/task_data/manifest.json` is committed.

## Using it

```python
from datasets import load_dataset   # or pandas.read_parquet("hf://datasets/LonnAI/lonn-task-data/train.parquet")
data = load_dataset("LonnAI/lonn-task-data")   # LonnAI members; log in with `hf auth login`
```

Columns: `text`, `task_type`, `difficulty` (easy / medium / hard or empty), `variant`, `source`, `source_id`, `id`, `group`, `split`. Rows sharing a `group` are always in the same split.

## Building, sharing and cleaning up

```bash
python3 scripts/build_task_data.py --include-gpqa                    # ~12 min the first time, 9 MB on disk
python3 scripts/build_task_data.py --push LonnAI/lonn-task-data      # rebuild and upload (GPQA never uploaded)
python3 scripts/build_task_data.py --clean                           # delete everything local except manifest.json
```

**Nothing large stays on disk.** New sources are read from Hugging Face straight into memory (`HfFileSystem`), so the Hugging Face cache does not grow. LiveCodeBench v1–v5 is 4.3 GB, almost all hidden tests. It is streamed line by line and only the question fields are kept, in `data/task_data/cache/` (1.4 MB), so a rebuild skips the stream. The builder needs the benchmark prompt files from `scripts/benchmark/build_data.py` for the exclusion check.

## What is in it (2026-10-22)

38,375 rows: 34,552 train and 3,823 validation (10% by prompt), plus `everyday_eval.parquet` (265 prompts). Each task type is capped at 3,000 prompts, sampled deterministically.

| Task type | Prompts | Sources |
|---|---:|---|
| `math_problem` | 3,000 | full MATH (train + test), levels as difficulty |
| `code_function` | 1,054 | HumanEval+ (unsampled), full MBPP |
| `code_algorithmic` | 935 | LiveCodeBench v1–v6, easy/medium/hard as difficulty |
| `knowledge_quantitative` | 3,000 | MMLU-Pro calculation subjects |
| `knowledge_recall` | 3,000 | MMLU-Pro recall subjects (medium), ARC-Challenge (easy), Dolly open_qa |
| `instruction_following` | 3,000 | IFEval (unsampled), RLVR-IFeval |
| `business_writing` | 399 | WritingBench English business, marketing, academic, law, education |
| `science_expert` | 496 | GPQA extended, **local only** (`restricted/gpqa.parquet`) |
| `summarization` | 1,186 | Dolly |
| `translation` | 3,000 | Dolly answers wrapped in a translation request (synthetic) |
| `information_extraction` | 3,000 | Dolly information_extraction and closed_qa |
| `creative_writing` | 791 | Dolly creative_writing, WritingBench Literature & Arts |
| `conversation` | 3,000 | Dolly brainstorming and general_qa |
| `tool_calling` | 880 | BFCL v3 simple, multiple, parallel, irrelevance (prompt + tool list) |
| `other` | 2,132 | Dolly classification |

**Variants (decision Q21):** multiple-choice and code prompts appear twice:
- `as_benchmark`: rendered exactly as in the benchmark (`templates.user_message`);
- `plain`: without options, answer-format lines or harness instructions, so the classifier learns the task, not the format.

Other sources have a single `as_benchmark` row.

**No benchmark question is used.** Every question sampled into the benchmark (850, including BFCL) is excluded by source ID and by normalized text, 811 matches in total. A check that searched for every benchmark text inside every training row found none. A further 733 duplicates across sources were dropped.

## Everyday prompts (WildChat)

The first version held only benchmark-style prompts, and the classifier misread casual ones. `scripts/everyday_prompts.py` adds real first user turns from [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (ODC-BY).

**Sampling (free).** About 70 random row groups are read into memory, keeping only prompts that are:
- English, with at least 90% Latin letters;
- not toxic, not redacted and not flagged by moderation, so no provider flags the account;
- 15–6,000 characters long;
- not a near-duplicate of templated mass prompts (by their first 80 characters).

At most 3 prompts are taken per user, and the evaluation and training sets come from **different users**.

**Labelling (paid OpenRouter calls, 2.26 USD in total).**

| Set | Prompts | Labels | Cost |
|---|---:|---|---:|
| `everyday_eval.parquet` (held out, never trained on) | 265 of 300 | **gold**: Claude Sonnet 5.5 and Gemini 3.8 Flash agree (265 of 300). The other 35 are in the local `everyday/review.csv` for a human to decide | 1.40 USD |
| training rows, `source` `wildchat` | 2,971 | deepseek-v4.1-flash with low reasoning effort, which agrees with the gold labels on 90.2% of the evaluation set (that check cost 0.07 USD) | 0.79 USD |

WildChat training rows are never dropped by the per-type cap. The text stays out of git: everything is under the gitignored `data/task_data/everyday/` and in the private HF dataset.

```bash
python3 scripts/everyday_prompts.py sample                     # free
python3 scripts/everyday_prompts.py label --set eval --model anthropic/claude-sonnet-5.5 --dry-run
python3 scripts/everyday_prompts.py label --set train --model deepseek/deepseek-v4.1-flash --reasoning low --budget 1.2
python3 scripts/everyday_prompts.py status                     # agreement, spend, writes review.csv
```

## Licences and limits

- **Licences:** MIT (MATH, MMLU-Pro), Apache-2.0 (HumanEval+, IFEval, BFCL, WritingBench), CC-BY-4.0 (MBPP), CC-BY-SA (ARC, Dolly), ODC-BY (RLVR-IFeval). LiveCodeBench's licence is unspecified, so the dataset is **private and internal only**.
- **GPQA** is gated, and its authors ask that examples are not published online. It is built only with `--include-gpqa`, into the local `restricted/` folder, and never uploaded. A training run elsewhere must add it from the gated source itself.
- **These are prompts, not model outputs**, so the open question about training on model outputs (`docs/benchmark_sources.md`) does not apply to this data. It does apply to the routing table.
- **Distribution:** apart from Dolly, the prompts are benchmark-style. Real chat traffic (for example WildChat, ODC-BY) would make `conversation` and `other` more realistic.
- **Thin types:** `business_writing` (399), `creative_writing` (791), `code_algorithmic` (935) and `science_expert` (496) are smaller than the rest. Training should weight classes.
