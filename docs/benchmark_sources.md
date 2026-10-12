# Benchmark sources

Sources for the routing benchmark (issue #16), checked 2026-10-12 against the Hugging Face API and GitHub. `scripts/benchmark/build_data.py` pins every source to the revision below, downloads it to `data/benchmark/raw/<category>/<dataset>/`, and writes seeded samples to `data/benchmark/prompts/<category>/<dataset>.jsonl`. The categories are `tool_calling` (160 prompts), `coding` (200), `general_knowledge` (200), `math` (150) and `writing` (140). Both folders are gitignored. Only `data/benchmark/manifest.json` is committed: it records revisions, file hashes, the seed, stratum counts and the selected source IDs, and `--check` rebuilds and compares against it.

```bash
python3 scripts/benchmark/build_data.py              # all datasets
python3 scripts/benchmark/build_data.py math500      # one dataset
python3 scripts/benchmark/build_data.py --check      # verify a rebuild is identical
```

## Datasets (13 datasets from 10 sources, 850 prompts)

| Dataset key | Source (pinned revision) | Used file | Size | n | Sampling | Licence | Notes |
|---|---|---|---|---|---|---|---|
| `bfcl_simple` | HF `gorilla-llm/Berkeley-Function-Calling-Leaderboard` @ `61fc060` | `BFCL_v3_simple.json` + `possible_answer/` | 400 rows | 40 | random | Apache-2.0 | BFCL **v3** files (repo updated 2026-04). Level: easy |
| `bfcl_multiple` | same | `BFCL_v3_multiple.json` + answers | 200 | 40 | random | Apache-2.0 | medium |
| `bfcl_parallel` | same | `BFCL_v3_parallel.json` + answers | 200 | 40 | random | Apache-2.0 | medium |
| `bfcl_irrelevance` | same | `BFCL_v3_irrelevance.json` | 240 | 40 | random | Apache-2.0 | No gold answer: correct = no tool call |
| `humaneval_plus` | HF `evalplus/humanevalplus` @ `d32357c` | parquet | 164 | 40 | random | Apache-2.0 | Includes the EvalPlus extended tests |
| `mbpp_plus` | HF `evalplus/mbppplus` @ `b2d74c9` | parquet | 378 | 40 | random | Apache-2.0 | The first assert is kept as an example so models know the function name |
| `livecodebench` | HF `livecodebench/code_generation_lite` @ `0fe84c3` | `test6.jsonl` only (134 MB) | 175 (43 easy / 52 medium / 80 hard) | 120 | 30 easy / 40 medium / 50 hard | "cc" (unspecified on card) | release_v6 additions, contests 2025-01-04 to 2025-04-06. Full repo is 4.5 GB, so only v6 is downloaded. Hidden tests decoded with a pickle loader that refuses objects |
| `mmlu_pro` | HF `TIGER-Lab/MMLU-Pro` @ `b189ec7` | test parquet | 12,032 | 120 | equal per category (14 → 8–9 each) | MIT | Up to 10 options |
| `gpqa_diamond` | HF `Idavidrein/gpqa` @ `83022ce` | `gpqa_diamond.csv` | 198 | 50 | random | CC-BY-4.0 | **Gated** (auto-approved after accepting terms). Authors ask that examples are not published in plain text. Choices are shuffled per question with the seed, because the source lists the correct answer first |
| `arc_challenge` | HF `allenai/ai2_arc` @ `210d026` | ARC-Challenge test parquet | 1,172 | 30 | random | CC-BY-SA-4.0 | Easy control set |
| `math500` | HF `HuggingFaceH4/MATH-500` @ `6e4ed1a` | `test.jsonl` | 500 (43 / 90 / 105 / 128 / 134 per level) | 150 | 20 / 20 / 30 / 40 / 40 for levels 1–5 | none on card (upstream MATH is MIT) | Coarse difficulty: levels 1–2 easy, 3 medium, 4–5 hard |
| `ifeval` | HF `google/IFEval` @ `966cd89` | `ifeval_input_data.jsonl` | 541 | 80 | random | Apache-2.0 | `instruction_id_list` + `kwargs` drive the official checkers |
| `writingbench_business` | GitHub `X-PLUG/WritingBench` @ `ae2d517` | `benchmark_query/benchmark_all.jsonl` | 1,000 (189 English business/marketing) | 60 | 30 Finance & Business + 30 Advertising & Marketing | Apache-2.0 | Each query has its own 5-criterion rubric (`checklist`, 1–10 scale) for the LLM judge. Chosen over Arena-Hard v2, whose HF repo is 3.3 GB of mostly model answers and has no per-prompt rubric |

Seed: `20261012`. Difficulty split, including GPQA: easy 220 (26%), medium 450 (53%), hard 180 (21%). AIME was dropped (2026-10-12) because of its non-commercial licence. Its 30 prompts went to MATH-500, and MATH-500 and LiveCodeBench were weighted toward hard problems. The set is still medium-heavy compared with the planned 30/40/30, because MMLU-Pro, IFEval, WritingBench and three of the four BFCL subsets all count as medium.

## Prompt record

One JSON line per prompt, sorted by `prompt_id` (`<dataset>:<source_id>`):

- **Identity and source:** `dataset`, `category` (its folder), `source_dataset`, `source_revision`, `source_id`, `source_split`.
- **Classification:**
  - `task_family`: `tool_calling` | `coding` | `gk` | `math` | `instruction_following` | `business_writing`.
  - `topic`: a `TopicCategory` value. IFEval and WritingBench map to `creative_writing`. BFCL is `null` (decision 2026-10-12: tool use is the `has_tools` flag, not a fifth topic).
  - `level_label`: the benchmark's own level.
  - `difficulty`: `easy` | `medium` | `hard`.
  - `subject`: category, domain or platform, where the source has one.
- **Tools:** `has_tools` and `tools`, with BFCL function schemas kept as given.
- **Content:** `input` holds the structured question fields (question, choices, problem, messages and so on). `reference` holds the gold answer, tests, checker arguments or rubric. Rendering `input` into chat messages is the harness's job (#17).
- **Build info:** `seed`, `manifest_version`.

Dev/test splits and CV folds are not assigned here.

## Contamination

- MMLU-Pro (2024), MATH-500, HumanEval+, MBPP+, ARC, IFEval (2023) and BFCL are older than every model in the pool, so assume they are contaminated. They serve as stable, comparable anchors, not as measures of true capability.
- The LiveCodeBench v6 problems (Jan–Apr 2025) are the most recent. Even so, the 2026 models in the pool may have seen them. No newer LiveCodeBench release is on Hugging Face (repo last updated 2025-06).
- GPQA's gating limits, but does not rule out, contamination.

## Licence and terms findings (for a decision by the team, not legal advice)

- **AIME 2026 (MathArena, CC-BY-NC-SA-4.0) was dropped** because of its non-commercial licence (decision 2026-10-12).
- **GPQA:** fine for internal use, but don't commit its content or put it online in plain text. The builder keeps it in gitignored folders.
- **The LiveCodeBench licence is unspecified** ("cc" on the card). The problems come from LeetCode and AtCoder, so treat the data as internal-only.

**Using model outputs as router training data.** Checked 2026-10-12; only the items marked verified were read.

| Terms | Finding |
|---|---|
| OpenRouter Terms (verified, openrouter.ai/terms) | Output ownership follows each model's own terms (§6.1). The user must review each provider's terms (§5.1, §5.8). §7 forbids "reselling API access to Models or otherwise developing a competing service". **Lönn AI is itself an LLM router, so the team should check whether that clause applies to the product.** Nothing explicit about training on outputs |
| Anthropic Commercial Terms (verified via search) | Customers own their outputs, but may not use the services "to build a competing product or service, including to train competing AI models" |
| Gemini API Additional Terms (archived versions to Nov 2025, via search) | "You may not use the Services to develop models that compete with the Services" |
| DeepSeek Open Platform Terms (partly, via search) | Outputs may be integrated downstream. Reverse engineering of models is forbidden. No explicit clause on training was found |
| gpt-oss (OpenAI open weights, Apache-2.0), GLM (Z.ai), Muse Spark (Meta) | **Not checked** |

A router (classifier) trained on correctness labels is arguably not a "competing AI model" in the Anthropic and Google sense, but this needs a decision by the team, ideally with legal advice. Store the raw outputs either way (#17). The decision only affects how they are used later.
