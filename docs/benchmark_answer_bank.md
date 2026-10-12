# Benchmark answer bank

Issue #17. Every active model in `config/models.yaml` answers every benchmark question from #16 (850 questions, see `docs/benchmark_sources.md`). Each answer is saved with whether it was correct and what it cost. The point is to evaluate and train routing by **replaying** (question, model) pairs instead of calling the APIs again.

## Running

```bash
python3 -m pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt_tab')"   # once, for IFEval
python3 scripts/benchmark/build_data.py                 # prompt files (#16)

python3 scripts/benchmark/run.py --dry-run              # plan and cost estimate, no calls
python3 scripts/benchmark/run.py --limit 10 --budget 3  # pilot: the same 10 questions per dataset for every model
python3 scripts/benchmark/run.py                        # everything (resumes)
python3 scripts/benchmark/run.py --category math --model gpt-oss-20b
python3 scripts/benchmark/run.py --regrade              # regrade saved answers, no calls (not while a run is going)
python3 scripts/benchmark/run.py --summary              # accuracy and cost of saved answers (add fields, e.g. model_id dataset)
python3 scripts/benchmark/check_graders.py              # graders accept each benchmark's own reference answers
```

- **Tool calling (BFCL) is deferred** (decision 2026-10-12). Default runs, pilots and dry runs skip the four BFCL datasets (690 questions remain, 5,520 calls). Asking for them with `--dataset` or `--category` stops with an error unless `--include-tool-calling` is given. The templates and grader for BFCL are ready.
- **API key:** `OPENROUTER_API_KEY` comes from the environment or `.env`.
- **Direct calls:** requests go straight to OpenRouter's chat completions API, not through `src/lonn/executor.py`.
- **Defaults:**
  - `max_tokens` 8192, temperature 0 and seed 20261012.
  - 8 parallel calls and a 300-second timeout per call.
  - The models' default reasoning (the reasoning on/off lever is out of scope).
- **Order:** models run one after another, then datasets in order, with the questions of each (model, dataset) block answered in parallel.
- **Progress:** a line every 10 answers or 30 seconds, showing the current model, the dataset, the count done, accuracy so far, errors and spend.
- **Saving and resuming:** each answer is appended to its file as soon as it arrives. Rerunning the same command skips saved answers and retries saved errors. Ctrl-C waits for the calls in flight and saves them.
- **Budget:** `--budget` (default 45 USD) caps the **total recorded spend in the results folder, earlier runs included**. The run stops sending new calls once that spend, plus the worst case for the calls in flight, would pass the cap.
- **Retries:** rate limits, 5xx errors and timeouts are retried up to 3 times (5, 15 and 45 seconds, with jitter). If retries run out, an error row is saved, and that question is retried on the next run.
- **Stopping:** 401, 402 (no credits) and 403 stop the run.

## Prompts

`scripts/benchmark/templates.py` builds every request. Each task family has one system prompt, **identical for every model**, that states the exact output format:

| Family | Datasets | Expected output |
|---|---|---|
| multiple_choice | MMLU-Pro, GPQA, ARC | a final line `Answer: X` |
| math | MATH-500 | the final answer in `\boxed{}` |
| code | HumanEval+, MBPP+, LiveCodeBench | exactly one ```` ```python ```` block |
| tool_calling | BFCL × 4 | native tool calls (OpenRouter `tools`), or no call when nothing fits |
| instruction_following | IFEval | the response itself |
| writing | WritingBench | the requested text |

BFCL schemas are converted to JSON Schema (`dict` → `object`, `float` → `number`, …), and dots in function names become `_`, because tool names may not contain dots. **Change any wording and you must bump `TEMPLATE_VERSION`.** Rows store the template version, not the request text; the request is rebuilt from the prompt files with `build_request()`. That keeps the gated GPQA questions out of git.

## Results

`results/benchmark/<category>/<dataset>/<model>.jsonl`, committed to git. There is one line per (question, model), and a later line for the same pair replaces an earlier one.

| Field | Meaning |
|---|---|
| `prompt_id`, `dataset`, `category`, `difficulty`, `model_id` | which question and model (`prompt_id` is the short ID, e.g. `math500-017`) |
| `prompt_template_version`, `max_tokens`, `temperature`, `seed` | request settings |
| `response`, `reasoning`, `tool_calls` | the final text, the reasoning text if the provider returns it, and the tool calls (name plus JSON arguments) |
| `finish_reason`, `native_finish_reason` | `length` means the answer was cut off at `max_tokens` |
| `correct`, `extracted_answer`, `grade_details`, `grader`, `grader_version` | the grade: `correct` is `null` when ungraded (WritingBench, call errors) |
| `input_tokens`, `output_tokens`, `reasoning_tokens` | usage reported by OpenRouter |
| `cost_usd`, `cost_source` | OpenRouter's billed cost (`openrouter`), or configured prices × tokens if it was missing (`estimate`) |
| `latency_ms`, `retries`, `provider`, `generation_id`, `timestamp` | call details (`provider` is the upstream provider OpenRouter picked) |
| `error` | `{message, status}` when the call failed after retries |

```python
import sys; sys.path.insert(0, 'scripts')
from benchmark.results import ResultBank
bank = ResultBank()
row = bank.lookup('math500-017', 'openrouter/openai/gpt-oss-20b')
row['correct'], row['cost_usd'], row['latency_ms']
bank.summary(by=('model_id', 'category'))   # accuracy, cost, truncations per group
```

## Grading (`scripts/benchmark/graders.py`)

| Family | Rule |
|---|---|
| multiple_choice | the last `Answer: X` line must equal the key (`Final answer:`, `The answer:` and list or bold markers are accepted, but the label must end the line or be followed by a delimiter) |
| math | the last `\boxed{}` must be equivalent to the reference under `math-verify` (so `0.5` = `\frac12`) |
| code | the last Python block must pass the official tests: HumanEval+ `check()`, MBPP+ asserts, LiveCodeBench public + private tests. For HumanEval+, MBPP+ and LeetCode-style problems, the code is imported as `solution.py` (its `__main__` block does not run), and **a pass needs exit code 0 plus a random token printed after the last test**, so code that exits early never passes. A HumanEval+ answer with only the function body gets the given signature added. LiveCodeBench stdin programs run as `__main__`, and their output is compared line by line: integers exactly, decimals with 1e-6 tolerance. Limits: 6 s per test (actual time counts against the total) and 120 s per answer |
| tool_calling | the same number of calls as the ground truth, matched in any order. Each call needs the right name and only known arguments, each with an allowed value. Strings are compared case-insensitively, ignoring spaces and punctuation; `""` marks an optional argument. For `irrelevance`, correct means no call |
| instruction_following | Google's IFEval checkers (vendored in `scripts/benchmark/ifeval/`), strict and prompt-level: every instruction must be followed |
| writing | not graded yet (`correct: null`). An Opus 5.5 judge with the WritingBench rubric is deferred until the budget allows |

Before grading, `<think>…</think>` blocks inside the response are removed; an unfinished `<think>` counts as no answer. Grading runs **in the main thread, one answer at a time**: API calls stay parallel, but math-verify's 5-second timeouts only work there, and sandboxed tests can't slow each other into false timeouts. If grading crashes, the paid answer is still saved (`grader: error`), and `--regrade` grades it later.

**Sandbox:** generated code runs under macOS `sandbox-exec`.
- **Blocked:** network access, writes outside a temporary folder, and reading the home folder (so `.env` and tokens can't be read), except the virtualenv.
- **Environment:** a clean one, with no API keys.
- **Limits:** CPU time, no forking, and output written to files capped at 64 MB, so endless printing is cut off. Memory **cannot** be capped reliably on macOS.

`check_graders.py` confirms that every reference answer the data has is graded correct: BFCL, HumanEval+, MBPP+, multiple choice and MATH-500. LiveCodeBench and IFEval have no reference answers, so unit tests cover them.

## Cost

`--dry-run` estimates cost from configured prices. It uses measured output tokens where answers are already saved, and rough guesses otherwise; before the pilot the estimate is about 8.6–36.8 USD for all 6,800 calls. Pool prices were corrected against OpenRouter's live list on 2026-10-12. Spend tracking uses OpenRouter's billed `cost`, not those prices.
