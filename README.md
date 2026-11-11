<p align="center">
  <img src="assets/Lonnlogo.png" width="300" alt="Lonn AI">
</p>

<p align="center">
  <strong>Every prompt, routed to the cheapest model that's good enough.</strong><br>
  <a href="https://storaptit.github.io/lonn-ai-website/">Website</a> ·
  <a href="https://storaptit.github.io/lonn-ai-website/prestudy.html">Pre-study</a>
</p>

## Overview

Lönn AI is an LLM routing API. Instead of sending every request to one model,
Lönn looks at each prompt and sends it to the model that is good enough for the
task at the lowest cost. That might be a frontier model for hard reasoning or a
low-cost model for everyday work. One setting, λ (lambda), balances cost against
quality. Read more about the product on the
[Lönn AI website](https://storaptit.github.io/lonn-ai-website/).

**This repository is the prototype of the product.** It is a student project at
KTH, built to test whether routing really saves money without losing quality,
before a production service exists.

### How the prototype routes a request

1. **Classify:** a small, fast classifier (a frozen `bge-small` text embedder
   with a linear layer, running on CPU) estimates how likely the prompt is to
   be each of 15 task types.
2. **Score:** each model in the pool gets a score from its measured accuracy
   and real billed cost on those task types: `accuracy − λ · cost`.
3. **Route:** the request goes to the best-scoring model, through
   [LiteLLM](https://github.com/BerriAI/litellm) and OpenRouter.
4. **Check:** if the answer comes back empty, cut off, or fails a code example
   given in the prompt, the request is sent once more to another model, when
   that is worth its expected cost.

### Results so far

We measured the router on our own benchmark of 690 prompts from public sets
(MATH-500, HumanEval+, MMLU-Pro, IFEval and WritingBench), answered by a pool
of eight models. Routing decisions were replayed with cross-validation, so no
test prompt informed its own route. At λ = 20 the router answered 85.8%
correctly for $1.58, compared with 88.1% for $7.58 when always using Claude
Sonnet 5.5. At λ = 10 it beat Sonnet (89.6%) at about half the cost. These are
benchmark results, not real customer traffic. See
[the lambda sweep](results/router_eval/lambda-sweep-0-100-cascade/lambda_sweep.md)
and [the task router docs](docs/task_router.md).

The prototype includes a Streamlit app, an HTTP API (FastAPI) and a Python SDK,
all described below.

## Repository layout

```text
src/lonn/                     # Runtime code
    task_classifier.py        # Prompt → probability per task type (bge-small + linear layer)
    task_types.py             # Load the task-type taxonomy
    routing_table.py          # Measured accuracy and cost per (task type, difficulty, model)
    task_router.py            # Rank models by accuracy − λ · cost; pick an escalation target
    executor.py               # Call the chosen model through LiteLLM
    answer_check.py           # Check an answer: cut off, empty, or failing the prompt's code examples
    code_sandbox.py           # Run generated code in a macOS sandbox (used by the answer check)
    model_pool.py             # Load and validate config/models.yaml
    api.py                    # HTTP API (FastAPI)
    schemas.py                # Shared data contracts
    complexity.py, router.py  # Original Laya-based pipeline, kept for benchmark comparisons
config/
    models.yaml               # The curated model pool (the only hand-edited model config)
    task_types.yaml           # The 15 task types
    routing_table.json        # Generated from the answer bank, don't hand-edit
    task_classifier.json      # Generated classifier weights, don't hand-edit
examples/app.py               # Streamlit prototype app
sdk/python/                   # Python SDK (package lonnrouter)
scripts/benchmark/            # Benchmark data, answer bank, grading, routing table, evaluation
scripts/                      # Classifier training data and training, pool discovery and checks
results/                      # Committed benchmark answers, evaluations and bake-offs
data/                         # Manifests for locally built data (the data itself is gitignored)
docs/                         # Architecture, router, benchmark and classifier docs
legacy/                       # Old Laya scorer tooling: demos, fine-tuning, labels and docs
tests/                        # Automated tests
reports/                      # Generated reports (created on demand, gitignored)
```

## Setup

Requires Python 3.10 or newer.

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

   This installs the runtime, the Streamlit app, the HTTP API and the Python SDK
   (in editable mode, from `sdk/python`).

2. Copy `.env.example` to `.env` if `.env` does not already exist.
3. Add your OpenRouter API key (https://openrouter.ai/keys) as
   `OPENROUTER_API_KEY=sk-or-...`. The active pool runs on OpenRouter. The
   inactive `nvidia_nim/...` entries need `NVIDIA_NIM_API_KEY=nvapi-...`
   (https://build.nvidia.com) if you enable them. Only the key for the provider
   actually called is required.
4. `.env` is gitignored. Never commit it.

The first automatic classification downloads the embedding model
(`BAAI/bge-small-en-v1.5`, about 130 MB) from Hugging Face. After that it runs
locally on the CPU.

## Run the prototype web app

From the repository root, with `OPENROUTER_API_KEY` set in `.env`:

```bash
source .venv/bin/activate
python3 -m streamlit run examples/app.py --server.address localhost
```

Open http://localhost:8501. In the sidebar:

- **Cost weight (lambda)** sets λ. 0 picks the best expected accuracy; higher
  values pick cheaper models. The default, 20, is the benchmarked setting.
- **Prompt analysis** is either **Automatic (classifier)** or **Manual task
  type**. Manual mode skips the classifier, which is handy for a quick test.
- **Output token budget** and **Request timeout** default to 4,096 tokens and
  120 seconds.

**Analyze & route** shows the task-type probabilities and the ranked models,
with expected accuracy, expected cost per request and score, without calling a
provider. **Route & call model** also sends the prompt, falling back to the next
model if a call fails, and shows the response, token usage, latency and cost.
The app does not run the answer check; the HTTP API does.

Reasoning models can use up the output budget before writing a final answer.
The app reports this as truncation; increase the budget and try again.

## HTTP API

Start the API from the repository root:

```bash
source .venv/bin/activate
python3 -m uvicorn lonn.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Interactive documentation is at http://127.0.0.1:8000/docs. The API has no
authentication, so keep it on loopback. To call it from another device, see
[network access and SSH tunnel instructions](docs/Prototype_architecture.md#9-calling-from-another-device--new).

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health; does not load the classifier or call providers |
| `GET /models` | The active model pool and its metadata |
| `POST /route` | Classify the prompt and rank up to five models, without a provider call |
| `POST /generate` | Route, call the best model, check the answer, and escalate once if needed |

Both `POST` endpoints take:

- `prompt` (required);
- `lam`, the cost weight λ (default 20);
- `signal` (optional), your own `task_scores` (probabilities per task type that
  sum to 1) and an optional `complexity` in [0, 1]. Supplying a signal skips the
  classifier. Task types are listed in [config/task_types.yaml](config/task_types.yaml).

`/generate` also takes `max_tokens` (default 4096), `timeout` (provider timeout
in seconds, default 120, maximum 600) and `verify` (default `true`).

Routing only, with a supplied signal:

```bash
curl http://127.0.0.1:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a sorting function","lam":20,"signal":{"task_scores":{"code_function":1.0}}}'
```

The response holds `models` (best first), `estimates` (expected accuracy,
expected cost in USD and score per candidate), `signal`, `lam`,
`analysis_source` (`classifier` or `provided`) and `routing_latency_ms`.

End-to-end call from Python:

```python
import httpx

result = httpx.post(
    "http://127.0.0.1:8000/generate",
    json={"prompt": "What is 17 * 23?", "lam": 20, "max_tokens": 4096, "timeout": 120},
    timeout=300,  # Also allow time for classification and a possible escalation.
)
result.raise_for_status()
data = result.json()
print(data["model"]["id"], data["response"]["content"])
print(data["escalated"], data["total_cost_usd"], data["total_latency_ms"])
```

How `/generate` works:

1. It tries the ranked models in order until one call succeeds, skipping models
   whose context window can't fit `max_tokens`.
2. With `verify` on, it checks the answer: cut off (`finish_reason` is
   `length`), empty, or code that fails the worked examples in the prompt. The
   code check runs in a `sandbox-exec` sandbox and is skipped on systems other
   than macOS.
3. If the check fails and another model is worth its expected cost at this λ,
   that model answers instead. This happens at most once.

The response adds `response` (content, model, token usage, cost, latency and
`finish_reason`), `attempts` (each model call with its check result, cost and
latency), `escalated`, `total_cost_usd` and `total_latency_ms`. Costs are
token counts times the configured prices, not provider invoices.

Errors: invalid requests return 422, analysis or routing failures 503,
provider rate limits 429, provider timeouts 504, and other provider failures
502. Error bodies leave out raw provider details.

The pool and the routing table load at startup, so restart the server after
editing `config/models.yaml` or rebuilding the table. The classifier loads on
first use; send one `/route` request before measuring steady-state latency.
Classifier calls are serialized within a server process, while provider calls
can overlap. This is a simple JSON API, not an OpenAI-compatible endpoint.

## Python SDK

`requirements.txt` already installs the SDK. For a client-only environment,
install the lightweight package from this checkout:

```bash
python -m pip install ./sdk/python
```

```python
from lonnrouter import LonnRouter

with LonnRouter(base_url="http://127.0.0.1:8000") as router:
    result = router.generate("Explain binary search", lam=20)
    print(result.content)
    print(result.model.id, result.total_cost_usd, result.escalated)
```

You can set `LONN_BASE_URL` instead and use `LonnRouter()`. The SDK only calls
the HTTP API; it does not start the server, and it depends only on HTTPX and
Pydantic. Use `router.route(...)` to rank models without calling one, and
`AsyncLonnRouter` in async code. See the [SDK guide](sdk/python/README.md) for
more. The package is not on PyPI yet.

## Using the router in Python

With `src` on the import path (`PYTHONPATH=src`), the pipeline is three calls:

```python
from lonn.executor import execute
from lonn.model_pool import load_model_pool
from lonn.routing_table import RoutingTable
from lonn.task_classifier import TaskClassifier
from lonn.task_router import rank_models

models = load_model_pool()
table = RoutingTable.load()
signal = TaskClassifier().classify(prompt)              # probability per task type
ranked = rank_models(signal, table, models, lam=20)     # best first
response = execute(prompt, ranked[0], timeout=120, max_tokens=4096)
print(response.content, response.usage.total_cost, response.finish_reason)
```

The router never calls an API, and models missing from the routing table are
not ranked. The executor loads the repository `.env` without overriding
existing environment variables and needs only the key for the chosen model's
provider. It makes no retries; provider errors reach the caller. See
[docs/task_router.md](docs/task_router.md) for the routing method.

## Benchmark and evaluation

The routing table comes from an answer bank: every pool model answers every
benchmark prompt, and each answer is graded and its billed cost recorded.

```bash
python3 scripts/benchmark/build_data.py               # build the prompt sets (gitignored data/benchmark/)
python3 scripts/benchmark/run.py --dry-run            # answer bank; paid calls, so dry-run first
python3 scripts/benchmark/build_routing_table.py      # rebuild config/routing_table.json
python3 scripts/benchmark/answer_checks.py            # run the answer check on every saved answer
python3 scripts/benchmark/eval_task_router.py --name task-true-cascade --cascade   # cross-fitted replay, no calls
```

Router evaluations are cross-fitted: the table used to route each prompt is
built without that prompt's answers. Read more in
[benchmark sources](docs/benchmark_sources.md),
[the answer bank](docs/benchmark_answer_bank.md),
[router evaluation](docs/router_evaluation.md) and
[the oracle gap](docs/oracle_gap.md).

The task classifier is trained on prompts streamed from Hugging Face, separate
from the benchmark prompts:

```bash
python3 scripts/build_task_data.py --include-gpqa    # build the training data
python3 scripts/classifier_bakeoff.py                # compare classifiers (about 45 min)
python3 scripts/train_task_classifier.py             # write config/task_classifier.json
```

See [task types](docs/task_types.md), [task data](docs/task_data.md) and
[the classifier bake-off](docs/classifier_bakeoff.md).

## Model pool

`config/models.yaml` is the only curated model configuration. Eight OpenRouter
models are active: Claude Sonnet 5.5, Gemini 3.8 Flash, Muse Spark 1.3,
DeepSeek V4 Pro, DeepSeek V4.1 Flash, GLM 5.3 Flash, gpt-oss-120b and
gpt-oss-20b. Each entry needs a provider (`openrouter` or `nvidia_nim`), a full
ID such as `openrouter/openai/gpt-oss-120b`, prices, a `price_basis`, a context
window and four topic aptitudes. The loader rejects unknown values instead of
treating them as zero. Record the source of any metadata you add; see
[model metadata](docs/model_metadata.md).

Two offline scripts help maintain the pool. They write only to `reports/` and
never change `config/models.yaml`:

```bash
python3 scripts/discover_models.py                                     # catalog → reports/model_catalog.json
python3 scripts/check_models.py --model nvidia_nim/openai/gpt-oss-20b  # live availability check
```

A successful check doesn't mean the model supports tool calls or answers well.

## Legacy: the Laya complexity scorer

The first prototype routed on four topics and a complexity score from
[Laya](https://pypi.org/project/laya/) (`src/lonn/complexity.py` and
`src/lonn/router.py`). The app and the API no longer use it, but the benchmark
evaluation still compares against it, so it stays in `src/`. Its demos,
fine-tuning scripts, labels and docs are in `legacy/`; start with
[legacy/docs/fine_tuned_laya.md](legacy/docs/fine_tuned_laya.md).

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests mock provider calls and inject a fake classifier and a fake Laya, so
they need no credentials or network access. The one live Laya test is skipped
when `laya` is not installed. `tests/try_sdk.py` is a manual smoke script
against a running server, not a unit test.

## Further reading

- [Prototype architecture](docs/Prototype_architecture.md): contracts and module boundaries
- [Task router](docs/task_router.md): how the router decides, and its results
- [Lambda sweep](results/router_eval/lambda-sweep-0-100-cascade/lambda_sweep.md): accuracy and cost for λ from 0 to 100
