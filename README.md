<p align="center">
  <img src="assets/Lonnlogo.png" width="300" alt="Lonn AI">
</p>


AI infrastructure for intelligent LLM routing, cost optimisation and observability.

Prototype 4.0 uses 15 task types: eight benchmark-measured model scores, six
user-supplied estimates, and `other`, derived from the other 14 scores. The
estimates are not benchmark measurements. See [Prototype 4.0](docs/prototype4.md)
for provenance, reproduction steps, and the comparison with Prototype 3.0.

## Repository layout

```text
config/models.yaml          # Model pool, prices and legacy topic scores
config/routing_table.json   # Benchmark-measured task scores and costs
config/task_aptitudes.yaml  # User-supplied estimates for six task types
src/lonn/                   # Runtime code
    model_pool.py           # Load and validate configuration into Model objects
    task_classifier.py      # Classify prompts into 15 task types
    routing_table.py        # Combine measured, supplied and derived scores
    task_router.py          # Select a model
    executor.py             # Call the selected model
    schemas.py              # Shared data contracts
scripts/discover_models.py  # Optional catalog discovery
scripts/check_models.py     # Optional live text-availability checks
scripts/finetune_laya.py    # Fine-tune Laya's decision head on labelled prompts
scripts/push_laya.py        # Upload a fine-tuned checkpoint to Hugging Face
examples/                   # App and usage examples
tests/                      # Automated tests
docs/                       # Architecture
reports/                    # Generated reports (created on demand, gitignored)
```

## Setup

Requires Python 3.10 or newer.

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` if `.env` does not already exist.
3. Fill in your OpenRouter API key (https://openrouter.ai/keys) as
   `OPENROUTER_API_KEY=sk-or-...`. The active pool runs on OpenRouter. The
   inactive `nvidia_nim/...` entries need `NVIDIA_NIM_API_KEY=nvapi-...`
   (https://build.nvidia.com) if you enable them. Only the key for the provider
   actually called is required.
4. `.env` is gitignored — never commit it.

## Run the prototype web app

From the repository root, after installing `requirements.txt` and setting
`OPENROUTER_API_KEY` in `.env`:

```bash
source .venv/bin/activate
python3 -m streamlit run examples/app.py --server.address localhost
```

Open http://localhost:8501. Choose **Manual task type** for a quick test without
the classifier, or **Automatic (classifier)** for the full analysis flow. First automatic use
may download classifier weights. **Analyze & route** displays routing without
calling a provider; **Route & call model** sends the prompt to the chosen model.
Adjust lambda, output token budget, and timeout in the sidebar. The app displays
task probabilities, each model's 15 scores and their provenance, selected model,
response, token usage, latency, and estimated cost. Settings changes do not repeat
a submitted API request.

Reasoning models may exhaust the output budget before producing final text.
The app reports truncation; increase the budget and retry if needed. The default
is 4,096 tokens and a 120-second timeout. API availability and account access
still require a live test. The UI keeps the API key on the Python server.

## Python SDK for apps and benchmarks

Installing the root `requirements.txt` also installs this SDK in editable mode.
For a client-only environment, install the lightweight package from this
checkout:

```bash
python -m pip install ./sdk/python
```

```python
from lonnrouter import LonnRouter

with LonnRouter(base_url="http://192.168.1.42:8000") as router:
    result = router.generate("Explain binary search", lam=20)
    print(result.content)
    print(result.model.id, result.usage.total_cost)
```

Replace the example address with your running Lonn server. You can instead set
`LONN_BASE_URL` and use `LonnRouter()`. The SDK calls the HTTP API; it does not
start the server. Only the server needs the task classifier, the selected provider's
credentials, and the model configuration. The client package depends only on
HTTPX and Pydantic.

Use `router.route(...)` for selection without generation, or `AsyncLonnRouter`
with `await router.generate(...)` inside an async app. Results expose typed
attributes and `model_dump()` for saving benchmark records. The client reuses
connections and performs no automatic retries.

See the [SDK guide](sdk/python/README.md) for async examples, installation,
timeouts, supplied signals, and error handling. This package is not yet published
to PyPI; install from the checkout or a built wheel.

## Benchmark HTTP API

Install `requirements.txt`, then start FastAPI from the repository root:

```bash
source .venv/bin/activate
python3 -m uvicorn lonn.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Interactive request/response documentation: http://127.0.0.1:8000/docs.
This API has no authentication; loopback is the default for local use. For a
benchmark runner on another device, see
[network access and SSH tunnel instructions](docs/Prototype_architecture.md#9-calling-from-another-device--new).

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health; does not check classifier or provider availability |
| `GET /models` | Active model pool and 15 task aptitudes with their sources |
| `POST /route` | Analyze a prompt and return up to five ranked models without a provider call |
| `POST /generate` | Analyze, then try up to five ranked models until one succeeds |

Selection-only example, supplying a signal to skip the classifier:

```bash
curl http://127.0.0.1:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a sorting function","lam":20,"signal":{"task_scores":{"code_function":1.0}}}'
```

Omit `signal` to analyze the prompt automatically. `task_scores` contains task
probabilities in [0, 1] that sum to one. `lam` defaults to 20. Complexity is
off by default. The router maximizes the probability-weighted task score minus
`lam` times the expected request cost. Supplied scores are estimates rather than
measured pass rates. `/route` includes ranked `models`, `estimates`, `signal`,
`lam`, `analysis_source`, and `routing_latency_ms`. Each model's `task_aptitudes`
identifies measured, supplied, and derived scores; the older `topic_aptitudes`
are retained only for compatibility with the original router.

End-to-end benchmark call from Python:

```python
import httpx

result = httpx.post(
    "http://127.0.0.1:8000/generate",
    json={"prompt": "What is 17 * 23?", "lam": 20,
          "max_tokens": 4096, "timeout": 120},
    timeout=300,  # Also allow time for local prompt analysis.
)
result.raise_for_status()
data = result.json()
print(data["model"]["id"], data["response"]["content"])
print(data["response"]["usage"], data["total_latency_ms"])
```

`/generate` accepts the same optional `signal`. It additionally returns `response`
(the existing `ModelResponse`: content, model, usage, provider latency, and
finish reason) and `total_latency_ms` for server-side analysis, routing, and
execution. Neither timing includes HTTP transport; measure client elapsed time
separately. Costs are configured estimates. Preserve `finish_reason` in benchmark
results: `length` indicates truncation, potentially with empty final text.

The output budget defaults to 4096 tokens and must be below a candidate model's
context window. Prompt token counting is not yet implemented. Provider timeout
defaults to 120 seconds (maximum 600); it does not limit classifier time.
Execution requires the API key for the selected model's provider in the server
environment or repository `.env`: `OPENROUTER_API_KEY` for `openrouter/...`
models, or `NVIDIA_NIM_API_KEY` for `nvidia_nim/...` models.
`/generate` attempts up to five ranked candidates, skipping candidates whose
context window cannot fit the requested output budget. If all attempts fail,
the API returns a sanitized error. Invalid requests return 422;
analysis/routing failures 503; provider rate limits 429; provider timeouts 504;
other execution failures 502. Error bodies omit raw provider details.

The pool is loaded and validated at startup; restart after changing configuration.
The classifier loads on the first automatic request and is reused. Warm it up with an
automatic `/route` call before measuring steady-state latency. Classifier calls
are serialized within each server process; provider calls can overlap. Each
additional Uvicorn worker loads its own classifier. This is a simple JSON API,
not an OpenAI chat-completions endpoint.

## Legacy complexity scoring

This older Laya scorer is retained for experiments; the API and app use the
task-type classifier and router described above.

The complexity scorer uses [Laya](https://pypi.org/project/laya/) to produce a
`ComplexityResult`: a topic probability distribution over `coding`,
`math_reasoning`, `creative_writing`, and `general_knowledge_extraction`, plus a
complexity score in [0, 1]: Laya's probability that the task is hard, from a
yes/no question. Both come from one Laya call. Laya is loaded on first use.

```python
from lonn.complexity import ComplexityScorer

result = ComplexityScorer().score_prompt(prompt)
print(result.complexity, result.topic_scores)
```

Two scripts in `examples/` let you try the scorer with the real Laya model:

- `scorer_demo.py` loads Laya once, then scores each prompt you type and shows
  how long it took (add `--debug` to see Laya's raw output):

  ```bash
  python3 examples/scorer_demo.py
  ```

- `scorer_benchmark.py` measures load time and per-prompt speed on your machine:

  ```bash
  python3 examples/scorer_benchmark.py
  ```

Prompts longer than Laya's window (about 370 tokens for English, after the
questions) are shortened to their beginning and end, where the actual request
usually is.

### Fine-tuned Laya checkpoint

The team's fine-tuned model is `LonnAI/laya-lonn` (private). See
[docs/fine_tuned_laya.md](docs/fine_tuned_laya.md) for access, setup, what it
changes, and its results.

To use a fine-tuned checkpoint for English prompts, set `LONN_LAYA_MODEL` in
`.env` to its Hugging Face repo id (or a local directory). For a private repo,
run `hf auth login` once with an account that has access. Unset, the stock
`convaiinnovations/laya` checkpoint is used. Non-English prompts always go to
Laya's multilingual checkpoint.

To train one from labelled prompts (format and rules in
`docs/labeling_guide.md`):

```bash
python3 scripts/finetune_laya.py --data data/labels/pilot.jsonl --out models/laya-lonn
```

Only the decision head is trained; the encoder stays frozen. A stratified 20%
of prompts is held out, and the script prints base versus fine-tuned topic
accuracy and how well complexity ranks the labelled levels (Spearman and AUC,
next to prompt length alone as a baseline). The output directory (gitignored) also
holds `metrics.json` and a model card. Upload it to a private repo with:

```bash
hf auth login
python3 scripts/push_laya.py --repo <hf-user>/laya-lonn
```

The checkpoint is trained for the exact question wording in
`src/lonn/complexity.py`; retrain after changing it.

## Execution

With `src` on Python's import path (`PYTHONPATH=src`), pass the router's
selected `Model` to the executor:

```python
from lonn.executor import ModelExecutor
from lonn.schemas import ModelResponse

response: ModelResponse = ModelExecutor().execute(
    prompt, selected_model, timeout=120, max_tokens=4096
)
print(response.content)
print(response.usage.total_cost, response.latency_ms)
```

The executor loads the repository `.env` without overriding existing environment
variables. The selected model ID must be the full `nvidia_nim/...` or `openrouter/...` ID
from `config/models.yaml`, and the matching key (`NVIDIA_NIM_API_KEY` or
`OPENROUTER_API_KEY`) must be set. Costs are computed from configured
per-million prices, not from provider billing. `finish_reason` exposes truncation.
Provider errors propagate to the caller without rerouting or automatic retries.

The pool loader and local Streamlit app are implemented. The active task scores
and their provenance are documented in [Prototype 4.0](docs/prototype4.md).

## NVIDIA model pool

Discover the catalog without excluding models based on third-party pricing:

```bash
python3 scripts/discover_models.py
```

This writes `reports/model_catalog.json`, including unpriced models.
It is a candidate inventory, not a guarantee of text-generation availability.
Test a specific candidate with a small live request:

```bash
python3 scripts/check_models.py --model nvidia_nim/openai/gpt-oss-20b
```

With no arguments the checker probes active entries in `config/models.yaml`.
`--catalog reports/model_catalog.json` probes the entire discovered
inventory, including non-chat models; it can make many API requests. The checker
uses a 120-second timeout and at most two attempts per model, retrying transient
failures. Results go to `reports/model_availability.json`; the pool is never
rewritten. It distinguishes timeouts, rate limits, access failures, and 404s.
A successful text response does not imply tool-call support or high aptitude.

Maintain `config/models.yaml` with active models, documented context limits, and
the legacy four-topic scores. `price_basis` must identify `provider`, `free_tier`,
or `estimate`. The active OpenRouter prices and inactive NIM estimates have
different sources; see [model metadata](docs/model_metadata.md). To use the
current task router:

```python
from lonn.model_pool import load_model_pool
from lonn.routing_table import load_runtime_table
from lonn.schemas import TaskSignal
from lonn.task_router import rank_models

models = load_model_pool()
table = load_runtime_table()
selected_model = rank_models(TaskSignal({"code_function": 1.0}), table, models)[0]
```

Discovery and probes are offline preparation steps, not part of each routed
request. Generated reports are ignored by Git; supplied task scores are maintained
separately in `config/task_aptitudes.yaml`.
See [the architecture and branch changes](docs/Prototype_architecture.md) for
the responsibility boundaries, HTTP request flow, and highlighted FastAPI additions.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Executor and app tests mock provider calls and require no credentials or network access.
Complexity tests use a fake Laya; the one live Laya test is skipped when `laya`
is not installed.
